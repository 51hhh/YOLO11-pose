#include "assembly.hpp"
#include "runtime_config.hpp"

#include "volleyball/services/stereo_pipeline_runner.hpp"

#include <rclcpp/rclcpp.hpp>

#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace app = volleyball::apps::nx_mainline;
namespace services = volleyball::services;

namespace {

volatile std::sig_atomic_t stop_requested = 0;

extern "C" void requestStop(int) {
    stop_requested = 1;
}

const char* statusName(services::RunStatus status) noexcept {
    switch (status) {
    case services::RunStatus::completed:
        return "completed";
    case services::RunStatus::stopped:
        return "stopped";
    case services::RunStatus::already_finished:
        return "already_finished";
    case services::RunStatus::source_timeout:
        return "source_timeout";
    case services::RunStatus::source_error:
        return "source_error";
    case services::RunStatus::source_protocol_error:
        return "source_protocol_error";
    case services::RunStatus::processing_error:
        return "processing_error";
    case services::RunStatus::finalize_error:
        return "finalize_error";
    case services::RunStatus::camera_start_error:
        return "camera_start_error";
    case services::RunStatus::trigger_start_error:
        return "trigger_start_error";
    case services::RunStatus::lifecycle_error:
        return "lifecycle_error";
    }
    return "unknown";
}

void printUsage(const char* executable) {
    std::cerr << "usage: " << executable << " --config ABSOLUTE_YAML_PATH\n";
}

class RosContext final {
public:
    RosContext() = default;
    ~RosContext() {
        if (initialized_ && rclcpp::ok()) rclcpp::shutdown();
    }

    RosContext(const RosContext&) = delete;
    RosContext& operator=(const RosContext&) = delete;

    void initialize(const char* executable) {
        int argc = 1;
        const char* argv[] = {executable};
        rclcpp::init(argc, argv);
        initialized_ = true;
    }

private:
    bool initialized_{false};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string{argv[1]} != "--config" ||
        std::string{argv[2]}.empty()) {
        printUsage(argv[0]);
        return 2;
    }

    app::MainlineRuntimeConfig config;
    std::string error;
    if (!app::loadRuntimeConfig(
            std::filesystem::path{argv[2]}, config, error)) {
        std::cerr << "mainline configuration rejected: " << error << '\n';
        return 2;
    }

    RosContext ros;
    try {
        if (config.publisher.enabled) ros.initialize(argv[0]);
        if (std::signal(SIGINT, requestStop) == SIG_ERR ||
            std::signal(SIGTERM, requestStop) == SIG_ERR) {
            throw std::runtime_error("cannot install process signal handlers");
        }
        const services::RunSummary summary = app::runMainlineApplication(
            config, [] { return stop_requested != 0; });
        std::cout << "mainline status=" << statusName(summary.status)
                  << " profile=" << config.profile
                  << " frames=" << summary.frames_received
                  << " observations=" << summary.observations
                  << " no_matches=" << summary.no_matches
                  << " timeouts=" << summary.timeouts
                  << " dropped=" << summary.frames_dropped
                  << " queue_high_watermark="
                  << summary.queue_high_watermark;
        if (!summary.message.empty()) {
            std::cout << " message=" << summary.message;
        }
        std::cout << '\n';
        if (summary.status == services::RunStatus::completed) return 0;
        if (summary.status == services::RunStatus::stopped) {
            return config.runtime.mode == "runtime" ? 0 : 130;
        }
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "mainline startup/runtime failure: "
                  << exception.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "mainline startup/runtime failure: unknown exception\n";
        return 1;
    }
}
