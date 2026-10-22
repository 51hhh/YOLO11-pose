#include "main_cli_options.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

MainCliOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    return parseMainCliOptions(static_cast<int>(argv.size()), argv.data());
}

}  // namespace

int main() {
    {
        const auto options = parse({"stereo_pipeline"});
        check(!options.should_exit, "default invocation must remain runnable");
        check(options.config_path == "config/base_pipeline.yaml",
              "default invocation must use the compatibility template path");
    }

    {
        const auto options = parse({
            "stereo_pipeline", "--config", "/tmp/generated.yaml",
            "--visualize"});
        check(!options.should_exit, "supported runtime options must parse");
        check(options.config_path == "/tmp/generated.yaml",
              "explicit config path must be preserved");
        check(options.enable_display, "visualization flag must be preserved");
    }

    for (const char* retired : {
             "--recording-out",
             "--record-baseline-clip",
             "--debug-realtime-dump",
             "--debug-feature-matches",
             "--debug-feature-matches-dir",
             "--display",
             "--visualizels",
         }) {
        const auto options = parse({"stereo_pipeline", retired});
        check(options.should_exit && options.exit_code != 0,
              "retired direct-write options must be rejected");
    }

    {
        const auto options = parse({"stereo_pipeline", "--config"});
        check(options.should_exit && options.exit_code != 0,
              "missing config value must fail closed");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "main CLI option tests passed\n";
    return EXIT_SUCCESS;
}
