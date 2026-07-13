#include "source_epoch.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>

namespace stereo3d {
namespace {

uint32_t randomSourceEpoch() {
    std::random_device rd;
    uint32_t epoch = rd() ^ uint32_t(::getpid()) ^
        uint32_t(std::chrono::system_clock::now().time_since_epoch().count());
    return epoch == 0 ? 1 : epoch;
}

uint32_t existingSourceEpoch(const std::filesystem::path& path) {
    std::ifstream input(path);
    uint64_t value = 0;
    if (!(input >> value) || value == 0 ||
        value > std::numeric_limits<uint32_t>::max()) {
        return 0;
    }
    return static_cast<uint32_t>(value);
}

}  // namespace

uint32_t createSourceEpoch(const std::string& path) {
    if (path.empty()) {
        throw std::runtime_error("NX source_epoch_file must not be empty");
    }

    const std::filesystem::path epoch_path(path);
    const auto parent = epoch_path.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error && !std::filesystem::exists(parent)) {
            throw std::runtime_error(
                "cannot create source epoch directory " + parent.string() +
                ": " + error.message());
        }
    }

    const uint32_t previous_epoch = existingSourceEpoch(epoch_path);
    uint32_t epoch = randomSourceEpoch();
    if (epoch == previous_epoch) {
        epoch = epoch == std::numeric_limits<uint32_t>::max() ? 1 : epoch + 1;
    }

    const std::filesystem::path temporary_path =
        epoch_path.string() + ".tmp." + std::to_string(::getpid());
    const int fd = ::open(
        temporary_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("cannot create NX source epoch file " + path);
    }

    const std::string content = std::to_string(epoch) + "\n";
    const ssize_t written = ::write(fd, content.data(), content.size());
    const int sync_result = written == static_cast<ssize_t>(content.size())
        ? ::fsync(fd) : -1;
    const int close_result = ::close(fd);
    if (written != static_cast<ssize_t>(content.size()) ||
        sync_result != 0 || close_result != 0) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw std::runtime_error("cannot persist NX source epoch to " + path);
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, epoch_path, rename_error);
    if (rename_error) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw std::runtime_error(
            "cannot atomically publish NX source epoch to " + path +
            ": " + rename_error.message());
    }
    return epoch;
}

}  // namespace stereo3d
