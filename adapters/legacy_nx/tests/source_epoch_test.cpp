#include "../src/ros/source_epoch.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("nx_source_epoch_test_" + std::to_string(::getpid()));
    const std::filesystem::path path = directory / "nx_source_epoch";

    const uint32_t first = stereo3d::createSourceEpoch(path.string());
    std::ifstream first_input(path);
    uint32_t first_on_disk = 0;
    first_input >> first_on_disk;
    assert(first != 0);
    assert(first_on_disk == first);

    const uint32_t second = stereo3d::createSourceEpoch(path.string());
    std::ifstream second_input(path);
    uint32_t second_on_disk = 0;
    second_input >> second_on_disk;
    assert(second != 0);
    assert(second != first);
    assert(second_on_disk == second);

    std::filesystem::remove_all(directory);
    return 0;
}
