#include "volleyball/adapters/ros2/source_epoch.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("nx_source_epoch_test_" + std::to_string(::getpid()));
    const std::filesystem::path path = directory / "nx_source_epoch";

    const std::uint32_t first =
        volleyball::adapters::ros2::createSourceEpoch(path.string());
    std::ifstream first_input(path);
    std::uint32_t first_on_disk = 0;
    first_input >> first_on_disk;
    check(first != 0, "first source epoch must never be zero");
    check(first_on_disk == first,
          "first source epoch must be persisted atomically");

    const std::uint32_t second =
        volleyball::adapters::ros2::createSourceEpoch(path.string());
    std::ifstream second_input(path);
    std::uint32_t second_on_disk = 0;
    second_input >> second_on_disk;
    check(second != 0, "second source epoch must never be zero");
    check(second != first, "process restart must generate a new source epoch");
    check(second_on_disk == second,
          "second source epoch must replace the persisted value");

    std::filesystem::remove_all(directory);
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "source epoch tests passed\n";
    return EXIT_SUCCESS;
}
