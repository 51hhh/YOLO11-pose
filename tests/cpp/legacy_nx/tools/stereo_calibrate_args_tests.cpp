#include "tools/stereo_calibrate/args.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

bool parse(std::vector<std::string> arguments,
           StereoCalibrateArgs& args) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parseStereoCalibrateArgs(
        static_cast<int>(argv.size()), argv.data(), args);
}

std::filesystem::path uniqueTemporaryPath() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    return std::filesystem::temp_directory_path() /
           ("stereo_calibrate_args_tests_" + std::to_string(suffix));
}

}  // namespace

int main() {
    {
        StereoCalibrateArgs args;
        check(parse({"stereo_calibrate"}, args),
              "default arguments must parse");
        check(!validateStereoCalibrateArgs(args),
              "square size must be explicitly provided");
    }

    {
        StereoCalibrateArgs args;
        check(parse({"stereo_calibrate", "--square-size", "26.0",
                     "--images-dir", "images", "--output", "candidate.yaml",
                     "--board-w", "8", "--board-h", "5", "--jobs", "4",
                     "--exhaustive"},
                    args),
              "explicit calibration arguments must parse");
        check(args.square_size == 26.0f && args.images_dir == "images" &&
                  args.output == "candidate.yaml" && args.board_w == 8 &&
                  args.board_h == 5 && args.jobs == 4 && args.use_sb &&
                  args.use_exhaustive,
              "explicit calibration arguments must be preserved and normalized");
        check(validateStereoCalibrateArgs(args),
              "explicit valid calibration arguments must pass validation");
    }

    for (const auto& invalid : {
             std::vector<std::string>{"calibrate", "--square-size", "26mm"},
             std::vector<std::string>{"calibrate", "--square-size", "nan"},
             std::vector<std::string>{"calibrate", "--board-w", "5x"},
             std::vector<std::string>{"calibrate", "--jobs", "many"},
             std::vector<std::string>{"calibrate", "--gpu-preprocess"},
         }) {
        StereoCalibrateArgs args;
        check(!parse(invalid, args),
              "malformed numeric arguments must fail closed");
    }

    {
        StereoCalibrateArgs args;
        check(parse({"calibrate", "--square-size", "26", "--board-h", "1"},
                    args),
              "board range validation must be separate from syntax parsing");
        check(!validateStereoCalibrateArgs(args),
              "a chessboard dimension below two must be rejected");
    }

    {
        StereoCalibrateArgs args;
        check(parse({"calibrate", "--help"}, args) && args.help_requested,
              "help must be a successful non-calibration action");
        check(validateStereoCalibrateArgs(args),
              "help must not require a square size");
    }

    const std::filesystem::path root = uniqueTemporaryPath();
    const std::filesystem::path candidate = root / "candidate.yaml";
    check(validateStereoCalibrateOutputPath(candidate.string()),
          "a candidate under an absent parent directory must be accepted");

    std::error_code error;
    check(std::filesystem::create_directory(root, error) && !error,
          "test output parent must be created");
    check(validateStereoCalibrateOutputPath(candidate.string()),
          "a missing candidate under an existing directory must be accepted");
    {
        std::ofstream file(candidate);
        file << "existing calibration";
    }
    check(!validateStereoCalibrateOutputPath(candidate.string()),
          "an existing calibration must be rejected");
    check(std::filesystem::remove(candidate, error) && !error,
          "test candidate must be removed");
    check(std::filesystem::remove(root, error) && !error,
          "test output parent must be removed");

    {
        std::ofstream file(root);
        file << "not a directory";
    }
    check(!validateStereoCalibrateOutputPath(candidate.string()),
          "a regular-file output parent must be rejected");
    check(std::filesystem::remove(root, error) && !error,
          "test regular-file parent must be removed");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Stereo calibrate argument tests passed\n";
    return EXIT_SUCCESS;
}
