#include "tools/capture_chessboard/args.h"

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
           CaptureChessboardArgs& args) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());
    return parseCaptureChessboardArgs(
        static_cast<int>(argv.size()), argv.data(), args);
}

std::filesystem::path uniqueTemporaryPath() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    return std::filesystem::temp_directory_path() /
           ("capture_chessboard_args_tests_" + std::to_string(suffix));
}

}  // namespace

int main() {
    {
        CaptureChessboardArgs args;
        check(parse({"capture_chessboard"}, args),
              "default arguments must parse");
        check(validateCaptureChessboardArgs(args),
              "default arguments must be valid");
        check(args.pwm_hz == 90.0 && args.trigger_chip == "gpiochip2" &&
                  args.trigger_line == 7,
              "capture defaults must match the production trigger profile");
    }

    {
        CaptureChessboardArgs args;
        check(parse({"capture_chessboard", "--output", "new-session",
                     "--exposure-us", "9867", "--gain-db", "11.99",
                     "--trigger-chip", "gpiochip4", "--trigger-line", "9",
                     "--pwm-hz", "120", "--image-node-num", "4"},
                    args),
              "explicit capture arguments must parse");
        check(args.output_dir == "new-session" && args.exposure_us == 9867 &&
                  args.trigger_chip == "gpiochip4" &&
                  args.trigger_line == 9 && args.pwm_hz == 120.0 &&
                  args.image_node_num == 4,
              "explicit capture arguments must be preserved");
        check(validateCaptureChessboardArgs(args),
              "explicit valid capture arguments must pass validation");
    }

    for (const auto& invalid : {
             std::vector<std::string>{"capture", "--exposure-us", "1us"},
             std::vector<std::string>{"capture", "--gain-db", "nan"},
             std::vector<std::string>{"capture", "--trigger-line", "-1"},
             std::vector<std::string>{"capture", "--pwm-hz", "inf"},
         }) {
        CaptureChessboardArgs args;
        check(!parse(invalid, args),
              "malformed numeric arguments must fail closed");
    }

    {
        CaptureChessboardArgs args;
        check(parse({"capture", "--image-node-num", "1"}, args),
              "FIFO range validation must be separate from syntax parsing");
        check(!validateCaptureChessboardArgs(args),
              "a one-slot SDK FIFO must be rejected");
    }

    {
        CaptureChessboardArgs args;
        check(parse({"capture", "--pwm-hz", "0.5"}, args),
              "PWM range validation must be separate from syntax parsing");
        check(!validateCaptureChessboardArgs(args),
              "a PWM rate that rounds to zero in the camera contract must be rejected");
    }

    {
        CaptureChessboardArgs args;
        check(parse({"capture", "--left-index", "2", "--right-index", "2"},
                    args),
              "equal camera indices are syntactically valid");
        check(!validateCaptureChessboardArgs(args),
              "equal camera identities must be rejected");
    }

    {
        CaptureChessboardArgs args;
        check(parse({"capture", "--help"}, args) && args.help_requested,
              "help must be a successful non-hardware action");
        check(validateCaptureChessboardArgs(args),
              "help must not require capture settings");
    }

    const std::filesystem::path output = uniqueTemporaryPath();
    std::error_code error;
    check(validateCaptureChessboardOutputDirectory(output.string()),
          "an absent output directory must be accepted");
    check(std::filesystem::create_directory(output, error) && !error,
          "test output directory must be created");
    check(validateCaptureChessboardOutputDirectory(output.string()),
          "an empty output directory must be accepted");

    const std::filesystem::path marker = output / "existing.png";
    {
        std::ofstream file(marker);
        file << "do not overwrite";
    }
    check(!validateCaptureChessboardOutputDirectory(output.string()),
          "a non-empty output directory must be rejected");
    check(std::filesystem::remove(marker, error) && !error,
          "test marker must be removed");
    check(std::filesystem::remove(output, error) && !error,
          "test output directory must be removed");

    {
        std::ofstream file(output);
        file << "not a directory";
    }
    check(!validateCaptureChessboardOutputDirectory(output.string()),
          "an output path that is a regular file must be rejected");
    check(std::filesystem::remove(output, error) && !error,
          "test regular file must be removed");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "capture_chessboard args tests passed\n";
    return EXIT_SUCCESS;
}
