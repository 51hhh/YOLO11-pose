#include "hik_frame_metadata_probe/args.h"

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

bool parse(std::vector<std::string> arguments,
           hik_metadata_probe::Args& args) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) argv.push_back(argument.data());
    return hik_metadata_probe::parseArgs(
        static_cast<int>(argv.size()), argv.data(), args);
}

}  // namespace

int main() {
    {
        hik_metadata_probe::Args args;
        check(parse({"hik_frame_metadata_probe"}, args),
              "default arguments must parse");
        check(hik_metadata_probe::validateArgs(args),
              "default arguments must be valid");
        check(args.pwm_hz == 90.0,
              "probe default must match the production trigger profile");
    }

    {
        hik_metadata_probe::Args args;
        check(parse({"probe", "--frames", "1000", "--timeout-ms", "250",
                     "--exposure-us", "9867.5", "--gain-db", "11.99",
                     "--trigger-line", "7", "--pwm-hz", "90"}, args),
              "explicit numeric arguments must parse");
        check(args.frames == 1000 && args.timeout_ms == 250 &&
                  args.trigger_line == 7 && args.pwm_hz == 90.0,
              "parsed numeric values must be preserved");
        check(hik_metadata_probe::validateArgs(args),
              "explicit valid arguments must pass validation");
    }

    for (const auto& invalid : {
             std::vector<std::string>{"probe", "--frames", "not-a-number"},
             std::vector<std::string>{"probe", "--timeout-ms", "1ms"},
             std::vector<std::string>{"probe", "--trigger-line", "-1"},
             std::vector<std::string>{"probe", "--pwm-hz", "nan"},
         }) {
        hik_metadata_probe::Args args;
        check(!parse(invalid, args),
              "malformed numeric arguments must fail closed");
    }

    {
        hik_metadata_probe::Args args;
        check(parse({"probe", "--frames", "1"}, args),
              "range validation must be separate from syntax parsing");
        check(!hik_metadata_probe::validateArgs(args),
              "one frame cannot validate counter continuity");
    }

    {
        hik_metadata_probe::Args args;
        check(parse({"probe", "--left-index", "2", "--right-index", "2"},
                    args),
              "equal indices are syntactically valid");
        check(!hik_metadata_probe::validateArgs(args),
              "equal camera identities must be rejected");
    }

    {
        hik_metadata_probe::Args args;
        check(parse({"probe", "--left-sn", "same", "--right-sn", "same"},
                    args),
              "equal serials are syntactically valid");
        check(!hik_metadata_probe::validateArgs(args),
              "equal serial numbers must be rejected");
    }

    {
        hik_metadata_probe::Args args;
        check(parse({"probe", "--help"}, args) && args.help_requested,
              "help must be a successful non-hardware action");
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "hik_frame_metadata_probe args tests passed\n";
    return EXIT_SUCCESS;
}
