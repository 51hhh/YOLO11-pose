#include "volleyball/adapters/cuda_depth/depth_candidate.hpp"

#include <cstdlib>
#include <iostream>

namespace adapter = volleyball::adapters::cuda_depth;
namespace core = volleyball::core;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

adapter::DepthCandidate candidate(const char* method, double disparity) {
    return {
        core::StereoDepthMethodId{method},
        {100.0, 50.0},
        disparity,
        0.2,
        6U,
        0.8,
    };
}

void testSelectsByConfiguredOrderNotCandidateOrder() {
    const auto result = adapter::selectDepthCandidate(
        {candidate("bbox_center", 8.0),
         candidate("circle_center", 10.0)},
        {core::StereoDepthMethodId{"circle_center"},
         core::StereoDepthMethodId{"bbox_center"}});
    check(result.valid_input && result.candidate &&
              result.candidate->method ==
                  core::StereoDepthMethodId{"circle_center"} &&
              result.candidate->disparity_px == 10.0,
          "selection must follow the frozen method order");
}

void testNoCandidateIsAValidRejection() {
    const auto result = adapter::selectDepthCandidate(
        {}, {core::StereoDepthMethodId{"circle_center"}});
    check(result.valid_input && !result.candidate,
          "no algorithmic measurement must remain a valid rejection");
}

void testRejectsAmbiguousOrUnsupportedInput() {
    for (const char* method : {"roi_multi_point", "center_patch"}) {
        const auto surface = adapter::selectDepthCandidate(
            {candidate(method, 10.0)},
            {core::StereoDepthMethodId{"bbox_center"}});
        check(
            !surface.valid_input,
            "surface matches must not be published as ball centers");
    }
    const auto unconfigured = adapter::selectDepthCandidate(
        {candidate("circle_center", 10.0)},
        {core::StereoDepthMethodId{"bbox_center"}});
    check(!unconfigured.valid_input,
          "computed candidates outside the frozen order must fail closed");
    const auto duplicate = adapter::selectDepthCandidate(
        {candidate("bbox_center", 9.0), candidate("bbox_center", 10.0)},
        {core::StereoDepthMethodId{"bbox_center"}});
    check(!duplicate.valid_input,
          "duplicate candidates must not be selected ambiguously");

    const auto unsupported = adapter::selectDepthCandidate(
        {}, {core::StereoDepthMethodId{"cuda_template_match"}});
    check(!unsupported.valid_input,
          "unimplemented final methods must fail configuration validation");
}

}  // namespace

int main() {
    testSelectsByConfiguredOrderNotCandidateOrder();
    testNoCandidateIsAValidRejection();
    testRejectsAmbiguousOrUnsupportedInput();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "CUDA depth candidate tests passed\n";
    return EXIT_SUCCESS;
}
