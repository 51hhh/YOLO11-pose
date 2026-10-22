#include "stereo/dual_yolo_depth_gpu.h"

#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" void launchDualYoloDepthCandidatesGpu(
    const uint8_t*, int,
    const uint8_t*, int,
    const uint8_t*, int,
    const uint8_t*, int,
    int, int,
    const stereo3d::DualYoloGpuDetectionPair* pairs,
    int num_pairs,
    stereo3d::DualYoloGpuCandidate* results,
    int, int, int, int, int, int,
    float, float, float, float, float, float,
    float, float, float, float, float, float, float, float, float, float,
    int, int, int, int, int, int, int, int, int, int, int, int,
    float, float, float, float, float,
    cudaStream_t) {
    for (int i = 0; i < num_pairs; ++i) {
        results[i].left_index = pairs[i].left_index;
        results[i].right_index = pairs[i].right_index;
    }
}

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

stereo3d::DualYoloGpuDetectionPair makePair() {
    stereo3d::DualYoloGpuDetectionPair pair;
    pair.left_index = 2;
    pair.right_index = 5;
    pair.left = {40.0f, 30.0f, 20.0f, 20.0f, 0.9f, 0};
    pair.right = {32.0f, 30.0f, 20.0f, 20.0f, 0.8f, 0};
    return pair;
}

void initMatcher(stereo3d::DualYoloDepthGpuMatcher* matcher) {
    stereo3d::DualYoloDepthGpuConfig config;
    check(matcher != nullptr, "matcher pointer must be valid");
    check(matcher->init(800.0f, 0.8f, 720.0f, 540.0f, config, 4),
          "fake GPU matcher must initialize");
}

void submitOne(stereo3d::DualYoloDepthGpuMatcher& matcher,
               FakeCudaStream* stream) {
    static uint8_t left[64]{};
    static uint8_t right[64]{};
    const std::vector<stereo3d::DualYoloGpuDetectionPair> pairs{makePair()};
    check(matcher.submitPairs(
              left, 8, right, 8,
              nullptr, 0, nullptr, 0,
              8, 8, pairs, stream),
          "valid fake batch must submit");
}

void testSuccessfulCollectIsExplicit() {
    fake_cuda::reset();
    stereo3d::DualYoloDepthGpuMatcher matcher;
    initMatcher(&matcher);
    FakeCudaStream stream;
    submitOne(matcher, &stream);

    std::vector<stereo3d::DualYoloGpuCandidate> output;
    check(matcher.collectPairs(&output), "successful collect must return true");
    check(output.size() == 1, "successful collect must preserve batch size");
    check(output[0].left_index == 2 && output[0].right_index == 5,
          "kernel output identity must reach host output");
    check(!matcher.hasPendingPairs(), "successful collect must clear pending state");
}

void testSubmitFailureCannotLookLikeEmptyCandidateSet() {
    fake_cuda::reset();
    stereo3d::DualYoloDepthGpuMatcher matcher;
    initMatcher(&matcher);
    FakeCudaStream stream;
    static uint8_t left[64]{};
    static uint8_t right[64]{};
    const std::vector<stereo3d::DualYoloGpuDetectionPair> pairs{makePair()};
    fake_cuda::next_memcpy_status = cudaErrorUnknown;
    check(!matcher.submitPairs(
              left, 8, right, 8,
              nullptr, 0, nullptr, 0,
              8, 8, pairs, &stream),
          "H2D failure must fail submission");
    check(!matcher.hasPendingPairs(), "failed submit must not leave pending batch");
}

void testKernelLaunchFailureDrainsSubmittedWork() {
    fake_cuda::reset();
    stereo3d::DualYoloDepthGpuMatcher matcher;
    initMatcher(&matcher);
    FakeCudaStream stream;
    static uint8_t left[64]{};
    static uint8_t right[64]{};
    const std::vector<stereo3d::DualYoloGpuDetectionPair> pairs{makePair()};
    fake_cuda::next_launch_status = cudaErrorUnknown;
    check(!matcher.submitPairs(
              left, 8, right, 8,
              nullptr, 0, nullptr, 0,
              8, 8, pairs, &stream),
          "kernel launch failure must fail submission");
    check(stream.sync_calls == 1,
          "failed submit after H2D enqueue must drain its stream");
    check(!matcher.hasPendingPairs(),
          "kernel launch failure must not leave pending batch");
}

void testCollectFailureClearsOutputAndReturnsFalse() {
    fake_cuda::reset();
    stereo3d::DualYoloDepthGpuMatcher matcher;
    initMatcher(&matcher);
    FakeCudaStream stream;
    submitOne(matcher, &stream);
    fake_cuda::event_sync_status = cudaErrorUnknown;

    std::vector<stereo3d::DualYoloGpuCandidate> output(3);
    check(!matcher.collectPairs(&output), "event sync failure must fail collect");
    check(output.empty(), "failed collect must not expose stale candidates");
    check(!matcher.hasPendingPairs(), "failed collect must clear pending state");
}

}  // namespace

int main() {
    testSuccessfulCollectIsExplicit();
    testSubmitFailureCannotLookLikeEmptyCandidateSet();
    testKernelLaunchFailureDrainsSubmittedWork();
    testCollectFailureClearsOutputAndReturnsFalse();
    std::cout << "dual YOLO GPU batch tests passed\n";
    return 0;
}
