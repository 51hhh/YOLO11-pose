#include "pipeline/sync.h"

#include <cstdlib>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void testAllStreamsSucceed() {
    FakeVPIStream rectification;
    FakeVPIStream stereo;
    FakeCudaStream detector;
    FakeCudaStream fusion;

    stereo3d::PipelineStreams streams;
    streams.vpiStreamPVA = &rectification;
    streams.vpiStreamGPU = &stereo;
    streams.cudaStreamDLA = &detector;
    streams.cudaStreamFuse = &fusion;
    check(streams.syncAll(), "all-success drain must return true");
}

void testVpiFailurePropagates() {
    FakeVPIStream rectification;
    rectification.sync_status = VPI_ERROR_INTERNAL;

    stereo3d::PipelineStreams streams;
    streams.vpiStreamPVA = &rectification;
    check(!streams.syncAll(), "VPI drain failure must return false");
}

void testCudaFailurePropagates() {
    FakeCudaStream detector;
    detector.sync_status = cudaErrorUnknown;

    stereo3d::PipelineStreams streams;
    streams.cudaStreamDLA = &detector;
    check(!streams.syncAll(), "CUDA drain failure must return false");
}

void testOptionalP2StreamsFollowRequirements() {
    stereo3d::PipelineStreams without_p2;
    check(without_p2.init({}), "base stream initialization must succeed");
    check(!without_p2.cudaStreamP2Ncc && !without_p2.cudaStreamP2XFeat &&
              !without_p2.cudaStreamP2SuperPoint &&
              !without_p2.cudaStreamP2Aliked,
          "disabled P2 algorithms must not allocate streams");
    without_p2.destroy();

    stereo3d::PipelineStreamRequirements requirements;
    requirements.p2_ncc = true;
    requirements.p2_aliked = true;
    stereo3d::PipelineStreams selected_p2;
    check(selected_p2.init(requirements),
          "selected P2 stream initialization must succeed");
    check(selected_p2.cudaStreamP2Ncc && !selected_p2.cudaStreamP2XFeat &&
              !selected_p2.cudaStreamP2SuperPoint &&
              selected_p2.cudaStreamP2Aliked,
          "only selected P2 algorithms may allocate streams");
    selected_p2.destroy();
}

}  // namespace

int main() {
    testAllStreamsSucceed();
    testVpiFailurePropagates();
    testCudaFailurePropagates();
    testOptionalP2StreamsFollowRequirements();
    std::cout << "pipeline stream tests passed\n";
    return 0;
}
