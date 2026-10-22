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

}  // namespace

int main() {
    testAllStreamsSucceed();
    testVpiFailurePropagates();
    testCudaFailurePropagates();
    std::cout << "pipeline stream tests passed\n";
    return 0;
}
