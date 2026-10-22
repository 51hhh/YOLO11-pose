#ifndef VOLLEYBALL_TEST_FAKE_VPI_STREAM_H_
#define VOLLEYBALL_TEST_FAKE_VPI_STREAM_H_

#include <cstdint>

enum VPIStatus {
    VPI_SUCCESS = 0,
    VPI_ERROR_INTERNAL = 1,
};

using VPIBackend = std::uint64_t;
constexpr VPIBackend VPI_BACKEND_PVA = 1u << 0;
constexpr VPIBackend VPI_BACKEND_CUDA = 1u << 1;
constexpr VPIBackend VPI_BACKEND_VIC = 1u << 2;

struct FakeVPIStream {
    VPIStatus sync_status = VPI_SUCCESS;
};

using VPIStream = FakeVPIStream*;

inline VPIStatus vpiStreamCreate(VPIBackend, VPIStream* stream) {
    static FakeVPIStream created;
    *stream = &created;
    return VPI_SUCCESS;
}

inline VPIStatus vpiStreamSync(VPIStream stream) {
    return stream ? stream->sync_status : VPI_SUCCESS;
}

inline void vpiStreamDestroy(VPIStream) {}

#endif  // VOLLEYBALL_TEST_FAKE_VPI_STREAM_H_
