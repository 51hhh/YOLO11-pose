#ifndef VOLLEYBALL_TEST_FAKE_GPU_CUDA_RUNTIME_H_
#define VOLLEYBALL_TEST_FAKE_GPU_CUDA_RUNTIME_H_

#include <cstddef>
#include <cstdlib>
#include <cstring>

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorUnknown = 999,
};

struct FakeCudaStream {
    cudaError_t sync_status = cudaSuccess;
    int sync_calls = 0;
};

struct FakeCudaEvent {
    bool recorded = false;
};

using cudaStream_t = FakeCudaStream*;
using cudaEvent_t = FakeCudaEvent*;

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
};

constexpr unsigned cudaEventDefault = 0u;
constexpr unsigned cudaHostAllocDefault = 0u;

namespace fake_cuda {

inline cudaError_t next_allocation_status = cudaSuccess;
inline cudaError_t next_memcpy_status = cudaSuccess;
inline cudaError_t next_launch_status = cudaSuccess;
inline cudaError_t next_event_record_status = cudaSuccess;
inline cudaError_t event_sync_status = cudaSuccess;

inline cudaError_t consume(cudaError_t* status) {
    const cudaError_t value = *status;
    *status = cudaSuccess;
    return value;
}

inline void reset() {
    next_allocation_status = cudaSuccess;
    next_memcpy_status = cudaSuccess;
    next_launch_status = cudaSuccess;
    next_event_record_status = cudaSuccess;
    event_sync_status = cudaSuccess;
}

}  // namespace fake_cuda

inline const char* cudaGetErrorString(cudaError_t status) {
    return status == cudaSuccess ? "cudaSuccess" : "cudaErrorUnknown";
}

inline cudaError_t cudaHostAlloc(void** output, std::size_t bytes, unsigned) {
    const cudaError_t injected =
        fake_cuda::consume(&fake_cuda::next_allocation_status);
    if (injected != cudaSuccess) return injected;
    *output = std::malloc(bytes);
    return *output ? cudaSuccess : cudaErrorUnknown;
}

inline cudaError_t cudaMalloc(void** output, std::size_t bytes) {
    const cudaError_t injected =
        fake_cuda::consume(&fake_cuda::next_allocation_status);
    if (injected != cudaSuccess) return injected;
    *output = std::malloc(bytes);
    return *output ? cudaSuccess : cudaErrorUnknown;
}

inline cudaError_t cudaFreeHost(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* output, unsigned) {
    *output = new FakeCudaEvent{};
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
    const cudaError_t injected =
        fake_cuda::consume(&fake_cuda::next_event_record_status);
    if (injected != cudaSuccess) return injected;
    if (!event) return cudaErrorUnknown;
    event->recorded = true;
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind,
    cudaStream_t) {
    const cudaError_t injected =
        fake_cuda::consume(&fake_cuda::next_memcpy_status);
    if (injected != cudaSuccess) return injected;
    if (!destination || !source) return cudaErrorUnknown;
    std::memcpy(destination, source, bytes);
    return cudaSuccess;
}

inline cudaError_t cudaGetLastError() {
    return fake_cuda::consume(&fake_cuda::next_launch_status);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (!stream) return cudaSuccess;
    ++stream->sync_calls;
    return stream->sync_status;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event || !event->recorded) return cudaErrorUnknown;
    return fake_cuda::event_sync_status;
}

inline cudaError_t cudaEventElapsedTime(
    float* elapsed_ms,
    cudaEvent_t start,
    cudaEvent_t end) {
    if (!elapsed_ms || !start || !end || !start->recorded || !end->recorded) {
        return cudaErrorUnknown;
    }
    *elapsed_ms = 0.1f;
    return cudaSuccess;
}

#endif  // VOLLEYBALL_TEST_FAKE_GPU_CUDA_RUNTIME_H_
