#ifndef VOLLEYBALL_TEST_FAKE_CUDA_RUNTIME_H_
#define VOLLEYBALL_TEST_FAKE_CUDA_RUNTIME_H_

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorUnknown = 999,
};

struct FakeCudaStream {
    cudaError_t sync_status = cudaSuccess;
};

using cudaStream_t = FakeCudaStream*;

constexpr unsigned cudaStreamNonBlocking = 1u;

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    static FakeCudaStream created;
    *stream = &created;
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return stream ? stream->sync_status : cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}

inline const char* cudaGetErrorString(cudaError_t status) {
    return status == cudaSuccess ? "cudaSuccess" : "cudaErrorUnknown";
}

#endif  // VOLLEYBALL_TEST_FAKE_CUDA_RUNTIME_H_
