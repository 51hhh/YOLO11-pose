#include "volleyball/tests/fakes/tensorrt_runtime.hpp"

#include "volleyball/adapters/tensorrt/runtime.hpp"

namespace volleyball::adapters::tensorrt {

TensorRtRuntime::TensorRtRuntime() = default;
TensorRtRuntime::~TensorRtRuntime() = default;

bool TensorRtRuntime::init(
    const std::string&,
    bool,
    int,
    float,
    float,
    const std::string&) {
    inputSize_ = tests::tensorrt_runtime_fake::input_size;
    outputContract_ = tests::tensorrt_runtime_fake::output_contract;
    classCount_ = tests::tensorrt_runtime_fake::class_count;
    return tests::tensorrt_runtime_fake::initialize_success;
}

PostprocessResult TensorRtRuntime::detectChecked(
    const void* image,
    int pitch,
    int width,
    int height,
    cudaStream_t) {
    ++tests::tensorrt_runtime_fake::detection_calls;
    tests::tensorrt_runtime_fake::last_pointer =
        reinterpret_cast<std::uintptr_t>(image);
    tests::tensorrt_runtime_fake::last_pitch = pitch;
    tests::tensorrt_runtime_fake::last_width = width;
    tests::tensorrt_runtime_fake::last_height = height;
    return tests::tensorrt_runtime_fake::detection_result;
}

}  // namespace volleyball::adapters::tensorrt
