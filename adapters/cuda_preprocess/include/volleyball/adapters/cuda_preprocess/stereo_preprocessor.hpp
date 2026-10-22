#pragma once

#include "volleyball/adapters/cuda_preprocess/remap_maps.hpp"
#include "volleyball/ports/stereo_preprocessor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace volleyball::adapters::cuda_preprocess {

struct CudaStereoPreprocessorConfig {
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::size_t buffer_pool_pairs{3};
};

// 将主机 BayerRG8 双目帧转换为 CUDA-device 校正 BGR8 与 Gray8。
// 左右相机使用独立非阻塞 stream；由于端口尚未携带 readiness event，
// prepare() 会在返回前同步两条 stream。一个实例允许多线程调用，但会串行处理帧。
class CudaStereoPreprocessor final : public ports::StereoPreprocessor {
public:
    explicit CudaStereoPreprocessor(CudaStereoPreprocessorConfig config);
    ~CudaStereoPreprocessor() override;

    CudaStereoPreprocessor(const CudaStereoPreprocessor&) = delete;
    CudaStereoPreprocessor& operator=(const CudaStereoPreprocessor&) = delete;

    [[nodiscard]] bool initialize(
        const StereoRemapMaps& maps,
        std::string& error);
    [[nodiscard]] bool isInitialized() const noexcept;
    [[nodiscard]] ports::PreprocessResult prepare(
        const ports::StereoFrame& frame) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::cuda_preprocess
