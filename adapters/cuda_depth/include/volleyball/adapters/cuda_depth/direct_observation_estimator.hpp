#pragma once

#include "volleyball/algorithms/stereo/triangulation.hpp"
#include "volleyball/ports/direct_observation_estimator.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace volleyball::adapters::cuda_depth {

struct CudaDirectObservationEstimatorConfig {
    core::RectifiedStereoCalibration calibration;
    algorithms::stereo::TriangulationConfig triangulation;
    std::vector<core::StereoDepthMethodId> method_order{
        core::StereoDepthMethodId{"circle_center"},
        core::StereoDepthMethodId{"radial_center"},
        core::StereoDepthMethodId{"edge_pair_center"},
        core::StereoDepthMethodId{"edge_centroid"},
        core::StereoDepthMethodId{"bbox_center"},
        core::StereoDepthMethodId{"bbox_edges"},
    };

    std::uint32_t max_disparity_px{2048};
    std::uint32_t circle_max_roi_pixels{18000};
    double anchor_stddev_px{0.5};
    double disparity_stddev_px{1.0};
    double min_depth_m{0.3};
    double max_depth_m{15.0};
};

[[nodiscard]] bool isValid(
    const CudaDirectObservationEstimatorConfig& config) noexcept;

// 校正 Gray8 必须位于 CUDA device memory。实例内部只有一个 matcher/stream，
// estimate 以互斥锁串行化；左右 detector 的并行不受该约束影响。
class CudaDirectObservationEstimator final
    : public ports::DirectObservationEstimator {
public:
    explicit CudaDirectObservationEstimator(
        CudaDirectObservationEstimatorConfig config);
    ~CudaDirectObservationEstimator() override;

    CudaDirectObservationEstimator(
        const CudaDirectObservationEstimator&) = delete;
    CudaDirectObservationEstimator& operator=(
        const CudaDirectObservationEstimator&) = delete;

    [[nodiscard]] ports::DirectObservationResult estimate(
        const ports::DirectObservationRequest& request) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volleyball::adapters::cuda_depth
