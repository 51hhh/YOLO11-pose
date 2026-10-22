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
        core::StereoDepthMethodId{"roi_multi_point"},
        core::StereoDepthMethodId{"circle_center"},
        core::StereoDepthMethodId{"radial_center"},
        core::StereoDepthMethodId{"edge_pair_center"},
        core::StereoDepthMethodId{"edge_centroid"},
        core::StereoDepthMethodId{"bbox_center"},
        core::StereoDepthMethodId{"bbox_edges"},
    };

    std::uint32_t max_disparity_px{2048};
    std::uint32_t patch_radius{5};
    std::uint32_t search_radius_px{8};
    std::uint32_t max_points{12};
    std::uint32_t min_points{4};
    std::uint32_t circle_max_roi_pixels{18000};
    double min_confidence{0.25};
    double max_disparity_delta_px{2.0};
    double max_disparity_delta_ratio{0.03};
    double max_depth_delta_m{0.5};
    double max_disparity_stddev_px{1.0};
    double min_disparity_stddev_px{0.05};
    double geometric_disparity_stddev_px{1.0};
    double anchor_stddev_px{0.5};
    double min_depth_m{0.3};
    double max_depth_m{15.0};

    double feature_y_tolerance_px{2.0};
    double feature_y_slope{0.0};
    double feature_y_offset_px{0.0};
    double feature_reverse_check_px{1.0};
    double feature_overlap_scale{0.55};
    double feature_mad_scale{2.5};
    double feature_ransac_gate_px{0.75};
    double feature_sphere_radius_m{0.10};
    double feature_sphere_radius_scale{1.8};
    double feature_sphere_margin_m{0.02};
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
