#include "volleyball/adapters/cuda_depth/direct_observation_estimator.hpp"

#include "volleyball/adapters/cuda_depth/depth_candidate.hpp"
#include "volleyball/adapters/cuda_depth/roi_matcher.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace volleyball::adapters::cuda_depth {
namespace {

using Method = core::StereoDepthMethodId;

bool finiteNonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool finitePositive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

bool fitsFloat(double value) noexcept {
    return std::isfinite(value) &&
           std::abs(value) <=
               static_cast<double>(std::numeric_limits<float>::max());
}

bool containsMethod(
    const std::vector<Method>& methods,
    std::string_view expected) noexcept {
    return std::any_of(
        methods.begin(), methods.end(), [&](const Method& method) {
            return method.value() == expected;
        });
}

bool requiresGpu(const std::vector<Method>& methods) noexcept {
    return std::any_of(methods.begin(), methods.end(), [](const Method& method) {
        const std::string_view value = method.value();
        return value != "bbox_center" && value != "bbox_edges";
    });
}

bool requiresGeometry(const std::vector<Method>& methods) noexcept {
    return containsMethod(methods, "circle_center") ||
           containsMethod(methods, "radial_center") ||
           containsMethod(methods, "edge_pair_center") ||
           containsMethod(methods, "edge_centroid");
}

bool sameFrame(
    const core::FrameReference& left,
    const core::FrameReference& right) noexcept {
    return left.camera_id == right.camera_id && left.role == right.role &&
           left.frame_index == right.frame_index;
}

bool validCudaGrayImage(
    const ports::ImageBufferPtr& image,
    std::string& error) {
    if (!image || image->memoryKind() != ports::MemoryKind::cuda_device ||
        image->pixelFormat() != core::PixelFormat::gray8 ||
        image->nativeHandle() == 0U || image->width() == 0U ||
        image->height() == 0U) {
        error = "geometry image must be a non-empty CUDA Gray8 buffer";
        return false;
    }
    if (image->width() >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        image->height() >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        image->strideBytes() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        image->strideBytes() < image->width()) {
        error = "geometry image dimensions/pitch exceed the CUDA matcher contract";
        return false;
    }
    const std::size_t rows_before_last =
        static_cast<std::size_t>(image->height() - 1U);
    if (rows_before_last != 0U &&
        image->strideBytes() >
            (std::numeric_limits<std::size_t>::max() - image->width()) /
                rows_before_last) {
        error = "geometry image byte extent overflows size_t";
        return false;
    }
    const std::size_t minimum_size =
        rows_before_last * image->strideBytes() + image->width();
    if (image->sizeBytes() < minimum_size) {
        error = "geometry image storage does not cover its pitched extent";
        return false;
    }
    return true;
}

bool validDetectionForImage(
    const core::Detection2D& detection,
    const ports::ImageBuffer& image) noexcept {
    if (!core::validate(detection) || !detection.bounding_box) {
        return false;
    }
    const auto& box = *detection.bounding_box;
    const double width = static_cast<double>(image.width());
    const double height = static_cast<double>(image.height());
    return box.x < width && box.y < height && box.x + box.width > 0.0 &&
           box.y + box.height > 0.0 && box.width <= width &&
           box.height <= height;
}

std::optional<double> pairConfidence(
    const core::Detection2D& left,
    const core::Detection2D& right) noexcept {
    if (!left.score || !right.score) {
        return std::nullopt;
    }
    return std::min(*left.score, *right.score);
}

bool depthInRange(
    double disparity,
    const CudaDirectObservationEstimatorConfig& config) noexcept {
    if (!finitePositive(disparity) ||
        disparity < config.triangulation.min_disparity_px) {
        return false;
    }
    const double depth =
        config.calibration.left.fx * config.calibration.baseline_m / disparity;
    return std::isfinite(depth) && depth >= config.min_depth_m &&
           depth <= config.max_depth_m;
}

bool appendCandidate(
    DepthCandidate candidate,
    std::vector<DepthCandidate>& output,
    std::string& error) {
    if (!isValid(candidate)) {
        error = "CUDA depth backend produced an invalid candidate";
        return false;
    }
    output.push_back(std::move(candidate));
    return true;
}

bool appendGpuDisparity(
    const stereo3d::DualYoloGpuDisparity& input,
    const char* method,
    const CudaDirectObservationEstimatorConfig& config,
    std::vector<DepthCandidate>& output,
    std::string& error) {
    if (input.valid == 0) {
        return true;
    }
    const double disparity = static_cast<double>(input.disparity);
    if (!depthInRange(disparity, config)) {
        error = "CUDA depth backend marked an out-of-range disparity valid";
        return false;
    }
    const double raw_stddev = static_cast<double>(input.stddev);
    if (!finiteNonnegative(raw_stddev) || input.support <= 0) {
        error = "CUDA depth backend returned invalid uncertainty/support";
        return false;
    }
    DepthCandidate candidate;
    candidate.method = Method{method};
    candidate.left_anchor_px = {
        static_cast<double>(input.anchor_cx),
        static_cast<double>(input.anchor_cy),
    };
    candidate.disparity_px = disparity;
    candidate.disparity_stddev_px =
        std::max(raw_stddev, config.min_disparity_stddev_px);
    candidate.support_points = static_cast<std::uint32_t>(input.support);
    candidate.confidence = static_cast<double>(input.confidence);
    return appendCandidate(std::move(candidate), output, error);
}

bool appendPointPair(
    const stereo3d::DualYoloGpuPointMeasure& left,
    const stereo3d::DualYoloGpuPointMeasure& right,
    const char* method,
    const CudaDirectObservationEstimatorConfig& config,
    std::vector<DepthCandidate>& output,
    std::string& error) {
    if (left.valid == 0 || right.valid == 0) {
        return true;
    }
    DepthCandidate candidate;
    candidate.method = Method{method};
    candidate.left_anchor_px = {
        static_cast<double>(left.cx), static_cast<double>(left.cy)};
    candidate.disparity_px =
        static_cast<double>(left.cx) - static_cast<double>(right.cx);
    if (!depthInRange(candidate.disparity_px, config)) {
        return true;
    }
    candidate.disparity_stddev_px =
        config.geometric_disparity_stddev_px;
    candidate.support_points = 1U;
    candidate.confidence = std::min(
        static_cast<double>(left.confidence),
        static_cast<double>(right.confidence));
    return appendCandidate(std::move(candidate), output, error);
}

bool appendCirclePair(
    const stereo3d::DualYoloGpuCircle& left,
    const stereo3d::DualYoloGpuCircle& right,
    const CudaDirectObservationEstimatorConfig& config,
    std::vector<DepthCandidate>& output,
    std::string& error) {
    if (left.valid == 0 || right.valid == 0) {
        return true;
    }
    DepthCandidate candidate;
    candidate.method = Method{"circle_center"};
    candidate.left_anchor_px = {
        static_cast<double>(left.cx), static_cast<double>(left.cy)};
    candidate.disparity_px =
        static_cast<double>(left.cx) - static_cast<double>(right.cx);
    if (!depthInRange(candidate.disparity_px, config)) {
        return true;
    }
    candidate.disparity_stddev_px =
        config.geometric_disparity_stddev_px;
    candidate.support_points = 1U;
    candidate.confidence = std::min(
        static_cast<double>(left.confidence),
        static_cast<double>(right.confidence));
    return appendCandidate(std::move(candidate), output, error);
}

bool appendBoundingBoxCandidates(
    const core::Detection2D& left,
    const core::Detection2D& right,
    const CudaDirectObservationEstimatorConfig& config,
    std::vector<DepthCandidate>& output,
    std::string& error) {
    const auto confidence = pairConfidence(left, right);
    const double center_disparity = left.center_px.x - right.center_px.x;
    if (containsMethod(config.method_order, "bbox_center") &&
        depthInRange(center_disparity, config)) {
        DepthCandidate candidate;
        candidate.method = Method{"bbox_center"};
        candidate.left_anchor_px = left.center_px;
        candidate.disparity_px = center_disparity;
        candidate.disparity_stddev_px =
            config.geometric_disparity_stddev_px;
        candidate.support_points = 1U;
        candidate.confidence = confidence;
        if (!appendCandidate(std::move(candidate), output, error)) {
            return false;
        }
    }

    if (!containsMethod(config.method_order, "bbox_edges")) {
        return true;
    }

    const auto& left_box = *left.bounding_box;
    const auto& right_box = *right.bounding_box;
    const double left_edge_disparity = left_box.x - right_box.x;
    const double right_edge_disparity =
        (left_box.x + left_box.width) -
        (right_box.x + right_box.width);
    std::optional<double> edge_disparity;
    const bool left_valid = depthInRange(left_edge_disparity, config);
    const bool right_valid = depthInRange(right_edge_disparity, config);
    if (left_valid && right_valid) {
        edge_disparity = 0.5 *
            (left_edge_disparity + right_edge_disparity);
    } else if (left_valid) {
        edge_disparity = left_edge_disparity;
    } else if (right_valid) {
        edge_disparity = right_edge_disparity;
    }
    if (edge_disparity) {
        DepthCandidate candidate;
        candidate.method = Method{"bbox_edges"};
        candidate.left_anchor_px = left.center_px;
        candidate.disparity_px = *edge_disparity;
        candidate.disparity_stddev_px =
            config.geometric_disparity_stddev_px;
        candidate.support_points = 1U;
        candidate.confidence = confidence;
        if (!appendCandidate(std::move(candidate), output, error)) {
            return false;
        }
    }
    return true;
}

stereo3d::DualYoloDepthGpuConfig matcherConfig(
    const CudaDirectObservationEstimatorConfig& config) {
    stereo3d::DualYoloDepthGpuConfig output;
    output.max_disparity = static_cast<int>(config.max_disparity_px);
    output.patch_radius = static_cast<int>(config.patch_radius);
    output.search_radius_px = static_cast<int>(config.search_radius_px);
    output.max_points = static_cast<int>(config.max_points);
    output.min_points = static_cast<int>(config.min_points);
    output.circle_max_roi_pixels =
        static_cast<int>(config.circle_max_roi_pixels);
    output.min_confidence = static_cast<float>(config.min_confidence);
    output.max_disp_delta_px =
        static_cast<float>(config.max_disparity_delta_px);
    output.max_disp_delta_ratio =
        static_cast<float>(config.max_disparity_delta_ratio);
    output.max_depth_delta_m =
        static_cast<float>(config.max_depth_delta_m);
    output.max_stddev_px =
        static_cast<float>(config.max_disparity_stddev_px);
    output.epipolar_y_tolerance =
        static_cast<float>(config.feature_y_tolerance_px);
    output.feature_y_tolerance_px =
        static_cast<float>(config.feature_y_tolerance_px);
    output.feature_y_slope = static_cast<float>(config.feature_y_slope);
    output.feature_y_offset_px =
        static_cast<float>(config.feature_y_offset_px);
    output.feature_reverse_check_px =
        static_cast<float>(config.feature_reverse_check_px);
    output.feature_overlap_scale =
        static_cast<float>(config.feature_overlap_scale);
    output.feature_mad_scale =
        static_cast<float>(config.feature_mad_scale);
    output.feature_ransac_gate_px =
        static_cast<float>(config.feature_ransac_gate_px);
    output.feature_sphere_radius_m =
        static_cast<float>(config.feature_sphere_radius_m);
    output.feature_sphere_radius_scale =
        static_cast<float>(config.feature_sphere_radius_scale);
    output.feature_sphere_margin_m =
        static_cast<float>(config.feature_sphere_margin_m);
    output.disparity_zero_offset = 0.0F;
    output.min_depth = static_cast<float>(config.min_depth_m);
    output.max_depth = static_cast<float>(config.max_depth_m);
    output.compute_geometry = requiresGeometry(config.method_order);
    output.compute_center_patch =
        containsMethod(config.method_order, "center_patch");
    output.compute_multi_point =
        containsMethod(config.method_order, "roi_multi_point");
    return output;
}

ports::DirectObservationResult failed(
    const Method& method,
    std::string message) {
    return {
        ports::DirectObservationStatus::failed,
        std::nullopt,
        method,
        std::move(message),
    };
}

ports::DirectObservationResult rejected(
    const Method& method,
    std::string message) {
    return {
        ports::DirectObservationStatus::rejected,
        std::nullopt,
        method,
        std::move(message),
    };
}

}  // namespace

bool isValid(
    const CudaDirectObservationEstimatorConfig& config) noexcept {
    if (!algorithms::stereo::isValidTriangulationConfig(
            config.calibration, config.triangulation) ||
        config.method_order.empty() || config.max_disparity_px == 0U ||
        config.max_disparity_px >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        config.patch_radius < 2U || config.patch_radius > 8U ||
        config.search_radius_px == 0U ||
        config.search_radius_px >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        config.max_points == 0U || config.max_points > 32U ||
        config.min_points == 0U ||
        config.min_points > config.max_points ||
        config.circle_max_roi_pixels == 0U ||
        config.circle_max_roi_pixels >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    for (std::size_t index = 0; index < config.method_order.size(); ++index) {
        if (!isSupportedSelectionMethod(config.method_order[index])) {
            return false;
        }
        for (std::size_t other = index + 1U;
             other < config.method_order.size(); ++other) {
            if (config.method_order[index] == config.method_order[other]) {
                return false;
            }
        }
    }
    const bool numeric_values_valid =
        config.min_confidence >= 0.0 && config.min_confidence <= 1.0 &&
        finitePositive(config.max_disparity_delta_px) &&
        finiteNonnegative(config.max_disparity_delta_ratio) &&
        finiteNonnegative(config.max_depth_delta_m) &&
        finitePositive(config.max_disparity_stddev_px) &&
        finiteNonnegative(config.min_disparity_stddev_px) &&
        config.min_disparity_stddev_px <=
            config.max_disparity_stddev_px &&
        finiteNonnegative(config.geometric_disparity_stddev_px) &&
        finiteNonnegative(config.anchor_stddev_px) &&
        finitePositive(config.min_depth_m) &&
        finitePositive(config.max_depth_m) &&
        config.max_depth_m > config.min_depth_m &&
        finitePositive(config.feature_y_tolerance_px) &&
        fitsFloat(config.feature_y_slope) &&
        fitsFloat(config.feature_y_offset_px) &&
        fitsFloat(config.feature_reverse_check_px) &&
        config.feature_reverse_check_px >= -1.0 &&
        finitePositive(config.feature_overlap_scale) &&
        config.feature_overlap_scale <= 1.0 &&
        finitePositive(config.feature_mad_scale) &&
        finitePositive(config.feature_ransac_gate_px) &&
        finitePositive(config.feature_sphere_radius_m) &&
        finitePositive(config.feature_sphere_radius_scale) &&
        finiteNonnegative(config.feature_sphere_margin_m);
    return numeric_values_valid && fitsFloat(config.calibration.left.fx) &&
           fitsFloat(config.calibration.baseline_m) &&
           fitsFloat(config.calibration.left.cx) &&
           fitsFloat(config.calibration.left.cy);
}

struct CudaDirectObservationEstimator::Impl {
    explicit Impl(CudaDirectObservationEstimatorConfig value)
        : config(std::move(value)), gpu_required(requiresGpu(config.method_order)) {
        if (!gpu_required) {
            return;
        }
        const cudaError_t stream_error =
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_error != cudaSuccess) {
            throw std::runtime_error(
                std::string{"failed to create CUDA depth stream: "} +
                cudaGetErrorString(stream_error));
        }
        if (!matcher.init(
                static_cast<float>(config.calibration.left.fx),
                static_cast<float>(config.calibration.baseline_m),
                static_cast<float>(config.calibration.left.cx),
                static_cast<float>(config.calibration.left.cy),
                matcherConfig(config),
                1)) {
            cudaStreamDestroy(stream);
            stream = nullptr;
            throw std::runtime_error("failed to initialize CUDA ROI depth matcher");
        }
    }

    ~Impl() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    CudaDirectObservationEstimatorConfig config;
    bool gpu_required{false};
    cudaStream_t stream{nullptr};
    stereo3d::DualYoloDepthGpuMatcher matcher;
    std::mutex mutex;
};

CudaDirectObservationEstimator::CudaDirectObservationEstimator(
    CudaDirectObservationEstimatorConfig config) {
    if (!isValid(config)) {
        throw std::invalid_argument(
            "invalid CUDA direct observation estimator configuration");
    }
    impl_ = std::make_unique<Impl>(std::move(config));
}

CudaDirectObservationEstimator::~CudaDirectObservationEstimator() = default;

ports::DirectObservationResult CudaDirectObservationEstimator::estimate(
    const ports::DirectObservationRequest& request) {
    const Method attempted_method = impl_->config.method_order.front();
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto& left_frame = request.rectified_frame.left;
        const auto& right_frame = request.rectified_frame.right;
        if (!sameFrame(left_frame.source, request.left_detection.frame) ||
            !sameFrame(right_frame.source, request.right_detection.frame) ||
            left_frame.source.role != core::CameraRole::left ||
            right_frame.source.role != core::CameraRole::right ||
            !core::validate(request.association) ||
            request.association.decision !=
                core::AssociationDecision::accepted ||
            request.association.left_frame_index !=
                request.left_detection.frame.frame_index ||
            request.association.right_frame_index !=
                request.right_detection.frame.frame_index ||
            request.association.left_candidate_id !=
                request.left_detection.candidate_id ||
            request.association.right_candidate_id !=
                request.right_detection.candidate_id) {
            return failed(attempted_method,
                          "depth request identities/association are invalid");
        }

        std::string error;
        if (!validCudaGrayImage(left_frame.geometry_gray, error) ||
            !validCudaGrayImage(right_frame.geometry_gray, error)) {
            return failed(attempted_method, std::move(error));
        }
        if (left_frame.geometry_gray->width() !=
                right_frame.geometry_gray->width() ||
            left_frame.geometry_gray->height() !=
                right_frame.geometry_gray->height()) {
            return failed(attempted_method,
                          "left/right geometry images have different extents");
        }
        if (!validDetectionForImage(
                request.left_detection, *left_frame.geometry_gray) ||
            !validDetectionForImage(
                request.right_detection, *right_frame.geometry_gray)) {
            return rejected(attempted_method,
                            "selected detections have no usable image ROI");
        }

        std::vector<DepthCandidate> candidates;
        candidates.reserve(8U);
        if (impl_->gpu_required) {
            const auto& left_detection = request.left_detection;
            const auto& right_detection = request.right_detection;
            const auto& left_box = *left_detection.bounding_box;
            const auto& right_box = *right_detection.bounding_box;
            stereo3d::DualYoloGpuDetectionPair pair;
            pair.left = {
                static_cast<float>(left_detection.center_px.x),
                static_cast<float>(left_detection.center_px.y),
                static_cast<float>(left_box.width),
                static_cast<float>(left_box.height),
                static_cast<float>(left_detection.score.value_or(0.0)),
                0,
            };
            pair.right = {
                static_cast<float>(right_detection.center_px.x),
                static_cast<float>(right_detection.center_px.y),
                static_cast<float>(right_box.width),
                static_cast<float>(right_box.height),
                static_cast<float>(right_detection.score.value_or(0.0)),
                0,
            };
            pair.left_index = 0;
            pair.right_index = 0;
            pair.epipolar_y_delta_px = static_cast<float>(
                left_detection.center_px.y - right_detection.center_px.y);

            const auto* left_device = reinterpret_cast<const std::uint8_t*>(
                left_frame.geometry_gray->nativeHandle());
            const auto* right_device = reinterpret_cast<const std::uint8_t*>(
                right_frame.geometry_gray->nativeHandle());
            if (!impl_->matcher.submitPairs(
                    left_device,
                    static_cast<int>(left_frame.geometry_gray->strideBytes()),
                    right_device,
                    static_cast<int>(right_frame.geometry_gray->strideBytes()),
                    nullptr,
                    0,
                    nullptr,
                    0,
                    static_cast<int>(left_frame.geometry_gray->width()),
                    static_cast<int>(left_frame.geometry_gray->height()),
                    {pair},
                    impl_->stream)) {
                return failed(attempted_method,
                              "failed to submit CUDA ROI depth matcher");
            }
            std::vector<stereo3d::DualYoloGpuCandidate> gpu_results;
            if (!impl_->matcher.collectPairs(&gpu_results) ||
                gpu_results.size() != 1U ||
                gpu_results.front().left_index != 0 ||
                gpu_results.front().right_index != 0) {
                return failed(attempted_method,
                              "failed to collect an identified CUDA depth result");
            }
            const auto& gpu = gpu_results.front();
            // 几何 CUDA 核一次会计算一组相关候选；只把冻结配置声明的
            // 方法提升到正式选择层，未配置输出不能污染证据集合。
            const auto configured = [&](std::string_view method) {
                return containsMethod(impl_->config.method_order, method);
            };
            if ((configured("roi_multi_point") &&
                 !appendGpuDisparity(
                     gpu.multi_point,
                     "roi_multi_point",
                     impl_->config,
                     candidates,
                     error)) ||
                (configured("center_patch") &&
                 !appendGpuDisparity(
                     gpu.center_patch,
                     "center_patch",
                     impl_->config,
                     candidates,
                     error)) ||
                (configured("circle_center") &&
                 !appendCirclePair(
                     gpu.left_circle,
                     gpu.right_circle,
                     impl_->config,
                     candidates,
                     error)) ||
                (configured("radial_center") &&
                 !appendPointPair(
                     gpu.left_radial_center,
                     gpu.right_radial_center,
                     "radial_center",
                     impl_->config,
                     candidates,
                     error)) ||
                (configured("edge_pair_center") &&
                 !appendPointPair(
                     gpu.left_edge_pair_center,
                     gpu.right_edge_pair_center,
                     "edge_pair_center",
                     impl_->config,
                     candidates,
                     error)) ||
                (configured("edge_centroid") &&
                 !appendPointPair(
                     gpu.left_edge_centroid,
                     gpu.right_edge_centroid,
                     "edge_centroid",
                     impl_->config,
                     candidates,
                     error))) {
                return failed(attempted_method, std::move(error));
            }
        }
        if (!appendBoundingBoxCandidates(
                request.left_detection,
                request.right_detection,
                impl_->config,
                candidates,
                error)) {
            return failed(attempted_method, std::move(error));
        }

        auto selection =
            selectDepthCandidate(candidates, impl_->config.method_order);
        if (!selection.valid_input) {
            return failed(attempted_method, std::move(selection.message));
        }
        if (!selection.candidate) {
            return rejected(attempted_method, std::move(selection.message));
        }
        const auto& candidate = *selection.candidate;
        algorithms::stereo::RefinedDisparityMeasurement measurement;
        measurement.left_anchor_px = candidate.left_anchor_px;
        measurement.disparity_px = candidate.disparity_px;
        measurement.disparity_stddev_px =
            candidate.disparity_stddev_px.value_or(
                impl_->config.geometric_disparity_stddev_px);
        measurement.anchor_stddev_px = impl_->config.anchor_stddev_px;
        measurement.confidence = candidate.confidence;
        auto triangulated = algorithms::stereo::triangulateRefinedDisparity(
            request.left_detection,
            request.right_detection,
            request.association,
            measurement,
            impl_->config.calibration,
            impl_->config.triangulation,
            request.observation_id);
        if (!triangulated) {
            return rejected(candidate.method, std::move(triangulated.message));
        }

        ports::DirectObservationEvidence evidence;
        evidence.method = candidate.method;
        evidence.left_anchor_px = candidate.left_anchor_px;
        evidence.disparity_px = candidate.disparity_px;
        evidence.disparity_stddev_px = candidate.disparity_stddev_px;
        evidence.support_points = candidate.support_points;
        return {
            ports::DirectObservationStatus::observation,
            ports::EstimatedDirectObservation{
                std::move(*triangulated.observation), std::move(evidence)},
            Method{},
            {},
        };
    } catch (const std::exception& exception) {
        return failed(
            attempted_method,
            std::string{"CUDA depth estimator exception: "} + exception.what());
    } catch (...) {
        return failed(attempted_method,
                      "CUDA depth estimator raised an unknown exception");
    }
}

}  // namespace volleyball::adapters::cuda_depth
