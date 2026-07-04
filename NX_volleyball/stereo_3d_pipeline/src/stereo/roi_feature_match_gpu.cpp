#include "roi_feature_match_gpu.h"

#include "roi_feature_match_common.h"
#include "roi_feature_match_gpu_reduce.h"
#include "../utils/logger.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>

#ifdef HAS_OPENCV_CUDAFEATURES2D
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

namespace stereo3d {

namespace {

cv::Rect clampRectToImage(int x, int y, int w, int h, int img_w, int img_h)
{
    return cv::Rect(x, y, w, h) & cv::Rect(0, 0, img_w, img_h);
}

enum class CudaDenseStereoMode {
    BM,
    SGM,
};

struct CudaDenseStereoWorkRects {
    cv::Rect left;
    cv::Rect right;
    int min_disparity = 0;
    int num_disparities = 64;
};

struct CudaDenseStereoScratch {
    int bm_num_disparities = 0;
    int bm_block_size = 0;
    int sgm_num_disparities = 0;
    cv::Ptr<cv::cuda::StereoBM> bm;
    cv::Ptr<cv::cuda::StereoSGM> sgm;
    cv::cuda::GpuMat left_work;
    cv::cuda::GpuMat right_work;
    cv::cuda::GpuMat disparity;

    void ensureBM(int num_disparities, int block_size) {
        if (!bm || bm_num_disparities != num_disparities ||
            bm_block_size != block_size) {
            bm_num_disparities = num_disparities;
            bm_block_size = block_size;
            bm = cv::cuda::createStereoBM(num_disparities, block_size);
            bm->setMinDisparity(0);
            bm->setTextureThreshold(0);
            bm->setUniquenessRatio(5);
        }
    }

    void ensureSGM(int num_disparities) {
        if (!sgm || sgm_num_disparities != num_disparities) {
            sgm_num_disparities = num_disparities;
            sgm = cv::cuda::createStereoSGM(0, num_disparities, 10, 120, 5);
        }
    }
};

int denseStereoNumDisparities(float initial_disp)
{
    (void)initial_disp;
    // Keep the real-time P2 test bounded. The global disparity prior is folded
    // into the crop offset, so the CUDA matcher only searches a local range.
    return 64;
}

bool buildDenseStereoWorkRects(
    const Detection& left_det,
    float initial_disp,
    int img_w,
    int img_h,
    int num_disparities,
    int block_size,
    CudaDenseStereoWorkRects& out)
{
    if (num_disparities <= 0 || block_size <= 0 ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return false;
    }

    const int block_margin = block_size / 2 + 4;
    const cv::Rect ball_roi = featureROIFromDetectionCPU(
        left_det, img_w, img_h, block_margin, 0.62f, 0);
    if (ball_roi.empty()) return false;

    const int min_disp = std::max(
        0,
        static_cast<int>(std::floor(initial_disp -
                                    static_cast<float>(num_disparities) * 0.5f)));
    const int pad_left = num_disparities + block_margin;
    const int pad_right = block_margin;
    const int x = ball_roi.x - pad_left;
    const int y = ball_roi.y - block_margin;
    const int w = ball_roi.width + pad_left + pad_right;
    const int h = ball_roi.height + block_margin * 2;
    const cv::Rect full(0, 0, img_w, img_h);
    const cv::Rect left_rect(x, y, w, h);
    const cv::Rect right_rect(x - min_disp, y, w, h);
    if ((left_rect & full) != left_rect ||
        (right_rect & full) != right_rect ||
        left_rect.width <= num_disparities + block_size ||
        left_rect.height <= block_size) {
        return false;
    }

    out.left = left_rect;
    out.right = right_rect;
    out.min_disparity = min_disp;
    out.num_disparities = num_disparities;
    return true;
}

float readDenseLocalDisparity(const cv::Mat& disparity, int x, int y)
{
    if (disparity.empty() || x < 0 || y < 0 ||
        x >= disparity.cols || y >= disparity.rows) {
        return -1.0f;
    }
    switch (disparity.type()) {
    case CV_8UC1:
        return static_cast<float>(disparity.at<uint8_t>(y, x));
    case CV_16SC1:
        return static_cast<float>(disparity.at<int16_t>(y, x)) * (1.0f / 16.0f);
    case CV_32FC1:
        return disparity.at<float>(y, x);
    default:
        return -1.0f;
    }
}

void setDenseStereoDebugPatch(SparseFeatureDisparityResult& result,
                              const cv::Mat& disparity_cpu,
                              const CudaDenseStereoWorkRects& rects)
{
    if (disparity_cpu.empty() || rects.left.empty()) {
        return;
    }
    const int out_w = std::clamp(disparity_cpu.cols, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    const int out_h = std::clamp(disparity_cpu.rows, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    auto& patch = result.debug_patch;
    patch = SparseFeatureDebugPatch{};
    patch.valid = true;
    patch.width = out_w;
    patch.height = out_h;
    patch.left_x0 = static_cast<float>(rects.left.x);
    patch.left_y0 = static_cast<float>(rects.left.y);
    patch.step_x = static_cast<float>(disparity_cpu.cols) /
                   static_cast<float>(out_w);
    patch.step_y = static_cast<float>(disparity_cpu.rows) /
                   static_cast<float>(out_h);
    patch.disparity.assign(static_cast<size_t>(out_w * out_h),
                           std::numeric_limits<float>::quiet_NaN());
    patch.disparity_min = std::numeric_limits<float>::max();
    patch.disparity_max = std::numeric_limits<float>::lowest();

    int finite_count = 0;
    for (int oy = 0; oy < out_h; ++oy) {
        const int sy = std::clamp(
            static_cast<int>(std::floor(
                (static_cast<float>(oy) + 0.5f) * patch.step_y)),
            0, disparity_cpu.rows - 1);
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = std::clamp(
                static_cast<int>(std::floor(
                    (static_cast<float>(ox) + 0.5f) * patch.step_x)),
                0, disparity_cpu.cols - 1);
            const float local_disp = readDenseLocalDisparity(disparity_cpu, sx, sy);
            if (!std::isfinite(local_disp) || local_disp <= 0.5f ||
                local_disp >= static_cast<float>(rects.num_disparities) - 1.0f) {
                continue;
            }
            const float disparity =
                static_cast<float>(rects.min_disparity) + local_disp;
            patch.disparity[static_cast<size_t>(oy * out_w + ox)] = disparity;
            patch.disparity_min = std::min(patch.disparity_min, disparity);
            patch.disparity_max = std::max(patch.disparity_max, disparity);
            ++finite_count;
        }
    }
    if (finite_count == 0) {
        patch = SparseFeatureDebugPatch{};
    }
}

void setScoreDebugPatch(SparseFeatureDisparityResult& result,
                        const cv::Mat& score_cpu,
                        const cv::Rect& search_rect,
                        int patch_radius)
{
    if (score_cpu.empty() || search_rect.empty() ||
        score_cpu.channels() != 1 || score_cpu.depth() != CV_32F) {
        return;
    }
    const int out_w = std::clamp(score_cpu.cols, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    const int out_h = std::clamp(score_cpu.rows, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    auto& patch = result.debug_patch;
    patch = SparseFeatureDebugPatch{};
    patch.valid = true;
    patch.disparity_is_score = true;
    patch.width = out_w;
    patch.height = out_h;
    patch.left_x0 = static_cast<float>(search_rect.x + patch_radius);
    patch.left_y0 = static_cast<float>(search_rect.y + patch_radius);
    patch.step_x = static_cast<float>(score_cpu.cols) /
                   static_cast<float>(out_w);
    patch.step_y = static_cast<float>(score_cpu.rows) /
                   static_cast<float>(out_h);
    patch.disparity.assign(static_cast<size_t>(out_w * out_h),
                           std::numeric_limits<float>::quiet_NaN());
    patch.disparity_min = std::numeric_limits<float>::max();
    patch.disparity_max = std::numeric_limits<float>::lowest();

    int finite_count = 0;
    for (int oy = 0; oy < out_h; ++oy) {
        const int sy = std::clamp(
            static_cast<int>(std::floor(
                (static_cast<float>(oy) + 0.5f) * patch.step_y)),
            0, score_cpu.rows - 1);
        const auto* row = score_cpu.ptr<float>(sy);
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = std::clamp(
                static_cast<int>(std::floor(
                    (static_cast<float>(ox) + 0.5f) * patch.step_x)),
                0, score_cpu.cols - 1);
            const float score = row[sx];
            if (!std::isfinite(score)) {
                continue;
            }
            const float clipped = std::clamp(score, -1.0f, 1.0f);
            patch.disparity[static_cast<size_t>(oy * out_w + ox)] = clipped;
            patch.disparity_min = std::min(patch.disparity_min, clipped);
            patch.disparity_max = std::max(patch.disparity_max, clipped);
            ++finite_count;
        }
    }
    if (finite_count == 0) {
        patch = SparseFeatureDebugPatch{};
    }
}

SparseFeatureDisparityResult aggregateDenseStereoDisparity(
    const cv::Mat& disparity_cpu,
    const CudaDenseStereoWorkRects& rects,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline)
{
    SparseFeatureDisparityResult result;
    if (disparity_cpu.empty()) {
        result.low_confidence = true;
        return result;
    }
    if (cfg.debug_patch_enabled) {
        setDenseStereoDebugPatch(result, disparity_cpu, rects);
    }

    const float max_delta = computeFeatureDeltaGate(
        initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int max_samples = std::clamp(
        std::max(cfg.subpixel_max_points * 4, 24), 16, 96);
    const int min_points = std::clamp(
        std::max(4, cfg.subpixel_min_points), 4, max_samples);
    const int ball_w = std::max(1, static_cast<int>(std::round(left_det.width)));
    const int ball_h = std::max(1, static_cast<int>(std::round(left_det.height)));
    const int approx_area = std::max(1, ball_w * ball_h);
    const int sample_step = std::max(
        1,
        static_cast<int>(std::ceil(std::sqrt(
            static_cast<double>(approx_area) /
            static_cast<double>(max_samples)))));

    std::vector<RobustMatchSample> samples;
    samples.reserve(static_cast<size_t>(max_samples) * 2U);
    int attempted = 0;
    for (int y = 0; y < disparity_cpu.rows; y += sample_step) {
        const float left_y = static_cast<float>(rects.left.y + y);
        if (left_y < left_det.cy - left_det.height * 0.65f ||
            left_y > left_det.cy + left_det.height * 0.65f) {
            continue;
        }
        for (int x = 0; x < disparity_cpu.cols; x += sample_step) {
            const float left_x = static_cast<float>(rects.left.x + x);
            if (!pointInsideDetectionEllipse(left_det, left_x, left_y, 0.62f)) {
                continue;
            }
            ++attempted;
            const float local_disp = readDenseLocalDisparity(disparity_cpu, x, y);
            if (!std::isfinite(local_disp) ||
                local_disp <= 0.5f ||
                local_disp >= static_cast<float>(rects.num_disparities) - 1.0f) {
                continue;
            }
            const float disparity =
                static_cast<float>(rects.min_disparity) + local_disp;
            if (disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            RobustMatchSample sample;
            sample.left_x = left_x;
            sample.left_y = left_y;
            sample.right_x = left_x - disparity;
            sample.right_y = left_y;
            sample.disparity = disparity;
            sample.score = 1.0f - std::min(
                1.0f,
                std::abs(disparity - initial_disp) /
                    std::max(1.0f, max_delta));
            sample.score = std::max(0.15f, sample.score);
            if (std::abs(featureYResidual(sample, left_det, cfg)) >
                    strictFeatureYTolerance(cfg) ||
                !passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
            }
            samples.push_back(sample);
        }
    }
    result.attempted = attempted;
    if (static_cast<int>(samples.size()) < min_points) {
        result.low_confidence = true;
        return result;
    }

    if (static_cast<int>(samples.size()) > max_samples) {
        std::vector<RobustMatchSample> limited;
        limited.reserve(static_cast<size_t>(max_samples));
        for (int i = 0; i < max_samples; ++i) {
            const size_t idx =
                static_cast<size_t>(i) * samples.size() /
                static_cast<size_t>(max_samples);
            limited.push_back(samples[std::min(idx, samples.size() - 1U)]);
        }
        samples.swap(limited);
    }

    const RobustAggregate robust = aggregateRobustMatches(
        samples, min_points, max_samples, initial_disp, max_delta,
        max_stddev, cfg);
    if (!robust.valid) {
        result.low_confidence = true;
        return result;
    }

    const float support_ratio =
        static_cast<float>(robust.support) /
        static_cast<float>(std::max(1, max_samples));
    const float consistency = std::clamp(
        1.0f / (1.0f + robust.stddev), 0.0f, 1.0f);
    const float delta_conf = 1.0f -
        std::min(1.0f, std::abs(robust.disparity - initial_disp) /
                           std::max(0.25f, max_delta));
    result.valid = true;
    result.disparity = robust.disparity;
    result.anchor_cx = robust.anchor_x;
    result.anchor_cy = robust.anchor_y;
    result.right_anchor_cx = robust.right_anchor_x;
    result.right_anchor_cy = robust.right_anchor_y;
    result.support = robust.support;
    copyDebugMatches(robust, result);
    result.stddev = robust.stddev;
    result.confidence = std::clamp(
        0.35f * support_ratio +
        0.40f * consistency +
        0.25f * delta_conf,
        0.0f, 1.0f);
    if (result.confidence < cfg.subpixel_min_confidence) {
        result.valid = false;
        result.low_confidence = true;
    }
    return result;
}

}  // namespace

#ifdef HAS_OPENCV_CUDAFEATURES2D
namespace {

struct OpenCVCudaOrbScratch {
    int max_points = 0;
    int edge_threshold = 0;
    int patch_size = 0;
    cv::Ptr<cv::cuda::ORB> orb;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
    cv::cuda::GpuMat left_proc;
    cv::cuda::GpuMat right_proc;
    cv::cuda::GpuMat left_keypoints;
    cv::cuda::GpuMat right_keypoints;
    cv::cuda::GpuMat left_desc;
    cv::cuda::GpuMat right_desc;
    cv::cuda::GpuMat knn;
    cv::cuda::GpuMat reverse_knn;

    void ensure(int requested_max_points,
                int requested_edge_threshold,
                int requested_patch_size) {
        if (!orb ||
            max_points != requested_max_points ||
            edge_threshold != requested_edge_threshold ||
            patch_size != requested_patch_size) {
            max_points = requested_max_points;
            edge_threshold = requested_edge_threshold;
            patch_size = requested_patch_size;
            orb = cv::cuda::ORB::create(
                max_points, 1.2f, 1, edge_threshold, 0, 2,
                cv::ORB::HARRIS_SCORE, patch_size, 12, false);
        }
        if (!matcher) {
            matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        }
    }
};

}  // namespace
#endif

namespace {

struct CudaTemplatePeakScratch {
    CudaTemplateScorePeak* device = nullptr;
    CudaTemplateScorePeak* host = nullptr;

    ~CudaTemplatePeakScratch() {
        if (device) {
            cudaFree(device);
            device = nullptr;
        }
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
        }
    }

    bool ensure() {
        if (!device) {
            const cudaError_t err =
                cudaMalloc(reinterpret_cast<void**>(&device),
                           sizeof(CudaTemplateScorePeak));
            if (err != cudaSuccess) {
                device = nullptr;
                return false;
            }
        }
        if (!host) {
            const cudaError_t err =
                cudaHostAlloc(reinterpret_cast<void**>(&host),
                              sizeof(CudaTemplateScorePeak),
                              cudaHostAllocDefault);
            if (err != cudaSuccess) {
                host = nullptr;
                return false;
            }
        }
        return true;
    }
};

bool findTemplatePeakOnGpu(const cv::cuda::GpuMat& score_gpu,
                           cudaStream_t stream,
                           double* max_value,
                           cv::Point* max_loc) {
    if (score_gpu.empty() || score_gpu.type() != CV_32FC1 ||
        !stream || !max_value || !max_loc) {
        return false;
    }
    thread_local CudaTemplatePeakScratch scratch;
    if (!scratch.ensure()) {
        return false;
    }
    cudaError_t err = findCudaTemplateScorePeak(
        score_gpu.ptr<float>(),
        score_gpu.step,
        score_gpu.cols,
        score_gpu.rows,
        scratch.device,
        scratch.host,
        stream);
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess || !scratch.host->valid) {
        return false;
    }
    *max_value = static_cast<double>(scratch.host->value);
    *max_loc = cv::Point(scratch.host->x, scratch.host->y);
    return true;
}

bool findTemplatePeakOnCpu(const cv::cuda::GpuMat& score_gpu,
                           cv::cuda::Stream& cv_stream,
                           double* max_value,
                           cv::Point* max_loc) {
    if (score_gpu.empty() || !max_value || !max_loc) {
        return false;
    }
    cv::Mat score_cpu;
    score_gpu.download(score_cpu, cv_stream);
    cv_stream.waitForCompletion();
    if (score_cpu.empty()) {
        return false;
    }
    double min_value = 0.0;
    cv::Point min_loc;
    cv::minMaxLoc(score_cpu, &min_value, max_value, &min_loc, max_loc);
    return true;
}

}  // namespace

SparseFeatureDisparityResult matchOpenCVORBDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream)
{
    SparseFeatureDisparityResult result;
#ifndef HAS_OPENCV_CUDAFEATURES2D
    (void)left_gpu;
    (void)left_pitch;
    (void)right_gpu;
    (void)right_pitch;
    (void)img_w;
    (void)img_h;
    (void)left_det;
    (void)right_det;
    (void)initial_disp;
    (void)cfg;
    (void)max_disparity;
    (void)focal;
    (void)baseline;
    (void)stream;
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
        LOG_WARN("OpenCV CUDA ORB requested but cudafeatures2d was not available at build time");
    });
    result.low_confidence = true;
    return result;
#else
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return result;
    }

    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 2, 10);
    const int max_points = std::clamp(std::max(cfg.subpixel_max_points * 4, 48),
                                      16, 160);
    const int min_points = std::clamp(std::max(3, cfg.subpixel_min_points),
                                      3, max_points);
    const int search_radius = std::max(1, cfg.subpixel_search_radius_px);
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int extra_margin = search_radius + static_cast<int>(std::ceil(max_delta)) + 2;
    const int border = std::max(2, patch_radius);
    const cv::Rect left_roi = featureROIFromDetectionCPU(
        left_det, img_w, img_h, border, 0.56f, 2);
    const cv::Rect right_roi = featureROIFromDetectionCPU(
        right_det, img_w, img_h, border, 0.62f, extra_margin);
    if (left_roi.empty() || right_roi.empty()) {
        result.low_confidence = true;
        return result;
    }

    try {
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        const int patch_size = std::max(9, patch_radius * 2 + 1);
        const int edge_threshold = std::clamp(patch_radius + 3, 5, 16);
        thread_local OpenCVCudaOrbScratch scratch;
        scratch.ensure(max_points, edge_threshold, patch_size);

        cv::cuda::GpuMat left_view(left_full, left_roi);
        cv::cuda::GpuMat right_view(right_full, right_roi);
        left_view.copyTo(scratch.left_proc, cv_stream);
        right_view.copyTo(scratch.right_proc, cv_stream);

        scratch.orb->detectAndComputeAsync(scratch.left_proc, cv::noArray(),
                                           scratch.left_keypoints, scratch.left_desc,
                                           false, cv_stream);
        scratch.orb->detectAndComputeAsync(scratch.right_proc, cv::noArray(),
                                           scratch.right_keypoints, scratch.right_desc,
                                           false, cv_stream);
        cv_stream.waitForCompletion();

        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> right_keypoints;
        scratch.orb->convert(scratch.left_keypoints, left_keypoints);
        scratch.orb->convert(scratch.right_keypoints, right_keypoints);
        if (left_keypoints.size() < static_cast<size_t>(min_points) ||
            right_keypoints.size() < static_cast<size_t>(min_points) ||
            scratch.left_desc.empty() || scratch.right_desc.empty() ||
            scratch.left_desc.type() != CV_8U || scratch.right_desc.type() != CV_8U) {
            result.low_confidence = true;
            result.attempted = static_cast<int>(left_keypoints.size());
            return result;
        }

        scratch.matcher->knnMatchAsync(scratch.left_desc, scratch.right_desc,
                                       scratch.knn, 2, cv::noArray(), cv_stream);
        scratch.matcher->knnMatchAsync(scratch.right_desc, scratch.left_desc,
                                       scratch.reverse_knn, 2, cv::noArray(), cv_stream);
        cv_stream.waitForCompletion();

        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<std::vector<cv::DMatch>> reverse_knn_matches;
        scratch.matcher->knnMatchConvert(scratch.knn, knn_matches);
        scratch.matcher->knnMatchConvert(scratch.reverse_knn, reverse_knn_matches);
        result.attempted = static_cast<int>(knn_matches.size());
        if (result.attempted < min_points) {
            result.low_confidence = true;
            return result;
        }

        const float ratio_thresh = 0.78f;
        std::vector<int> reverse_best(right_keypoints.size(), -1);
        for (const auto& pair : reverse_knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(right_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(left_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            reverse_best[static_cast<size_t>(best.queryIdx)] = best.trainIdx;
        }

        const float max_hamming = static_cast<float>(
            std::max(1, scratch.left_desc.cols * 8));
        const float min_score = std::max(
            0.45f,
            0.35f + cfg.subpixel_min_confidence * 0.45f);
        std::vector<RobustMatchSample> samples;
        samples.reserve(knn_matches.size());

        for (const auto& pair : knn_matches) {
            if (pair.empty()) continue;
            const cv::DMatch& best = pair[0];
            if (best.queryIdx < 0 || best.trainIdx < 0 ||
                best.queryIdx >= static_cast<int>(left_keypoints.size()) ||
                best.trainIdx >= static_cast<int>(right_keypoints.size())) {
                continue;
            }
            if (pair.size() > 1 && pair[1].distance > 0.0f &&
                best.distance > ratio_thresh * pair[1].distance) {
                continue;
            }
            if (cfg.feature_reverse_check_px >= 0.0f &&
                (best.trainIdx >= static_cast<int>(reverse_best.size()) ||
                 reverse_best[static_cast<size_t>(best.trainIdx)] != best.queryIdx)) {
                continue;
            }

            const cv::KeyPoint& kl = left_keypoints[static_cast<size_t>(best.queryIdx)];
            const cv::KeyPoint& kr = right_keypoints[static_cast<size_t>(best.trainIdx)];
            const float lx = static_cast<float>(left_roi.x) + kl.pt.x;
            const float ly = static_cast<float>(left_roi.y) + kl.pt.y;
            const float rx = static_cast<float>(right_roi.x) + kr.pt.x;
            const float ry = static_cast<float>(right_roi.y) + kr.pt.y;
            const float disparity = lx - rx;
            if (disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }

            const float score = 1.0f - std::min(1.0f, best.distance / max_hamming);
            if (score < min_score) continue;

            RobustMatchSample sample;
            sample.left_x = lx;
            sample.left_y = ly;
            sample.right_x = rx;
            sample.right_y = ry;
            sample.disparity = disparity;
            sample.score = score;
            if (std::abs(featureYResidual(sample, left_det, cfg)) >
                    strictFeatureYTolerance(cfg) ||
                !passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
            }
            samples.push_back(sample);
        }

        if (static_cast<int>(samples.size()) < min_points) {
            result.low_confidence = true;
            return result;
        }

        const RobustAggregate robust = aggregateRobustMatches(
            samples, min_points, max_points, initial_disp, max_delta,
            max_stddev, cfg);
        if (!robust.valid) {
            result.low_confidence = true;
            return result;
        }

        result.disparity = robust.disparity;
        result.anchor_cx = robust.anchor_x;
        result.anchor_cy = robust.anchor_y;
        result.right_anchor_cx = robust.right_anchor_x;
        result.right_anchor_cy = robust.right_anchor_y;
        result.support = robust.support;
        copyDebugMatches(robust, result);
        result.stddev = robust.stddev;
        if (result.stddev > max_stddev ||
            std::abs(result.disparity - initial_disp) > max_delta ||
            result.disparity <= 0.5f ||
            result.disparity > static_cast<float>(max_disparity)) {
            result.low_confidence = true;
            return result;
        }

        const float support_ratio = static_cast<float>(robust.support) /
                                    static_cast<float>(std::max(1, max_points));
        const float score_conf = std::clamp((robust.mean_score - min_score) /
                                            std::max(0.01f, 1.0f - min_score),
                                            0.0f, 1.0f);
        const float consistency = std::clamp(1.0f / (1.0f + result.stddev),
                                             0.0f, 1.0f);
        const float delta_conf = 1.0f -
            std::min(1.0f, std::abs(result.disparity - initial_disp) / max_delta);
        result.confidence = std::clamp(0.30f * support_ratio +
                                       0.35f * score_conf +
                                       0.25f * consistency +
                                       0.10f * delta_conf,
                                       0.0f, 1.0f);
        if (result.confidence < cfg.subpixel_min_confidence) {
            result.low_confidence = true;
            return result;
        }
        result.valid = true;
        return result;
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV CUDA ORB ROI feature match failed: %s", e.what());
        return result;
    }
#endif
}

SparseFeatureDisparityResult matchCudaTemplateDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream)
{
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        left_det.width < 6.0f || left_det.height < 6.0f ||
        right_det.width < 6.0f || right_det.height < 6.0f) {
        return result;
    }

    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 3, 16);
    const int patch_size = patch_radius * 2 + 1;
    const int search_radius = std::clamp(cfg.subpixel_search_radius_px, 2, 64);
    const int y_radius = std::max(
        1,
        static_cast<int>(std::ceil(strictFeatureYTolerance(cfg))));
    const float max_delta = computeFeatureDeltaGate(initial_disp, focal, baseline, cfg);
    const float anchor_x = left_det.cx;
    const float anchor_y = left_det.cy;
    const int templ_x = static_cast<int>(std::round(anchor_x)) - patch_radius;
    const int templ_y = static_cast<int>(std::round(anchor_y)) - patch_radius;
    const cv::Rect templ_rect =
        clampRectToImage(templ_x, templ_y, patch_size, patch_size, img_w, img_h);
    if (templ_rect.width != patch_size || templ_rect.height != patch_size) {
        result.low_confidence = true;
        return result;
    }

    const float predicted_right_x = anchor_x - initial_disp;
    const int search_x = static_cast<int>(std::round(predicted_right_x)) -
                         patch_radius - search_radius;
    const int search_y = static_cast<int>(std::round(anchor_y)) -
                         patch_radius - y_radius;
    const int search_w = patch_size + search_radius * 2;
    const int search_h = patch_size + y_radius * 2;
    const cv::Rect search_rect =
        clampRectToImage(search_x, search_y, search_w, search_h, img_w, img_h);
    if (search_rect.width != search_w || search_rect.height != search_h ||
        search_rect.width < patch_size || search_rect.height < patch_size) {
        result.low_confidence = true;
        return result;
    }

    try {
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        cv::cuda::GpuMat templ(left_full, templ_rect);
        cv::cuda::GpuMat search(right_full, search_rect);

        thread_local cv::Ptr<cv::cuda::TemplateMatching> matcher;
        if (!matcher) {
            matcher = cv::cuda::createTemplateMatching(
                CV_8UC1, cv::TM_CCOEFF_NORMED);
        }
        thread_local cv::cuda::GpuMat score_gpu;
        matcher->match(search, templ, score_gpu, cv_stream);
        cv_stream.waitForCompletion();
        if (score_gpu.empty()) {
            result.low_confidence = true;
            return result;
        }

        double max_val = 0.0;
        cv::Point max_loc;
        if (!findTemplatePeakOnGpu(score_gpu, stream, &max_val, &max_loc) &&
            !findTemplatePeakOnCpu(score_gpu, cv_stream, &max_val, &max_loc)) {
            result.low_confidence = true;
            return result;
        }
        if (cfg.debug_patch_enabled) {
            cv::Mat score_cpu;
            score_gpu.download(score_cpu, cv_stream);
            cv_stream.waitForCompletion();
            setScoreDebugPatch(result, score_cpu, search_rect, patch_radius);
        }

        const float right_anchor_x =
            static_cast<float>(search_rect.x + max_loc.x + patch_radius);
        const float right_anchor_y =
            static_cast<float>(search_rect.y + max_loc.y + patch_radius);
        const float disparity = anchor_x - right_anchor_x;
        RobustMatchSample sample;
        sample.left_x = anchor_x;
        sample.left_y = anchor_y;
        sample.right_x = right_anchor_x;
        sample.right_y = right_anchor_y;
        sample.disparity = disparity;
        sample.score = static_cast<float>(std::clamp(max_val, 0.0, 1.0));
        result.disparity = disparity;
        result.confidence = sample.score;
        result.anchor_cx = anchor_x;
        result.anchor_cy = anchor_y;
        result.right_anchor_cx = right_anchor_x;
        result.right_anchor_cy = right_anchor_y;
        setSingleDebugMatch(sample, result);

        result.attempted = 1;
        if (sample.score < std::max(0.05f, cfg.subpixel_min_confidence) ||
            disparity <= 0.5f ||
            disparity > static_cast<float>(max_disparity) ||
            std::abs(disparity - initial_disp) > max_delta ||
            std::abs(featureYResidual(sample, left_det, cfg)) >
                strictFeatureYTolerance(cfg) ||
            !passesFeatureOverlapGate(sample, left_det, right_det,
                                      initial_disp, cfg) ||
            !passesSphereRadiusGate(sample, left_det, initial_disp,
                                    focal, baseline, cfg)) {
            result.low_confidence = true;
            return result;
        }

        result.valid = true;
        result.stddev = 0.0f;
        result.support = 1;
        return result;
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV CUDA TemplateMatching ROI match failed: %s", e.what());
        return result;
    }
}

namespace {

SparseFeatureDisparityResult matchCudaDenseStereoDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream,
    CudaDenseStereoMode mode)
{
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        left_det.width < 8.0f || left_det.height < 8.0f ||
        right_det.width < 8.0f || right_det.height < 8.0f) {
        return result;
    }

    const int num_disparities = denseStereoNumDisparities(initial_disp);
    const int block_size = std::clamp(
        std::max(5, cfg.subpixel_patch_radius * 2 + 1), 5, 15) | 1;
    CudaDenseStereoWorkRects rects;
    if (!buildDenseStereoWorkRects(left_det, initial_disp, img_w, img_h,
                                   num_disparities, block_size, rects)) {
        result.low_confidence = true;
        return result;
    }

    try {
        cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        cv::cuda::GpuMat left_view(left_full, rects.left);
        cv::cuda::GpuMat right_view(right_full, rects.right);

        thread_local CudaDenseStereoScratch scratch;
        left_view.copyTo(scratch.left_work, cv_stream);
        right_view.copyTo(scratch.right_work, cv_stream);

        if (mode == CudaDenseStereoMode::BM) {
            scratch.ensureBM(num_disparities, block_size);
            scratch.bm->compute(scratch.left_work, scratch.right_work,
                                scratch.disparity, cv_stream);
        } else {
            scratch.ensureSGM(num_disparities);
            scratch.sgm->compute(scratch.left_work, scratch.right_work,
                                 scratch.disparity, cv_stream);
        }

        cv::Mat disparity_cpu;
        scratch.disparity.download(disparity_cpu, cv_stream);
        cv_stream.waitForCompletion();
        return aggregateDenseStereoDisparity(
            disparity_cpu, rects, left_det, right_det,
            initial_disp, cfg, max_disparity, focal, baseline);
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV CUDA dense ROI stereo match failed: %s", e.what());
        return result;
    }
}

}  // namespace

SparseFeatureDisparityResult matchCudaStereoBMDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream)
{
    return matchCudaDenseStereoDisparityGPU(
        left_gpu, left_pitch, right_gpu, right_pitch,
        img_w, img_h, left_det, right_det, initial_disp, cfg,
        max_disparity, focal, baseline, stream, CudaDenseStereoMode::BM);
}

SparseFeatureDisparityResult matchCudaStereoSGMDisparityGPU(
    const uint8_t* left_gpu, int left_pitch,
    const uint8_t* right_gpu, int right_pitch,
    int img_w, int img_h,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline,
    cudaStream_t stream)
{
    return matchCudaDenseStereoDisparityGPU(
        left_gpu, left_pitch, right_gpu, right_pitch,
        img_w, img_h, left_det, right_det, initial_disp, cfg,
        max_disparity, focal, baseline, stream, CudaDenseStereoMode::SGM);
}

}  // namespace stereo3d
