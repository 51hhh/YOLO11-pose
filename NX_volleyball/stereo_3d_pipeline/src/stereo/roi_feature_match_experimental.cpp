#include "roi_feature_match_gpu.h"

#include "roi_feature_match_common.h"
#include "roi_feature_match_gpu_reduce.h"
#include "../utils/logger.h"

#include <cuda.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <vpi/CUDAInterop.h>
#include <vpi/Image.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/TemplateMatching.h>

#ifdef HAS_OPENCV_CUDAOPTFLOW
#include <opencv2/cudaoptflow.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace stereo3d {
namespace {

cv::Rect clampRectToImage(int x, int y, int w, int h, int img_w, int img_h) {
    return cv::Rect(x, y, w, h) & cv::Rect(0, 0, img_w, img_h);
}

bool buildShiftedWorkRects(const Detection& left_det,
                           float initial_disp,
                           int img_w,
                           int img_h,
                           int border,
                           float scale,
                           cv::Rect& left_rect,
                           cv::Rect& right_rect) {
    if (!std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return false;
    }
    left_rect = featureROIFromDetectionCPU(
        left_det, img_w, img_h, border, scale, border);
    if (left_rect.empty() || left_rect.width < 12 || left_rect.height < 12) {
        return false;
    }
    right_rect = cv::Rect(
        static_cast<int>(std::round(static_cast<float>(left_rect.x) -
                                    initial_disp)),
        left_rect.y,
        left_rect.width,
        left_rect.height);
    const cv::Rect full(0, 0, img_w, img_h);
    if ((right_rect & full) != right_rect) {
        return false;
    }
    return true;
}

bool buildResidualSearchWorkRects(const Detection& left_det,
                                  float initial_disp,
                                  float residual_min,
                                  int img_w,
                                  int img_h,
                                  int border,
                                  float scale,
                                  cv::Rect& left_rect,
                                  cv::Rect& right_rect,
                                  float* crop_shift_out) {
    if (!std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        !std::isfinite(residual_min)) {
        return false;
    }
    left_rect = featureROIFromDetectionCPU(
        left_det, img_w, img_h, border, scale, border);
    if (left_rect.empty() || left_rect.width < 12 || left_rect.height < 12) {
        return false;
    }
    const float crop_shift = initial_disp + residual_min;
    right_rect = cv::Rect(
        static_cast<int>(std::round(static_cast<float>(left_rect.x) -
                                    crop_shift)),
        left_rect.y,
        left_rect.width,
        left_rect.height);
    const cv::Rect full(0, 0, img_w, img_h);
    if ((right_rect & full) != right_rect) {
        return false;
    }
    if (crop_shift_out) {
        *crop_shift_out = static_cast<float>(left_rect.x - right_rect.x);
    }
    return true;
}

VPIImageData makeCudaPitchImageData(const uint8_t* ptr,
                                    int pitch,
                                    int width,
                                    int height) {
    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    data.buffer.pitch.format = VPI_IMAGE_FORMAT_U8;
    data.buffer.pitch.numPlanes = 1;
    data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
    data.buffer.pitch.planes[0].width = width;
    data.buffer.pitch.planes[0].height = height;
    data.buffer.pitch.planes[0].pitchBytes = pitch;
    data.buffer.pitch.planes[0].data = const_cast<uint8_t*>(ptr);
    return data;
}

void destroyVpiImage(VPIImage& image) {
    if (image) {
        vpiImageDestroy(image);
        image = nullptr;
    }
}

void destroyVpiPayload(VPIPayload& payload) {
    if (payload) {
        vpiPayloadDestroy(payload);
        payload = nullptr;
    }
}

void destroyVpiStream(VPIStream& stream) {
    if (stream) {
        vpiStreamDestroy(stream);
        stream = nullptr;
    }
}

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

bool findVpiCudaScorePeak(VPIImage score,
                          cudaStream_t stream,
                          float* best,
                          int* best_x,
                          int* best_y) {
    if (!score || !stream || !best || !best_x || !best_y) {
        return false;
    }
    VPIImageData score_data{};
    VPIStatus st = vpiImageLockData(score, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR,
                                    &score_data);
    if (st != VPI_SUCCESS) {
        return false;
    }
    const int out_w = score_data.buffer.pitch.planes[0].width;
    const int out_h = score_data.buffer.pitch.planes[0].height;
    const int pitch = score_data.buffer.pitch.planes[0].pitchBytes;
    const auto* score_gpu = static_cast<const float*>(
        score_data.buffer.pitch.planes[0].data);

    thread_local CudaTemplatePeakScratch scratch;
    bool ok = false;
    if (score_gpu && scratch.ensure()) {
        cudaError_t err = findCudaTemplateScorePeak(
            score_gpu, static_cast<size_t>(pitch), out_w, out_h,
            scratch.device, scratch.host, stream);
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(stream);
        }
        ok = err == cudaSuccess && scratch.host->valid != 0;
    }
    vpiImageUnlock(score);
    if (!ok) {
        return false;
    }
    *best = scratch.host->value;
    *best_x = scratch.host->x;
    *best_y = scratch.host->y;
    return true;
}

struct VpiTemplateScratch {
    CUstream cuda_stream = nullptr;
    VPIStream stream = nullptr;
    VPIPayload payload = nullptr;
    VPIImage source = nullptr;
    VPIImage templ = nullptr;
    VPIImage score = nullptr;
    int source_w = 0;
    int source_h = 0;
    int templ_w = 0;
    int templ_h = 0;
    int score_w = 0;
    int score_h = 0;

    ~VpiTemplateScratch() { reset(); }

    void reset() {
        destroyVpiImage(score);
        destroyVpiImage(templ);
        destroyVpiImage(source);
        destroyVpiPayload(payload);
        destroyVpiStream(stream);
        cuda_stream = nullptr;
        source_w = source_h = templ_w = templ_h = score_w = score_h = 0;
    }

    bool ensure(CUstream cu_stream,
                const uint8_t* source_ptr,
                int source_pitch,
                int requested_source_w,
                int requested_source_h,
                const uint8_t* templ_ptr,
                int templ_pitch,
                int requested_templ_w,
                int requested_templ_h,
                int requested_score_w,
                int requested_score_h) {
        const bool dims_changed =
            source_w != requested_source_w ||
            source_h != requested_source_h ||
            templ_w != requested_templ_w ||
            templ_h != requested_templ_h ||
            score_w != requested_score_w ||
            score_h != requested_score_h;
        if (cuda_stream != cu_stream || dims_changed) {
            reset();
        }
        if (!stream &&
            vpiStreamCreateWrapperCUDA(cu_stream, VPI_BACKEND_CUDA,
                                       &stream) != VPI_SUCCESS) {
            reset();
            return false;
        }

        const VPIImageData source_data = makeCudaPitchImageData(
            source_ptr, source_pitch, requested_source_w, requested_source_h);
        const VPIImageData templ_data = makeCudaPitchImageData(
            templ_ptr, templ_pitch, requested_templ_w, requested_templ_h);
        if (!source) {
            if (vpiImageCreateWrapper(&source_data, nullptr,
                                      VPI_BACKEND_CUDA, &source) !=
                VPI_SUCCESS) {
                reset();
                return false;
            }
        } else if (vpiImageSetWrapper(source, &source_data) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!templ) {
            if (vpiImageCreateWrapper(&templ_data, nullptr,
                                      VPI_BACKEND_CUDA, &templ) !=
                VPI_SUCCESS) {
                reset();
                return false;
            }
        } else if (vpiImageSetWrapper(templ, &templ_data) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!payload &&
            vpiCreateTemplateMatching(VPI_BACKEND_CUDA,
                                      requested_source_w,
                                      requested_source_h,
                                      &payload) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!score &&
            vpiImageCreate(requested_score_w, requested_score_h,
                           VPI_IMAGE_FORMAT_F32,
                           VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                           &score) != VPI_SUCCESS) {
            reset();
            return false;
        }
        cuda_stream = cu_stream;
        source_w = requested_source_w;
        source_h = requested_source_h;
        templ_w = requested_templ_w;
        templ_h = requested_templ_h;
        score_w = requested_score_w;
        score_h = requested_score_h;
        return true;
    }
};

struct VpiStereoScratch {
    CUstream cuda_stream = nullptr;
    VPIStream stream = nullptr;
    VPIPayload payload = nullptr;
    VPIImage left = nullptr;
    VPIImage right = nullptr;
    VPIImage disparity = nullptr;
    VPIImage confidence = nullptr;
    int width = 0;
    int height = 0;
    int max_disparity = 0;

    ~VpiStereoScratch() { reset(); }

    void reset() {
        destroyVpiImage(confidence);
        destroyVpiImage(disparity);
        destroyVpiImage(right);
        destroyVpiImage(left);
        destroyVpiPayload(payload);
        destroyVpiStream(stream);
        cuda_stream = nullptr;
        width = height = max_disparity = 0;
    }

    bool ensure(CUstream cu_stream,
                const uint8_t* left_ptr,
                int left_pitch,
                const uint8_t* right_ptr,
                int right_pitch,
                int requested_w,
                int requested_h,
                int requested_max_disparity) {
        const bool dims_changed =
            width != requested_w ||
            height != requested_h ||
            max_disparity != requested_max_disparity;
        if (cuda_stream != cu_stream || dims_changed) {
            reset();
        }
        if (!stream &&
            vpiStreamCreateWrapperCUDA(cu_stream, VPI_BACKEND_CUDA,
                                       &stream) != VPI_SUCCESS) {
            reset();
            return false;
        }

        const VPIImageData left_data = makeCudaPitchImageData(
            left_ptr, left_pitch, requested_w, requested_h);
        const VPIImageData right_data = makeCudaPitchImageData(
            right_ptr, right_pitch, requested_w, requested_h);
        if (!left) {
            if (vpiImageCreateWrapper(&left_data, nullptr,
                                      VPI_BACKEND_CUDA, &left) !=
                VPI_SUCCESS) {
                reset();
                return false;
            }
        } else if (vpiImageSetWrapper(left, &left_data) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!right) {
            if (vpiImageCreateWrapper(&right_data, nullptr,
                                      VPI_BACKEND_CUDA, &right) !=
                VPI_SUCCESS) {
                reset();
                return false;
            }
        } else if (vpiImageSetWrapper(right, &right_data) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!payload) {
            VPIStereoDisparityEstimatorCreationParams create_params{};
            vpiInitStereoDisparityEstimatorCreationParams(&create_params);
            create_params.maxDisparity = requested_max_disparity;
            create_params.includeDiagonals = 0;
            if (vpiCreateStereoDisparityEstimator(
                    VPI_BACKEND_CUDA, requested_w, requested_h,
                    VPI_IMAGE_FORMAT_U8, &create_params, &payload) !=
                VPI_SUCCESS) {
                reset();
                return false;
            }
        }
        if (!disparity &&
            vpiImageCreate(requested_w, requested_h, VPI_IMAGE_FORMAT_S16,
                           VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                           &disparity) != VPI_SUCCESS) {
            reset();
            return false;
        }
        if (!confidence &&
            vpiImageCreate(requested_w, requested_h, VPI_IMAGE_FORMAT_U16,
                           VPI_BACKEND_CUDA | VPI_BACKEND_CPU,
                           &confidence) != VPI_SUCCESS) {
            reset();
            return false;
        }
        cuda_stream = cu_stream;
        width = requested_w;
        height = requested_h;
        max_disparity = requested_max_disparity;
        return true;
    }
};

SparseFeatureDisparityResult aggregateGpuPointMatches(
    const std::vector<cv::Point2f>& left_pts,
    const std::vector<cv::Point2f>& right_pts,
    const std::vector<uint8_t>& status,
    const cv::Rect& left_rect,
    const cv::Rect& right_rect,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline) {
    SparseFeatureDisparityResult result;
    const size_t n = std::min(left_pts.size(), right_pts.size());
    if (n == 0) {
        result.low_confidence = true;
        return result;
    }
    const float max_delta = computeFeatureDeltaGate(
        initial_disp, focal, baseline, cfg);
    const float max_stddev = std::max(0.05f, cfg.subpixel_max_stddev_px);
    const int max_points = std::clamp(
        std::max(cfg.subpixel_max_points * 4, 24), 16, 128);
    const int min_points = std::clamp(
        std::max(3, cfg.subpixel_min_points), 3, max_points);

    std::vector<RobustMatchSample> samples;
    samples.reserve(std::min(n, static_cast<size_t>(max_points)));
    for (size_t i = 0; i < n; ++i) {
        if (!status.empty() &&
            (i >= status.size() || !status[i])) {
            continue;
        }
        const float lx = static_cast<float>(left_rect.x) + left_pts[i].x;
        const float ly = static_cast<float>(left_rect.y) + left_pts[i].y;
        const float rx = static_cast<float>(right_rect.x) + right_pts[i].x;
        const float ry = static_cast<float>(right_rect.y) + right_pts[i].y;
        const float disparity = lx - rx;
        if (!std::isfinite(disparity) ||
            disparity <= 0.5f ||
            disparity > static_cast<float>(max_disparity) ||
            std::abs(disparity - initial_disp) > max_delta) {
            continue;
        }
        RobustMatchSample sample;
        sample.left_x = lx;
        sample.left_y = ly;
        sample.right_x = rx;
        sample.right_y = ry;
        sample.disparity = disparity;
        sample.score = 1.0f - std::min(
            1.0f, std::abs(disparity - initial_disp) /
                      std::max(0.25f, max_delta));
        sample.score = std::max(0.10f, sample.score);
        if (std::abs(featureYResidual(sample, left_det, cfg)) >
                strictFeatureYTolerance(cfg) ||
            !passesFeatureOverlapGate(sample, left_det, right_det,
                                      initial_disp, cfg) ||
            !passesSphereRadiusGate(sample, left_det, initial_disp,
                                    focal, baseline, cfg)) {
            continue;
        }
        samples.push_back(sample);
        if (static_cast<int>(samples.size()) >= max_points) {
            break;
        }
    }
    result.attempted = static_cast<int>(n);
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
    result.valid = true;
    result.disparity = robust.disparity;
    result.anchor_cx = robust.anchor_x;
    result.anchor_cy = robust.anchor_y;
    result.right_anchor_cx = robust.right_anchor_x;
    result.right_anchor_cy = robust.right_anchor_y;
    result.support = robust.support;
    result.stddev = robust.stddev;
    const float support_ratio =
        static_cast<float>(robust.support) /
        static_cast<float>(std::max(1, max_points));
    const float consistency =
        std::clamp(1.0f / (1.0f + robust.stddev), 0.0f, 1.0f);
    result.confidence = std::clamp(
        0.45f * consistency + 0.35f * support_ratio +
        0.20f * std::max(0.0f, robust.mean_score),
        0.0f, 1.0f);
    if (result.confidence < cfg.subpixel_min_confidence) {
        result.valid = false;
        result.low_confidence = true;
    }
    return result;
}

SparseFeatureDisparityResult unsupportedBackend(const char* name) {
    static std::mutex mutex;
    static std::set<std::string> warned;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (warned.insert(name ? name : "unknown").second) {
            LOG_WARN("%s requested but this backend is not available in this build",
                     name ? name : "unknown");
        }
    }
    SparseFeatureDisparityResult result;
    result.low_confidence = true;
    return result;
}

}  // namespace

SparseFeatureDisparityResult matchOpenCVCudaGfttLkDisparityGPU(
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
    cudaStream_t stream) {
    SparseFeatureDisparityResult result;
#ifndef HAS_OPENCV_CUDAOPTFLOW
    (void)left_gpu; (void)left_pitch; (void)right_gpu; (void)right_pitch;
    (void)img_w; (void)img_h; (void)left_det; (void)right_det;
    (void)initial_disp; (void)cfg; (void)max_disparity; (void)focal;
    (void)baseline; (void)stream;
    return unsupportedBackend("OpenCV CUDA GFTT/LK");
#else
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return result;
    }
    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 3, 12);
    cv::Rect left_rect;
    cv::Rect right_rect;
    if (!buildShiftedWorkRects(left_det, initial_disp, img_w, img_h,
                               patch_radius + 6, 1.05f,
                               left_rect, right_rect)) {
        result.low_confidence = true;
        return result;
    }
    try {
        cv::cuda::Stream cv_stream =
            cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        cv::cuda::GpuMat left_view(left_full, left_rect);
        cv::cuda::GpuMat right_view(right_full, right_rect);
        thread_local cv::cuda::GpuMat left_work;
        thread_local cv::cuda::GpuMat right_work;
        left_view.copyTo(left_work, cv_stream);
        right_view.copyTo(right_work, cv_stream);

        const int max_corners =
            std::clamp(std::max(cfg.subpixel_max_points * 8, 64), 32, 192);
        thread_local cv::Ptr<cv::cuda::CornersDetector> gftt;
        thread_local cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk;
        thread_local int gftt_max = 0;
        thread_local int lk_win = 0;
        const int win = std::clamp(patch_radius * 2 + 1, 9, 25) | 1;
        if (!gftt || gftt_max != max_corners) {
            gftt_max = max_corners;
            gftt = cv::cuda::createGoodFeaturesToTrackDetector(
                CV_8UC1, max_corners, 0.01, 3.0, 3, true, 0.04);
        }
        if (!lk || lk_win != win) {
            lk_win = win;
            lk = cv::cuda::SparsePyrLKOpticalFlow::create(
                cv::Size(win, win), 1, 12, false);
        }

        cv::cuda::GpuMat prev_pts_gpu;
        cv::cuda::GpuMat next_pts_gpu;
        cv::cuda::GpuMat status_gpu;
        cv::cuda::GpuMat err_gpu;
        gftt->detect(left_work, prev_pts_gpu, cv::noArray(), cv_stream);
        lk->calc(left_work, right_work, prev_pts_gpu, next_pts_gpu,
                 status_gpu, err_gpu, cv_stream);

        cv::Mat prev_pts_mat;
        cv::Mat next_pts_mat;
        cv::Mat status_mat;
        prev_pts_gpu.download(prev_pts_mat, cv_stream);
        next_pts_gpu.download(next_pts_mat, cv_stream);
        status_gpu.download(status_mat, cv_stream);
        cv_stream.waitForCompletion();
        if (prev_pts_mat.empty() || next_pts_mat.empty()) {
            result.low_confidence = true;
            return result;
        }

        const auto* prev_ptr = prev_pts_mat.ptr<cv::Point2f>();
        const auto* next_ptr = next_pts_mat.ptr<cv::Point2f>();
        const size_t count =
            static_cast<size_t>(prev_pts_mat.total());
        std::vector<cv::Point2f> prev(prev_ptr, prev_ptr + count);
        std::vector<cv::Point2f> next(next_ptr, next_ptr + count);
        std::vector<uint8_t> status;
        if (!status_mat.empty()) {
            const auto* status_ptr = status_mat.ptr<uint8_t>();
            status.assign(status_ptr, status_ptr + status_mat.total());
        }
        return aggregateGpuPointMatches(
            prev, next, status, left_rect, right_rect,
            left_det, right_det, initial_disp, cfg, max_disparity,
            focal, baseline);
    } catch (const cv::Exception& e) {
        LOG_WARN("OpenCV CUDA GFTT/LK ROI match failed: %s", e.what());
        return result;
    }
#endif
}

SparseFeatureDisparityResult matchCudaCannyHoughCircleDisparityGPU(
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
    cudaStream_t stream) {
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return result;
    }
    cv::Rect left_rect;
    cv::Rect right_rect;
    if (!buildShiftedWorkRects(left_det, initial_disp, img_w, img_h,
                               8, 0.92f, left_rect, right_rect)) {
        result.low_confidence = true;
        return result;
    }
    try {
        cv::cuda::Stream cv_stream =
            cv::cuda::StreamAccessor::wrapStream(stream);
        cv::cuda::GpuMat left_full(img_h, img_w, CV_8UC1,
                                   const_cast<uint8_t*>(left_gpu),
                                   static_cast<size_t>(left_pitch));
        cv::cuda::GpuMat right_full(img_h, img_w, CV_8UC1,
                                    const_cast<uint8_t*>(right_gpu),
                                    static_cast<size_t>(right_pitch));
        cv::cuda::GpuMat left_view(left_full, left_rect);
        cv::cuda::GpuMat right_view(right_full, right_rect);

        const int radius_px = std::max(
            4, static_cast<int>(std::round(
                   0.25f * (left_det.width + left_det.height))));
        const int min_radius = std::max(3, static_cast<int>(radius_px * 0.65f));
        const int max_radius = std::max(min_radius + 2,
                                        static_cast<int>(radius_px * 1.35f));
        thread_local cv::Ptr<cv::cuda::HoughCirclesDetector> hough;
        thread_local int cached_min_radius = 0;
        thread_local int cached_max_radius = 0;
        if (!hough || cached_min_radius != min_radius ||
            cached_max_radius != max_radius) {
            cached_min_radius = min_radius;
            cached_max_radius = max_radius;
            hough = cv::cuda::createHoughCirclesDetector(
                1.0f, static_cast<float>(std::max(8, radius_px)),
                120, 10, min_radius, max_radius, 8);
        }
        cv::cuda::GpuMat left_circles_gpu;
        cv::cuda::GpuMat right_circles_gpu;
        hough->detect(left_view, left_circles_gpu, cv_stream);
        hough->detect(right_view, right_circles_gpu, cv_stream);
        cv::Mat left_circles;
        cv::Mat right_circles;
        left_circles_gpu.download(left_circles, cv_stream);
        right_circles_gpu.download(right_circles, cv_stream);
        cv_stream.waitForCompletion();
        if (left_circles.empty() || right_circles.empty() ||
            left_circles.channels() != 3 || right_circles.channels() != 3 ||
            left_circles.depth() != CV_32F ||
            right_circles.depth() != CV_32F) {
            result.low_confidence = true;
            return result;
        }
        const auto* lc = left_circles.ptr<cv::Vec3f>();
        const auto* rc = right_circles.ptr<cv::Vec3f>();
        const cv::Vec3f left_circle = lc[0];
        const cv::Vec3f right_circle = rc[0];
        const float lx = static_cast<float>(left_rect.x) + left_circle[0];
        const float ly = static_cast<float>(left_rect.y) + left_circle[1];
        const float rx = static_cast<float>(right_rect.x) + right_circle[0];
        const float ry = static_cast<float>(right_rect.y) + right_circle[1];
        const float disparity = lx - rx;
        const float max_delta = computeFeatureDeltaGate(
            initial_disp, focal, baseline, cfg);
        RobustMatchSample sample;
        sample.left_x = lx;
        sample.left_y = ly;
        sample.right_x = rx;
        sample.right_y = ry;
        sample.disparity = disparity;
        sample.score = 1.0f - std::min(
            1.0f, std::abs(disparity - initial_disp) /
                      std::max(0.25f, max_delta));
        result.attempted = 1;
        if (disparity <= 0.5f ||
            disparity > static_cast<float>(max_disparity) ||
            std::abs(disparity - initial_disp) > max_delta ||
            std::abs(ly - ry) > strictFeatureYTolerance(cfg) ||
            !passesFeatureOverlapGate(sample, left_det, right_det,
                                      initial_disp, cfg)) {
            result.low_confidence = true;
            return result;
        }
        result.valid = true;
        result.disparity = disparity;
        result.confidence = std::clamp(sample.score, 0.0f, 1.0f);
        result.stddev = 0.0f;
        result.support = 1;
        result.anchor_cx = lx;
        result.anchor_cy = ly;
        result.right_anchor_cx = rx;
        result.right_anchor_cy = ry;
        return result;
    } catch (const cv::Exception& e) {
        LOG_WARN("CUDA Canny/Hough circle ROI match failed: %s", e.what());
        return result;
    }
}

SparseFeatureDisparityResult matchVpiTemplateDisparityGPU(
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
    cudaStream_t stream) {
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return result;
    }
    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 3, 16);
    const int patch_size = patch_radius * 2 + 1;
    const int search_radius = std::clamp(cfg.subpixel_search_radius_px, 2, 64);
    const int y_radius =
        std::max(1, static_cast<int>(std::ceil(strictFeatureYTolerance(cfg))));
    const float max_delta = computeFeatureDeltaGate(
        initial_disp, focal, baseline, cfg);
    const int templ_x = static_cast<int>(std::round(left_det.cx)) -
                        patch_radius;
    const int templ_y = static_cast<int>(std::round(left_det.cy)) -
                        patch_radius;
    const cv::Rect templ_rect =
        clampRectToImage(templ_x, templ_y, patch_size, patch_size,
                         img_w, img_h);
    if (templ_rect.width != patch_size || templ_rect.height != patch_size) {
        result.low_confidence = true;
        return result;
    }
    const float predicted_right_x = left_det.cx - initial_disp;
    const int search_x = static_cast<int>(std::round(predicted_right_x)) -
                         patch_radius - search_radius;
    const int search_y = static_cast<int>(std::round(left_det.cy)) -
                         patch_radius - y_radius;
    const int search_w = patch_size + search_radius * 2;
    const int search_h = patch_size + y_radius * 2;
    const cv::Rect search_rect =
        clampRectToImage(search_x, search_y, search_w, search_h,
                         img_w, img_h);
    if (search_rect.width != search_w || search_rect.height != search_h) {
        result.low_confidence = true;
        return result;
    }

    const uint8_t* source_ptr =
        right_gpu + static_cast<size_t>(search_rect.y) *
                        static_cast<size_t>(right_pitch) +
        search_rect.x;
    const uint8_t* templ_ptr =
        left_gpu + static_cast<size_t>(templ_rect.y) *
                       static_cast<size_t>(left_pitch) +
        templ_rect.x;
    thread_local VpiTemplateScratch scratch;
    const int score_w = search_rect.width - templ_rect.width + 1;
    const int score_h = search_rect.height - templ_rect.height + 1;
    if (!scratch.ensure(reinterpret_cast<CUstream>(stream),
                        source_ptr, right_pitch,
                        search_rect.width, search_rect.height,
                        templ_ptr, left_pitch,
                        templ_rect.width, templ_rect.height,
                        score_w, score_h)) {
        result.low_confidence = true;
        return result;
    }
    VPIStatus st = vpiTemplateMatchingSetSourceImage(
        scratch.stream, VPI_BACKEND_CUDA, scratch.payload, scratch.source);
    if (st == VPI_SUCCESS) {
        st = vpiTemplateMatchingSetTemplateImage(
            scratch.stream, VPI_BACKEND_CUDA, scratch.payload,
            scratch.templ, nullptr);
    }
    if (st == VPI_SUCCESS) {
        st = vpiSubmitTemplateMatching(scratch.stream, VPI_BACKEND_CUDA,
                                       scratch.payload, scratch.score,
                                       VPI_TEMPLATE_MATCHING_NCC);
    }
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (st != VPI_SUCCESS) {
        result.low_confidence = true;
        return result;
    }

    float best = -1.0f;
    int best_x = -1;
    int best_y = -1;
    if (!findVpiCudaScorePeak(scratch.score, stream, &best, &best_x,
                              &best_y)) {
        result.low_confidence = true;
        return result;
    }

    if (best_x < 0 || best < std::max(0.05f, cfg.subpixel_min_confidence)) {
        result.low_confidence = true;
        return result;
    }
    const float rx = static_cast<float>(search_rect.x + best_x + patch_radius);
    const float ry = static_cast<float>(search_rect.y + best_y + patch_radius);
    const float disparity = left_det.cx - rx;
    RobustMatchSample sample;
    sample.left_x = left_det.cx;
    sample.left_y = left_det.cy;
    sample.right_x = rx;
    sample.right_y = ry;
    sample.disparity = disparity;
    sample.score = std::clamp(best, 0.0f, 1.0f);
    result.attempted = 1;
    if (disparity <= 0.5f ||
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
    result.disparity = disparity;
    result.confidence = sample.score;
    result.stddev = 0.0f;
    result.support = 1;
    result.anchor_cx = left_det.cx;
    result.anchor_cy = left_det.cy;
    result.right_anchor_cx = rx;
    result.right_anchor_cy = ry;
    return result;
}

SparseFeatureDisparityResult matchVpiStereoDisparityGPU(
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
    cudaStream_t stream) {
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f) {
        return result;
    }
    constexpr int local_max_disp = 64;
    const float max_delta = computeFeatureDeltaGate(
        initial_disp, focal, baseline, cfg);
    constexpr float residual_half = static_cast<float>(local_max_disp) * 0.5f;
    cv::Rect left_rect;
    cv::Rect right_rect;
    float crop_shift = -1.0f;
    if (!buildResidualSearchWorkRects(
            left_det, initial_disp, -residual_half, img_w, img_h,
            local_max_disp + 8, 0.72f, left_rect, right_rect,
            &crop_shift)) {
        result.low_confidence = true;
        return result;
    }

    const uint8_t* left_ptr =
        left_gpu + static_cast<size_t>(left_rect.y) *
                       static_cast<size_t>(left_pitch) +
        left_rect.x;
    const uint8_t* right_ptr =
        right_gpu + static_cast<size_t>(right_rect.y) *
                        static_cast<size_t>(right_pitch) +
        right_rect.x;
    thread_local VpiStereoScratch scratch;
    if (!scratch.ensure(reinterpret_cast<CUstream>(stream),
                        left_ptr, left_pitch,
                        right_ptr, right_pitch,
                        left_rect.width, left_rect.height,
                        local_max_disp)) {
        result.low_confidence = true;
        return result;
    }
    VPIStereoDisparityEstimatorParams params{};
    vpiInitStereoDisparityEstimatorParams(&params);
    params.maxDisparity = local_max_disp;
    params.minDisparity = 0;
    params.confidenceThreshold = 0;
    params.confidenceType = VPI_STEREO_CONFIDENCE_ABSOLUTE;
    params.p1 = 10;
    params.p2 = 120;
    VPIStatus st = vpiSubmitStereoDisparityEstimator(
        scratch.stream, VPI_BACKEND_CUDA, scratch.payload,
        scratch.left, scratch.right,
        scratch.disparity, scratch.confidence, &params);
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (st != VPI_SUCCESS) {
        result.low_confidence = true;
        return result;
    }

    VPIImageData disp_data{};
    VPIImageData conf_data{};
    bool disp_locked = false;
    bool conf_locked = false;
    if (vpiImageLockData(scratch.disparity, VPI_LOCK_READ,
                         VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                         &disp_data) == VPI_SUCCESS) {
        disp_locked = true;
    }
    if (vpiImageLockData(scratch.confidence, VPI_LOCK_READ,
                         VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                         &conf_data) == VPI_SUCCESS) {
        conf_locked = true;
    }
    if (!disp_locked || !conf_locked) {
        if (conf_locked) {
            vpiImageUnlock(scratch.confidence);
        }
        if (disp_locked) {
            vpiImageUnlock(scratch.disparity);
        }
        result.low_confidence = true;
        return result;
    }
    const auto* disp_base = static_cast<const uint8_t*>(
        disp_data.buffer.pitch.planes[0].data);
    const auto* conf_base = static_cast<const uint8_t*>(
        conf_data.buffer.pitch.planes[0].data);
    const int disp_pitch = disp_data.buffer.pitch.planes[0].pitchBytes;
    const int conf_pitch = conf_data.buffer.pitch.planes[0].pitchBytes;
    std::vector<RobustMatchSample> samples;
    const int sample_step = std::max(
        1, static_cast<int>(std::ceil(
               std::sqrt(std::max(1.0f, left_det.width * left_det.height) /
                         64.0f))));
    int attempted = 0;
    for (int y = 0; y < left_rect.height; y += sample_step) {
        const float ly = static_cast<float>(left_rect.y + y);
        if (ly < left_det.cy - left_det.height * 0.65f ||
            ly > left_det.cy + left_det.height * 0.65f) {
            continue;
        }
        const auto* disp_row = reinterpret_cast<const int16_t*>(
            disp_base + static_cast<size_t>(y) *
                            static_cast<size_t>(disp_pitch));
        const auto* conf_row = reinterpret_cast<const uint16_t*>(
            conf_base + static_cast<size_t>(y) *
                            static_cast<size_t>(conf_pitch));
        for (int x = 0; x < left_rect.width; x += sample_step) {
            const float lx = static_cast<float>(left_rect.x + x);
            if (!pointInsideDetectionEllipse(left_det, lx, ly, 0.62f)) {
                continue;
            }
            ++attempted;
            const float local_disp = static_cast<float>(disp_row[x]) / 32.0f;
            const float disparity = crop_shift + local_disp;
            if (!std::isfinite(local_disp) ||
                local_disp < 0.0f ||
                local_disp >= static_cast<float>(local_max_disp) - 1.0f ||
                !std::isfinite(disparity) ||
                disparity <= 0.5f ||
                disparity > static_cast<float>(max_disparity) ||
                std::abs(disparity - initial_disp) > max_delta) {
                continue;
            }
            RobustMatchSample sample;
            sample.left_x = lx;
            sample.left_y = ly;
            sample.right_x = lx - disparity;
            sample.right_y = ly;
            sample.disparity = disparity;
            sample.score = std::clamp(
                static_cast<float>(conf_row[x]) / 65535.0f, 0.05f, 1.0f);
            if (!passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
            }
            samples.push_back(sample);
        }
    }
    vpiImageUnlock(scratch.confidence);
    vpiImageUnlock(scratch.disparity);

    result.attempted = attempted;
    const RobustAggregate robust = aggregateRobustMatches(
        samples, std::max(3, cfg.subpixel_min_points), 96,
        initial_disp, max_delta, std::max(0.05f, cfg.subpixel_max_stddev_px),
        cfg);
    if (!robust.valid) {
        result.low_confidence = true;
        return result;
    }
    result.valid = true;
    result.disparity = robust.disparity;
    result.anchor_cx = robust.anchor_x;
    result.anchor_cy = robust.anchor_y;
    result.right_anchor_cx = robust.right_anchor_x;
    result.right_anchor_cy = robust.right_anchor_y;
    result.support = robust.support;
    result.stddev = robust.stddev;
    result.confidence = std::clamp(
        0.6f / (1.0f + robust.stddev) + 0.4f * robust.mean_score,
        0.0f, 1.0f);
    if (result.confidence < cfg.subpixel_min_confidence) {
        result.valid = false;
        result.low_confidence = true;
    }
    return result;
}

SparseFeatureDisparityResult matchVpiHarrisLkDisparityGPU(
    const uint8_t*, int, const uint8_t*, int, int, int,
    const Detection&, const Detection&, float,
    const ROIFeatureMatchConfig&, int, float, float, cudaStream_t) {
    return unsupportedBackend("VPI Harris+PyrLK");
}

SparseFeatureDisparityResult matchVpiOrbDisparityGPU(
    const uint8_t*, int, const uint8_t*, int, int, int,
    const Detection&, const Detection&, float,
    const ROIFeatureMatchConfig&, int, float, float, cudaStream_t) {
    return unsupportedBackend("VPI ORB");
}

SparseFeatureDisparityResult matchFixstarsLibSgmDisparityGPU(
    const uint8_t*, int, const uint8_t*, int, int, int,
    const Detection&, const Detection&, float,
    const ROIFeatureMatchConfig&, int, float, float, cudaStream_t) {
    return unsupportedBackend("Fixstars libSGM");
}

SparseFeatureDisparityResult matchCudaSiftDisparityGPU(
    const uint8_t*, int, const uint8_t*, int, int, int,
    const Detection&, const Detection&, float,
    const ROIFeatureMatchConfig&, int, float, float, cudaStream_t) {
    return unsupportedBackend("CUDA-SIFT");
}

}  // namespace stereo3d
