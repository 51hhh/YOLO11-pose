#include "roi_feature_match_gpu.h"

#include "roi_feature_match_common.h"
#include "roi_feature_match_gpu_reduce.h"
#include "../utils/logger.h"

#include <cuda.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <vpi/CUDAInterop.h>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/algo/BruteForceMatcher.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <vpi/algo/ORB.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/TemplateMatching.h>

#ifdef HAS_OPENCV_CUDAOPTFLOW
#include <opencv2/cudaoptflow.hpp>
#endif

#ifdef HAS_FIXSTARS_LIBSGM
#include <libsgm.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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

void destroyVpiArray(VPIArray& array) {
    if (array) {
        vpiArrayDestroy(array);
        array = nullptr;
    }
}

void destroyVpiPyramid(VPIPyramid& pyramid) {
    if (pyramid) {
        vpiPyramidDestroy(pyramid);
        pyramid = nullptr;
    }
}

bool logVpiFailure(VPIStatus status, const char* op) {
    if (status == VPI_SUCCESS) {
        return false;
    }
    char message[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiGetLastStatusMessage(message, sizeof(message));
    LOG_WARN("%s failed: %s (%s)",
             op ? op : "VPI operation",
             vpiStatusGetName(status),
             message);
    return true;
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

void setScoreDebugPatchFromVpiImage(SparseFeatureDisparityResult& result,
                                    VPIImage score,
                                    const cv::Rect& search_rect,
                                    int patch_radius) {
    if (!score || search_rect.empty()) {
        return;
    }
    VPIImageData score_data{};
    VPIStatus st = vpiImageLockData(score, VPI_LOCK_READ,
                                    VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR,
                                    &score_data);
    if (st != VPI_SUCCESS) {
        return;
    }

    const int score_w = score_data.buffer.pitch.planes[0].width;
    const int score_h = score_data.buffer.pitch.planes[0].height;
    const int pitch = score_data.buffer.pitch.planes[0].pitchBytes;
    const auto* base = static_cast<const uint8_t*>(
        score_data.buffer.pitch.planes[0].data);
    if (!base || score_w <= 0 || score_h <= 0 || pitch <= 0) {
        vpiImageUnlock(score);
        return;
    }

    const int out_w = std::clamp(score_w, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    const int out_h = std::clamp(score_h, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    auto& patch = result.debug_patch;
    patch = SparseFeatureDebugPatch{};
    patch.valid = true;
    patch.disparity_is_score = true;
    patch.width = out_w;
    patch.height = out_h;
    patch.left_x0 = static_cast<float>(search_rect.x + patch_radius);
    patch.left_y0 = static_cast<float>(search_rect.y + patch_radius);
    patch.step_x = static_cast<float>(score_w) / static_cast<float>(out_w);
    patch.step_y = static_cast<float>(score_h) / static_cast<float>(out_h);
    patch.disparity.assign(static_cast<size_t>(out_w * out_h),
                           std::numeric_limits<float>::quiet_NaN());
    patch.disparity_min = std::numeric_limits<float>::max();
    patch.disparity_max = std::numeric_limits<float>::lowest();

    int finite_count = 0;
    for (int oy = 0; oy < out_h; ++oy) {
        const int sy = std::clamp(
            static_cast<int>(std::floor(
                (static_cast<float>(oy) + 0.5f) * patch.step_y)),
            0, score_h - 1);
        const auto* row = reinterpret_cast<const float*>(
            base + static_cast<size_t>(sy) * static_cast<size_t>(pitch));
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = std::clamp(
                static_cast<int>(std::floor(
                    (static_cast<float>(ox) + 0.5f) * patch.step_x)),
                0, score_w - 1);
            const float score_value = row[sx];
            if (!std::isfinite(score_value)) {
                continue;
            }
            const float clipped = std::clamp(score_value, -1.0f, 1.0f);
            patch.disparity[static_cast<size_t>(oy * out_w + ox)] = clipped;
            patch.disparity_min = std::min(patch.disparity_min, clipped);
            patch.disparity_max = std::max(patch.disparity_max, clipped);
            ++finite_count;
        }
    }
    vpiImageUnlock(score);
    if (finite_count == 0) {
        patch = SparseFeatureDebugPatch{};
    }
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

struct VpiHarrisLkScratch {
    CUstream cuda_stream = nullptr;
    VPIStream stream = nullptr;
    VPIImage left = nullptr;
    VPIImage right = nullptr;
    VPIPyramid left_pyr = nullptr;
    VPIPyramid right_pyr = nullptr;
    VPIPayload harris = nullptr;
    VPIPayload lk = nullptr;
    VPIArray prev_points = nullptr;
    VPIArray cur_points = nullptr;
    VPIArray scores = nullptr;
    VPIArray status = nullptr;
    int width = 0;
    int height = 0;
    int capacity = 0;
    int levels = 0;

    ~VpiHarrisLkScratch() { reset(); }

    void reset() {
        destroyVpiArray(status);
        destroyVpiArray(scores);
        destroyVpiArray(cur_points);
        destroyVpiArray(prev_points);
        destroyVpiPayload(lk);
        destroyVpiPayload(harris);
        destroyVpiPyramid(right_pyr);
        destroyVpiPyramid(left_pyr);
        destroyVpiImage(right);
        destroyVpiImage(left);
        destroyVpiStream(stream);
        cuda_stream = nullptr;
        width = height = capacity = levels = 0;
    }

    bool ensure(CUstream cu_stream,
                const uint8_t* left_ptr,
                int left_pitch,
                const uint8_t* right_ptr,
                int right_pitch,
                int requested_w,
                int requested_h,
                int requested_capacity,
                int requested_levels) {
        const bool changed =
            cuda_stream != cu_stream ||
            width != requested_w ||
            height != requested_h ||
            capacity != requested_capacity ||
            levels != requested_levels;
        if (changed) {
            reset();
        }
        if (!stream &&
            logVpiFailure(vpiStreamCreateWrapperCUDA(
                              cu_stream, VPI_BACKEND_CUDA, &stream),
                          "vpiStreamCreateWrapperCUDA")) {
            reset();
            return false;
        }

        const VPIImageData left_data = makeCudaPitchImageData(
            left_ptr, left_pitch, requested_w, requested_h);
        const VPIImageData right_data = makeCudaPitchImageData(
            right_ptr, right_pitch, requested_w, requested_h);
        if (!left) {
            if (logVpiFailure(vpiImageCreateWrapper(&left_data, nullptr,
                                                    VPI_BACKEND_CUDA, &left),
                              "vpiImageCreateWrapper(left)")) {
                reset();
                return false;
            }
        } else if (logVpiFailure(vpiImageSetWrapper(left, &left_data),
                                 "vpiImageSetWrapper(left)")) {
            reset();
            return false;
        }
        if (!right) {
            if (logVpiFailure(vpiImageCreateWrapper(&right_data, nullptr,
                                                    VPI_BACKEND_CUDA, &right),
                              "vpiImageCreateWrapper(right)")) {
                reset();
                return false;
            }
        } else if (logVpiFailure(vpiImageSetWrapper(right, &right_data),
                                 "vpiImageSetWrapper(right)")) {
            reset();
            return false;
        }
        if (!left_pyr &&
            logVpiFailure(vpiPyramidCreate(
                              requested_w, requested_h, VPI_IMAGE_FORMAT_U8,
                              requested_levels, 0.5f, VPI_BACKEND_CUDA,
                              &left_pyr),
                          "vpiPyramidCreate(left)")) {
            reset();
            return false;
        }
        if (!right_pyr &&
            logVpiFailure(vpiPyramidCreate(
                              requested_w, requested_h, VPI_IMAGE_FORMAT_U8,
                              requested_levels, 0.5f, VPI_BACKEND_CUDA,
                              &right_pyr),
                          "vpiPyramidCreate(right)")) {
            reset();
            return false;
        }
        const uint64_t array_flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
        if (!prev_points &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_KEYPOINT_F32,
                              array_flags, &prev_points),
                          "vpiArrayCreate(prev_points)")) {
            reset();
            return false;
        }
        if (!cur_points &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_KEYPOINT_F32,
                              array_flags, &cur_points),
                          "vpiArrayCreate(cur_points)")) {
            reset();
            return false;
        }
        if (!scores &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_U32,
                              array_flags, &scores),
                          "vpiArrayCreate(scores)")) {
            reset();
            return false;
        }
        if (!status &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_U8,
                              array_flags, &status),
                          "vpiArrayCreate(status)")) {
            reset();
            return false;
        }
        if (!harris &&
            logVpiFailure(vpiCreateHarrisCornerDetector(
                              VPI_BACKEND_CUDA,
                              requested_w, requested_h, &harris),
                          "vpiCreateHarrisCornerDetector")) {
            reset();
            return false;
        }
        if (!lk &&
            logVpiFailure(vpiCreateOpticalFlowPyrLK(
                              VPI_BACKEND_CUDA,
                              requested_w, requested_h, VPI_IMAGE_FORMAT_U8,
                              requested_levels, 0.5f, &lk),
                          "vpiCreateOpticalFlowPyrLK")) {
            reset();
            return false;
        }

        cuda_stream = cu_stream;
        width = requested_w;
        height = requested_h;
        capacity = requested_capacity;
        levels = requested_levels;
        return true;
    }
};

struct VpiOrbScratch {
    CUstream cuda_stream = nullptr;
    VPIStream stream = nullptr;
    VPIImage left = nullptr;
    VPIImage right = nullptr;
    VPIPyramid left_pyr = nullptr;
    VPIPyramid right_pyr = nullptr;
    VPIPayload left_orb = nullptr;
    VPIPayload right_orb = nullptr;
    VPIArray left_points = nullptr;
    VPIArray right_points = nullptr;
    VPIArray left_descriptors = nullptr;
    VPIArray right_descriptors = nullptr;
    VPIArray matches = nullptr;
    int width = 0;
    int height = 0;
    int capacity = 0;
    int levels = 0;
    int internal_capacity = 0;

    ~VpiOrbScratch() { reset(); }

    void reset() {
        destroyVpiArray(matches);
        destroyVpiArray(right_descriptors);
        destroyVpiArray(left_descriptors);
        destroyVpiArray(right_points);
        destroyVpiArray(left_points);
        destroyVpiPayload(right_orb);
        destroyVpiPayload(left_orb);
        destroyVpiPyramid(right_pyr);
        destroyVpiPyramid(left_pyr);
        destroyVpiImage(right);
        destroyVpiImage(left);
        destroyVpiStream(stream);
        cuda_stream = nullptr;
        width = height = capacity = levels = internal_capacity = 0;
    }

    bool ensure(CUstream cu_stream,
                const uint8_t* left_ptr,
                int left_pitch,
                const uint8_t* right_ptr,
                int right_pitch,
                int requested_w,
                int requested_h,
                int requested_capacity,
                int requested_levels,
                int requested_internal_capacity) {
        const bool changed =
            cuda_stream != cu_stream ||
            width != requested_w ||
            height != requested_h ||
            capacity != requested_capacity ||
            levels != requested_levels ||
            internal_capacity != requested_internal_capacity;
        if (changed) {
            reset();
        }
        if (!stream &&
            logVpiFailure(vpiStreamCreateWrapperCUDA(
                              cu_stream, VPI_BACKEND_CUDA, &stream),
                          "vpiStreamCreateWrapperCUDA")) {
            reset();
            return false;
        }
        const VPIImageData left_data = makeCudaPitchImageData(
            left_ptr, left_pitch, requested_w, requested_h);
        const VPIImageData right_data = makeCudaPitchImageData(
            right_ptr, right_pitch, requested_w, requested_h);
        if (!left) {
            if (logVpiFailure(vpiImageCreateWrapper(&left_data, nullptr,
                                                    VPI_BACKEND_CUDA, &left),
                              "vpiImageCreateWrapper(orb left)")) {
                reset();
                return false;
            }
        } else if (logVpiFailure(vpiImageSetWrapper(left, &left_data),
                                 "vpiImageSetWrapper(orb left)")) {
            reset();
            return false;
        }
        if (!right) {
            if (logVpiFailure(vpiImageCreateWrapper(&right_data, nullptr,
                                                    VPI_BACKEND_CUDA, &right),
                              "vpiImageCreateWrapper(orb right)")) {
                reset();
                return false;
            }
        } else if (logVpiFailure(vpiImageSetWrapper(right, &right_data),
                                 "vpiImageSetWrapper(orb right)")) {
            reset();
            return false;
        }
        if (!left_pyr &&
            logVpiFailure(vpiPyramidCreate(
                              requested_w, requested_h, VPI_IMAGE_FORMAT_U8,
                              requested_levels, 0.5f, VPI_BACKEND_CUDA,
                              &left_pyr),
                          "vpiPyramidCreate(orb left)")) {
            reset();
            return false;
        }
        if (!right_pyr &&
            logVpiFailure(vpiPyramidCreate(
                              requested_w, requested_h, VPI_IMAGE_FORMAT_U8,
                              requested_levels, 0.5f, VPI_BACKEND_CUDA,
                              &right_pyr),
                          "vpiPyramidCreate(orb right)")) {
            reset();
            return false;
        }
        const uint64_t array_flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;
        if (!left_points &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_PYRAMIDAL_KEYPOINT_F32,
                              array_flags, &left_points),
                          "vpiArrayCreate(orb left points)")) {
            reset();
            return false;
        }
        if (!right_points &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_PYRAMIDAL_KEYPOINT_F32,
                              array_flags, &right_points),
                          "vpiArrayCreate(orb right points)")) {
            reset();
            return false;
        }
        if (!left_descriptors &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                              array_flags, &left_descriptors),
                          "vpiArrayCreate(orb left descriptors)")) {
            reset();
            return false;
        }
        if (!right_descriptors &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR,
                              array_flags, &right_descriptors),
                          "vpiArrayCreate(orb right descriptors)")) {
            reset();
            return false;
        }
        if (!matches &&
            logVpiFailure(vpiArrayCreate(
                              requested_capacity,
                              VPI_ARRAY_TYPE_MATCHES,
                              array_flags, &matches),
                          "vpiArrayCreate(orb matches)")) {
            reset();
            return false;
        }
        if (!left_orb &&
            logVpiFailure(vpiCreateORBFeatureDetector(
                              VPI_BACKEND_CUDA,
                              requested_internal_capacity, &left_orb),
                          "vpiCreateORBFeatureDetector(left)")) {
            reset();
            return false;
        }
        if (!right_orb &&
            logVpiFailure(vpiCreateORBFeatureDetector(
                              VPI_BACKEND_CUDA,
                              requested_internal_capacity, &right_orb),
                          "vpiCreateORBFeatureDetector(right)")) {
            reset();
            return false;
        }

        cuda_stream = cu_stream;
        width = requested_w;
        height = requested_h;
        capacity = requested_capacity;
        levels = requested_levels;
        internal_capacity = requested_internal_capacity;
        return true;
    }
};

int vpiPyramidLevelsForRoi(int width, int height) {
    if (width >= 80 && height >= 80) {
        return 3;
    }
    if (width >= 48 && height >= 48) {
        return 2;
    }
    return 1;
}

float pyramidalPointToBase(float value, float octave) {
    if (!std::isfinite(value) || !std::isfinite(octave)) {
        return value;
    }
    return value * std::pow(2.0f, std::max(0.0f, octave));
}

bool trimVpiHarrisKeypoints(VPIArray keypoints,
                            VPIArray scores,
                            const Detection& left_det,
                            const cv::Rect& left_rect,
                            int max_points) {
    VPIArrayData pts_data{};
    VPIArrayData score_data{};
    if (logVpiFailure(vpiArrayLockData(
                          keypoints, VPI_LOCK_READ_WRITE,
                          VPI_ARRAY_BUFFER_HOST_AOS, &pts_data),
                      "vpiArrayLockData(harris points)")) {
        return false;
    }
    if (logVpiFailure(vpiArrayLockData(
                          scores, VPI_LOCK_READ_WRITE,
                          VPI_ARRAY_BUFFER_HOST_AOS, &score_data),
                      "vpiArrayLockData(harris scores)")) {
        vpiArrayUnlock(keypoints);
        return false;
    }

    auto& pts = pts_data.buffer.aos;
    auto& sc = score_data.buffer.aos;
    const int count = pts.sizePointer ? *pts.sizePointer : 0;
    auto* point_ptr = static_cast<VPIKeypointF32*>(pts.data);
    auto* score_ptr = static_cast<uint32_t*>(sc.data);
    std::vector<int> indices;
    indices.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        const float gx = static_cast<float>(left_rect.x) + point_ptr[i].x;
        const float gy = static_cast<float>(left_rect.y) + point_ptr[i].y;
        if (pointInsideDetectionEllipse(left_det, gx, gy, 0.85f)) {
            indices.push_back(i);
        }
    }
    std::stable_sort(indices.begin(), indices.end(),
                     [score_ptr](int a, int b) {
                         return score_ptr[a] > score_ptr[b];
                     });
    if (static_cast<int>(indices.size()) > max_points) {
        indices.resize(max_points);
    }
    std::vector<VPIKeypointF32> selected;
    std::vector<uint32_t> selected_scores;
    selected.reserve(indices.size());
    selected_scores.reserve(indices.size());
    for (int idx : indices) {
        selected.push_back(point_ptr[idx]);
        selected_scores.push_back(score_ptr[idx]);
    }
    std::copy(selected.begin(), selected.end(), point_ptr);
    std::copy(selected_scores.begin(), selected_scores.end(), score_ptr);
    if (pts.sizePointer) {
        *pts.sizePointer = static_cast<int32_t>(selected.size());
    }
    if (sc.sizePointer) {
        *sc.sizePointer = static_cast<int32_t>(selected.size());
    }

    vpiArrayUnlock(scores);
    vpiArrayUnlock(keypoints);
    return !selected.empty();
}

void setDenseDisparityDebugPatch(SparseFeatureDisparityResult& result,
                                 const cv::Mat& disparity_cpu,
                                 const cv::Rect& left_rect,
                                 int min_disparity,
                                 float invalid_disparity,
                                 float disparity_scale) {
    if (disparity_cpu.empty() || disparity_cpu.depth() != CV_16U ||
        left_rect.empty() || disparity_scale <= 0.0f) {
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
    patch.left_x0 = static_cast<float>(left_rect.x);
    patch.left_y0 = static_cast<float>(left_rect.y);
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
        const auto* row = disparity_cpu.ptr<uint16_t>(sy);
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = std::clamp(
                static_cast<int>(std::floor(
                    (static_cast<float>(ox) + 0.5f) * patch.step_x)),
                0, disparity_cpu.cols - 1);
            const float raw = static_cast<float>(row[sx]);
            if (std::abs(raw - invalid_disparity) < 0.5f) {
                continue;
            }
            const float disparity =
                static_cast<float>(min_disparity) + raw / disparity_scale;
            if (!std::isfinite(disparity) || disparity <= 0.5f) {
                continue;
            }
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

void setVpiStereoDebugPatch(SparseFeatureDisparityResult& result,
                            const uint8_t* disp_base,
                            int disp_pitch,
                            const uint8_t* conf_base,
                            int conf_pitch,
                            const cv::Rect& left_rect,
                            float crop_shift,
                            int local_max_disp) {
    if (!disp_base || !conf_base || disp_pitch <= 0 || conf_pitch <= 0 ||
        left_rect.empty() || local_max_disp <= 0) {
        return;
    }
    const int out_w = std::clamp(left_rect.width, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    const int out_h = std::clamp(left_rect.height, 1,
                                 kMaxSparseFeatureDebugPatchSide);
    auto& patch = result.debug_patch;
    patch = SparseFeatureDebugPatch{};
    patch.valid = true;
    patch.has_confidence = true;
    patch.width = out_w;
    patch.height = out_h;
    patch.left_x0 = static_cast<float>(left_rect.x);
    patch.left_y0 = static_cast<float>(left_rect.y);
    patch.step_x = static_cast<float>(left_rect.width) /
                   static_cast<float>(out_w);
    patch.step_y = static_cast<float>(left_rect.height) /
                   static_cast<float>(out_h);
    patch.disparity.assign(static_cast<size_t>(out_w * out_h),
                           std::numeric_limits<float>::quiet_NaN());
    patch.confidence.assign(static_cast<size_t>(out_w * out_h),
                            std::numeric_limits<float>::quiet_NaN());
    patch.disparity_min = std::numeric_limits<float>::max();
    patch.disparity_max = std::numeric_limits<float>::lowest();
    patch.confidence_min = 0.0f;
    patch.confidence_max = 1.0f;

    int finite_count = 0;
    for (int oy = 0; oy < out_h; ++oy) {
        const int sy = std::clamp(
            static_cast<int>(std::floor(
                (static_cast<float>(oy) + 0.5f) * patch.step_y)),
            0, left_rect.height - 1);
        const auto* disp_row = reinterpret_cast<const int16_t*>(
            disp_base + static_cast<size_t>(sy) *
                            static_cast<size_t>(disp_pitch));
        const auto* conf_row = reinterpret_cast<const uint16_t*>(
            conf_base + static_cast<size_t>(sy) *
                            static_cast<size_t>(conf_pitch));
        for (int ox = 0; ox < out_w; ++ox) {
            const int sx = std::clamp(
                static_cast<int>(std::floor(
                    (static_cast<float>(ox) + 0.5f) * patch.step_x)),
                0, left_rect.width - 1);
            const float local_disp = static_cast<float>(disp_row[sx]) / 32.0f;
            const float confidence =
                std::clamp(static_cast<float>(conf_row[sx]) / 65535.0f,
                           0.0f, 1.0f);
            patch.confidence[static_cast<size_t>(oy * out_w + ox)] = confidence;
            if (!std::isfinite(local_disp) || local_disp < 0.0f ||
                local_disp >= static_cast<float>(local_max_disp) - 1.0f) {
                continue;
            }
            const float disparity = crop_shift + local_disp;
            if (!std::isfinite(disparity) || disparity <= 0.5f) {
                continue;
            }
            patch.disparity[static_cast<size_t>(oy * out_w + ox)] = disparity;
            patch.disparity_min = std::min(patch.disparity_min, disparity);
            patch.disparity_max = std::max(patch.disparity_max, disparity);
            ++finite_count;
        }
    }
    (void)finite_count;
}

SparseFeatureDisparityResult aggregateDenseDisparityMap(
    const cv::Mat& disparity_cpu,
    const cv::Rect& left_rect,
    int min_disparity,
    float invalid_disparity,
    float disparity_scale,
    const Detection& left_det,
    const Detection& right_det,
    float initial_disp,
    const ROIFeatureMatchConfig& cfg,
    int max_disparity,
    float focal,
    float baseline) {
    SparseFeatureDisparityResult result;
    if (disparity_cpu.empty() || disparity_cpu.depth() != CV_16U ||
        disparity_scale <= 0.0f) {
        result.low_confidence = true;
        return result;
    }
    if (cfg.debug_patch_enabled) {
        setDenseDisparityDebugPatch(result, disparity_cpu, left_rect,
                                    min_disparity, invalid_disparity,
                                    disparity_scale);
    }
    const float max_delta = computeFeatureDeltaGate(
        initial_disp, focal, baseline, cfg);
    const int sample_step = std::max(
        1, static_cast<int>(std::ceil(
               std::sqrt(std::max(1.0f, left_det.width * left_det.height) /
                         64.0f))));
    std::vector<RobustMatchSample> samples;
    int attempted = 0;
    for (int y = 0; y < disparity_cpu.rows; y += sample_step) {
        const float ly = static_cast<float>(left_rect.y + y);
        if (ly < left_det.cy - left_det.height * 0.65f ||
            ly > left_det.cy + left_det.height * 0.65f) {
            continue;
        }
        const auto* row = disparity_cpu.ptr<uint16_t>(y);
        for (int x = 0; x < disparity_cpu.cols; x += sample_step) {
            const float lx = static_cast<float>(left_rect.x + x);
            if (!pointInsideDetectionEllipse(left_det, lx, ly, 0.62f)) {
                continue;
            }
            ++attempted;
            const float raw = static_cast<float>(row[x]);
            if (std::abs(raw - invalid_disparity) < 0.5f) {
                continue;
            }
            const float disparity =
                static_cast<float>(min_disparity) + raw / disparity_scale;
            if (!std::isfinite(disparity) ||
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
            sample.score = 1.0f - std::min(
                1.0f, std::abs(disparity - initial_disp) /
                          std::max(0.25f, max_delta));
            sample.score = std::max(0.05f, sample.score);
            if (!passesFeatureOverlapGate(sample, left_det, right_det,
                                          initial_disp, cfg) ||
                !passesSphereRadiusGate(sample, left_det, initial_disp,
                                        focal, baseline, cfg)) {
                continue;
            }
            samples.push_back(sample);
        }
    }

    result.attempted = attempted;
    const int min_points = std::max(3, cfg.subpixel_min_points);
    const RobustAggregate robust = aggregateRobustMatches(
        samples, min_points, 96, initial_disp, max_delta,
        std::max(0.05f, cfg.subpixel_max_stddev_px), cfg);
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
    copyDebugMatches(robust, result);
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

#ifdef HAS_FIXSTARS_LIBSGM
struct LibSgmScratch {
    int width = 0;
    int height = 0;
    int disparity_size = 0;
    int src_pitch_pixels = 0;
    int dst_pitch_pixels = 0;
    std::unique_ptr<sgm::StereoSGM> matcher;
    cv::cuda::GpuMat disparity_gpu;

    bool ensure(int requested_w,
                int requested_h,
                int requested_disparity_size,
                int requested_src_pitch_pixels) {
        disparity_gpu.create(requested_h, requested_w, CV_16UC1);
        const int requested_dst_pitch_pixels =
            static_cast<int>(disparity_gpu.step / sizeof(uint16_t));
        const bool changed =
            !matcher ||
            width != requested_w ||
            height != requested_h ||
            disparity_size != requested_disparity_size ||
            src_pitch_pixels != requested_src_pitch_pixels ||
            dst_pitch_pixels != requested_dst_pitch_pixels;
        if (changed) {
            sgm::StereoSGM::Parameters params(
                10, 120, 0.95f, true, sgm::PathType::SCAN_4PATH,
                0, 1, sgm::CensusType::SYMMETRIC_CENSUS_9x7);
            matcher = std::make_unique<sgm::StereoSGM>(
                requested_w, requested_h, requested_disparity_size,
                8, 16, requested_src_pitch_pixels,
                requested_dst_pitch_pixels,
                sgm::EXECUTE_INOUT_CUDA2CUDA, params);
            width = requested_w;
            height = requested_h;
            disparity_size = requested_disparity_size;
            src_pitch_pixels = requested_src_pitch_pixels;
            dst_pitch_pixels = requested_dst_pitch_pixels;
        }
        return matcher != nullptr && !disparity_gpu.empty();
    }
};
#endif

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
    copyDebugMatches(robust, result);
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
    result.unsupported = true;
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
        setSingleDebugMatch(sample, result);
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
    if (cfg.debug_patch_enabled) {
        setScoreDebugPatchFromVpiImage(result, scratch.score,
                                       search_rect, patch_radius);
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
    result.disparity = disparity;
    result.confidence = sample.score;
    result.anchor_cx = left_det.cx;
    result.anchor_cy = left_det.cy;
    result.right_anchor_cx = rx;
    result.right_anchor_cy = ry;
    setSingleDebugMatch(sample, result);
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
    result.stddev = 0.0f;
    result.support = 1;
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
    if (cfg.debug_patch_enabled) {
        setVpiStereoDebugPatch(result, disp_base, disp_pitch, conf_base, conf_pitch,
                               left_rect, crop_shift, local_max_disp);
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
    copyDebugMatches(robust, result);
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
    const int patch_radius = std::clamp(cfg.subpixel_patch_radius, 3, 12);
    cv::Rect left_rect;
    cv::Rect right_rect;
    if (!buildShiftedWorkRects(left_det, initial_disp, img_w, img_h,
                               patch_radius + 8, 1.05f,
                               left_rect, right_rect)) {
        result.low_confidence = true;
        return result;
    }

    const int max_points =
        std::clamp(std::max(cfg.subpixel_max_points * 8, 64), 32, 192);
    const int levels = vpiPyramidLevelsForRoi(left_rect.width,
                                              left_rect.height);
    const uint8_t* left_ptr =
        left_gpu + static_cast<size_t>(left_rect.y) *
                       static_cast<size_t>(left_pitch) +
        left_rect.x;
    const uint8_t* right_ptr =
        right_gpu + static_cast<size_t>(right_rect.y) *
                        static_cast<size_t>(right_pitch) +
        right_rect.x;

    thread_local VpiHarrisLkScratch scratch;
    if (!scratch.ensure(reinterpret_cast<CUstream>(stream),
                        left_ptr, left_pitch,
                        right_ptr, right_pitch,
                        left_rect.width, left_rect.height,
                        max_points, levels)) {
        result.low_confidence = true;
        return result;
    }

    VPIHarrisCornerDetectorParams harris_params{};
    if (logVpiFailure(vpiInitHarrisCornerDetectorParams(&harris_params),
                      "vpiInitHarrisCornerDetectorParams")) {
        result.low_confidence = true;
        return result;
    }
    harris_params.gradientSize = 3;
    harris_params.blockSize = 3;
    harris_params.strengthThresh = 0.5f;
    harris_params.sensitivity = 0.01f;
    harris_params.minNMSDistance = 3.0f;

    VPIStatus st = vpiSubmitHarrisCornerDetector(
        scratch.stream, VPI_BACKEND_CUDA, scratch.harris,
        scratch.left, scratch.prev_points, scratch.scores, &harris_params);
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (logVpiFailure(st, "vpiSubmitHarrisCornerDetector")) {
        result.low_confidence = true;
        return result;
    }
    if (!trimVpiHarrisKeypoints(scratch.prev_points, scratch.scores,
                                left_det, left_rect, max_points)) {
        result.low_confidence = true;
        return result;
    }

    st = vpiSubmitGaussianPyramidGenerator(
        scratch.stream, VPI_BACKEND_CUDA, scratch.left,
        scratch.left_pyr, VPI_BORDER_CLAMP);
    if (st == VPI_SUCCESS) {
        st = vpiSubmitGaussianPyramidGenerator(
            scratch.stream, VPI_BACKEND_CUDA, scratch.right,
            scratch.right_pyr, VPI_BORDER_CLAMP);
    }
    if (logVpiFailure(st, "vpiSubmitGaussianPyramidGenerator(HarrisLK)")) {
        result.low_confidence = true;
        return result;
    }

    VPIOpticalFlowPyrLKParams lk_params{};
#if NV_VPI_VERSION_API_AT_MOST(3, 1)
    st = vpiInitOpticalFlowPyrLKParams(&lk_params);
#else
    st = vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &lk_params);
#endif
    if (logVpiFailure(st, "vpiInitOpticalFlowPyrLKParams")) {
        result.low_confidence = true;
        return result;
    }
    lk_params.useInitialFlow = 0;
    lk_params.numIterations = 10;
    lk_params.windowDimension =
        std::clamp(patch_radius * 2 + 1, 7, 25);
    st = vpiSubmitOpticalFlowPyrLK(
        scratch.stream, VPI_BACKEND_CUDA, scratch.lk,
        scratch.left_pyr, scratch.right_pyr,
        scratch.prev_points, scratch.cur_points,
        scratch.status, &lk_params);
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (logVpiFailure(st, "vpiSubmitOpticalFlowPyrLK")) {
        result.low_confidence = true;
        return result;
    }

    VPIArrayData prev_data{};
    VPIArrayData cur_data{};
    VPIArrayData status_data{};
    bool prev_locked = false;
    bool cur_locked = false;
    bool status_locked = false;
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.prev_points, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &prev_data),
                       "vpiArrayLockData(prev LK)")) {
        prev_locked = true;
    }
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.cur_points, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &cur_data),
                       "vpiArrayLockData(cur LK)")) {
        cur_locked = true;
    }
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.status, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &status_data),
                       "vpiArrayLockData(status LK)")) {
        status_locked = true;
    }
    if (!prev_locked || !cur_locked || !status_locked) {
        if (status_locked) vpiArrayUnlock(scratch.status);
        if (cur_locked) vpiArrayUnlock(scratch.cur_points);
        if (prev_locked) vpiArrayUnlock(scratch.prev_points);
        result.low_confidence = true;
        return result;
    }

    const auto& prev_aos = prev_data.buffer.aos;
    const auto& cur_aos = cur_data.buffer.aos;
    const auto& status_aos = status_data.buffer.aos;
    const int prev_count = prev_aos.sizePointer ? *prev_aos.sizePointer : 0;
    const int cur_count = cur_aos.sizePointer ? *cur_aos.sizePointer : 0;
    const int status_count =
        status_aos.sizePointer ? *status_aos.sizePointer : 0;
    const int count = std::min({prev_count, cur_count, status_count});
    const auto* prev_pts = static_cast<const VPIKeypointF32*>(prev_aos.data);
    const auto* cur_pts = static_cast<const VPIKeypointF32*>(cur_aos.data);
    const auto* status_ptr = static_cast<const uint8_t*>(status_aos.data);
    std::vector<cv::Point2f> left_points;
    std::vector<cv::Point2f> right_points;
    std::vector<uint8_t> valid_status;
    left_points.reserve(std::max(0, count));
    right_points.reserve(std::max(0, count));
    valid_status.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        left_points.emplace_back(prev_pts[i].x, prev_pts[i].y);
        right_points.emplace_back(cur_pts[i].x, cur_pts[i].y);
        valid_status.push_back(status_ptr[i] == 0 ? 1 : 0);
    }
    vpiArrayUnlock(scratch.status);
    vpiArrayUnlock(scratch.cur_points);
    vpiArrayUnlock(scratch.prev_points);

    return aggregateGpuPointMatches(
        left_points, right_points, valid_status, left_rect, right_rect,
        left_det, right_det, initial_disp, cfg, max_disparity,
        focal, baseline);
}

SparseFeatureDisparityResult matchVpiOrbDisparityGPU(
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
                               12, 1.10f, left_rect, right_rect)) {
        result.low_confidence = true;
        return result;
    }

    const int levels = vpiPyramidLevelsForRoi(left_rect.width,
                                              left_rect.height);
    const int max_points =
        std::clamp(std::max(cfg.subpixel_max_points * 8, 64), 32, 192);
    const int per_level = std::max(16, (max_points + levels - 1) / levels);
    const int capacity = per_level * levels;
    const int internal_capacity = std::clamp(per_level * 20, 128, 1024);
    const uint8_t* left_ptr =
        left_gpu + static_cast<size_t>(left_rect.y) *
                       static_cast<size_t>(left_pitch) +
        left_rect.x;
    const uint8_t* right_ptr =
        right_gpu + static_cast<size_t>(right_rect.y) *
                        static_cast<size_t>(right_pitch) +
        right_rect.x;

    thread_local VpiOrbScratch scratch;
    if (!scratch.ensure(reinterpret_cast<CUstream>(stream),
                        left_ptr, left_pitch,
                        right_ptr, right_pitch,
                        left_rect.width, left_rect.height,
                        capacity, levels, internal_capacity)) {
        result.low_confidence = true;
        return result;
    }

    VPIORBParams orb_params{};
    VPIStatus st = vpiInitORBParams(&orb_params);
    if (logVpiFailure(st, "vpiInitORBParams")) {
        result.low_confidence = true;
        return result;
    }
    orb_params.fastParams.circleRadius = 3;
    orb_params.fastParams.arcLength = 9;
    orb_params.fastParams.intensityThreshold = 8.0f;
    orb_params.fastParams.nonMaxSuppression = 1;
    orb_params.maxFeaturesPerLevel = per_level;
    orb_params.maxPyramidLevels = levels;
    orb_params.scoreType = VPI_CORNER_SCORE_HARRIS;
    orb_params.flags = 0;

    st = vpiSubmitGaussianPyramidGenerator(
        scratch.stream, VPI_BACKEND_CUDA, scratch.left,
        scratch.left_pyr, VPI_BORDER_CLAMP);
    if (st == VPI_SUCCESS) {
        st = vpiSubmitGaussianPyramidGenerator(
            scratch.stream, VPI_BACKEND_CUDA, scratch.right,
            scratch.right_pyr, VPI_BORDER_CLAMP);
    }
    if (st == VPI_SUCCESS) {
        st = vpiSubmitORBFeatureDetector(
            scratch.stream, VPI_BACKEND_CUDA, scratch.left_orb,
            scratch.left_pyr, scratch.left_points,
            scratch.left_descriptors, &orb_params, VPI_BORDER_LIMITED);
    }
    if (st == VPI_SUCCESS) {
        st = vpiSubmitORBFeatureDetector(
            scratch.stream, VPI_BACKEND_CUDA, scratch.right_orb,
            scratch.right_pyr, scratch.right_points,
            scratch.right_descriptors, &orb_params, VPI_BORDER_LIMITED);
    }
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (logVpiFailure(st, "VPI ORB feature detector")) {
        result.low_confidence = true;
        return result;
    }
    int32_t left_descriptor_count = 0;
    int32_t right_descriptor_count = 0;
    if (logVpiFailure(vpiArrayGetSize(scratch.left_descriptors,
                                      &left_descriptor_count),
                      "vpiArrayGetSize(orb left descriptors)") ||
        logVpiFailure(vpiArrayGetSize(scratch.right_descriptors,
                                      &right_descriptor_count),
                      "vpiArrayGetSize(orb right descriptors)") ||
        left_descriptor_count <= 0 || right_descriptor_count <= 0) {
        result.low_confidence = true;
        return result;
    }
    st = vpiSubmitBruteForceMatcher(
        scratch.stream, VPI_BACKEND_CUDA,
        scratch.left_descriptors, scratch.right_descriptors,
        VPI_NORM_HAMMING, 1, scratch.matches,
        VPI_ENABLE_CROSS_CHECK);
    if (st == VPI_SUCCESS) {
        st = vpiStreamSync(scratch.stream);
    }
    if (logVpiFailure(st, "VPI BruteForceMatcher")) {
        result.low_confidence = true;
        return result;
    }

    VPIArrayData left_data{};
    VPIArrayData right_data{};
    VPIArrayData matches_data{};
    bool left_locked = false;
    bool right_locked = false;
    bool matches_locked = false;
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.left_points, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &left_data),
                       "vpiArrayLockData(orb left points)")) {
        left_locked = true;
    }
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.right_points, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &right_data),
                       "vpiArrayLockData(orb right points)")) {
        right_locked = true;
    }
    if (!logVpiFailure(vpiArrayLockData(
                           scratch.matches, VPI_LOCK_READ,
                           VPI_ARRAY_BUFFER_HOST_AOS, &matches_data),
                       "vpiArrayLockData(orb matches)")) {
        matches_locked = true;
    }
    if (!left_locked || !right_locked || !matches_locked) {
        if (matches_locked) vpiArrayUnlock(scratch.matches);
        if (right_locked) vpiArrayUnlock(scratch.right_points);
        if (left_locked) vpiArrayUnlock(scratch.left_points);
        result.low_confidence = true;
        return result;
    }

    const auto& left_aos = left_data.buffer.aos;
    const auto& right_aos = right_data.buffer.aos;
    const auto& match_aos = matches_data.buffer.aos;
    const int left_count = left_aos.sizePointer ? *left_aos.sizePointer : 0;
    const int right_count =
        right_aos.sizePointer ? *right_aos.sizePointer : 0;
    const int match_count =
        match_aos.sizePointer ? *match_aos.sizePointer : 0;
    const int count = std::min(left_count, match_count);
    const auto* left_pts =
        static_cast<const VPIPyramidalKeypointF32*>(left_aos.data);
    const auto* right_pts =
        static_cast<const VPIPyramidalKeypointF32*>(right_aos.data);
    const auto* matches = static_cast<const VPIMatches*>(match_aos.data);
    std::vector<cv::Point2f> left_points;
    std::vector<cv::Point2f> right_points;
    std::vector<uint8_t> status;
    left_points.reserve(std::max(0, count));
    right_points.reserve(std::max(0, count));
    status.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        const int ref = matches[i].refIndex[0];
        if (ref < 0 || ref >= right_count ||
            !std::isfinite(matches[i].distance[0])) {
            continue;
        }
        const float lx = pyramidalPointToBase(left_pts[i].x,
                                              left_pts[i].octave);
        const float ly = pyramidalPointToBase(left_pts[i].y,
                                              left_pts[i].octave);
        const float rx = pyramidalPointToBase(right_pts[ref].x,
                                              right_pts[ref].octave);
        const float ry = pyramidalPointToBase(right_pts[ref].y,
                                              right_pts[ref].octave);
        left_points.emplace_back(lx, ly);
        right_points.emplace_back(rx, ry);
        status.push_back(1);
    }
    vpiArrayUnlock(scratch.matches);
    vpiArrayUnlock(scratch.right_points);
    vpiArrayUnlock(scratch.left_points);

    return aggregateGpuPointMatches(
        left_points, right_points, status, left_rect, right_rect,
        left_det, right_det, initial_disp, cfg, max_disparity,
        focal, baseline);
}

SparseFeatureDisparityResult matchFixstarsLibSgmDisparityGPU(
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
#ifndef HAS_FIXSTARS_LIBSGM
    (void)left_gpu; (void)left_pitch; (void)right_gpu; (void)right_pitch;
    (void)img_w; (void)img_h; (void)left_det; (void)right_det;
    (void)initial_disp; (void)cfg; (void)max_disparity; (void)focal;
    (void)baseline; (void)stream;
    return unsupportedBackend("Fixstars libSGM");
#else
    SparseFeatureDisparityResult result;
    if (!left_gpu || !right_gpu || left_pitch <= 0 || right_pitch <= 0 ||
        left_pitch != right_pitch || img_w <= 0 || img_h <= 0 || !stream ||
        !std::isfinite(initial_disp) || initial_disp <= 0.5f ||
        left_det.width < 8.0f || left_det.height < 8.0f ||
        right_det.width < 8.0f || right_det.height < 8.0f) {
        return result;
    }

    constexpr int disparity_size = 64;
    const int block_margin =
        std::clamp(cfg.subpixel_patch_radius + 6, 8, 20);
    const cv::Rect ball_roi = featureROIFromDetectionCPU(
        left_det, img_w, img_h, block_margin, 0.62f, 0);
    if (ball_roi.empty()) {
        result.low_confidence = true;
        return result;
    }
    const int min_disp = std::max(
        0,
        static_cast<int>(std::floor(
            initial_disp - static_cast<float>(disparity_size) * 0.5f)));
    const int pad_left = disparity_size + block_margin;
    const int pad_right = block_margin;
    const cv::Rect full(0, 0, img_w, img_h);
    const cv::Rect left_rect(
        ball_roi.x - pad_left,
        ball_roi.y - block_margin,
        ball_roi.width + pad_left + pad_right,
        ball_roi.height + block_margin * 2);
    const cv::Rect right_rect(
        left_rect.x - min_disp,
        left_rect.y,
        left_rect.width,
        left_rect.height);
    if ((left_rect & full) != left_rect ||
        (right_rect & full) != right_rect ||
        left_rect.width <= disparity_size + 8 ||
        left_rect.height <= 12) {
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

    try {
        thread_local LibSgmScratch scratch;
        if (!scratch.ensure(left_rect.width, left_rect.height,
                            disparity_size, left_pitch)) {
            result.low_confidence = true;
            return result;
        }
        scratch.matcher->execute(left_ptr, right_ptr,
                                 scratch.disparity_gpu.ptr<uint16_t>());
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG_WARN("Fixstars libSGM execute failed: %s",
                     cudaGetErrorString(err));
            result.low_confidence = true;
            return result;
        }
        cv::cuda::Stream cv_stream =
            cv::cuda::StreamAccessor::wrapStream(stream);
        cv::Mat disparity_cpu;
        scratch.disparity_gpu.download(disparity_cpu, cv_stream);
        cv_stream.waitForCompletion();
        const float invalid = static_cast<float>(
            static_cast<uint16_t>(scratch.matcher->get_invalid_disparity()));
        return aggregateDenseDisparityMap(
            disparity_cpu, left_rect, min_disp, invalid,
            static_cast<float>(sgm::StereoSGM::SUBPIXEL_SCALE),
            left_det, right_det, initial_disp, cfg, max_disparity,
            focal, baseline);
    } catch (const std::exception& e) {
        LOG_WARN("Fixstars libSGM ROI match failed: %s", e.what());
        return result;
    }
#endif
}

SparseFeatureDisparityResult matchCudaSiftDisparityGPU(
    const uint8_t*, int, const uint8_t*, int, int, int,
    const Detection&, const Detection&, float,
    const ROIFeatureMatchConfig&, int, float, float, cudaStream_t) {
    return unsupportedBackend("CUDA-SIFT");
}

}  // namespace stereo3d
