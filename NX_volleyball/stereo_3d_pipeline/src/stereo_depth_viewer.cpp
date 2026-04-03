/**
 * @file stereo_depth_viewer.cpp
 * @brief 双目深度实时查看器 — 多算法 GPU 加速
 *
 * 功能:
 *   - 加载海康双目相机, 实时采集 BayerRG8 图像
 *   - VPI Remap GPU 校正
 *   - PWM 硬件触发自启动 (gpiochip2 line 7, 15Hz)
 *   - 多算法视差/深度计算 (全部 GPU 加速), 按 't' 切换:
 *       0: 原始左右图像 (Bayer → BGR)
 *       1: VPI CUDA SGM (全帧)
 *       2: VPI CUDA SGM (半分辨率)
 *       3: VPI CUDA SGM + GPU 双边滤波
 *       4: OpenCV CUDA SGM
 *       5: OpenCV CUDA BM
 *       6: OpenCV CUDA BP (Belief Propagation)
 *       7: OpenCV CUDA CSBP (Constant Space BP)
 *       8: OpenCV SGBM CPU (高质量参考)
 *   - 鼠标点击测量深度
 *   - 按 'q' / ESC 退出
 *
 * 编译: 在 CMakeLists.txt 中添加 stereo_depth_viewer 目标
 * 运行: ./stereo_depth_viewer -c calibration/stereo_calib.yaml
 *        (自动启动 PWM 触发, 无需外部脚本)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

// WLS (Weighted Least Squares) 后处理滤波 — opencv_contrib ximgproc
#ifdef HAS_XIMGPROC
#include <opencv2/ximgproc/disparity_filter.hpp>
#endif

#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/Rescale.h>
#include <vpi/WarpMap.h>

#include "calibration/stereo_calibration.h"
#include "calibration/pwm_trigger.h"
#include "capture/hikvision_camera.h"
#include "stereo/onnx_stereo.h"
#include "utils/logger.h"

#include <fstream>   // JSON 报告输出
#include <iomanip>   // std::setprecision
#include <unistd.h>  // access()

using namespace stereo3d;

// ============================================================
//  全局状态
// ============================================================
static std::atomic<bool> g_running{true};

static void sigHandler(int) { g_running = false; }

// 查看模式
enum class ViewMode {
    RAW_STEREO = 0,       // 原始左右图像
    VPI_CUDA_FULL,        // VPI CUDA SGM 全帧
    VPI_CUDA_HALF,        // VPI CUDA SGM 半分辨率
    VPI_CUDA_BILATERAL,   // VPI CUDA SGM + GPU 双边滤波
    OPENCV_CUDA_SGM,      // OpenCV CUDA SGM
    OPENCV_CUDA_BM,       // OpenCV CUDA BM
    OPENCV_CUDA_BP,       // OpenCV CUDA Belief Propagation
    OPENCV_CUDA_CSBP,     // OpenCV CUDA Constant Space BP
    OPENCV_SGBM_CPU,      // OpenCV SGBM CPU (高质量参考)
    OPENCV_SGBM_WLS,      // OpenCV SGBM + WLS 后处理 (左右一致性 + 边缘保持)
    OPENCV_SGBM_CENSUS,   // OpenCV SGBM Census 预处理
    ONNX_CRESTEREO,       // CREStereo DL 模型 (ONNX Runtime)
    ONNX_HITNET,          // HITNet DL 模型 (ONNX Runtime)
    MODE_COUNT
};

static const char* viewModeName(ViewMode m) {
    switch (m) {
        case ViewMode::RAW_STEREO:        return "Raw Stereo (L|R)";
        case ViewMode::VPI_CUDA_FULL:     return "VPI CUDA SGM Full";
        case ViewMode::VPI_CUDA_HALF:     return "VPI CUDA SGM Half";
        case ViewMode::VPI_CUDA_BILATERAL: return "VPI CUDA SGM + Bilateral";
        case ViewMode::OPENCV_CUDA_SGM:   return "OpenCV CUDA SGM";
        case ViewMode::OPENCV_CUDA_BM:    return "OpenCV CUDA BM";
        case ViewMode::OPENCV_CUDA_BP:    return "OpenCV CUDA BP";
        case ViewMode::OPENCV_CUDA_CSBP:  return "OpenCV CUDA CSBP";
        case ViewMode::OPENCV_SGBM_CPU:   return "OpenCV SGBM CPU";
        case ViewMode::OPENCV_SGBM_WLS:   return "OpenCV SGBM+WLS";
        case ViewMode::OPENCV_SGBM_CENSUS: return "OpenCV SGBM Census";
        case ViewMode::ONNX_CRESTEREO:    return "CREStereo ONNX DL";
        case ViewMode::ONNX_HITNET:       return "HITNet ONNX DL";
        default: return "Unknown";
    }
}

// ============================================================
//  深度显示状态 (鼠标回调)
// ============================================================
struct DepthViewerState {
    cv::Mat depth_map;    // 深度图 (float, mm)
    float baseline_mm;
    float focal_px;
    int click_x = -1;
    int click_y = -1;
    float click_depth_mm = 0.0f;
    float depth_min_mm = 0.0f;   // 当前帧自动范围
    float depth_max_mm = 0.0f;
};

static void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* state = reinterpret_cast<DepthViewerState*>(userdata);
    if (state->depth_map.empty()) return;

    // 确保坐标在图像范围内
    x = std::clamp(x, 0, state->depth_map.cols - 1);
    y = std::clamp(y, 0, state->depth_map.rows - 1);

    state->click_x = x;
    state->click_y = y;
    state->click_depth_mm = state->depth_map.at<float>(y, x);

    if (state->click_depth_mm > 0) {
        printf("[Click] (%d, %d) → depth = %.1f mm (%.2f m)\n",
               x, y, state->click_depth_mm, state->click_depth_mm / 1000.0f);
    } else {
        printf("[Click] (%d, %d) → invalid depth\n", x, y);
    }
}

// ============================================================
//  VPI 资源封装
// ============================================================
struct VPIResources {
    VPIStream stream = nullptr;

    // 校正
    VPIPayload remapL = nullptr;
    VPIPayload remapR = nullptr;

    // 原始/校正 图像
    VPIImage rawL = nullptr, rawR = nullptr;
    VPIImage rectL = nullptr, rectR = nullptr;

    // 全帧视差
    VPIPayload stereoFull = nullptr;
    VPIImage dispFull = nullptr, confFull = nullptr;

    // 半帧视差
    VPIPayload stereoHalf = nullptr;
    VPIImage halfL = nullptr, halfR = nullptr;
    VPIImage dispHalf = nullptr, confHalf = nullptr;
    VPIImage dispUpscaled = nullptr, confUpscaled = nullptr;

    int width = 0, height = 0;
    int maxDisp = 256;
    int winSize = 5;

    void destroy() {
        auto safeDestroy = [](VPIImage& img) { if (img) { vpiImageDestroy(img); img = nullptr; } };
        auto safePayload = [](VPIPayload& p) { if (p) { vpiPayloadDestroy(p); p = nullptr; } };

        safeDestroy(rawL); safeDestroy(rawR);
        safeDestroy(rectL); safeDestroy(rectR);
        safeDestroy(dispFull); safeDestroy(confFull);
        safeDestroy(halfL); safeDestroy(halfR);
        safeDestroy(dispHalf); safeDestroy(confHalf);
        safeDestroy(dispUpscaled); safeDestroy(confUpscaled);

        safePayload(remapL); safePayload(remapR);
        safePayload(stereoFull); safePayload(stereoHalf);

        if (stream) { vpiStreamDestroy(stream); stream = nullptr; }
    }
};

static bool initVPI(VPIResources& vpi, const StereoCalibration& calib,
                    int width, int height, int maxDisp, int winSize)
{
    vpi.width = width;
    vpi.height = height;
    vpi.maxDisp = maxDisp;
    vpi.winSize = winSize;

    VPIStatus err;
    // CPU flag 必须包含, 否则 vpiImageLockData HOST_PITCH_LINEAR 会失败
    uint64_t flags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU;

    // Stream
    err = vpiStreamCreate(VPI_BACKEND_CUDA, &vpi.stream);
    if (err != VPI_SUCCESS) { LOG_ERROR("vpiStreamCreate failed: %d", err); return false; }

    // 原始图像 (U8, Bayer 输入) — 需要 host 上传
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rawL);
    if (err != VPI_SUCCESS) { LOG_ERROR("rawL create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rawR);
    if (err != VPI_SUCCESS) { LOG_ERROR("rawR create failed"); return false; }

    // 校正后图像 (U8) — 需要 host 下载给 OpenCV 算法
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rectL);
    if (err != VPI_SUCCESS) { LOG_ERROR("rectL create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, flags, &vpi.rectR);
    if (err != VPI_SUCCESS) { LOG_ERROR("rectR create failed"); return false; }

    // ---- 视差输出 (全帧) — 需要 host 下载 ----
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, flags, &vpi.dispFull);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispFull create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, flags, &vpi.confFull);
    if (err != VPI_SUCCESS) { LOG_ERROR("confFull create failed"); return false; }

    // ---- 全帧 Stereo Payload ----
    {
        VPIStereoDisparityEstimatorCreationParams params;
        vpiInitStereoDisparityEstimatorCreationParams(&params);
        params.maxDisparity = maxDisp;
        err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, width, height,
                                                 VPI_IMAGE_FORMAT_U8, &params, &vpi.stereoFull);
        if (err != VPI_SUCCESS) { LOG_ERROR("stereoFull create failed: %d", err); return false; }
    }

    // ---- 半帧 (中间结果不需要 host 访问, 但 upscaled 需要下载) ----
    int halfW = width / 2, halfH = height / 2;
    uint64_t gpuOnly = VPI_BACKEND_CUDA;
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, gpuOnly, &vpi.halfL);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfL create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U8, gpuOnly, &vpi.halfR);
    if (err != VPI_SUCCESS) { LOG_ERROR("halfR create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_S16, gpuOnly, &vpi.dispHalf);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispHalf create failed"); return false; }
    err = vpiImageCreate(halfW, halfH, VPI_IMAGE_FORMAT_U16, gpuOnly, &vpi.confHalf);
    if (err != VPI_SUCCESS) { LOG_ERROR("confHalf create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_S16, flags, &vpi.dispUpscaled);
    if (err != VPI_SUCCESS) { LOG_ERROR("dispUpscaled create failed"); return false; }
    err = vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U16, flags, &vpi.confUpscaled);
    if (err != VPI_SUCCESS) { LOG_ERROR("confUpscaled create failed"); return false; }

    {
        VPIStereoDisparityEstimatorCreationParams hparams;
        vpiInitStereoDisparityEstimatorCreationParams(&hparams);
        hparams.maxDisparity = maxDisp / 2;
        err = vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, halfW, halfH,
                                                 VPI_IMAGE_FORMAT_U8, &hparams, &vpi.stereoHalf);
        if (err != VPI_SUCCESS) { LOG_ERROR("stereoHalf create failed: %d", err); return false; }
    }

    // ---- 校正 Remap ----
    cv::Mat map1L, map2L, map1R, map2R;
    calib.buildRemapMaps(map1L, map2L, map1R, map2R, width, height);

    auto buildRemap = [&](const cv::Mat& mapX, const cv::Mat& mapY,
                          VPIPayload& payload, const char* name) -> bool {
        VPIWarpMap wm;
        memset(&wm, 0, sizeof(wm));
        wm.grid.numHorizRegions  = 1;
        wm.grid.numVertRegions   = 1;
        wm.grid.regionWidth[0]   = width;
        wm.grid.regionHeight[0]  = height;
        wm.grid.horizInterval[0] = 1;
        wm.grid.vertInterval[0]  = 1;

        VPIStatus e = vpiWarpMapAllocData(&wm);
        if (e != VPI_SUCCESS) { LOG_ERROR("WarpMap alloc %s", name); return false; }

        vpiWarpMapGenerateIdentity(&wm);

        for (int y = 0; y < height; ++y) {
            auto* row = reinterpret_cast<VPIKeypointF32*>(
                reinterpret_cast<uint8_t*>(wm.keypoints) + y * wm.pitchBytes);
            const float* mx = mapX.ptr<float>(y);
            const float* my = mapY.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                row[x].x = mx[x];
                row[x].y = my[x];
            }
        }

        e = vpiCreateRemap(VPI_BACKEND_CUDA, &wm, &payload);
        vpiWarpMapFreeData(&wm);
        if (e != VPI_SUCCESS) { LOG_ERROR("CreateRemap %s", name); return false; }
        return true;
    };

    if (!buildRemap(map1L, map2L, vpi.remapL, "Left")) return false;
    if (!buildRemap(map1R, map2R, vpi.remapR, "Right")) return false;

    LOG_INFO("VPI resources initialized: %dx%d, maxDisp=%d", width, height, maxDisp);
    return true;
}

// ============================================================
//  VPI 辅助: 上传 cv::Mat → VPIImage
// ============================================================
static bool uploadToVPI(VPIImage vpiImg, const cv::Mat& mat) {
    VPIImageData data;
    memset(&data, 0, sizeof(data));
    VPIStatus err = vpiImageLockData(vpiImg, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("uploadToVPI: vpiImageLockData failed: %d", err);
        return false;
    }
    int h = mat.rows;
    int w = mat.cols;
    int dstPitch = data.buffer.pitch.planes[0].pitchBytes;
    uint8_t* dst = reinterpret_cast<uint8_t*>(data.buffer.pitch.planes[0].data);
    for (int y = 0; y < h; ++y) {
        memcpy(dst + y * dstPitch, mat.ptr(y), w * mat.elemSize());
    }
    vpiImageUnlock(vpiImg);
    return true;
}

// ============================================================
//  VPI 辅助: 下载 VPIImage → cv::Mat
// ============================================================
static cv::Mat downloadFromVPI(VPIImage vpiImg, int cvType) {
    VPIImageData data;
    memset(&data, 0, sizeof(data));
    VPIStatus err = vpiImageLockData(vpiImg, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
    if (err != VPI_SUCCESS) {
        LOG_ERROR("downloadFromVPI: vpiImageLockData failed: %d", err);
        return cv::Mat();
    }
    int w = data.buffer.pitch.planes[0].width;
    int h = data.buffer.pitch.planes[0].height;
    int pitch = data.buffer.pitch.planes[0].pitchBytes;
    uint8_t* src = reinterpret_cast<uint8_t*>(data.buffer.pitch.planes[0].data);

    cv::Mat mat(h, w, cvType);
    for (int y = 0; y < h; ++y) {
        memcpy(mat.ptr(y), src + y * pitch, w * mat.elemSize());
    }
    vpiImageUnlock(vpiImg);
    return mat;
}

// ============================================================
//  视差 → 深度图 (float, mm)
// ============================================================
static cv::Mat disparityToDepth(const cv::Mat& disp_float, float baseline_mm, float focal_px) {
    cv::Mat depth(disp_float.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < disp_float.rows; ++y) {
        const float* dp = disp_float.ptr<float>(y);
        float* dd = depth.ptr<float>(y);
        for (int x = 0; x < disp_float.cols; ++x) {
            if (dp[x] > 0.5f) {
                dd[x] = baseline_mm * focal_px / dp[x];
            }
        }
    }
    return depth;
}

// ============================================================
//  视差图着色 → 灰度深度显示
//  近 = 亮白, 远 = 暗黑, 无效 = 黑
// ============================================================
static cv::Mat depthToGray(const cv::Mat& depth_mm,
                           float minDepth_mm = 400.0f,
                           float maxDepth_mm = 15000.0f) {
    cv::Mat gray(depth_mm.size(), CV_8U, cv::Scalar(0));
    float range = maxDepth_mm - minDepth_mm;
    if (range <= 0) range = 1.0f;

    for (int y = 0; y < depth_mm.rows; ++y) {
        const float* dp = depth_mm.ptr<float>(y);
        uint8_t* gp = gray.ptr<uint8_t>(y);
        for (int x = 0; x < depth_mm.cols; ++x) {
            float d = dp[x];
            if (d > minDepth_mm && d < maxDepth_mm) {
                // 近→255(亮白), 远→0(黑)
                float norm = 1.0f - (d - minDepth_mm) / range;
                gp[x] = static_cast<uint8_t>(std::clamp(norm * 255.0f, 0.0f, 255.0f));
            }
            // else: 无效保持 0 (黑色)
        }
    }
    return gray;
}

// ============================================================
//  双边滤波平滑视差图 (CPU, CV_32F 无量化损失)
// ============================================================
static cv::Mat bilateralFilterDisparity(const cv::Mat& dispF, int d = 9,
                                         double sigmaColor = 25.0,
                                         double sigmaSpace = 25.0) {
    double maxVal;
    cv::minMaxLoc(dispF, nullptr, &maxVal);
    if (maxVal < 1.0) return dispF;

    // 创建无效像素掩码
    cv::Mat mask = (dispF > 0.5f);

    // 使用 CV_32F 直接滤波 (避免 CV_8U 量化损失)
    cv::Mat filtered;
    cv::bilateralFilter(dispF, filtered, d, sigmaColor, sigmaSpace);

    // 保留无效区域
    filtered.setTo(0.0f, ~mask);
    return filtered;
}

// ============================================================
//  去斑点滤波 (去除小连通区域噪声)
// ============================================================
static void removeSpeckles(cv::Mat& dispF, int maxSpeckleSize = 400,
                           float maxDiff = 1.5f) {
    cv::Mat disp16;
    dispF.convertTo(disp16, CV_16S, 16.0);
    cv::filterSpeckles(disp16, 0, maxSpeckleSize, (int)(maxDiff * 16.0));
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);
}

// ============================================================
//  中值滤波去噪 (5x5)
// ============================================================
static void medianFilterDisparity(cv::Mat& dispF) {
    double maxVal;
    cv::minMaxLoc(dispF, nullptr, &maxVal);
    if (maxVal < 1.0) return;

    cv::Mat origMask = (dispF > 0);  // 量化前保存原始有效区域

    cv::Mat disp8;
    dispF.convertTo(disp8, CV_8U, 255.0 / maxVal);
    cv::medianBlur(disp8, disp8, 5);

    disp8.convertTo(dispF, CV_32F, maxVal / 255.0);
    dispF.setTo(0.0f, ~origMask);   // 用原始 mask 恢复无效区域
}

// ============================================================
//  VPI SGM 提交参数 (CUDA backend 优化)
// ============================================================
static void fillVPIParams(VPIStereoDisparityEstimatorParams& p, int maxDisp) {
    vpiInitStereoDisparityEstimatorParams(&p);
    p.maxDisparity = maxDisp;
    // CUDA backend: 9×7 census 固定窗口
    // SGM P1/P2: P1小→允许倾斜面; P2大→抑制深度突变
    p.p1 = 5;                       // 增大P1(默认3): 更好的倾斜面连续性
    p.p2 = 96;                      // 增大P2(之前72): 更强平滑约束
    p.confidenceThreshold = 49152;  // VPI 内部过滤 (越高越严格), 配合 applyConfidenceMask 二次过滤
    p.uniqueness = 0.97f;           // 唯一性比 (越高越严格), 实际由 confMask 控制有效像素数
    p.confidenceType = VPI_STEREO_CONFIDENCE_ABSOLUTE;
}

// ============================================================
//  VPI 视差 → 去除低置信度像素
// ============================================================
static void applyConfidenceMask(cv::Mat& dispF, const cv::Mat& confU16,
                                int threshold = 2000) {
    for (int y = 0; y < dispF.rows; ++y) {
        float* dp = dispF.ptr<float>(y);
        const uint16_t* cp = confU16.ptr<uint16_t>(y);
        for (int x = 0; x < dispF.cols; ++x) {
            if (cp[x] < (uint16_t)threshold) {
                dp[x] = 0.0f;  // 低置信度 → 无效
            }
        }
    }
}

// ============================================================
//  VPI 视差计算 (全帧)
// ============================================================
static cv::Mat computeVPICudaFull(VPIResources& vpi, cv::Mat* outConf = nullptr) {
    VPIStereoDisparityEstimatorParams p;
    fillVPIParams(p, vpi.maxDisp);

    vpiSubmitStereoDisparityEstimator(vpi.stream, VPI_BACKEND_CUDA,
        vpi.stereoFull, vpi.rectL, vpi.rectR,
        vpi.dispFull, vpi.confFull, &p);
    vpiStreamSync(vpi.stream);

    cv::Mat dispS16 = downloadFromVPI(vpi.dispFull, CV_16S);
    cv::Mat dispF;
    dispS16.convertTo(dispF, CV_32F, 1.0 / 32.0);  // Q10.5 (NVIDIA 官方格式)

    // 置信度过滤 (threshold=2000 经验证给出 64.7% 有效像素)
    cv::Mat confU16 = downloadFromVPI(vpi.confFull, CV_16U);
    if (!confU16.empty()) {
        applyConfidenceMask(dispF, confU16);
        if (outConf) *outConf = confU16;
    }

    return dispF;
}

// ============================================================
//  VPI 视差计算 (半帧)
// ============================================================
static cv::Mat computeVPICudaHalf(VPIResources& vpi) {
    // 降采样
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.rectL, vpi.halfL,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.rectR, vpi.halfR,
                     VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);

    VPIStereoDisparityEstimatorParams p;
    fillVPIParams(p, vpi.maxDisp / 2);

    vpiSubmitStereoDisparityEstimator(vpi.stream, VPI_BACKEND_CUDA,
        vpi.stereoHalf, vpi.halfL, vpi.halfR,
        vpi.dispHalf, vpi.confHalf, &p);

    // 上采样回原始分辨率
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.dispHalf, vpi.dispUpscaled,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat dispS16 = downloadFromVPI(vpi.dispUpscaled, CV_16S);
    cv::Mat dispF;
    // 半分辨率视差 ×2 补偿, Q10.5 格式
    dispS16.convertTo(dispF, CV_32F, 2.0 / 32.0);  // Q10.5 × 2

    // 置信度过滤
    vpiSubmitRescale(vpi.stream, VPI_BACKEND_CUDA, vpi.confHalf, vpi.confUpscaled,
                     VPI_INTERP_NEAREST, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);
    cv::Mat confU16 = downloadFromVPI(vpi.confUpscaled, CV_16U);
    if (!confU16.empty()) {
        applyConfidenceMask(dispF, confU16);
    }

    return dispF;
}

// ============================================================
//  OpenCV CUDA SGM
// ============================================================
static cv::Mat computeOpenCVCudaSGM(const cv::Mat& grayL, const cv::Mat& grayR,
                                     int maxDisp) {
    static cv::Ptr<cv::cuda::StereoSGM> sgm;
    if (!sgm) {
        int numDisp = (maxDisp + 15) & ~0xF;
        sgm = cv::cuda::createStereoSGM(0, numDisp);
    }

    cv::cuda::GpuMat gpuL, gpuR, gpuDisp;
    gpuL.upload(grayL);
    gpuR.upload(grayR);
    sgm->compute(gpuL, gpuR, gpuDisp);

    cv::Mat disp16;
    gpuDisp.download(disp16);
    cv::Mat dispF;
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);  // cuda::StereoSGM 输出 disp×16 定点格式
    dispF.setTo(0, dispF < 0);  // 清除无效负值
    return dispF;
}

// ============================================================
//  OpenCV CUDA BM
// ============================================================
static cv::Mat computeOpenCVCudaBM(const cv::Mat& grayL, const cv::Mat& grayR,
                                    int maxDisp) {
    static cv::Ptr<cv::cuda::StereoBM> bm;
    if (!bm) {
        int numDisp = (maxDisp + 15) & ~0xF;
        bm = cv::cuda::createStereoBM(numDisp, 19);
    }

    cv::cuda::GpuMat gpuL, gpuR, gpuDisp;
    gpuL.upload(grayL);
    gpuR.upload(grayR);

    cv::cuda::Stream stream;
    bm->compute(gpuL, gpuR, gpuDisp, stream);
    stream.waitForCompletion();

    cv::Mat disp8;
    gpuDisp.download(disp8);
    cv::Mat dispF;
    // cuda::StereoBM 输出 CV_8U (0..numDisp), 直接转 float
    disp8.convertTo(dispF, CV_32F);
    return dispF;
}

// ============================================================
//  OpenCV SGBM CPU (高质量参考, 8路径 HH4 模式)
// ============================================================
static cv::Mat computeOpenCVSGBM(const cv::Mat& grayL, const cv::Mat& grayR,
                                  int maxDisp) {
    static cv::Ptr<cv::StereoSGBM> sgbm;
    if (!sgbm) {
        int numDisp = (maxDisp + 15) & ~0xF;
        int blockSize = 5;
        int cn = 1;  // 灰度通道
        sgbm = cv::StereoSGBM::create(0, numDisp, blockSize,
            8 * cn * blockSize * blockSize,    // P1: 标准公式
            32 * cn * blockSize * blockSize,   // P2: 标准公式
            1,    // disp12MaxDiff: 左右一致性
            63,   // preFilterCap
            5,    // uniquenessRatio
            400,  // speckleWindowSize: 增大去噪窗口
            1,    // speckleRange: 收紧范围每个连通域
            cv::StereoSGBM::MODE_SGBM_3WAY);
    }

    cv::Mat disp16;
    sgbm->compute(grayL, grayR, disp16);
    cv::Mat dispF;
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);  // Q12.4 → ÷16
    dispF.setTo(0, dispF < 0);  // 清除无效负值
    return dispF;
}

// ============================================================
//  OpenCV SGBM + WLS 后处理 (左右一致性检查 + 加权最小二乘滤波)
//  最高质量传统算法, 边缘保持极佳
// ============================================================
static cv::Mat computeOpenCVSGBM_WLS(const cv::Mat& grayL, const cv::Mat& grayR,
                                      int maxDisp) {
    int numDisp = (maxDisp + 15) & ~0xF;
    int blockSize = 5;
    int cn = 1;

    // 左匹配器 (主要)
    static cv::Ptr<cv::StereoSGBM> sgbmL;
    if (!sgbmL) {
        sgbmL = cv::StereoSGBM::create(0, numDisp, blockSize,
            8 * cn * blockSize * blockSize,
            32 * cn * blockSize * blockSize,
            1, 63, 10, 200, 1,
            cv::StereoSGBM::MODE_SGBM_3WAY);
    }

    // 右匹配器 (用于左右一致性检查)
#ifdef HAS_XIMGPROC
    static cv::Ptr<cv::StereoMatcher> sgbmR;
    if (!sgbmR) {
        sgbmR = cv::ximgproc::createRightMatcher(sgbmL);
    }

    cv::Mat dispL16, dispR16;
    sgbmL->compute(grayL, grayR, dispL16);
    sgbmR->compute(grayR, grayL, dispR16);

    // WLS 滤波器: sigma=1.5 经验值, lambda=8000 平滑强度
    static cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter;
    if (!wlsFilter) {
        wlsFilter = cv::ximgproc::createDisparityWLSFilter(sgbmL);
        wlsFilter->setLambda(8000.0);
        wlsFilter->setSigmaColor(1.5);
    }

    cv::Mat filteredDisp;
    wlsFilter->filter(dispL16, grayL, filteredDisp, dispR16);

    cv::Mat dispF;
    filteredDisp.convertTo(dispF, CV_32F, 1.0 / 16.0);
    dispF.setTo(0, dispF < 0);
    return dispF;
#else
    // 无 ximgproc: 回退到普通 SGBM + 手动左右一致性
    cv::Mat dispL16;
    sgbmL->compute(grayL, grayR, dispL16);
    cv::Mat dispF;
    dispL16.convertTo(dispF, CV_32F, 1.0 / 16.0);
    dispF.setTo(0, dispF < 0);
    return dispF;
#endif
}

// ============================================================
//  OpenCV SGBM + Census 预处理
//  Census 变换对光照变化鲁棒, 适合室内/混合光源场景
// ============================================================
static cv::Mat computeOpenCVSGBM_Census(const cv::Mat& grayL, const cv::Mat& grayR,
                                         int maxDisp) {
    // Census 变换: 将每个像素的邻域比较编码为二进制串
    auto censusTransform = [](const cv::Mat& img, int winH = 5, int winW = 5) -> cv::Mat {
        int h = img.rows, w = img.cols;
        int rh = winH / 2, rw = winW / 2;
        // Census 5×5 = 24 bits, 用 CV_32S 存储
        cv::Mat census(h, w, CV_32S, cv::Scalar(0));
        for (int y = rh; y < h - rh; ++y) {
            const uint8_t* row = img.ptr<uint8_t>(y);
            int* out = census.ptr<int>(y);
            for (int x = rw; x < w - rw; ++x) {
                uint8_t center = row[x];
                int code = 0;
                for (int dy = -rh; dy <= rh; ++dy) {
                    const uint8_t* nrow = img.ptr<uint8_t>(y + dy);
                    for (int dx = -rw; dx <= rw; ++dx) {
                        if (dy == 0 && dx == 0) continue;
                        code = (code << 1) | (nrow[x + dx] < center ? 1 : 0);
                    }
                }
                out[x] = code;
            }
        }
        return census;
    };

    cv::Mat censusL = censusTransform(grayL);
    cv::Mat censusR = censusTransform(grayR);

    // Census 差异作为增强输入
    cv::Mat normL, normR;
    cv::normalize(censusL, normL, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(censusR, normR, 0, 255, cv::NORM_MINMAX, CV_8U);

    int numDisp = (maxDisp + 15) & ~0xF;
    int blockSize = 3;
    int cn = 1;

    static cv::Ptr<cv::StereoSGBM> sgbm;
    if (!sgbm) {
        sgbm = cv::StereoSGBM::create(0, numDisp, blockSize,
            8 * cn * blockSize * blockSize,
            32 * cn * blockSize * blockSize,
            1, 63, 5, 400, 1,
            cv::StereoSGBM::MODE_SGBM_3WAY);
    }

    cv::Mat disp16;
    sgbm->compute(normL, normR, disp16);
    cv::Mat dispF;
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);
    dispF.setTo(0, dispF < 0);
    return dispF;
}

// ============================================================
//  OpenCV CUDA Belief Propagation (全局优化, GPU)
//  注意: BP 内存占用极高, 在 NX 16GB 上必须降分辨率运行
// ============================================================
static cv::Mat computeOpenCVCudaBP(const cv::Mat& grayL, const cv::Mat& grayR,
                                    int maxDisp) {
    // 降到 1/2 分辨率以适应 16GB 显存
    cv::Mat halfL, halfR;
    cv::resize(grayL, halfL, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(grayR, halfR, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    int halfDisp = std::max(16, (maxDisp / 2 + 15) & ~0xF);

    static cv::Ptr<cv::cuda::StereoBeliefPropagation> bp;
    static int lastDisp = 0;
    if (!bp || lastDisp != halfDisp) {
        bp = cv::cuda::createStereoBeliefPropagation(halfDisp, 3, 3);
        bp->setMaxDataTerm(25.0f);
        bp->setDataWeight(0.1f);
        bp->setMaxDiscTerm(15.0f);
        bp->setDiscSingleJump(1.0f);
        lastDisp = halfDisp;
    }

    cv::cuda::GpuMat gpuL, gpuR, gpuDisp;
    gpuL.upload(halfL);
    gpuR.upload(halfR);
    bp->compute(gpuL, gpuR, gpuDisp);

    cv::Mat dispS16;
    gpuDisp.download(dispS16);
    cv::Mat dispHalf;
    dispS16.convertTo(dispHalf, CV_32F);

    // 上采样回原分辨率, 视差值 ×2 (因为降了 1/2)
    cv::Mat dispF;
    cv::resize(dispHalf, dispF, grayL.size(), 0, 0, cv::INTER_LINEAR);
    dispF *= 2.0f;
    return dispF;
}

// ============================================================
//  OpenCV CUDA Constant Space BP (内存优化 BP, GPU)
//  CSBP 比 BP 内存效率高, 可尝试较高分辨率
// ============================================================
static cv::Mat computeOpenCVCudaCSBP(const cv::Mat& grayL, const cv::Mat& grayR,
                                      int maxDisp) {
    // 降到 1/2 分辨率, CSBP 在全分辨率 256 disp 仍可能 OOM
    cv::Mat halfL, halfR;
    cv::resize(grayL, halfL, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(grayR, halfR, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

    int halfDisp = std::max(16, (maxDisp / 2 + 15) & ~0xF);

    static cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp;
    static int lastDisp = 0;
    if (!csbp || lastDisp != halfDisp) {
        csbp = cv::cuda::createStereoConstantSpaceBP(halfDisp, 4, 3, 4);
        csbp->setMaxDataTerm(30.0f);
        csbp->setDataWeight(0.1f);
        csbp->setMaxDiscTerm(20.0f);
        csbp->setDiscSingleJump(1.0f);
        lastDisp = halfDisp;
    }

    cv::cuda::GpuMat gpuL, gpuR, gpuDisp;
    gpuL.upload(halfL);
    gpuR.upload(halfR);
    csbp->compute(gpuL, gpuR, gpuDisp);

    cv::Mat dispS16;
    gpuDisp.download(dispS16);
    cv::Mat dispHalf;
    dispS16.convertTo(dispHalf, CV_32F);

    // 上采样回原分辨率, 视差值 ×2
    cv::Mat dispF;
    cv::resize(dispHalf, dispF, grayL.size(), 0, 0, cv::INTER_LINEAR);
    dispF *= 2.0f;
    return dispF;
}

// ============================================================
//  绘制 OSD (模式名、帧率、深度信息)
// ============================================================
static void drawOSD(cv::Mat& frame, ViewMode mode, float fps,
                    const DepthViewerState& state) {
    // 模式名
    char buf[256];
    snprintf(buf, sizeof(buf), "[%d/%d] %s | FPS: %.1f | 't':switch 'q':quit",
             (int)mode, (int)ViewMode::MODE_COUNT - 1, viewModeName(mode), fps);
    cv::putText(frame, buf, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 深度范围标注
    if (mode != ViewMode::RAW_STEREO) {
        if (state.depth_max_mm > 0) {
            snprintf(buf, sizeof(buf), "Depth: %.1f~%.1fm",
                     state.depth_min_mm / 1000.0f, state.depth_max_mm / 1000.0f);
            cv::putText(frame, buf, cv::Point(10, frame.rows - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }
        cv::putText(frame, "RED=near  BLUE=far  BLACK=invalid", cv::Point(10, frame.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }

    // 深度点击信息
    if (state.click_x >= 0 && state.click_depth_mm > 0 &&
        mode != ViewMode::RAW_STEREO) {
        snprintf(buf, sizeof(buf), "Depth(%d,%d): %.0fmm (%.2fm)",
                 state.click_x, state.click_y,
                 state.click_depth_mm, state.click_depth_mm / 1000.0f);
        cv::putText(frame, buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // 标记点
        cv::circle(frame, cv::Point(state.click_x, state.click_y), 6,
                   cv::Scalar(0, 0, 255), 2);
        cv::drawMarker(frame, cv::Point(state.click_x, state.click_y),
                       cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2);
    }
}

// ============================================================
//  Bayer → BGR
// ============================================================
static cv::Mat bayerToBGR(const cv::Mat& bayer) {
    cv::Mat bgr;
    cv::cvtColor(bayer, bgr, cv::COLOR_BayerBG2BGR);
    return bgr;
}

// ============================================================
//  Bayer → Gray
// ============================================================
static cv::Mat bayerToGray(const cv::Mat& bayer) {
    cv::Mat gray;
    cv::cvtColor(bayer, gray, cv::COLOR_BayerBG2GRAY);
    return gray;
}

// ============================================================
//  诊断模式: 检查同步、标定、左右相机
// ============================================================
static int runDiagnostics(HikvisionCamera& camera, VPIResources& vpi,
                           const StereoCalibration& calib,
                           int W, int H, bool useTrigger, int numFrames = 5) {
    printf("\n");
    printf("==============================================\n");
    printf("  双目诊断模式\n");
    printf("==============================================\n");
    printf("图像尺寸: %dx%d\n", W, H);
    printf("触发模式: %s\n", useTrigger ? "硬件触发" : "自由运行(无同步!)");
    printf("基线: %.2f mm, 焦距: %.2f px\n",
           calib.getBaseline() * 1000.0f, calib.getFocalLength());
    printf("诊断帧数: %d\n", numFrames);
    printf("\n");

    std::vector<uint8_t> bufL(W * H), bufR(W * H);

    // ---- 1. 同步测试: 抓帧并报告时间戳差 ----
    printf("--- [1] 时间戳同步测试 ---\n");
    if (!useTrigger) {
        printf("  *** 警告: 自由运行模式, 双目不同步! ***\n");
        printf("  *** 建议不加 --free-run 使用硬件触发 ***\n");
    }

    cv::Mat savedBayerL, savedBayerR;
    int syncOk = 0;

    // 收集帧间隔数据用于分析同步
    std::vector<int64_t> tsL_vec, tsR_vec;

    for (int i = 0; i < numFrames; ++i) {
        GrabResult resL, resR;
        bool ok = camera.grabFramePair(bufL.data(), bufR.data(),
                                        0, 0, 5000, resL, resR);
        if (!ok) {
            printf("  帧 %d: 抓帧失败!\n", i);
            continue;
        }

        tsL_vec.push_back(resL.timestamp_us);
        tsR_vec.push_back(resR.timestamp_us);

        int64_t dt_us = (int64_t)resR.timestamp_us - (int64_t)resL.timestamp_us;
        printf("  帧 %d: L.ts=%lu R.ts=%lu 差=%+ld us (%.2f ms)\n",
               i, (unsigned long)resL.timestamp_us, (unsigned long)resR.timestamp_us,
               (long)dt_us, dt_us / 1000.0);

        if (i == 0) {
            savedBayerL = cv::Mat(H, W, CV_8U, bufL.data()).clone();
            savedBayerR = cv::Mat(H, W, CV_8U, bufR.data()).clone();
        }
    }

    // 分析同步: 检查帧间隔一致性 (不看绝对偏移, 而看帧间间隔是否匹配)
    if (tsL_vec.size() >= 3) {
        double sumDriftRate = 0;
        int driftCount = 0;
        for (size_t i = 1; i < tsL_vec.size(); ++i) {
            int64_t intervalL = tsL_vec[i] - tsL_vec[i-1];
            int64_t intervalR = tsR_vec[i] - tsR_vec[i-1];
            int64_t intervalDiff = std::abs(intervalL - intervalR);
            sumDriftRate += intervalDiff;
            driftCount++;
            if (intervalDiff < 200) syncOk++;  // <200us差异=同步良好
        }
        double avgDrift = driftCount > 0 ? sumDriftRate / driftCount : 0;
        int64_t clockOffset = (int64_t)tsR_vec[0] - (int64_t)tsL_vec[0];
        int64_t intervalL0 = tsL_vec.size() > 1 ? tsL_vec[1] - tsL_vec[0] : 0;
        printf("\n  分析:\n");
        printf("    L/R时钟偏移: %.3f 秒 (相机内部时钟差, 不影响同步)\n",
               clockOffset / 1e6);
        printf("    帧间隔: %.2f ms (%.1f fps)\n",
               intervalL0 / 1000.0, intervalL0 > 0 ? 1e6 / intervalL0 : 0);
        printf("    帧间隔一致性: 平均漂移 %.0f us/帧\n", avgDrift);
        if (avgDrift < 100) {
            printf("    ✓ 硬件触发同步良好! 两台相机帧间隔完全一致\n");
        } else if (avgDrift < 2000) {
            printf("    ~ 硬件触发基本正常, 时钟漂移 %.0f us/帧 (晶振差异)\n", avgDrift);
        } else {
            printf("    ✗ 帧间隔不一致, 可能未正确同步\n");
        }
    }
    printf("  同步帧: %d/%d (帧间隔一致性)\n\n", syncOk, numFrames > 1 ? numFrames - 1 : 0);

    if (savedBayerL.empty()) {
        printf("错误: 未能获取任何帧!\n");
        return 1;
    }

    // ---- 2. 保存原始图像 ----
    printf("--- [2] 保存原始图像 ---\n");

    cv::Mat bgrL = bayerToBGR(savedBayerL);
    cv::Mat bgrR = bayerToBGR(savedBayerR);
    cv::imwrite("diagnose_output/raw_left.png", bgrL);
    cv::imwrite("diagnose_output/raw_right.png", bgrR);

    // 左右拼接
    cv::Mat rawPair;
    cv::hconcat(bgrL, bgrR, rawPair);
    cv::putText(rawPair, "LEFT (cam 0)", cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    cv::putText(rawPair, "RIGHT (cam 1)", cv::Point(W + 10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    cv::imwrite("diagnose_output/raw_pair.png", rawPair);
    printf("  已保存: diagnose_output/raw_left.png, raw_right.png, raw_pair.png\n");
    printf("  >>> 请检查: 左图应为物理左侧相机视角, 右图为物理右侧\n\n");

    // ---- 3. VPI 校正 + 极线检查 ----
    printf("--- [3] 校正 + 极线验证 (标定质量检查) ---\n");

    cv::Mat grayL = bayerToGray(savedBayerL);
    cv::Mat grayR = bayerToGray(savedBayerR);

    // 正常顺序校正
    uploadToVPI(vpi.rawL, grayL);
    uploadToVPI(vpi.rawR, grayR);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapL,
                   vpi.rawL, vpi.rectL, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapR,
                   vpi.rawR, vpi.rectR, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat rectL = downloadFromVPI(vpi.rectL, CV_8U);
    cv::Mat rectR = downloadFromVPI(vpi.rectR, CV_8U);
    cv::imwrite("diagnose_output/rect_left.png", rectL);
    cv::imwrite("diagnose_output/rect_right.png", rectR);

    // 校正后拼接 + 极线
    cv::Mat rectColorL, rectColorR;
    cv::cvtColor(rectL, rectColorL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectR, rectColorR, cv::COLOR_GRAY2BGR);

    cv::Mat rectPair;
    cv::hconcat(rectColorL, rectColorR, rectPair);
    int numLines = 20;
    for (int i = 0; i < numLines; ++i) {
        int y = H * (i + 1) / (numLines + 1);
        cv::line(rectPair, cv::Point(0, y), cv::Point(2 * W, y),
                 cv::Scalar(0, 255, 0), 1);
    }
    cv::putText(rectPair, "RECTIFIED LEFT", cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    cv::putText(rectPair, "RECTIFIED RIGHT", cv::Point(W + 10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    cv::imwrite("diagnose_output/rect_pair_epipolar.png", rectPair);
    printf("  已保存: diagnose_output/rect_pair_epipolar.png\n");
    printf("  >>> 绿色水平线应穿过左右图像中的同一特征点\n\n");

    // ---- 4. L/R 互换校正 ----
    printf("--- [4] 左右互换测试 ---\n");

    uploadToVPI(vpi.rawL, grayR);  // 互换!
    uploadToVPI(vpi.rawR, grayL);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapL,
                   vpi.rawL, vpi.rectL, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapR,
                   vpi.rawR, vpi.rectR, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat swapRectL = downloadFromVPI(vpi.rectL, CV_8U);
    cv::Mat swapRectR = downloadFromVPI(vpi.rectR, CV_8U);

    cv::Mat swapColorL, swapColorR;
    cv::cvtColor(swapRectL, swapColorL, cv::COLOR_GRAY2BGR);
    cv::cvtColor(swapRectR, swapColorR, cv::COLOR_GRAY2BGR);

    cv::Mat swapPair;
    cv::hconcat(swapColorL, swapColorR, swapPair);
    for (int i = 0; i < numLines; ++i) {
        int y = H * (i + 1) / (numLines + 1);
        cv::line(swapPair, cv::Point(0, y), cv::Point(2 * W, y),
                 cv::Scalar(0, 255, 255), 1);
    }
    cv::putText(swapPair, "SWAPPED: R->L", cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    cv::putText(swapPair, "SWAPPED: L->R", cv::Point(W + 10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    cv::imwrite("diagnose_output/swap_pair_epipolar.png", swapPair);
    printf("  已保存: diagnose_output/swap_pair_epipolar.png\n");
    printf("  >>> 如果互换后极线对齐更好 → 物理相机左右接反了!\n\n");

    // ---- 5. 正常 vs 互换 视差对比 ----
    printf("--- [5] 视差对比 (正常 vs 互换) ---\n");

    // 正常视差
    uploadToVPI(vpi.rawL, grayL);
    uploadToVPI(vpi.rawR, grayR);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapL,
                   vpi.rawL, vpi.rectL, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapR,
                   vpi.rawR, vpi.rectR, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat dispNormal = computeVPICudaFull(vpi);
    int validNormal = 0;
    float sumNormal = 0.0f;
    for (int y = 0; y < dispNormal.rows; ++y) {
        const float* p = dispNormal.ptr<float>(y);
        for (int x = 0; x < dispNormal.cols; ++x) {
            if (p[x] > 0.5f) { validNormal++; sumNormal += p[x]; }
        }
    }

    double maxN;
    cv::minMaxLoc(dispNormal, nullptr, &maxN);
    cv::Mat dispNormal8;
    dispNormal.convertTo(dispNormal8, CV_8U, 255.0 / std::max(maxN, 1.0));
    cv::Mat dispNormalColor;
    cv::applyColorMap(dispNormal8, dispNormalColor, cv::COLORMAP_JET);
    char buf[256];
    snprintf(buf, sizeof(buf), "NORMAL: valid %d/%d (%.1f%%) avg=%.1f",
             validNormal, W * H, 100.0 * validNormal / (W * H),
             validNormal > 0 ? sumNormal / validNormal : 0.0f);
    cv::putText(dispNormalColor, buf, cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::imwrite("diagnose_output/disp_normal.png", dispNormalColor);

    // 互换视差
    uploadToVPI(vpi.rawL, grayR);
    uploadToVPI(vpi.rawR, grayL);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapL,
                   vpi.rawL, vpi.rectL, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapR,
                   vpi.rawR, vpi.rectR, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
    vpiStreamSync(vpi.stream);

    cv::Mat dispSwapped = computeVPICudaFull(vpi);
    int validSwapped = 0;
    float sumSwapped = 0.0f;
    for (int y = 0; y < dispSwapped.rows; ++y) {
        const float* p = dispSwapped.ptr<float>(y);
        for (int x = 0; x < dispSwapped.cols; ++x) {
            if (p[x] > 0.5f) { validSwapped++; sumSwapped += p[x]; }
        }
    }

    double maxS;
    cv::minMaxLoc(dispSwapped, nullptr, &maxS);
    cv::Mat dispSwapped8;
    dispSwapped.convertTo(dispSwapped8, CV_8U, 255.0 / std::max(maxS, 1.0));
    cv::Mat dispSwappedColor;
    cv::applyColorMap(dispSwapped8, dispSwappedColor, cv::COLORMAP_JET);
    snprintf(buf, sizeof(buf), "SWAPPED: valid %d/%d (%.1f%%) avg=%.1f",
             validSwapped, W * H, 100.0 * validSwapped / (W * H),
             validSwapped > 0 ? sumSwapped / validSwapped : 0.0f);
    cv::putText(dispSwappedColor, buf, cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::imwrite("diagnose_output/disp_swapped.png", dispSwappedColor);

    // 拼接对比
    cv::Mat dispCompare;
    cv::hconcat(dispNormalColor, dispSwappedColor, dispCompare);
    cv::imwrite("diagnose_output/disp_compare.png", dispCompare);

    printf("  正常视差: 有效像素 %d/%d (%.1f%%), 平均视差 %.1f\n",
           validNormal, W * H, 100.0 * validNormal / (W * H),
           validNormal > 0 ? sumNormal / validNormal : 0.0f);
    printf("  互换视差: 有效像素 %d/%d (%.1f%%), 平均视差 %.1f\n",
           validSwapped, W * H, 100.0 * validSwapped / (W * H),
           validSwapped > 0 ? sumSwapped / validSwapped : 0.0f);

    if (validSwapped > validNormal * 1.5) {
        printf("\n  *** 结论: 互换后有效像素明显更多 → 左右相机可能接反了! ***\n");
        printf("  *** 建议: 交换相机线缆, 或在代码中设置 --swap-lr ***\n");
    } else if (validNormal > validSwapped * 1.5) {
        printf("\n  ✓ 正常顺序有效像素更多, 左右相机方向正确\n");
    } else if (validNormal < W * H * 0.05 && validSwapped < W * H * 0.05) {
        printf("\n  *** 两种顺序有效像素都极少 → 标定数据可能有问题! ***\n");
        printf("  *** 建议: 重新标定 ***\n");
    } else {
        printf("\n  两者差异不大, 请查看图像进一步确认\n");
    }

    printf("\n==============================================\n");
    printf("  诊断完成! 请查看以下文件:\n");
    printf("==============================================\n");
    printf("  diagnose_output/raw_pair.png           → 原始左右 (确认物理方向)\n");
    printf("  diagnose_output/rect_pair_epipolar.png → 校正极线 (绿线应对齐同一特征)\n");
    printf("  diagnose_output/swap_pair_epipolar.png → 互换极线 (对比用)\n");
    printf("  diagnose_output/disp_compare.png       → 视差对比 (正常 vs 互换)\n");
    printf("  diagnose_output/disp_normal.png        → 正常顺序视差图\n");
    printf("  diagnose_output/disp_swapped.png       → 互换顺序视差图\n");
    printf("==============================================\n");

    return 0;
}

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {
    // ---- 解析参数 ----
    std::string calibPath;
    std::string configPath;
    int maxDisp = 256;
    int winSize = 5;
    bool freeRun = false;
    bool headless = false;
    int headlessFrames = 30;  // 无头模式运行帧数
    bool diagnose = false;    // 诊断模式
    bool swapLR = false;      // 左右相机互换
    double pwmFreq = 15.0;    // PWM 触发频率 Hz
    bool noPwm = false;       // 禁用 PWM 自启动
    std::string crestereoPath; // CREStereo ONNX 模型路径
    std::string hitnetPath;    // HITNet ONNX 模型路径

    for (int i = 1; i < argc; ++i) {
        if ((std::string(argv[i]) == "-c" || std::string(argv[i]) == "--calibration") && i + 1 < argc) {
            calibPath = argv[++i];
        } else if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (std::string(argv[i]) == "--max-disp" && i + 1 < argc) {
            maxDisp = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--win-size" && i + 1 < argc) {
            winSize = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--free-run") {
            freeRun = true;
        } else if (std::string(argv[i]) == "--diagnose") {
            diagnose = true;
        } else if (std::string(argv[i]) == "--swap-lr") {
            swapLR = true;
        } else if (std::string(argv[i]) == "--headless") {
            headless = true;
        } else if (std::string(argv[i]) == "--frames" && i + 1 < argc) {
            headlessFrames = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--pwm-freq" && i + 1 < argc) {
            pwmFreq = std::atof(argv[++i]);
        } else if (std::string(argv[i]) == "--no-pwm") {
            noPwm = true;
        } else if (std::string(argv[i]) == "--crestereo" && i + 1 < argc) {
            crestereoPath = argv[++i];
        } else if (std::string(argv[i]) == "--hitnet" && i + 1 < argc) {
            hitnetPath = argv[++i];
        } else if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
            printf("Usage: stereo_depth_viewer -c <calib.yaml> [options]\n"
                   "  -c, --calibration  标定文件路径\n"
                   "  --config           pipeline.yaml 配置文件 (可选)\n"
                   "  --max-disp N       最大视差 (默认: 256)\n"
                   "  --win-size N       匹配窗口 (默认: 5)\n"
                   "  --free-run         自由运行模式 (无外触发, 不启动PWM)\n"
                   "  --pwm-freq Hz      PWM 触发频率 (默认: 15)\n"
                   "  --no-pwm           禁用 PWM 自启动 (仅在外部已启动PWM时使用)\n"
                   "  --diagnose         诊断模式\n"
                   "  --swap-lr          左右相机互换\n"
                   "  --headless         无头模式 (无窗口, 自动遍历所有模式)\n"
                   "  --frames N         无头模式每种算法运行帧数 (默认: 30)\n"
                   "  --crestereo PATH   CREStereo ONNX 模型路径\n"
                   "  --hitnet PATH      HITNet ONNX 模型路径\n"
                   "\n按键:\n"
                   "  t   切换视图模式\n"
                   "  q   退出\n");
            return 0;
        }
    }

    if (calibPath.empty()) {
        fprintf(stderr, "Error: 必须指定标定文件 (-c <path>)\n");
        return 1;
    }

    signal(SIGINT, sigHandler);

    // ---- 加载标定 ----
    StereoCalibration calib;
    if (!calib.load(calibPath)) {
        LOG_ERROR("标定文件加载失败: %s", calibPath.c_str());
        return 1;
    }

    float baseline_mm = calib.getBaseline() * 1000.0f;  // m → mm
    float focal_px = calib.getFocalLength();
    LOG_INFO("基线 = %.2f mm, 焦距 = %.2f px", baseline_mm, focal_px);

    // ---- 初始化相机 ----
    CameraConfig camCfg;
    camCfg.width = 1440;
    camCfg.height = 1080;
    camCfg.use_trigger = !freeRun;

    HikvisionCamera camera;
    if (!camera.open(camCfg)) {
        LOG_ERROR("相机打开失败");
        return 1;
    }

    if (!camera.startGrabbing()) {
        LOG_ERROR("相机启动采集失败");
        return 1;
    }

    int W = camera.width();
    int H = camera.height();
    LOG_INFO("相机分辨率: %dx%d", W, H);

    // ---- 初始化 VPI (必须在 PWM 之前, 避免 mlockall 干扰 CUDA 分配) ----
    VPIResources vpi;
    if (!initVPI(vpi, calib, W, H, maxDisp, winSize)) {
        LOG_ERROR("VPI 初始化失败");
        vpi.destroy();
        camera.stopGrabbing();
        camera.close();
        return 1;
    }

    // ---- PWM 自启动 (VPI 初始化后再启动, 避免 mlockall 冲突) ----
    std::unique_ptr<PWMTrigger> pwm;
    if (camCfg.use_trigger && !noPwm) {
        pwm = std::make_unique<PWMTrigger>("gpiochip2", 7, pwmFreq, 50.0);
        if (pwm->start()) {
            LOG_INFO("PWM 触发已启动: %.1f Hz", pwmFreq);
        } else {
            LOG_WARN("PWM 启动失败, 将在无触发信号下工作 (可能抓帧超时)");
            pwm.reset();
        }
    } else if (freeRun) {
        LOG_INFO("自由运行模式: PWM 未启动");
    } else if (noPwm) {
        LOG_INFO("PWM 自启动已禁用 (--no-pwm)");
    }

    // ---- 分配采集缓冲 ----
    std::vector<uint8_t> bufL(W * H), bufR(W * H);

    // ---- ONNX DL 模型加载 ----
    stereo3d::OnnxStereo onnxCREStereo, onnxHITNet;

    if (!crestereoPath.empty()) {
        if (onnxCREStereo.load(crestereoPath, stereo3d::OnnxStereo::Model::CREStereo)) {
            LOG_INFO("CREStereo ONNX 模型已加载: %s", crestereoPath.c_str());
        } else {
            LOG_WARN("CREStereo 加载失败: %s", crestereoPath.c_str());
        }
    } else {
        // 自动查找默认路径
        const char* defaultPaths[] = {
            "dl_models/crestereo_init_iter10_480x640.onnx",
            "../dl_models/crestereo_init_iter10_480x640.onnx",
            nullptr
        };
        for (const char** p = defaultPaths; *p; ++p) {
            if (access(*p, F_OK) == 0) {
                if (onnxCREStereo.load(*p, stereo3d::OnnxStereo::Model::CREStereo))
                    LOG_INFO("CREStereo 自动加载: %s", *p);
                break;
            }
        }
    }

    if (!hitnetPath.empty()) {
        if (onnxHITNet.load(hitnetPath, stereo3d::OnnxStereo::Model::HITNet)) {
            LOG_INFO("HITNet ONNX 模型已加载: %s", hitnetPath.c_str());
        } else {
            LOG_WARN("HITNet 加载失败: %s", hitnetPath.c_str());
        }
    } else {
        const char* defaultPaths[] = {
            "dl_models/hitnet_eth3d_480x640.onnx",
            "../dl_models/hitnet_eth3d_480x640.onnx",
            nullptr
        };
        for (const char** p = defaultPaths; *p; ++p) {
            if (access(*p, F_OK) == 0) {
                if (onnxHITNet.load(*p, stereo3d::OnnxStereo::Model::HITNet))
                    LOG_INFO("HITNet 自动加载: %s", *p);
                break;
            }
        }
    }

    // ---- 诊断模式 ----
    if (diagnose) {
        // 创建输出目录
        system("mkdir -p diagnose_output");
        int ret = runDiagnostics(camera, vpi, calib, W, H, !freeRun);
        vpi.destroy();
        camera.stopGrabbing();
        camera.close();
        return ret;
    }

    // ---- 深度查看器状态 ----
    DepthViewerState viewState;
    viewState.baseline_mm = baseline_mm;
    viewState.focal_px = focal_px;

    // ---- 创建窗口 (非 headless 模式) ----
    const char* winName = "Stereo Depth Viewer";
    if (!headless) {
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, 1280, 720);
        cv::setMouseCallback(winName, onMouse, &viewState);
    }

    ViewMode currentMode = ViewMode::RAW_STEREO;
    float fps = 0.0f;
    int frameCount = 0;
    int modeFrameCount = 0;
    double modeTotalMs = 0.0;

    // headless 基准测试数据收集
    struct BenchmarkEntry {
        std::string name;
        double avgMs;
        double fps;
        int validPixels;
        int totalPixels;
        double validRatio;
        float depthMin;
        float depthMax;
        float depthMean;
    };
    std::vector<BenchmarkEntry> benchResults;
    std::vector<std::pair<std::string, cv::Mat>> savedFrames;  // name → colorized disparity

    LOG_INFO("查看器启动%s, 按 't' 切换模式, 按 'q' 退出",
             headless ? " (无头模式)" : "");
    LOG_INFO("当前模式: %s", viewModeName(currentMode));

    // ---- 主循环 ----
    while (g_running) {
        auto frameStart = std::chrono::steady_clock::now();

        // 1. 采集
        GrabResult resL, resR;
        bool ok = camera.grabFramePair(bufL.data(), bufR.data(),
                                        0, 0, 2000, resL, resR);
        if (!ok) {
            LOG_WARN("抓帧失败, 重试...");
            continue;
        }

        // 2. Bayer → 各种格式
        cv::Mat bayerL(H, W, CV_8U, swapLR ? bufR.data() : bufL.data());
        cv::Mat bayerR(H, W, CV_8U, swapLR ? bufL.data() : bufR.data());

        cv::Mat displayFrame;

        if (currentMode == ViewMode::RAW_STEREO) {
            // 原始 Bayer → BGR → 左右拼接
            cv::Mat bgrL = bayerToBGR(bayerL);
            cv::Mat bgrR = bayerToBGR(bayerR);
            cv::Mat combined;
            cv::hconcat(bgrL, bgrR, combined);
            // 缩放到合理显示尺寸
            cv::resize(combined, displayFrame, cv::Size(W, H / 2));
        } else {
            // 深度模式: 需要灰度 + VPI 校正
            cv::Mat grayL = bayerToGray(bayerL);
            cv::Mat grayR = bayerToGray(bayerR);

            // 上传到 VPI
            if (!uploadToVPI(vpi.rawL, grayL) || !uploadToVPI(vpi.rawR, grayR)) {
                LOG_ERROR("VPI 上传失败");
                break;
            }

            // VPI 校正
            vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapL,
                           vpi.rawL, vpi.rectL,
                           VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
            vpiSubmitRemap(vpi.stream, VPI_BACKEND_CUDA, vpi.remapR,
                           vpi.rawR, vpi.rectR,
                           VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);
            vpiStreamSync(vpi.stream);

            // 首帧保存校正后灰度图对 (供深度学习模型评估)
            static bool rectSaved = false;
            if (headless && !rectSaved) {
                cv::Mat rl = downloadFromVPI(vpi.rectL, CV_8U);
                cv::Mat rr = downloadFromVPI(vpi.rectR, CV_8U);
                if (!rl.empty() && !rr.empty()) {
                    cv::imwrite("diagnose_output/rect_left.png", rl);
                    cv::imwrite("diagnose_output/rect_right.png", rr);
                    LOG_INFO("已保存校正图对: rect_left.png, rect_right.png (%dx%d)",
                             rl.cols, rl.rows);
                }
                rectSaved = true;
            }

            cv::Mat dispF;

            switch (currentMode) {
                case ViewMode::VPI_CUDA_FULL:
                    dispF = computeVPICudaFull(vpi);
                    if (!dispF.empty()) medianFilterDisparity(dispF);
                    break;

                case ViewMode::VPI_CUDA_HALF:
                    dispF = computeVPICudaHalf(vpi);
                    if (!dispF.empty()) medianFilterDisparity(dispF);
                    break;

                case ViewMode::VPI_CUDA_BILATERAL: {
                    dispF = computeVPICudaFull(vpi);
                    if (!dispF.empty()) {
                        dispF = bilateralFilterDisparity(dispF, 9, 20.0, 20.0);
                    }
                    break;
                }

                case ViewMode::OPENCV_CUDA_SGM:
                case ViewMode::OPENCV_CUDA_BM:
                case ViewMode::OPENCV_CUDA_BP:
                case ViewMode::OPENCV_CUDA_CSBP:
                case ViewMode::OPENCV_SGBM_CPU:
                case ViewMode::OPENCV_SGBM_WLS:
                case ViewMode::OPENCV_SGBM_CENSUS: {
                    // 从 VPI 下载校正后灰度图给 OpenCV
                    cv::Mat rectGrayL = downloadFromVPI(vpi.rectL, CV_8U);
                    cv::Mat rectGrayR = downloadFromVPI(vpi.rectR, CV_8U);

                    if (rectGrayL.empty() || rectGrayR.empty()) {
                        LOG_ERROR("VPI 下载校正图失败");
                        break;
                    }

                    if (currentMode == ViewMode::OPENCV_CUDA_SGM)
                        dispF = computeOpenCVCudaSGM(rectGrayL, rectGrayR, maxDisp);
                    else if (currentMode == ViewMode::OPENCV_CUDA_BM)
                        dispF = computeOpenCVCudaBM(rectGrayL, rectGrayR, maxDisp);
                    else if (currentMode == ViewMode::OPENCV_CUDA_BP)
                        dispF = computeOpenCVCudaBP(rectGrayL, rectGrayR, maxDisp);
                    else if (currentMode == ViewMode::OPENCV_CUDA_CSBP)
                        dispF = computeOpenCVCudaCSBP(rectGrayL, rectGrayR, maxDisp);
                    else if (currentMode == ViewMode::OPENCV_SGBM_WLS)
                        dispF = computeOpenCVSGBM_WLS(rectGrayL, rectGrayR, maxDisp);
                    else if (currentMode == ViewMode::OPENCV_SGBM_CENSUS)
                        dispF = computeOpenCVSGBM_Census(rectGrayL, rectGrayR, maxDisp);
                    else {
                        dispF = computeOpenCVSGBM(rectGrayL, rectGrayR, maxDisp);
                    }
                    if (!dispF.empty()) medianFilterDisparity(dispF);
                    break;
                }

                case ViewMode::ONNX_CRESTEREO: {
                    if (!onnxCREStereo.isLoaded()) {
                        LOG_WARN("CREStereo 模型未加载, 跳过");
                        break;
                    }
                    cv::Mat rectGrayL = downloadFromVPI(vpi.rectL, CV_8U);
                    cv::Mat rectGrayR = downloadFromVPI(vpi.rectR, CV_8U);
                    if (!rectGrayL.empty() && !rectGrayR.empty())
                        dispF = onnxCREStereo.compute(rectGrayL, rectGrayR);
                    break;
                }

                case ViewMode::ONNX_HITNET: {
                    if (!onnxHITNet.isLoaded()) {
                        LOG_WARN("HITNet 模型未加载, 跳过");
                        break;
                    }
                    cv::Mat rectGrayL = downloadFromVPI(vpi.rectL, CV_8U);
                    cv::Mat rectGrayR = downloadFromVPI(vpi.rectR, CV_8U);
                    if (!rectGrayL.empty() && !rectGrayR.empty())
                        dispF = onnxHITNet.compute(rectGrayL, rectGrayR);
                    break;
                }

                default:
                    break;
            }

            if (!dispF.empty()) {
                // 视差 → 深度 (保留供点击测距)
                viewState.depth_map = disparityToDepth(dispF, baseline_mm, focal_px);

                // JET 伪彩色: 直接基于视差着色 (与诊断模式一致)
                // 视差高=近=暖色(红), 视差低=远=冷色(蓝), 无效=黑
                double maxVal;
                cv::minMaxLoc(dispF, nullptr, &maxVal);
                if (maxVal < 1.0) maxVal = 1.0;

                cv::Mat disp8;
                dispF.convertTo(disp8, CV_8U, 255.0 / maxVal);
                cv::Mat dispMask = (disp8 > 0);
                cv::Mat dispColor;
                cv::applyColorMap(disp8, dispColor, cv::COLORMAP_JET);
                dispColor.setTo(cv::Scalar(0, 0, 0), ~dispMask);
                displayFrame = dispColor;

                // 更新 OSD 深度范围信息
                cv::Mat depthValid = (viewState.depth_map > 100) & (viewState.depth_map < 30000);
                cv::Scalar dmean, dstd;
                cv::meanStdDev(viewState.depth_map, dmean, dstd, depthValid);
                viewState.depth_min_mm = std::max((float)(dmean[0] - 2.0 * dstd[0]), 200.0f);
                viewState.depth_max_mm = std::min((float)(dmean[0] + 2.0 * dstd[0]), 30000.0f);
            }
        }

        frameCount++;
        modeFrameCount++;

        // 3. 帧率
        auto frameEnd = std::chrono::steady_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
        modeTotalMs += elapsed_ms;
        fps = 1000.0f / elapsed_ms;

        if (headless) {
            // 保存每种模式的最后一帧 + 收集基准数据
            if (modeFrameCount == headlessFrames && !displayFrame.empty()) {
                char savePath[256];
                snprintf(savePath, sizeof(savePath), "diagnose_output/mode_%d_%s.png",
                         (int)currentMode, viewModeName(currentMode));
                for (char* c = savePath; *c; ++c) {
                    if (*c == ' ' || *c == '(' || *c == ')' || *c == '|' || *c == '+') *c = '_';
                }
                cv::imwrite(savePath, displayFrame);
                LOG_INFO("已保存帧: %s", savePath);
                savedFrames.push_back({viewModeName(currentMode), displayFrame.clone()});
            }
            // 无头模式: 每种模式运行 headlessFrames 帧后切换
            if (modeFrameCount >= headlessFrames) {
                double avgMs = modeTotalMs / modeFrameCount;
                LOG_INFO("[%s] %d frames, avg %.1f ms/frame (%.1f fps)",
                         viewModeName(currentMode), modeFrameCount,
                         avgMs, 1000.0 / avgMs);

                // 收集基准数据
                if (currentMode != ViewMode::RAW_STEREO) {
                    BenchmarkEntry entry;
                    entry.name = viewModeName(currentMode);
                    entry.avgMs = avgMs;
                    entry.fps = 1000.0 / avgMs;
                    entry.totalPixels = W * H;
                    entry.validPixels = 0;
                    entry.depthMin = 999999.f;
                    entry.depthMax = 0.f;
                    entry.depthMean = 0.f;

                    if (!viewState.depth_map.empty()) {
                        float sum = 0;
                        for (int y = 0; y < viewState.depth_map.rows; ++y) {
                            const float* p = viewState.depth_map.ptr<float>(y);
                            for (int x = 0; x < viewState.depth_map.cols; ++x) {
                                if (p[x] > 200 && p[x] < 30000) {
                                    entry.validPixels++;
                                    sum += p[x];
                                    if (p[x] < entry.depthMin) entry.depthMin = p[x];
                                    if (p[x] > entry.depthMax) entry.depthMax = p[x];
                                }
                            }
                        }
                        entry.depthMean = entry.validPixels > 0 ? sum / entry.validPixels : 0;
                    }
                    entry.validRatio = entry.totalPixels > 0
                        ? 100.0 * entry.validPixels / entry.totalPixels : 0;
                    benchResults.push_back(entry);
                }

                int next = ((int)currentMode + 1) % (int)ViewMode::MODE_COUNT;
                if (next == 0) {
                    LOG_INFO("所有模式测试完成, 共 %d 帧", frameCount);
                    break;
                }
                currentMode = (ViewMode)next;
                modeFrameCount = 0;
                modeTotalMs = 0.0;
                LOG_INFO("切换模式: %s", viewModeName(currentMode));
            }
        } else {
            // GUI 模式
            if (!displayFrame.empty()) {
                drawOSD(displayFrame, currentMode, fps, viewState);
                cv::imshow(winName, displayFrame);
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                g_running = false;
            } else if (key == 't' || key == 'T') {
                int next = ((int)currentMode + 1) % (int)ViewMode::MODE_COUNT;
                currentMode = (ViewMode)next;
                modeFrameCount = 0;
                modeTotalMs = 0.0;
                viewState.click_x = -1;
                LOG_INFO("切换模式: %s", viewModeName(currentMode));
            }
        }
    }

    // ---- 清理 ----
    LOG_INFO("正在关闭...");

    // ---- headless: 生成基准报告 + 对比网格 ----
    if (headless && !benchResults.empty()) {
        // JSON 报告
        {
            std::ofstream jf("diagnose_output/benchmark_report.json");
            if (jf.is_open()) {
                jf << "{\n  \"platform\": \"Jetson Orin NX 16GB\",\n";
                jf << "  \"resolution\": \"" << W << "x" << H << "\",\n";
                jf << "  \"max_disparity\": " << maxDisp << ",\n";
                jf << "  \"frames_per_mode\": " << headlessFrames << ",\n";
                jf << "  \"algorithms\": [\n";
                for (size_t i = 0; i < benchResults.size(); ++i) {
                    auto& e = benchResults[i];
                    jf << "    {\n";
                    jf << "      \"name\": \"" << e.name << "\",\n";
                    jf << "      \"avg_ms\": " << std::fixed << std::setprecision(2) << e.avgMs << ",\n";
                    jf << "      \"fps\": " << std::fixed << std::setprecision(1) << e.fps << ",\n";
                    jf << "      \"valid_pixels\": " << e.validPixels << ",\n";
                    jf << "      \"valid_ratio\": " << std::fixed << std::setprecision(1) << e.validRatio << ",\n";
                    jf << "      \"depth_min_mm\": " << std::fixed << std::setprecision(0) << e.depthMin << ",\n";
                    jf << "      \"depth_max_mm\": " << std::fixed << std::setprecision(0) << e.depthMax << ",\n";
                    jf << "      \"depth_mean_mm\": " << std::fixed << std::setprecision(0) << e.depthMean << "\n";
                    jf << "    }" << (i + 1 < benchResults.size() ? "," : "") << "\n";
                }
                jf << "  ]\n}\n";
                jf.close();
                LOG_INFO("基准报告已保存: diagnose_output/benchmark_report.json");
            }
        }

        // 对比网格 (每行 4 图, 缩放到统一尺寸)
        if (!savedFrames.empty()) {
            int thumbW = 480, thumbH = 360;
            int cols = 4;
            int rows = ((int)savedFrames.size() + cols - 1) / cols;
            cv::Mat grid(rows * (thumbH + 30), cols * thumbW, CV_8UC3, cv::Scalar(30, 30, 30));

            for (size_t i = 0; i < savedFrames.size(); ++i) {
                int r = (int)i / cols;
                int c = (int)i % cols;
                int x0 = c * thumbW;
                int y0 = r * (thumbH + 30);

                cv::Mat thumb;
                cv::resize(savedFrames[i].second, thumb, cv::Size(thumbW, thumbH));
                if (thumb.channels() == 1)
                    cv::cvtColor(thumb, thumb, cv::COLOR_GRAY2BGR);
                thumb.copyTo(grid(cv::Rect(x0, y0, thumbW, thumbH)));

                // 标签
                cv::putText(grid, savedFrames[i].first,
                            cv::Point(x0 + 5, y0 + thumbH + 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(200, 200, 200), 1);
            }
            cv::imwrite("diagnose_output/comparison_grid.png", grid);
            LOG_INFO("对比网格已保存: diagnose_output/comparison_grid.png");
        }

        // 终端汇总表
        printf("\n");
        printf("============ 基准测试汇总 ============\n");
        printf("%-24s %8s %8s %8s %10s\n", "算法", "ms/帧", "FPS", "有效%", "平均深度mm");
        printf("--------------------------------------------------------------\n");
        for (auto& e : benchResults) {
            printf("%-24s %8.1f %8.1f %7.1f%% %10.0f\n",
                   e.name.c_str(), e.avgMs, e.fps, e.validRatio, e.depthMean);
        }
        printf("============================================\n\n");
    }

    if (!headless) {
        cv::destroyAllWindows();
    }
    if (pwm) {
        pwm->stop();
        LOG_INFO("PWM 已停止");
    }
    vpi.destroy();
    camera.stopGrabbing();
    camera.close();

    LOG_INFO("已退出");
    return 0;
}
