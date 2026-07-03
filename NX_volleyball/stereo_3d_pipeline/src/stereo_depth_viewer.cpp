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
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>
#include <utility>

#include <opencv2/opencv.hpp>

#include "calibration/stereo_calibration.h"
#include "calibration/pwm_trigger.h"
#include "capture/hikvision_camera.h"
#include "stereo/onnx_stereo.h"
#include "utils/logger.h"
#include "stereo_depth_viewer_vpi.h"
#include "stereo_depth_viewer_diagnostics.h"
#include "stereo_depth_viewer_algorithms.h"
#include "stereo_depth_viewer_args.h"
#include "stereo_depth_viewer_benchmark.h"
#include "stereo_depth_viewer_models.h"
#include "stereo_depth_viewer_ui.h"

using namespace stereo3d;

// ============================================================
//  全局状态
// ============================================================
static std::atomic<bool> g_running{true};

static void sigHandler(int) { g_running = false; }

// ============================================================
//  main
// ============================================================
int main(int argc, char** argv) {
    // ---- 解析参数 ----
    const StereoDepthViewerArgs args = parseStereoDepthViewerArgs(argc, argv);
    if (args.should_exit) return args.exit_code;
    const std::string& calibPath = args.calib_path;
    const int maxDisp = args.max_disp;
    const int winSize = args.win_size;
    const bool freeRun = args.free_run;
    const bool headless = args.headless;
    const int headlessFrames = args.headless_frames;
    const bool diagnose = args.diagnose;
    const bool swapLR = args.swap_lr;
    const double pwmFreq = args.pwm_freq;
    const bool noPwm = args.no_pwm;
    const std::string& crestereoPath = args.crestereo_path;
    const std::string& hitnetPath = args.hitnet_path;

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
    loadViewerOnnxModels(crestereoPath, hitnetPath,
                         onnxCREStereo, onnxHITNet);

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
        cv::setMouseCallback(winName, onDepthViewerMouse, &viewState);
    }

    ViewMode currentMode = ViewMode::RAW_STEREO;
    float fps = 0.0f;
    int frameCount = 0;
    int modeFrameCount = 0;
    double modeTotalMs = 0.0;

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
        writeHeadlessBenchmarkReport(W, H, maxDisp, headlessFrames,
                                     benchResults, savedFrames);
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
