#pragma once

#include "calibration/stereo_calibration.h"
#include "capture/hikvision_camera.h"
#include "stereo_depth_viewer_vpi.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

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
static int runDiagnostics(stereo3d::HikvisionCamera& camera, VPIResources& vpi,
                           const stereo3d::StereoCalibration& calib,
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
        stereo3d::GrabResult resL, resR;
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
