#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#ifdef HAS_XIMGPROC
#include <opencv2/ximgproc/disparity_filter.hpp>
#endif

#include <algorithm>
#include <cstdint>

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
