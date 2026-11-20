#include "aruco_stereo.h"
#include "aruco_board.h"
#include "images.h"
#include "../common/atomic_file.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool loadMonoIntrinsics(
    const std::string& path,
    const std::string& side,
    cv::Mat& K,
    cv::Mat& D,
    cv::Size& imageSize)
{
    try {
        cv::FileStorage file(
            path,
            cv::FileStorage::READ
        );

        if (!file.isOpened()) {
            fprintf(stderr,
                    "[ERROR] Cannot open intrinsics: %s\n",
                    path.c_str());
            return false;
        }

        // 与之前单目 YAML 保存的字段对应
        const std::string keyK =
            "camera_matrix_" + side;

        const std::string keyD =
            "distortion_coefficients_" + side;

        file[keyK] >> K;
        file[keyD] >> D;

        file["image_width"] >> imageSize.width;
        file["image_height"] >> imageSize.height;

        file.release();

        // 检查数据完整性
        if (K.rows != 3 || K.cols != 3 ||
            D.empty() ||
            imageSize.width <= 0 ||
            imageSize.height <= 0) {

            fprintf(stderr,
                    "[ERROR] Invalid intrinsics: %s\n",
                    path.c_str());
            return false;
        }

        // 保证矩阵使用 double 类型
        K.convertTo(K, CV_64F);
        D.convertTo(D, CV_64F);

        const size_t n = D.total();

        if ((n != 4 && n != 5 && n != 8 &&
             n != 12 && n != 14) ||
            !cv::checkRange(K) ||
            !cv::checkRange(D) ||
            K.at<double>(0, 0) <= 0 ||
            K.at<double>(1, 1) <= 0) {

            fprintf(stderr,
                    "[ERROR] Invalid K or D: %s\n",
                    path.c_str());
            return false;
        }

        return true;
    }
    catch (const cv::Exception& e) {
        fprintf(stderr,
                "[ERROR] Reading %s failed: %s\n",
                path.c_str(),
                e.what());
        return false;
    }
}

int runArucoStereoCalibration(
    const StereoCalibrateArgs& args)
{
    // 1. 加载左右相机内参
    cv::Mat K1, D1, K2, D2;

    cv::Size sizeL;
    cv::Size sizeR;

    if (!loadMonoIntrinsics(
            args.intrinsics_left,
            "left",
            K1,
            D1,
            sizeL)) {
        return 1;
    }

    if (!loadMonoIntrinsics(
            args.intrinsics_right,
            "right",
            K2,
            D2,
            sizeR)) {
        return 1;
    }

    if (sizeL != sizeR) {
        fprintf(stderr,
                "[ERROR] Left/right image sizes differ\n");
        return 1;
    }

    const cv::Size imgSize = sizeL;

    // 2. 获取左右相机图片
    auto leftFiles = globImages(
        fs::path(args.images_dir) / "left"
    );

    auto rightFiles = globImages(
        fs::path(args.images_dir) / "right"
    );

    // 3. 根据文件名配对
    auto pairs = pairImagesByStem(
        leftFiles,
        rightFiles
    );

    if (pairs.empty()) {
        fprintf(stderr,
                "[ERROR] No paired stereo images\n");
        return 1;
    }

    // 4. 准备标定数据
    std::vector<std::vector<cv::Point3f>> objPoints;

    std::vector<std::vector<cv::Point2f>> imgPointsL;
    std::vector<std::vector<cv::Point2f>> imgPointsR;

    std::vector<ImagePair> acceptedPairs;

    // 5. 逐组检测左右相机图像
    for (const auto& pair : pairs) {

        ArucoImageResult left =
            detectBoardTags(pair.left);

        ArucoImageResult right =
            detectBoardTags(pair.right);

        if (!left.read_ok || !right.read_ok) {
            printf("%s SKIP (image read failed)\n",
                   pair.stem.c_str());
            continue;
        }

        if (left.image_size != imgSize ||
            right.image_size != imgSize) {
            printf("%s SKIP (size mismatch)\n",
                   pair.stem.c_str());
            continue;
        }

        // 当前图像对的三维点与二维点
        std::vector<cv::Point3f> object;
        std::vector<cv::Point2f> pointsL;
        std::vector<cv::Point2f> pointsR;

        int commonTags = 0;

        // 6. 通过 ID 建立左右对应关系
        for (int id : {1, 2, 3, 4}) {

            auto itL = left.tags.find(id);
            auto itR = right.tags.find(id);

            // 左右都检测到相同 ID 才接受
            if (itL == left.tags.end() ||
                itR == right.tags.end()) {
                continue;
            }

            auto points3D = boardMarkerCorners(id);

            if (points3D.size() != 4 ||
                itL->second.size() != 4 ||
                itR->second.size() != 4) {
                continue;
            }

            // 保存同一个 Tag 的四个角点
            for (int j = 0; j < 4; ++j) {

                object.push_back(points3D[j]);

                pointsL.push_back(itL->second[j]);

                pointsR.push_back(itR->second[j]);
            }

            ++commonTags;
        }

        // 第一版要求至少两个共同 Tag
        if (commonTags < 2) {
            printf("%s SKIP (common tags: %d)\n",
                   pair.stem.c_str(),
                   commonTags);
            continue;
        }

        objPoints.push_back(object);
        imgPointsL.push_back(pointsL);
        imgPointsR.push_back(pointsR);

        acceptedPairs.push_back(pair);

        printf("%s OK: %d tags, %zu corners\n",
               pair.stem.c_str(),
               commonTags,
               object.size());
    }

    // 7. 检查有效图像对数量
    printf("\nValid stereo pairs: %zu / %zu\n",
           objPoints.size(),
           pairs.size());

    if (objPoints.size() < 8) {
        fprintf(stderr,
                "[ERROR] Need at least 8 valid stereo pairs\n");
        return 1;
    }

    // 8. 求解双目 R、T
    cv::Mat R, T, E, F;

    double rms = 0.0;

    try {
        rms = cv::stereoCalibrate(
            objPoints,
            imgPointsL,
            imgPointsR,
            K1,
            D1,
            K2,
            D2,
            imgSize,
            R,
            T,
            E,
            F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(
                cv::TermCriteria::COUNT |
                cv::TermCriteria::EPS,
                100,
                1e-6
            )
        );
    }
    catch (const cv::Exception& e) {
        fprintf(stderr,
                "[ERROR] stereoCalibrate failed: %s\n",
                e.what());
        return 1;
    }

    const double baseline = cv::norm(T);

    printf("\nStereo RMS: %.4f px\n", rms);
    printf("Baseline: %.3f mm\n", baseline);

    // 9. 检查外参求解结果
    if (!std::isfinite(rms) ||
        !std::isfinite(baseline) ||
        baseline <= 0.0 ||
        !cv::checkRange(R) ||
        !cv::checkRange(T) ||
        rms > 1.0) {

        fprintf(stderr,
                "[ERROR] Invalid stereo calibration "
                "or RMS > 1.0 px\n");
        return 1;
    }

    // 10. 计算双目校正矩阵
    cv::Mat R1, R2, P1, P2, Q;

    cv::Rect roiL, roiR;

    try {
        cv::stereoRectify(
            K1, D1,
            K2, D2,
            imgSize,
            R, T,
            R1, R2,
            P1, P2,
            Q,
            cv::CALIB_ZERO_DISPARITY,
            0.0,
            cv::Size(),
            &roiL,
            &roiR
        );
    }
    catch (const cv::Exception& e) {
        fprintf(stderr,
                "[ERROR] stereoRectify failed: %s\n",
                e.what());
        return 1;
    }

    if (roiL.empty() || roiR.empty() ||
        !cv::checkRange(P1) ||
        !cv::checkRange(P2) ||
        !cv::checkRange(Q) ||
        P1.at<double>(0, 0) <= 0.0) {

        fprintf(stderr,
                "[ERROR] Invalid rectification result\n");
        return 1;
    }

    // 11. 准备输出路径
    const fs::path outputPath(args.output);
    const fs::path parent = outputPath.parent_path();

    std::error_code error;

    if (!parent.empty()) {
        fs::create_directories(parent, error);

        if (error || !fs::is_directory(parent)) {
            fprintf(stderr,
                    "[ERROR] Cannot create output directory\n");
            return 1;
        }
    }

    // 使用与原项目相同的 .partial 临时文件机制
    const fs::path temporaryPath =
        outputPath.parent_path() /
        (outputPath.stem().string() +
         ".partial" +
         outputPath.extension().string());

    error.clear();

    if (fs::exists(temporaryPath, error) || error) {
        fprintf(stderr,
                "[ERROR] Temporary output file exists "
                "or cannot be inspected\n");
        return 1;
    }

    // 12. 保存完整双目标定结果
    try {
        cv::FileStorage storage(
            temporaryPath.string(),
            cv::FileStorage::WRITE
        );

        if (!storage.isOpened()) {
            fprintf(stderr,
                    "[ERROR] Cannot open output YAML\n");
            return 1;
        }

        storage << "image_width" << imgSize.width;
        storage << "image_height" << imgSize.height;

        storage << "baseline" << baseline;
        storage << "rms_error" << rms;

        storage << "camera_matrix_left" << K1;
        storage << "distortion_coefficients_left" << D1;

        storage << "camera_matrix_right" << K2;
        storage << "distortion_coefficients_right" << D2;

        storage << "rotation" << R;
        storage << "translation" << T;

        storage << "essential_matrix" << E;
        storage << "fundamental_matrix" << F;

        storage << "rectification_left" << R1;
        storage << "rectification_right" << R2;

        storage << "projection_left" << P1;
        storage << "projection_right" << P2;

        storage << "disparity_to_depth_map" << Q;

        storage << "valid_pairs"
                << static_cast<int>(acceptedPairs.size());

        storage.release();
    }
    catch (const cv::Exception& e) {
        fs::remove(temporaryPath, error);

        fprintf(stderr,
                "[ERROR] Saving YAML failed: %s\n",
                e.what());
        return 1;
    }

    // 13. 检查临时文件
    error.clear();

    const bool regular =
        fs::is_regular_file(temporaryPath, error);

    const auto bytes =
        regular && !error
        ? fs::file_size(temporaryPath, error)
        : 0;

    if (error || !regular || bytes == 0) {
        fprintf(stderr,
                "[ERROR] Temporary YAML invalid\n");

        error.clear();
        fs::remove(temporaryPath, error);
        return 1;
    }

    // 14. 安全发布标定文件
    std::string publishError;

    if (!volleyball::tools::hardware::publishFileNoReplace(
            temporaryPath,
            outputPath,
            publishError)) {

        error.clear();
        fs::remove(temporaryPath, error);

        fprintf(stderr,
                "[ERROR] Publishing YAML failed: %s\n",
                publishError.c_str());
        return 1;
    }

    printf("\nSaved: %s\n", args.output.c_str());

    return 0;
}