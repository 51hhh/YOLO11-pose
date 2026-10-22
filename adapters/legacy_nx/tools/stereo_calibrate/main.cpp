/**
 * @file main.cpp
 * @brief Stereo calibration tool using chessboard pattern
 *
 * Detects chessboard corners in left/right image pairs, performs
 * monocular + stereo calibration, and outputs an OpenCV YAML file
 * compatible with StereoCalibration::load().
 *
 * Usage:
 *   ./stereo_calibrate -s 26.0 --board-w 5 --board-h 8
 *   ./stereo_calibrate -s 26.0 --board-w 5 --board-h 8 -d images/ -o calib.yaml
 */

#include "args.h"
#include "images.h"
#include "../common/atomic_file.h"
#include "../common/parallel_tasks.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <cmath>
#include <exception>

namespace fs = std::filesystem;

static std::vector<cv::Point3f> makeObjectPoints(cv::Size boardSize, float squareSize) {
    std::vector<cv::Point3f> pts;
    const std::size_t corner_count =
        static_cast<std::size_t>(boardSize.width) *
        static_cast<std::size_t>(boardSize.height);
    pts.reserve(corner_count);
    for (int r = 0; r < boardSize.height; ++r)
        for (int c = 0; c < boardSize.width; ++c)
            pts.emplace_back(
                static_cast<float>(c) * squareSize,
                static_cast<float>(r) * squareSize,
                0.0f);
    return pts;
}

// ======================== Per-image reprojection error ========================

static std::vector<double> perImageErrors(
    const std::vector<std::vector<cv::Point3f>>& objPts,
    const std::vector<std::vector<cv::Point2f>>& imgPts,
    const cv::Mat& K, const cv::Mat& D,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs)
{
    std::vector<double> errs(objPts.size());
    for (size_t i = 0; i < objPts.size(); ++i) {
        std::vector<cv::Point2f> proj;
        cv::projectPoints(objPts[i], rvecs[i], tvecs[i], K, D, proj);
        errs[i] = cv::norm(imgPts[i], proj, cv::NORM_L2) / std::sqrt((double)proj.size());
    }
    return errs;
}

static void printWorstPerImageErrors(const std::vector<std::string>& stems,
                                     const std::vector<double>& errL,
                                     const std::vector<double>& errR,
                                     size_t limit) {
    if (stems.empty()) return;

    std::vector<size_t> order(stems.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::max(errL[a], errR[a]) > std::max(errL[b], errR[b]);
    });

    const size_t count = std::min(limit, order.size());
    printf("\nWorst per-image reprojection errors (reported, not removed):\n");
    for (size_t rank = 0; rank < count; ++rank) {
        const size_t i = order[rank];
        printf("  %zu. %s  L=%.3f px  R=%.3f px  max=%.3f px\n",
               rank + 1, stems[i].c_str(), errL[i], errR[i],
               std::max(errL[i], errR[i]));
    }
}

// ======================== Main ========================

int main(int argc, char* argv[]) {
    StereoCalibrateArgs args;
    if (!parseStereoCalibrateArgs(argc, argv, args)) return 1;
    if (args.help_requested) return 0;
    if (!validateStereoCalibrateArgs(args)) return 1;
    if (!validateStereoCalibrateOutputPath(args.output)) return 1;
    const cv::Size boardSize(args.board_w, args.board_h);
    const auto objTemplate = makeObjectPoints(boardSize, args.square_size);

    // ---- Collect corners ----
    auto leftFiles  = globImages(fs::path(args.images_dir) / "left");
    auto rightFiles = globImages(fs::path(args.images_dir) / "right");

    if (leftFiles.empty() || rightFiles.empty()) {
        fprintf(stderr, "[ERROR] No images in %s/left/ or right/\n", args.images_dir.c_str());
        return 1;
    }

    auto pairs = pairImagesByStem(leftFiles, rightFiles);
    if (leftFiles.size() != rightFiles.size() || pairs.size() != leftFiles.size()) {
        printf("[WARN] L/R count or filename mismatch: L=%zu R=%zu paired=%zu\n",
               leftFiles.size(), rightFiles.size(), pairs.size());
    }
    if (pairs.empty()) {
        fprintf(stderr, "[ERROR] No left/right image pairs with matching names\n");
        return 1;
    }

    std::vector<std::vector<cv::Point3f>> objPoints;
    std::vector<std::vector<cv::Point2f>> imgPointsL, imgPointsR;
    std::vector<std::string> acceptedStems;
    cv::Size imgSize;

    const size_t n = pairs.size();
    const int default_jobs = args.use_gpu_preprocess ? 1 : defaultJobCount();
    const int jobs = std::clamp(args.jobs > 0 ? args.jobs : default_jobs,
                                1, static_cast<int>(n));
    printf("\n%zu image pairs, detecting corners with %d worker(s)...\n\n", n, jobs);
    if (args.use_gpu_preprocess && args.jobs <= 0) {
        printf("[INFO] --gpu-preprocess enabled: defaulting to one detection worker\n");
    }

    std::vector<DetectionResult> results(n);
    std::string parallel_error;
    if (!legacy_nx_tools::runIndexedTasks(
            n,
            static_cast<std::size_t>(jobs),
            [&](std::size_t index) {
                results[index] = detectPairCorners(
                    pairs[index],
                    boardSize,
                    args.use_gpu_preprocess,
                    args.use_sb,
                    args.use_exhaustive);
            },
            parallel_error)) {
        fprintf(stderr, "[ERROR] Corner detection failed: %s\n",
                parallel_error.c_str());
        return 1;
    }
    printf("Detecting corners: %zu/%zu\n\n", n, n);

    for (const auto& result : results) {
        if (result.read_ok) {
            imgSize = result.left_size;
            break;
        }
    }
    if (imgSize.width == 0 || imgSize.height == 0) {
        fprintf(stderr, "[ERROR] Could not read any paired images\n");
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        const auto& result = results[i];
        if (!result.read_ok) {
            printf("[%3zu/%zu] %s SKIP (cannot read)\n",
                   i+1, n, pairs[i].stem.c_str());
            continue;
        }
        if (result.left_size != imgSize || result.right_size != imgSize) {
            printf("[%3zu/%zu] %s SKIP (image size mismatch L=%dx%d R=%dx%d expected=%dx%d)\n",
                   i+1, n, pairs[i].stem.c_str(),
                   result.left_size.width, result.left_size.height,
                   result.right_size.width, result.right_size.height,
                   imgSize.width, imgSize.height);
            continue;
        }
        if (!result.found_left || !result.found_right) {
            printf("[%3zu/%zu] %s -- pair not detected with same detector mode\n",
                   i+1, n, pairs[i].stem.c_str());
            continue;
        }

        objPoints.push_back(objTemplate);
        imgPointsL.push_back(result.corners_left);
        imgPointsR.push_back(result.corners_right);
        acceptedStems.push_back(pairs[i].stem);
        printf("[%3zu/%zu] %s OK\n", i+1, n, pairs[i].stem.c_str());
    }

    printf("\nDetected: %zu / %zu pairs\n", objPoints.size(), n);
    if (objPoints.size() != n) {
        printf("[WARN] %zu input pairs were not usable. For formal calibration, recapture them with capture-time quality checks instead of relying on post-processing selection.\n",
               n - objPoints.size());
    }
    if (objPoints.size() < 20) {
        printf("[WARN] Only %zu valid pairs. Formal long-baseline calibration should use at least 20 well-distributed pairs.\n",
               objPoints.size());
    }
    if (objPoints.size() < 5) {
        fprintf(stderr, "[ERROR] Need at least 5 valid pairs, got %zu\n", objPoints.size());
        return 1;
    }

    // ---- Monocular calibration ----
    printf("\n==================================================\n");
    printf("Monocular Calibration\n");
    printf("==================================================\n");

    cv::Mat K1, D1, K2, D2;
    std::vector<cv::Mat> rvecsL, tvecsL, rvecsR, tvecsR;

    double rmsL = 0.0;
    double rmsR = 0.0;
    if (!legacy_nx_tools::runIndexedTasks(
            2,
            2,
            [&](std::size_t index) {
                if (index == 0) {
                    rmsL = cv::calibrateCamera(
                        objPoints, imgPointsL, imgSize,
                        K1, D1, rvecsL, tvecsL);
                } else {
                    rmsR = cv::calibrateCamera(
                        objPoints, imgPointsR, imgSize,
                        K2, D2, rvecsR, tvecsR);
                }
            },
            parallel_error)) {
        fprintf(stderr, "[ERROR] Monocular calibration failed: %s\n",
                parallel_error.c_str());
        return 1;
    }
    printf("\n  [LEFT]  RMS=%.4f  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
           rmsL, K1.at<double>(0,0), K1.at<double>(1,1),
           K1.at<double>(0,2), K1.at<double>(1,2));

    printf("  [RIGHT] RMS=%.4f  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
           rmsR, K2.at<double>(0,0), K2.at<double>(1,1),
           K2.at<double>(0,2), K2.at<double>(1,2));

    // ---- Per-image diagnostics ----
    auto errL = perImageErrors(objPoints, imgPointsL, K1, D1, rvecsL, tvecsL);
    auto errR = perImageErrors(objPoints, imgPointsR, K2, D2, rvecsR, tvecsR);
    printWorstPerImageErrors(acceptedStems, errL, errR, 8);
    if (rmsL > 0.5 || rmsR > 0.5) {
        printf("[WARN] Monocular RMS is high for a controlled calibration set. Check focus, exposure, board coverage, corner ordering, and board rigidity before trusting stereo RMS.\n");
    }

    // ---- Stereo calibration ----
    printf("\n==================================================\n");
    printf("Stereo Calibration\n");
    printf("==================================================\n");

    cv::Mat R, T, E, F;
    cv::TermCriteria stereoCrit(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                                100, 1e-6);

    int stereo_flags = args.fix_intrinsics
        ? cv::CALIB_FIX_INTRINSIC
        : cv::CALIB_USE_INTRINSIC_GUESS;
    double rmsS = 0.0;
    double baseline = 0.0;
    try {
        rmsS = cv::stereoCalibrate(
            objPoints, imgPointsL, imgPointsR,
            K1, D1, K2, D2, imgSize,
            R, T, E, F,
            stereo_flags,
            stereoCrit);
        baseline = cv::norm(T);
    } catch (const cv::Exception& exception) {
        fprintf(stderr, "[ERROR] Stereo calibration failed: %s\n",
                exception.what());
        return 1;
    }

    printf("\n  Stereo RMS = %.4f px\n", rmsS);
    if (rmsS > 1.0)
        printf("  [!] RMS > 1.0, consider recapturing or checking board params\n");
    printf("  Baseline = %.2f mm (%.2f cm)\n", baseline, baseline / 10.0);

    // ---- Stereo rectification ----
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roiL, roiR;
    try {
        cv::stereoRectify(K1, D1, K2, D2, imgSize, R, T,
                          R1, R2, P1, P2, Q,
                          cv::CALIB_ZERO_DISPARITY, 0.0,
                          cv::Size(), &roiL, &roiR);
    } catch (const cv::Exception& exception) {
        fprintf(stderr, "[ERROR] Stereo rectification failed: %s\n",
                exception.what());
        return 1;
    }

    printf("  ROI Left:  [%d, %d, %d, %d]\n", roiL.x, roiL.y, roiL.width, roiL.height);
    printf("  ROI Right: [%d, %d, %d, %d]\n", roiR.x, roiR.y, roiR.width, roiR.height);

    const double focal = P1.at<double>(0, 0);
    if (rmsS > 1.0) {
        fprintf(stderr, "[ERROR] Stereo RMS %.4f px > 1.0 px; refusing to save calibration\n",
                rmsS);
        return 1;
    }
    if (roiL.empty() || roiR.empty()) {
        fprintf(stderr, "[ERROR] Rectification ROI is empty; refusing to save calibration\n");
        return 1;
    }
    if (!std::isfinite(focal) || focal <= 0.0 ||
        !std::isfinite(baseline) || baseline <= 0.0) {
        fprintf(stderr, "[ERROR] Invalid focal/baseline; refusing to save calibration\n");
        return 1;
    }

    // ---- Save calibration ----
    // Long-running calibration writes a peer temporary file first. The final
    // path appears only after the complete file closes and passes basic checks.
    {
        if (!validateStereoCalibrateOutputPath(args.output)) return 1;
        const fs::path output_path(args.output);
        const fs::path parent = output_path.parent_path();
        if (!parent.empty()) {
            std::error_code error;
            fs::create_directories(parent, error);
            if (error || !fs::is_directory(parent)) {
                fprintf(stderr, "[ERROR] Failed to create output directory %s: %s\n",
                        parent.string().c_str(),
                        error ? error.message().c_str()
                              : "path exists but is not a directory");
                return 1;
            }
        }
        const fs::path temporary_path = output_path.parent_path() /
            (output_path.stem().string() + ".partial" +
             output_path.extension().string());
        std::error_code error;
        const bool temporary_exists = fs::exists(temporary_path, error);
        if (error || temporary_exists) {
            fprintf(stderr,
                    "[ERROR] Refusing stale calibration temporary file %s%s%s\n",
                    temporary_path.string().c_str(),
                    error ? ": " : "",
                    error ? error.message().c_str() : "");
            return 1;
        }

        try {
            cv::FileStorage storage(
                temporary_path.string(), cv::FileStorage::WRITE);
            if (!storage.isOpened()) {
                fprintf(stderr,
                        "[ERROR] Failed to open temporary calibration file: %s\n",
                        temporary_path.string().c_str());
                fs::remove(temporary_path, error);
                return 1;
            }
            storage << "image_width"  << imgSize.width;
            storage << "image_height" << imgSize.height;
            storage << "baseline"     << baseline;
            storage << "rms_error"    << rmsS;

            storage << "camera_matrix_left"           << K1;
            storage << "distortion_coefficients_left" << D1;
            storage << "rectification_left"           << R1;
            storage << "projection_left"              << P1;

            storage << "camera_matrix_right"           << K2;
            storage << "distortion_coefficients_right" << D2;
            storage << "rectification_right"           << R2;
            storage << "projection_right"              << P2;

            storage << "rotation"               << R;
            storage << "translation"            << T;
            storage << "essential_matrix"       << E;
            storage << "fundamental_matrix"     << F;
            storage << "disparity_to_depth_map" << Q;
            storage.release();
        } catch (const cv::Exception& exception) {
            fs::remove(temporary_path, error);
            fprintf(stderr, "[ERROR] Failed to serialize calibration: %s\n",
                    exception.what());
            return 1;
        } catch (const std::exception& exception) {
            fs::remove(temporary_path, error);
            fprintf(stderr, "[ERROR] Failed to serialize calibration: %s\n",
                    exception.what());
            return 1;
        }

        error.clear();
        const bool regular = fs::is_regular_file(temporary_path, error);
        const auto bytes = regular && !error
            ? fs::file_size(temporary_path, error) : 0;
        if (error || !regular || bytes == 0) {
            fprintf(stderr,
                    "[ERROR] Temporary calibration is missing or empty: %s\n",
                    temporary_path.string().c_str());
            error.clear();
            fs::remove(temporary_path, error);
            return 1;
        }

        std::string publish_error;
        if (!legacy_nx_tools::publishFileNoReplace(
                temporary_path, output_path, publish_error)) {
            error.clear();
            fs::remove(temporary_path, error);
            fprintf(stderr, "[ERROR] Failed to publish calibration: %s\n",
                    publish_error.c_str());
            return 1;
        }
        printf("\nSaved: %s\n", args.output.c_str());
    }

    // ---- Depth accuracy report ----
    {
        printf("\nBaseline = %.2f mm, Focal = %.2f px\n", baseline, focal);
        printf("Depth accuracy (disparity precision 0.5 px):\n");
        for (double d_mm : {3000.0, 5000.0, 9000.0, 15000.0}) {
            double disp = baseline * focal / d_mm;
            double delta_z = (d_mm * d_mm) / (baseline * focal) * 0.5;
            printf("  %.0fm: disparity=%.2fpx, error=+/-%.1fmm (+/-%.2fcm)\n",
                   d_mm / 1000.0, disp, delta_z, delta_z / 10.0);
        }
    }

    // ---- Visualization ----
    if (!args.no_vis && !pairs.empty()) {
        printf("\nRectification preview (press any key / ESC to skip)\n");

        cv::Mat mapLx, mapLy, mapRx, mapRy;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, mapLx, mapLy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, mapRx, mapRy);

        const std::size_t sample_count = std::min<std::size_t>(3, pairs.size());
        for (std::size_t i = 0; i < sample_count; ++i) {
            cv::Mat imgL = cv::imread(pairs[i].left, cv::IMREAD_UNCHANGED);
            cv::Mat imgR = cv::imread(pairs[i].right, cv::IMREAD_UNCHANGED);
            if (imgL.empty() || imgR.empty()) continue;

            // 海康 BayerRG8 sensor → OpenCV BayerBG convention
            if (imgL.channels() == 1) cv::cvtColor(imgL, imgL, cv::COLOR_BayerBG2BGR);
            if (imgR.channels() == 1) cv::cvtColor(imgR, imgR, cv::COLOR_BayerBG2BGR);

            cv::Mat rectL, rectR;
            cv::remap(imgL, rectL, mapLx, mapLy, cv::INTER_LINEAR);
            cv::remap(imgR, rectR, mapRx, mapRy, cv::INTER_LINEAR);

            for (int y = 0; y < rectL.rows; y += 30) {
                cv::line(rectL, cv::Point(0,y), cv::Point(rectL.cols,y), cv::Scalar(0,255,0), 1);
                cv::line(rectR, cv::Point(0,y), cv::Point(rectR.cols,y), cv::Scalar(0,255,0), 1);
            }

            cv::Mat combined;
            cv::hconcat(rectL, rectR, combined);
            if (combined.cols > 1920) {
                double s = 1920.0 / combined.cols;
                cv::resize(combined, combined, cv::Size(), s, s);
            }
            cv::imshow("Rectification (green lines should be horizontal)", combined);
            if (cv::waitKey(0) == 27) break;
        }
        cv::destroyAllWindows();
    }

    printf("\nCalibration candidate written to: %s\n", args.output.c_str());
    printf("Validate known distances and review it before replacing ");
    printf("configs/calibration/stereo_calib.yaml.\n");

    return 0;
}
