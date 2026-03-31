/**
 * @file stereo_calibrate.cpp
 * @brief Stereo calibration tool using chessboard pattern
 *
 * Detects chessboard corners in left/right image pairs, performs
 * monocular + stereo calibration with outlier rejection, and outputs
 * an OpenCV YAML file compatible with StereoCalibration::load().
 *
 * Usage:
 *   ./stereo_calibrate -s 30.0                       # square size in mm
 *   ./stereo_calibrate -s 25.0 -d images/ -o calib.yaml
 */

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ======================== Defaults ========================
static constexpr int BOARD_W = 9;
static constexpr int BOARD_H = 6;
static const char* DEFAULT_DIR = "calibration_images";
static const char* DEFAULT_OUT = "stereo_calib.yaml";

// ======================== Args ========================
struct Args {
    float       square_size = 0.0f;
    std::string images_dir  = DEFAULT_DIR;
    std::string output      = DEFAULT_OUT;
    int         board_w     = BOARD_W;
    int         board_h     = BOARD_H;
    bool        no_vis      = false;
};

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--square-size") && i+1<argc)
            a.square_size = std::stof(argv[++i]);
        else if ((arg == "-d" || arg == "--images-dir") && i+1<argc)
            a.images_dir = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i+1<argc)
            a.output = argv[++i];
        else if (arg == "--board-w" && i+1<argc)
            a.board_w = std::atoi(argv[++i]);
        else if (arg == "--board-h" && i+1<argc)
            a.board_h = std::atoi(argv[++i]);
        else if (arg == "--no-vis")
            a.no_vis = true;
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -s SQUARE_SIZE [options]\n"
                   "  -s, --square-size MM   Square size in mm (required)\n"
                   "  -d, --images-dir DIR   Image directory (default: calibration_images)\n"
                   "  -o, --output FILE      Output YAML (default: stereo_calib.yaml)\n"
                   "  --board-w N            Inner corners width (default: 9)\n"
                   "  --board-h N            Inner corners height (default: 6)\n"
                   "  --no-vis               Skip visualization\n"
                   "  -h, --help             Show help\n",
                   argv[0]);
            std::exit(0);
        }
    }
    if (a.square_size <= 0.0f) {
        fprintf(stderr, "[ERROR] Square size required: -s <mm>\n");
        std::exit(1);
    }
    return a;
}

// ======================== Helpers ========================

static std::vector<std::string> globImages(const fs::path& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir)) return files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static cv::Mat toGray(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return {};
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 1 && img.type() == CV_8UC1) {
        // Try BayerRG decode
        cv::Mat bgr;
        try {
            cv::cvtColor(img, bgr, cv::COLOR_BayerRG2BGR);
            cv::cvtColor(bgr, img, cv::COLOR_BGR2GRAY);
        } catch (...) {
            // Already grayscale
        }
    }
    return img;
}

// Chessboard detection flags (strict to relaxed)
static const std::vector<int> CB_FLAGS_LIST = {
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FILTER_QUADS,
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE,
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK,
    0,
};

static bool findCorners(const cv::Mat& gray, cv::Size boardSize,
                        std::vector<cv::Point2f>& corners) {
    static const cv::TermCriteria subpixCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    for (int flags : CB_FLAGS_LIST) {
        bool found = cv::findChessboardCorners(gray, boardSize, corners, flags);
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11,11),
                             cv::Size(-1,-1), subpixCrit);
            return true;
        }
    }
    return false;
}

static std::vector<cv::Point3f> makeObjectPoints(cv::Size boardSize, float squareSize) {
    std::vector<cv::Point3f> pts;
    pts.reserve(boardSize.width * boardSize.height);
    for (int r = 0; r < boardSize.height; ++r)
        for (int c = 0; c < boardSize.width; ++c)
            pts.emplace_back(c * squareSize, r * squareSize, 0.0f);
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

// ======================== Main ========================

int main(int argc, char* argv[]) {
    Args args = parseArgs(argc, argv);
    const cv::Size boardSize(args.board_w, args.board_h);
    const auto objTemplate = makeObjectPoints(boardSize, args.square_size);

    // ---- Collect corners ----
    auto leftFiles  = globImages(fs::path(args.images_dir) / "left");
    auto rightFiles = globImages(fs::path(args.images_dir) / "right");

    if (leftFiles.empty() || rightFiles.empty()) {
        fprintf(stderr, "[ERROR] No images in %s/left/ or right/\n", args.images_dir.c_str());
        return 1;
    }

    size_t n = std::min(leftFiles.size(), rightFiles.size());
    if (leftFiles.size() != rightFiles.size())
        printf("[WARN] L/R count mismatch: L=%zu R=%zu\n", leftFiles.size(), rightFiles.size());

    std::vector<std::vector<cv::Point3f>> objPoints;
    std::vector<std::vector<cv::Point2f>> imgPointsL, imgPointsR;
    cv::Size imgSize;

    printf("\n%zu image pairs, detecting corners...\n\n", n);

    for (size_t i = 0; i < n; ++i) {
        cv::Mat gL = toGray(leftFiles[i]);
        cv::Mat gR = toGray(rightFiles[i]);
        if (gL.empty() || gR.empty()) {
            printf("[%3zu/%zu] SKIP (cannot read)\n", i+1, n);
            continue;
        }
        if (imgSize.width == 0)
            imgSize = gL.size();

        std::vector<cv::Point2f> cL, cR;
        bool fL = findCorners(gL, boardSize, cL);
        bool fR = findCorners(gR, boardSize, cR);

        if (!fL || !fR) {
            printf("[%3zu/%zu] %s -- %s not detected\n", i+1, n,
                   fs::path(leftFiles[i]).stem().string().c_str(),
                   !fL ? "LEFT" : "RIGHT");
            continue;
        }

        objPoints.push_back(objTemplate);
        imgPointsL.push_back(cL);
        imgPointsR.push_back(cR);
        printf("[%3zu/%zu] %s OK\n", i+1, n,
               fs::path(leftFiles[i]).stem().string().c_str());
    }

    printf("\nDetected: %zu / %zu pairs\n", objPoints.size(), n);
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

    double rmsL = cv::calibrateCamera(objPoints, imgPointsL, imgSize,
                                      K1, D1, rvecsL, tvecsL);
    printf("\n  [LEFT]  RMS=%.4f  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
           rmsL, K1.at<double>(0,0), K1.at<double>(1,1),
           K1.at<double>(0,2), K1.at<double>(1,2));

    double rmsR = cv::calibrateCamera(objPoints, imgPointsR, imgSize,
                                      K2, D2, rvecsR, tvecsR);
    printf("  [RIGHT] RMS=%.4f  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
           rmsR, K2.at<double>(0,0), K2.at<double>(1,1),
           K2.at<double>(0,2), K2.at<double>(1,2));

    // ---- Outlier rejection ----
    auto errL = perImageErrors(objPoints, imgPointsL, K1, D1, rvecsL, tvecsL);
    auto errR = perImageErrors(objPoints, imgPointsR, K2, D2, rvecsR, tvecsR);

    std::vector<double> maxErr(objPoints.size());
    for (size_t i = 0; i < maxErr.size(); ++i)
        maxErr[i] = std::max(errL[i], errR[i]);

    double mean = std::accumulate(maxErr.begin(), maxErr.end(), 0.0) / maxErr.size();
    double sq_sum = 0.0;
    for (double e : maxErr) sq_sum += (e - mean) * (e - mean);
    double stddev = std::sqrt(sq_sum / maxErr.size());
    double threshold = mean + 2.0 * stddev;

    std::vector<size_t> keep;
    for (size_t i = 0; i < maxErr.size(); ++i)
        if (maxErr[i] < threshold) keep.push_back(i);

    int nRemoved = static_cast<int>(maxErr.size()) - static_cast<int>(keep.size());
    if (nRemoved > 0 && keep.size() >= 5) {
        printf("\nOutlier rejection: removed %d pairs (threshold=%.3f), keeping %zu\n",
               nRemoved, threshold, keep.size());

        std::vector<std::vector<cv::Point3f>> filtObj;
        std::vector<std::vector<cv::Point2f>> filtL, filtR;
        for (size_t idx : keep) {
            filtObj.push_back(objPoints[idx]);
            filtL.push_back(imgPointsL[idx]);
            filtR.push_back(imgPointsR[idx]);
        }
        objPoints  = filtObj;
        imgPointsL = filtL;
        imgPointsR = filtR;

        // Re-calibrate monocular
        rmsL = cv::calibrateCamera(objPoints, imgPointsL, imgSize,
                                   K1, D1, rvecsL, tvecsL);
        rmsR = cv::calibrateCamera(objPoints, imgPointsR, imgSize,
                                   K2, D2, rvecsR, tvecsR);
        printf("  Re-calibrated: L RMS=%.4f, R RMS=%.4f\n", rmsL, rmsR);
    }

    // ---- Stereo calibration ----
    printf("\n==================================================\n");
    printf("Stereo Calibration\n");
    printf("==================================================\n");

    cv::Mat R, T, E, F;
    cv::TermCriteria stereoCrit(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                                100, 1e-6);

    double rmsS = cv::stereoCalibrate(
        objPoints, imgPointsL, imgPointsR,
        K1, D1, K2, D2, imgSize,
        R, T, E, F,
        cv::CALIB_USE_INTRINSIC_GUESS,
        stereoCrit);

    double baseline = cv::norm(T);

    printf("\n  Stereo RMS = %.4f px\n", rmsS);
    if (rmsS > 1.0)
        printf("  [!] RMS > 1.0, consider recapturing or checking board params\n");
    printf("  Baseline = %.2f mm (%.2f cm)\n", baseline, baseline / 10.0);

    // ---- Stereo rectification ----
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roiL, roiR;
    cv::stereoRectify(K1, D1, K2, D2, imgSize, R, T,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0,
                      cv::Size(), &roiL, &roiR);

    printf("  ROI Left:  [%d, %d, %d, %d]\n", roiL.x, roiL.y, roiL.width, roiL.height);
    printf("  ROI Right: [%d, %d, %d, %d]\n", roiR.x, roiR.y, roiR.width, roiR.height);

    // ---- Save calibration ----
    // Output format compatible with StereoCalibration::load()
    {
        cv::FileStorage fs(args.output, cv::FileStorage::WRITE);
        fs << "image_width"  << imgSize.width;
        fs << "image_height" << imgSize.height;
        fs << "baseline"     << baseline;
        fs << "rms_error"    << rmsS;

        fs << "camera_matrix_left"              << K1;
        fs << "distortion_coefficients_left"    << D1;
        fs << "rectification_left"              << R1;
        fs << "projection_left"                 << P1;

        fs << "camera_matrix_right"             << K2;
        fs << "distortion_coefficients_right"   << D2;
        fs << "rectification_right"             << R2;
        fs << "projection_right"                << P2;

        fs << "rotation"                << R;
        fs << "translation"             << T;
        fs << "essential_matrix"        << E;
        fs << "fundamental_matrix"      << F;
        fs << "disparity_to_depth_map"  << Q;
        fs.release();

        printf("\nSaved: %s\n", args.output.c_str());
    }

    // ---- Depth accuracy report ----
    {
        double focal = P1.at<double>(0, 0);
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
    if (!args.no_vis && !leftFiles.empty()) {
        printf("\nRectification preview (press any key / ESC to skip)\n");

        cv::Mat mapLx, mapLy, mapRx, mapRy;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, mapLx, mapLy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, mapRx, mapRy);

        int nSample = std::min(3, (int)std::min(leftFiles.size(), rightFiles.size()));
        for (int i = 0; i < nSample; ++i) {
            cv::Mat imgL = cv::imread(leftFiles[i]);
            cv::Mat imgR = cv::imread(rightFiles[i]);
            if (imgL.empty() || imgR.empty()) continue;

            // Handle single-channel BayerRG
            if (imgL.channels() == 1) cv::cvtColor(imgL, imgL, cv::COLOR_BayerRG2BGR);
            if (imgR.channels() == 1) cv::cvtColor(imgR, imgR, cv::COLOR_BayerRG2BGR);

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

    printf("\nCalibration complete!\n");
    printf("Copy to pipeline config directory:\n");
    printf("  cp %s calibration/stereo_calib.yaml\n", args.output.c_str());

    return 0;
}
