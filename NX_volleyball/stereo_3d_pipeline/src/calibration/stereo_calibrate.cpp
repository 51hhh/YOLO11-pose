/**
 * @file stereo_calibrate.cpp
 * @brief Stereo calibration tool using chessboard pattern
 *
 * Detects chessboard corners in left/right image pairs, performs
 * monocular + stereo calibration, and outputs an OpenCV YAML file
 * compatible with StereoCalibration::load().
 *
 * Usage:
 *   ./stereo_calibrate -s 30.0                       # square size in mm
 *   ./stereo_calibrate -s 25.0 -d images/ -o calib.yaml
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <cmath>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <future>
#include <limits>
#include <map>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

// ======================== Defaults ========================
static constexpr int BOARD_W = 5;
static constexpr int BOARD_H = 8;
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
    bool        use_gpu_preprocess = false;
    bool        fix_intrinsics = true;
    bool        use_sb = false;
    bool        use_exhaustive = false;
    int         jobs        = 0;
};

static bool parseIntValue(const char* text, int& out) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

static bool parseFloatValue(const char* text, float& out) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

static const char* requireValue(int& i, int argc, char* argv[], const std::string& arg) {
    if (i + 1 >= argc) {
        fprintf(stderr, "[ERROR] %s requires a value\n", arg.c_str());
        std::exit(1);
    }
    return argv[++i];
}

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--square-size") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseFloatValue(value, a.square_size)) {
                fprintf(stderr, "[ERROR] Invalid square size: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "-d" || arg == "--images-dir") {
            a.images_dir = requireValue(i, argc, argv, arg);
        } else if (arg == "-o" || arg == "--output") {
            a.output = requireValue(i, argc, argv, arg);
        } else if (arg == "--board-w") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.board_w)) {
                fprintf(stderr, "[ERROR] Invalid --board-w value: %s\n", value);
                std::exit(1);
            }
        } else if (arg == "--board-h") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.board_h)) {
                fprintf(stderr, "[ERROR] Invalid --board-h value: %s\n", value);
                std::exit(1);
            }
        }
        else if (arg == "--no-vis")
            a.no_vis = true;
        else if (arg == "--gpu-preprocess")
            a.use_gpu_preprocess = true;
        else if (arg == "--optimize-intrinsics")
            a.fix_intrinsics = false;
        else if (arg == "--sb")
            a.use_sb = true;
        else if (arg == "--exhaustive")
            a.use_exhaustive = true;
        else if (arg == "--jobs") {
            const char* value = requireValue(i, argc, argv, arg);
            if (!parseIntValue(value, a.jobs)) {
                fprintf(stderr, "[ERROR] Invalid --jobs value: %s\n", value);
                std::exit(1);
            }
        }
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -s SQUARE_SIZE [options]\n"
                   "  -s, --square-size MM   Square size in mm (required)\n"
                   "  -d, --images-dir DIR   Image directory (default: calibration_images)\n"
                   "  -o, --output FILE      Output YAML (default: stereo_calib.yaml)\n"
                   "  --board-w N            Inner corners width (default: 5)\n"
                   "  --board-h N            Inner corners height (default: 8)\n"
                   "  --no-vis               Skip visualization\n"
                   "  --gpu-preprocess       Use CUDA for Bayer/BGR to gray preprocessing when available\n"
                   "  --optimize-intrinsics  Let stereoCalibrate refine intrinsics (default: fix monocular intrinsics)\n"
                   "  --sb                   Enable SB chessboard fallback after classic detectors\n"
                   "  --exhaustive           Enable slow SB exhaustive fallback for difficult boards\n"
                   "  --jobs N               Parallel image-pair detection jobs (default: CPU cores; 1 with --gpu-preprocess)\n"
                   "  -h, --help             Show help\n",
                   argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "[ERROR] Unknown or incomplete argument: %s\n", arg.c_str());
            std::exit(1);
        }
    }
    if (a.square_size <= 0.0f) {
        fprintf(stderr, "[ERROR] Square size required: -s <mm>\n");
        std::exit(1);
    }
    if (a.board_w < 2 || a.board_h < 2) {
        fprintf(stderr, "[ERROR] Invalid chessboard inner corners: %dx%d\n",
                a.board_w, a.board_h);
        std::exit(1);
    }
    if (a.jobs < 0) {
        fprintf(stderr, "[ERROR] --jobs must be >= 0, got %d\n", a.jobs);
        std::exit(1);
    }
    if (a.use_exhaustive) {
        a.use_sb = true;
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

struct ImagePair {
    std::string stem;
    std::string left;
    std::string right;
};

static std::vector<ImagePair> pairImagesByStem(const std::vector<std::string>& leftFiles,
                                               const std::vector<std::string>& rightFiles) {
    std::map<std::string, std::string> leftByStem;
    std::map<std::string, std::string> rightByStem;
    for (const auto& path : leftFiles) {
        const std::string stem = fs::path(path).stem().string();
        if (!leftByStem.emplace(stem, path).second) {
            printf("[WARN] Duplicate left image stem ignored: %s\n", stem.c_str());
        }
    }
    for (const auto& path : rightFiles) {
        const std::string stem = fs::path(path).stem().string();
        if (!rightByStem.emplace(stem, path).second) {
            printf("[WARN] Duplicate right image stem ignored: %s\n", stem.c_str());
        }
    }

    std::vector<ImagePair> pairs;
    for (const auto& [stem, left] : leftByStem) {
        auto it = rightByStem.find(stem);
        if (it == rightByStem.end()) {
            printf("[WARN] Missing right image for %s, skipping\n", stem.c_str());
            continue;
        }
        pairs.push_back({stem, left, it->second});
    }
    for (const auto& [stem, right] : rightByStem) {
        if (leftByStem.find(stem) == leftByStem.end()) {
            printf("[WARN] Missing left image for %s, skipping\n", stem.c_str());
        }
    }
    return pairs;
}

static cv::Mat toGrayCPU(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return {};
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 1 && img.type() == CV_8UC1) {
        // 海康 BayerRG8 sensor → OpenCV BayerBG convention → grayscale
        try {
            cv::cvtColor(img, img, cv::COLOR_BayerBG2GRAY);
        } catch (...) {
            // Already grayscale
        }
    }
    return img;
}

static cv::Mat toGrayGPU(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) return {};
    try {
        cv::cuda::GpuMat gpu(img);
        cv::cuda::GpuMat gray_gpu;
        if (img.channels() == 3) {
            cv::cuda::cvtColor(gpu, gray_gpu, cv::COLOR_BGR2GRAY);
        } else if (img.channels() == 1 && img.type() == CV_8UC1) {
            cv::cuda::cvtColor(gpu, gray_gpu, cv::COLOR_BayerBG2GRAY);
        } else {
            return toGrayCPU(path);
        }
        cv::Mat gray;
        gray_gpu.download(gray);
        return gray;
    } catch (const cv::Exception&) {
        return toGrayCPU(path);
    }
}

static cv::Mat toGray(const std::string& path, bool use_gpu_preprocess) {
    return use_gpu_preprocess ? toGrayGPU(path) : toGrayCPU(path);
}

enum class ChessboardDetectorMode {
    CLASSIC_FAST,
    CLASSIC_FILTER_QUADS,
    CLASSIC_NORMALIZED,
    CLASSIC_PLAIN,
    SB_NORMALIZED,
    SB_EXHAUSTIVE,
};

static bool findCornersWithMode(const cv::Mat& gray, cv::Size boardSize,
                                ChessboardDetectorMode mode,
                                std::vector<cv::Point2f>& corners) {
    static const cv::TermCriteria subpixCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    corners.clear();
    auto refine_if_found = [&](bool found) {
        if (!found) return false;
        cv::cornerSubPix(gray, corners, cv::Size(11,11),
                         cv::Size(-1,-1), subpixCrit);
        return true;
    };

    switch (mode) {
    case ChessboardDetectorMode::CLASSIC_FAST:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FAST_CHECK));
    case ChessboardDetectorMode::CLASSIC_FILTER_QUADS:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_FILTER_QUADS));
    case ChessboardDetectorMode::CLASSIC_NORMALIZED:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE));
    case ChessboardDetectorMode::CLASSIC_PLAIN:
        return refine_if_found(cv::findChessboardCorners(
            gray, boardSize, corners, 0));
    case ChessboardDetectorMode::SB_NORMALIZED:
        return refine_if_found(cv::findChessboardCornersSB(
            gray, boardSize, corners, cv::CALIB_CB_NORMALIZE_IMAGE));
    case ChessboardDetectorMode::SB_EXHAUSTIVE:
        return refine_if_found(cv::findChessboardCornersSB(
            gray, boardSize, corners,
            cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE));
    }
    return false;
}

struct DetectionResult {
    bool read_ok = false;
    bool found_left = false;
    bool found_right = false;
    cv::Size left_size;
    cv::Size right_size;
    std::vector<cv::Point2f> corners_left;
    std::vector<cv::Point2f> corners_right;
};

static DetectionResult detectPairCorners(const ImagePair& pair,
                                         cv::Size boardSize,
                                         bool use_gpu_preprocess,
                                         bool use_sb,
                                         bool use_exhaustive) {
    DetectionResult result;
    cv::Mat gL = toGray(pair.left, use_gpu_preprocess);
    cv::Mat gR = toGray(pair.right, use_gpu_preprocess);
    if (gL.empty() || gR.empty()) return result;
    result.read_ok = true;
    result.left_size = gL.size();
    result.right_size = gR.size();
    // Try each detector as a pair. Accepting left/right corners from different
    // detector families risks inconsistent ordering on rotated or oblique boards.
    std::vector<ChessboardDetectorMode> modes = {
        ChessboardDetectorMode::CLASSIC_FAST,
        ChessboardDetectorMode::CLASSIC_FILTER_QUADS,
        ChessboardDetectorMode::CLASSIC_NORMALIZED,
    };
    if (use_sb) {
        modes.push_back(ChessboardDetectorMode::SB_NORMALIZED);
    }
    if (use_exhaustive) {
        modes.push_back(ChessboardDetectorMode::SB_EXHAUSTIVE);
    }
    for (ChessboardDetectorMode mode : modes) {
        std::vector<cv::Point2f> cL;
        const bool fL = findCornersWithMode(gL, boardSize, mode, cL);
        if (!fL) continue;
        std::vector<cv::Point2f> cR;
        const bool fR = findCornersWithMode(gR, boardSize, mode, cR);
        if (fL && fR) {
            result.found_left = true;
            result.found_right = true;
            result.corners_left = std::move(cL);
            result.corners_right = std::move(cR);
            return result;
        }
    }
    return result;
}

static int defaultJobCount() {
    unsigned int hw = std::thread::hardware_concurrency();
    return std::max(1, static_cast<int>(hw == 0 ? 1 : hw));
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
    std::atomic<size_t> next{0};
    std::atomic<size_t> completed{0};
    std::vector<std::thread> workers;
    workers.reserve(jobs);
    for (int j = 0; j < jobs; ++j) {
        workers.emplace_back([&]() {
            while (true) {
                const size_t i = next.fetch_add(1);
                if (i >= n) break;
                results[i] = detectPairCorners(pairs[i], boardSize,
                                               args.use_gpu_preprocess,
                                               args.use_sb,
                                               args.use_exhaustive);
                completed.fetch_add(1);
            }
        });
    }
    while (completed.load() < n) {
        printf("\rDetecting corners: %zu/%zu", completed.load(), n);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for (auto& worker : workers) worker.join();
    printf("\rDetecting corners: %zu/%zu\n\n", n, n);

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

    auto calibrate_left = std::async(std::launch::async, [&]() {
        return cv::calibrateCamera(objPoints, imgPointsL, imgSize,
                                   K1, D1, rvecsL, tvecsL);
    });
    double rmsR = cv::calibrateCamera(objPoints, imgPointsR, imgSize,
                                      K2, D2, rvecsR, tvecsR);
    double rmsL = calibrate_left.get();
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
    double rmsS = cv::stereoCalibrate(
        objPoints, imgPointsL, imgPointsR,
        K1, D1, K2, D2, imgSize,
        R, T, E, F,
        stereo_flags,
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
        const fs::path output_path(args.output);
        const fs::path parent = output_path.parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec) {
                fprintf(stderr, "[ERROR] Failed to create output directory %s: %s\n",
                        parent.string().c_str(), ec.message().c_str());
                return 1;
            }
        }
        cv::FileStorage fs(args.output, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            fprintf(stderr, "[ERROR] Failed to open output calibration file: %s\n",
                    args.output.c_str());
            return 1;
        }
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
    if (!args.no_vis && !pairs.empty()) {
        printf("\nRectification preview (press any key / ESC to skip)\n");

        cv::Mat mapLx, mapLy, mapRx, mapRy;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, mapLx, mapLy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, mapRx, mapRy);

        int nSample = std::min(3, (int)pairs.size());
        for (int i = 0; i < nSample; ++i) {
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

    printf("\nCalibration complete!\n");
    printf("Copy to pipeline config directory:\n");
    printf("  cp %s calibration/stereo_calib.yaml\n", args.output.c_str());

    return 0;
}
