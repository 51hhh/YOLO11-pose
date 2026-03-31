/**
 * @file capture_chessboard.cpp
 * @brief Chessboard image pair capture tool for stereo calibration
 *
 * Uses the same HikvisionCamera API and libgpiod PWM trigger as the main
 * pipeline. Outputs numbered PNG pairs to calibration_images/left/ and right/.
 *
 * Usage:
 *   ./capture_chessboard                          # default: PWM trigger
 *   ./capture_chessboard --free-run               # no trigger
 *   ./capture_chessboard -o /path/to/output -e 2000
 *
 * Keys:
 *   SPACE - capture current pair (both must detect chessboard)
 *   q/ESC - quit
 *   c     - clear captured images
 */

#include "../capture/hikvision_camera.h"
#include "pwm_trigger.h"

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <filesystem>

namespace fs = std::filesystem;

// ======================== Defaults ========================
static constexpr int    BOARD_W       = 9;
static constexpr int    BOARD_H       = 6;
static constexpr float  SQUARE_SIZE   = 30.0f;   // mm
static constexpr int    EXPOSURE_US   = 2000;
static constexpr double PWM_FREQ      = 10.0;
static constexpr double PWM_DUTY      = 50.0;
static const char*      DEFAULT_CHIP  = "gpiochip2";
static constexpr unsigned LINE_OFFSET = 7;

static std::atomic<bool> g_quit{false};
static void sigHandler(int) { g_quit.store(true); }

// ======================== Argument parser ========================
struct Args {
    std::string output_dir   = "calibration_images";
    int         exposure_us  = EXPOSURE_US;
    bool        free_run     = false;
    bool        no_pwm       = false;
    int         cam_left_idx = 0;
    int         cam_right_idx= 1;
    int         board_w      = BOARD_W;
    int         board_h      = BOARD_H;
    int         cam_width    = 1440;
    int         cam_height   = 1080;
};

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--free-run")          { a.free_run = true; }
        else if (arg == "--no-pwm")       { a.no_pwm = true; }
        else if (arg == "-o" && i+1<argc) { a.output_dir = argv[++i]; }
        else if (arg == "-e" && i+1<argc) { a.exposure_us = std::atoi(argv[++i]); }
        else if (arg == "--left" && i+1<argc)  { a.cam_left_idx = std::atoi(argv[++i]); }
        else if (arg == "--right" && i+1<argc) { a.cam_right_idx = std::atoi(argv[++i]); }
        else if (arg == "--board-w" && i+1<argc) { a.board_w = std::atoi(argv[++i]); }
        else if (arg == "--board-h" && i+1<argc) { a.board_h = std::atoi(argv[++i]); }
        else if (arg == "--width" && i+1<argc)   { a.cam_width = std::atoi(argv[++i]); }
        else if (arg == "--height" && i+1<argc)  { a.cam_height = std::atoi(argv[++i]); }
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [options]\n"
                   "  --free-run          Free-run mode (no trigger)\n"
                   "  --no-pwm            Disable PWM output\n"
                   "  -o DIR              Output directory (default: calibration_images)\n"
                   "  -e US               Exposure time in us (default: 2000)\n"
                   "  --left IDX          Left camera index (default: 0)\n"
                   "  --right IDX         Right camera index (default: 1)\n"
                   "  --board-w N         Chessboard inner corners width (default: 9)\n"
                   "  --board-h N         Chessboard inner corners height (default: 6)\n"
                   "  --width W           Camera image width (default: 1440)\n"
                   "  --height H          Camera image height (default: 1080)\n"
                   "  -h, --help          Show this help\n",
                   argv[0]);
            std::exit(0);
        }
    }
    return a;
}

// ======================== Main ========================
int main(int argc, char* argv[]) {
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    Args args = parseArgs(argc, argv);
    const cv::Size boardSize(args.board_w, args.board_h);

    // Create output directories
    fs::create_directories(fs::path(args.output_dir) / "left");
    fs::create_directories(fs::path(args.output_dir) / "right");

    // ---- PWM ----
    std::unique_ptr<stereo3d::PWMTrigger> pwm;
    if (!args.free_run && !args.no_pwm) {
        pwm = std::make_unique<stereo3d::PWMTrigger>(
            DEFAULT_CHIP, LINE_OFFSET, PWM_FREQ, PWM_DUTY);
        if (!pwm->start()) {
            fprintf(stderr, "[WARN] PWM start failed, continuing without PWM\n");
            pwm.reset();
        }
    }

    // ---- Camera ----
    stereo3d::HikvisionCamera camera;
    stereo3d::CameraConfig camCfg;
    camCfg.camera_index_left  = args.cam_left_idx;
    camCfg.camera_index_right = args.cam_right_idx;
    camCfg.exposure_us  = static_cast<float>(args.exposure_us);
    camCfg.gain_db      = 0.0f;
    camCfg.width        = args.cam_width;
    camCfg.height       = args.cam_height;

    if (args.free_run) {
        camCfg.use_trigger = false;
    } else {
        camCfg.use_trigger = true;
        camCfg.trigger_source = "Line0";
        camCfg.trigger_activation = "RisingEdge";
    }

    if (!camera.open(camCfg)) {
        fprintf(stderr, "[ERROR] Failed to open cameras\n");
        return 1;
    }
    if (!camera.startGrabbing()) {
        fprintf(stderr, "[ERROR] Failed to start grabbing\n");
        camera.close();
        return 1;
    }

    const int W = camera.width();
    const int H = camera.height();
    fprintf(stderr, "[INFO] Camera ready: %dx%d\n", W, H);

    // Allocate grab buffers
    std::vector<uint8_t> bufL(W * H);
    std::vector<uint8_t> bufR(W * H);

    // Chessboard detection flags
    const int cbFlags = cv::CALIB_CB_ADAPTIVE_THRESH
                      | cv::CALIB_CB_NORMALIZE_IMAGE
                      | cv::CALIB_CB_FILTER_QUADS;
    const cv::TermCriteria subpixCrit(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    int captureCount = 0;
    auto lastCapture = std::chrono::steady_clock::now() - std::chrono::seconds(2);

    printf("==================================================\n");
    printf("Stereo Chessboard Capture\n");
    printf("==================================================\n");
    printf("Board:  %dx%d inner corners\n", args.board_w, args.board_h);
    printf("Output: %s/\n", args.output_dir.c_str());
    printf("Mode:   %s\n", args.free_run ? "Free-run" : "PWM trigger");
    printf("Keys:   SPACE=capture  q/ESC=quit  c=clear\n");
    printf("==================================================\n");

    while (!g_quit.load()) {
        stereo3d::GrabResult resL, resR;
        bool ok = camera.grabFramePair(
            bufL.data(), bufR.data(), 0, 0, 1000, resL, resR);

        if (!ok) {
            printf("\rWaiting for trigger...");
            fflush(stdout);
            continue;
        }

        // BayerRG8 -> BGR for display, Gray for detection
        cv::Mat bayerL(H, W, CV_8UC1, bufL.data());
        cv::Mat bayerR(H, W, CV_8UC1, bufR.data());
        cv::Mat bgrL, bgrR, grayL, grayR;
        cv::cvtColor(bayerL, bgrL, cv::COLOR_BayerRG2BGR);
        cv::cvtColor(bayerR, bgrR, cv::COLOR_BayerRG2BGR);
        cv::cvtColor(bgrL, grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(bgrR, grayR, cv::COLOR_BGR2GRAY);

        // Detect chessboard corners
        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL = cv::findChessboardCorners(grayL, boardSize, cornersL, cbFlags);
        bool foundR = cv::findChessboardCorners(grayR, boardSize, cornersR, cbFlags);

        if (foundL) {
            cv::cornerSubPix(grayL, cornersL, cv::Size(11,11), cv::Size(-1,-1), subpixCrit);
        }
        if (foundR) {
            cv::cornerSubPix(grayR, cornersR, cv::Size(11,11), cv::Size(-1,-1), subpixCrit);
        }

        // Draw preview
        cv::Mat drawL = bgrL.clone(), drawR = bgrR.clone();
        if (foundL) cv::drawChessboardCorners(drawL, boardSize, cornersL, true);
        if (foundR) cv::drawChessboardCorners(drawR, boardSize, cornersR, true);

        char label[64];
        snprintf(label, sizeof(label), "L:%s R:%s | %d",
                 foundL ? "OK" : "--", foundR ? "OK" : "--", captureCount);
        cv::Scalar color = (foundL && foundR)
            ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        cv::putText(drawL, label, cv::Point(10,30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

        cv::Mat display;
        cv::hconcat(drawL, drawR, display);
        if (display.cols > 1920) {
            double s = 1920.0 / display.cols;
            cv::resize(display, display, cv::Size(), s, s);
        }
        cv::imshow("Stereo Capture", display);

        int key = cv::waitKey(1) & 0xFF;
        if (key == ' ') {
            if (!(foundL && foundR)) {
                printf("\nChessboard not detected in both images\n");
            } else {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - lastCapture).count();
                if (elapsed < 1.0) {
                    printf("\nToo fast, wait 1 second\n");
                } else {
                    char name[32];
                    snprintf(name, sizeof(name), "%04d.png", captureCount);
                    std::string lpPath = (fs::path(args.output_dir) / "left"  / name).string();
                    std::string rpPath = (fs::path(args.output_dir) / "right" / name).string();

                    // Save raw BayerRG8 as grayscale PNG (lossless)
                    cv::imwrite(lpPath, bayerL);
                    cv::imwrite(rpPath, bayerR);

                    captureCount++;
                    lastCapture = now;
                    printf("\n[Captured] Pair #%d\n", captureCount);
                }
            }
        } else if (key == 'q' || key == 27) {
            break;
        } else if (key == 'c') {
            // Clear all captured images
            for (const auto& sub : {"left", "right"}) {
                auto dir = fs::path(args.output_dir) / sub;
                for (auto& entry : fs::directory_iterator(dir)) {
                    if (entry.is_regular_file()) fs::remove(entry.path());
                }
            }
            captureCount = 0;
            printf("[Cleared] All images\n");
        }
    }

    cv::destroyAllWindows();
    camera.stopGrabbing();
    camera.close();
    if (pwm) pwm->stop();

    printf("\nTotal captured: %d pairs\n", captureCount);
    if (captureCount >= 10) {
        printf("\nReady for calibration:\n");
        printf("  ./stereo_calibrate -s %.1f -d %s\n", SQUARE_SIZE, args.output_dir.c_str());
    }

    return 0;
}
