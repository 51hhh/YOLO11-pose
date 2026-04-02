/**
 * @file capture_chessboard.cpp
 * @brief Stereo image pair capture tool for calibration
 *
 * Pure image acquisition — no chessboard detection during capture.
 * Chessboard detection is done separately by stereo_calibrate.
 *
 * Usage:
 *   ./capture_chessboard -o calibration_images      # HW trigger (default)
 *   ./capture_chessboard --free-run -o images        # free-run mode
 *
 * Keys:
 *   SPACE  - save current frame pair
 *   q/ESC  - quit
 *   c      - clear all saved images
 */

#include "../capture/hikvision_camera.h"
#include "pwm_trigger.h"

#include <opencv2/opencv.hpp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <filesystem>

namespace fs = std::filesystem;

static constexpr int    EXPOSURE_US  = 9867;
static constexpr float  GAIN_DB      = 11.9906f;
static constexpr double PWM_FREQ     = 100.0;
static constexpr double PWM_DUTY     = 50.0;
static const char*      GPIO_CHIP    = "gpiochip2";
static constexpr unsigned GPIO_LINE  = 7;

static std::atomic<bool> g_quit{false};
static void sigHandler(int) { g_quit.store(true); }

struct Args {
    std::string output_dir  = "calibration_images";
    int         exposure_us = EXPOSURE_US;
    float       gain_db     = GAIN_DB;
    bool        free_run    = false;
    bool        no_pwm      = false;
    bool        headless    = false;
    int         auto_count  = 0;
    int         cam_width   = 1440;
    int         cam_height  = 1080;
};

static Args parseArgs(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--free-run")              a.free_run = true;
        else if (arg == "--no-pwm")                a.no_pwm   = true;
        else if (arg == "--headless")              a.headless = true;
        else if (arg == "-n" && i+1 < argc)      { a.auto_count = std::atoi(argv[++i]); a.headless = true; }
        else if (arg == "-o" && i+1 < argc)        a.output_dir = argv[++i];
        else if (arg == "-e" && i+1 < argc)        a.exposure_us = std::atoi(argv[++i]);
        else if (arg == "-g" && i+1 < argc)        a.gain_db = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--width"  && i+1 < argc)  a.cam_width  = std::atoi(argv[++i]);
        else if (arg == "--height" && i+1 < argc)  a.cam_height = std::atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s [options]\n"
                   "  -o DIR          Output directory [calibration_images]\n"
                   "  -e US           Exposure time in us [9867]\n"
                   "  -g DB           Gain in dB [11.9906]\n"
                   "  --free-run      Free-run mode (no HW trigger)\n"
                   "  --no-pwm        Disable PWM output\n"
                   "  --headless      No GUI (for SSH sessions)\n"
                   "  -n COUNT        Auto-capture COUNT pairs then exit (implies --headless)\n"
                   "  --width W       Camera width [1440]\n"
                   "  --height H      Camera height [1080]\n"
                   "  -h, --help      Show this help\n",
                   argv[0]);
            std::exit(0);
        }
    }
    return a;
}

int main(int argc, char* argv[]) {
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    Args args = parseArgs(argc, argv);

    fs::create_directories(fs::path(args.output_dir) / "left");
    fs::create_directories(fs::path(args.output_dir) / "right");

    // PWM trigger
    std::unique_ptr<stereo3d::PWMTrigger> pwm;
    if (!args.free_run && !args.no_pwm) {
        pwm = std::make_unique<stereo3d::PWMTrigger>(
            GPIO_CHIP, GPIO_LINE, PWM_FREQ, PWM_DUTY);
        if (!pwm->start()) {
            fprintf(stderr, "[WARN] PWM start failed, continuing without trigger\n");
            pwm.reset();
        }
    }

    // Camera
    stereo3d::HikvisionCamera camera;
    stereo3d::CameraConfig cfg;
    cfg.exposure_us = static_cast<float>(args.exposure_us);
    cfg.gain_db     = args.gain_db;
    cfg.width       = args.cam_width;
    cfg.height      = args.cam_height;

    if (args.free_run) {
        cfg.use_trigger = false;
    } else {
        cfg.use_trigger        = true;
        cfg.trigger_source     = "Line0";
        cfg.trigger_activation = "RisingEdge";
    }

    if (!camera.open(cfg)) {
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

    std::vector<uint8_t> bufL(W * H);
    std::vector<uint8_t> bufR(W * H);

    int captureCount = 0;
    auto lastCapture = std::chrono::steady_clock::now() - std::chrono::seconds(2);

    printf("==================================================\n");
    printf("Stereo Image Capture\n");
    printf("==================================================\n");
    printf("Output: %s/\n", args.output_dir.c_str());
    printf("Camera: %dx%d  exp=%dus  gain=%.2fdB  BayerRG8\n",
           W, H, args.exposure_us, args.gain_db);
    printf("Mode:   %s\n", args.free_run ? "Free-run" : "HW trigger (Line0, RisingEdge)");
    if (!args.headless)
        printf("Keys:   SPACE=save  q/ESC=quit  c=clear\n");
    printf("==================================================\n");

    while (!g_quit.load()) {
        stereo3d::GrabResult resL, resR;
        bool ok = camera.grabFramePair(
            bufL.data(), bufR.data(), 0, 0, 1000, resL, resR);

        if (!ok) {
            if (!args.headless) {
                printf("\rWaiting for trigger...");
                fflush(stdout);
            }
            continue;
        }

        // Display (BayerRG8 -> BGR)
        if (!args.headless) {
            cv::Mat bayerL(H, W, CV_8UC1, bufL.data());
            cv::Mat bayerR(H, W, CV_8UC1, bufR.data());
            cv::Mat bgrL, bgrR;
            cv::cvtColor(bayerL, bgrL, cv::COLOR_BayerRG2BGR);
            cv::cvtColor(bayerR, bgrR, cv::COLOR_BayerRG2BGR);

            char label[64];
            snprintf(label, sizeof(label), "Captured: %d", captureCount);
            cv::putText(bgrL, label, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::Mat display;
            cv::hconcat(bgrL, bgrR, display);
            if (display.cols > 1920) {
                double s = 1920.0 / display.cols;
                cv::resize(display, display, cv::Size(), s, s);
            }
            cv::imshow("Stereo Capture", display);
        }

        int key = args.headless ? -1 : (cv::waitKey(1) & 0xFF);

        // Auto-capture in headless mode
        bool doSave = false;
        if (args.headless && args.auto_count > 0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - lastCapture).count() >= 1.0)
                doSave = true;
        }

        if (key == ' ' || doSave) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - lastCapture).count() < 0.5) {
                printf("\rToo fast, wait...");
                fflush(stdout);
                continue;
            }

            char name[32];
            snprintf(name, sizeof(name), "%04d.png", captureCount);
            std::string pathL = (fs::path(args.output_dir) / "left"  / name).string();
            std::string pathR = (fs::path(args.output_dir) / "right" / name).string();

            cv::Mat rawL(H, W, CV_8UC1, bufL.data());
            cv::Mat rawR(H, W, CV_8UC1, bufR.data());
            cv::imwrite(pathL, rawL);
            cv::imwrite(pathR, rawR);

            captureCount++;
            lastCapture = now;
            printf("\r[Saved] Pair #%d  %s\n", captureCount, name);

            if (args.auto_count > 0 && captureCount >= args.auto_count) {
                printf("[Auto] Reached %d pairs\n", args.auto_count);
                break;
            }
        } else if (key == 'q' || key == 27) {
            break;
        } else if (key == 'c') {
            for (const auto& sub : {"left", "right"}) {
                auto dir = fs::path(args.output_dir) / sub;
                for (auto& entry : fs::directory_iterator(dir))
                    if (entry.is_regular_file()) fs::remove(entry.path());
            }
            captureCount = 0;
            printf("\r[Cleared] All images\n");
        }
    }

    if (!args.headless) cv::destroyAllWindows();
    camera.stopGrabbing();
    camera.close();
    if (pwm) pwm->stop();

    printf("\nTotal: %d pairs in %s/\n", captureCount, args.output_dir.c_str());
    if (captureCount > 0) {
        printf("Next step: ./stereo_calibrate -s <square_mm> -d %s\n",
               args.output_dir.c_str());
    }

    return 0;
}
