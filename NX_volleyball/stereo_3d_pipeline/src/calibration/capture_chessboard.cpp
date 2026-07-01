/**
 * @file capture_chessboard.cpp
 * @brief Stereo image pair capture tool for calibration
 *
 * Pure image acquisition by default. Use --live-check to overlay a lightweight
 * chessboard detection preview while positioning the board.
 *
 * Usage:
 *   ./capture_chessboard -o calibration_images      # HW trigger (default)
 *   ./capture_chessboard --free-run -o images        # free-run mode
 *
 * Keys:
 *   SPACE  - save current frame pair
 *   q/ESC  - quit
 *   c      - clear all saved images
 *   l      - toggle live chessboard preview
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
#include <fstream>
#include <cstdint>
#include <algorithm>

namespace fs = std::filesystem;

static constexpr int    EXPOSURE_US  = 9867;
static constexpr float  GAIN_DB      = 11.9906f;
static constexpr double PWM_FREQ     = 100.0;
static constexpr double PWM_DUTY     = 50.0;
static constexpr double GUI_PREVIEW_FPS = 30.0;
static constexpr int    LIVE_CHECK_INTERVAL = 10;
static constexpr int    LIVE_CHECK_MAX_WIDTH = 960;
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
    int         left_index   = 0;
    int         right_index  = 1;
    int         image_node_num = 3;
    int         board_w = 6;
    int         board_h = 9;
    bool        live_check = false;
    std::string serial_left;
    std::string serial_right;
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
        else if (arg == "--left-index" && i+1 < argc)  a.left_index = std::atoi(argv[++i]);
        else if (arg == "--right-index" && i+1 < argc) a.right_index = std::atoi(argv[++i]);
        else if (arg == "--serial-left" && i+1 < argc) a.serial_left = argv[++i];
        else if (arg == "--serial-right" && i+1 < argc) a.serial_right = argv[++i];
        else if (arg == "--image-node-num" && i+1 < argc) a.image_node_num = std::atoi(argv[++i]);
        else if (arg == "--board-w" && i+1 < argc) a.board_w = std::atoi(argv[++i]);
        else if (arg == "--board-h" && i+1 < argc) a.board_h = std::atoi(argv[++i]);
        else if (arg == "--live-check")             a.live_check = true;
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
                   "  --left-index N  Left camera index [0]\n"
                   "  --right-index N Right camera index [1]\n"
                   "  --serial-left S Bind left camera by serial number\n"
                   "  --serial-right S Bind right camera by serial number\n"
                   "  --image-node-num N SDK FIFO depth [3]\n"
                   "  --board-w N     Inner corners width for --live-check [6]\n"
                   "  --board-h N     Inner corners height for --live-check [9]\n"
                   "  --live-check    Overlay chessboard corner status in GUI preview\n"
                   "  -h, --help      Show this help\n",
                   argv[0]);
            std::exit(0);
        }
    }
    return a;
}

static bool validateArgs(const Args& a) {
    if (a.exposure_us <= 0) {
        fprintf(stderr, "[ERROR] Exposure must be positive, got %d\n", a.exposure_us);
        return false;
    }
    if (a.cam_width <= 0 || a.cam_height <= 0) {
        fprintf(stderr, "[ERROR] Invalid image size: %dx%d\n", a.cam_width, a.cam_height);
        return false;
    }
    if (a.board_w < 2 || a.board_h < 2) {
        fprintf(stderr, "[ERROR] Invalid chessboard inner corners: %dx%d\n",
                a.board_w, a.board_h);
        return false;
    }
    if (a.auto_count < 0) {
        fprintf(stderr, "[ERROR] Auto capture count must be >= 0, got %d\n", a.auto_count);
        return false;
    }
    if (!a.serial_left.empty() && !a.serial_right.empty() &&
        a.serial_left == a.serial_right) {
        fprintf(stderr, "[ERROR] Left/right serial numbers must be different: %s\n",
                a.serial_left.c_str());
        return false;
    }
    if ((a.serial_left.empty() || a.serial_right.empty()) &&
        a.left_index == a.right_index) {
        fprintf(stderr, "[ERROR] Left/right camera indices must be different when serial binding is incomplete: %d\n",
                a.left_index);
        return false;
    }
    return true;
}

struct LiveCheckState {
    bool left_found = false;
    bool right_found = false;
};

static void writeMetadataHeader(std::ofstream& metadata) {
    metadata << "pair,file,left_frame_number,right_frame_number,"
             << "left_frame_counter,right_frame_counter,left_trigger_index,right_trigger_index,"
             << "left_timestamp_ns,right_timestamp_ns,frame_number_delta,frame_counter_delta,"
             << "trigger_delta,timestamp_delta_ns\n";
}

static bool findPreviewCorners(const cv::Mat& bgr,
                               const cv::Size& board_size) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    if (gray.cols > LIVE_CHECK_MAX_WIDTH) {
        const double scale = static_cast<double>(LIVE_CHECK_MAX_WIDTH) / gray.cols;
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    std::vector<cv::Size> candidates{board_size};
    const cv::Size swapped(board_size.height, board_size.width);
    if (swapped != board_size) {
        candidates.push_back(swapped);
    }

    const int fast_flags = cv::CALIB_CB_ADAPTIVE_THRESH |
                           cv::CALIB_CB_NORMALIZE_IMAGE |
                           cv::CALIB_CB_FAST_CHECK;
    const int robust_flags = cv::CALIB_CB_ADAPTIVE_THRESH |
                             cv::CALIB_CB_NORMALIZE_IMAGE |
                             cv::CALIB_CB_FILTER_QUADS;
    std::vector<cv::Point2f> corners;
    for (const cv::Size& size : candidates) {
        corners.clear();
        if (cv::findChessboardCorners(gray, size, corners, fast_flags)) {
            return true;
        }
        corners.clear();
        if (cv::findChessboardCornersSB(gray, size, corners, cv::CALIB_CB_NORMALIZE_IMAGE)) {
            return true;
        }
        corners.clear();
        if (cv::findChessboardCorners(gray, size, corners, robust_flags)) {
            return true;
        }
    }
    return false;
}

static void drawLiveCheck(cv::Mat& image,
                          bool found,
                          const char* side) {
    const cv::Scalar color = found ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 255);
    char text[64];
    snprintf(text, sizeof(text), "%s: %s", side, found ? "OK" : "MISS");
    cv::putText(image, text, cv::Point(10, 32),
                cv::FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
}

int main(int argc, char* argv[]) {
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    Args args = parseArgs(argc, argv);
    if (!validateArgs(args)) return 1;

    fs::create_directories(fs::path(args.output_dir) / "left");
    fs::create_directories(fs::path(args.output_dir) / "right");

    // PWM trigger is started after both cameras enter grabbing state.
    // Starting PWM earlier can give the two USB cameras different local
    // FrameCounter phases, which breaks strict watermarked pair matching.
    std::unique_ptr<stereo3d::PWMTrigger> pwm;

    // Camera
    stereo3d::HikvisionCamera camera;
    stereo3d::CameraConfig cfg;
    cfg.exposure_us = static_cast<float>(args.exposure_us);
    cfg.gain_db     = args.gain_db;
    cfg.width       = args.cam_width;
    cfg.height      = args.cam_height;
    cfg.camera_index_left = args.left_index;
    cfg.camera_index_right = args.right_index;
    cfg.serial_left = args.serial_left;
    cfg.serial_right = args.serial_right;
    cfg.image_node_num = std::max(2, args.image_node_num);
    cfg.trigger_frequency_hz = static_cast<int>(PWM_FREQ);
    cfg.embedded_info_clear_rows = 2;

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

    if (!args.free_run && !args.no_pwm) {
        pwm = std::make_unique<stereo3d::PWMTrigger>(
            GPIO_CHIP, GPIO_LINE, PWM_FREQ, PWM_DUTY);
        if (!pwm->start()) {
            fprintf(stderr, "[WARN] PWM start failed; waiting for an external trigger source\n");
            pwm.reset();
        }
    } else if (args.no_pwm && !args.free_run) {
        fprintf(stderr, "[WARN] PWM disabled. Ensure the external trigger starts after both cameras are grabbing.\n");
    }

    const int W = camera.width();
    const int H = camera.height();

    std::vector<uint8_t> bufL(W * H);
    std::vector<uint8_t> bufR(W * H);

    int captureCount = 0;
    auto lastCapture = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    auto lastPreview = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    bool live_check_enabled = args.live_check && !args.headless;
    int preview_frame = 0;
    LiveCheckState live_state;
    const cv::Size board_size(args.board_w, args.board_h);
    const fs::path metadata_path = fs::path(args.output_dir) / "capture_metadata.csv";
    std::ofstream metadata(metadata_path.string(), std::ios::out | std::ios::trunc);
    if (!metadata.is_open()) {
        fprintf(stderr, "[ERROR] Failed to open metadata file: %s\n",
                metadata_path.string().c_str());
        camera.stopGrabbing();
        camera.close();
        if (pwm) pwm->stop();
        return 1;
    }
    writeMetadataHeader(metadata);

    printf("==================================================\n");
    printf("Stereo Image Capture\n");
    printf("==================================================\n");
    printf("Output: %s/\n", args.output_dir.c_str());
    printf("Camera: %dx%d  exp=%dus  gain=%.2fdB  BayerRG8\n",
           W, H, args.exposure_us, args.gain_db);
    printf("Mode:   %s\n", args.free_run ? "Free-run" : "HW trigger (Line0, RisingEdge)");
    printf("Sync:   %s\n", args.free_run ? "not guaranteed" : "watermarked FrameCounter");
    if (!args.serial_left.empty() || !args.serial_right.empty()) {
        printf("Serial: L=%s R=%s\n", args.serial_left.c_str(), args.serial_right.c_str());
    } else {
        printf("Index:  L=%d R=%d (set serials for repeatable left/right binding)\n",
               args.left_index, args.right_index);
    }
    if (!args.headless)
        printf("Keys:   SPACE=save  q/ESC=quit  c=clear  l=live-check\n");
    printf("Trigger: %s\n",
           args.free_run ? "camera free-run" : "100.0 Hz PWM hardware trigger");
    if (!args.headless) {
        printf("Preview: display capped at %.0f fps", GUI_PREVIEW_FPS);
        if (live_check_enabled) {
            printf(", live-check every %d preview frames (%dx%d, swapped tried)",
                   LIVE_CHECK_INTERVAL, args.board_w, args.board_h);
        }
        printf("\n");
    }
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

        // Display is throttled independently from 100Hz capture/trigger.
        if (!args.headless) {
            auto preview_now = std::chrono::steady_clock::now();
            const double preview_dt =
                std::chrono::duration<double>(preview_now - lastPreview).count();
            if (preview_dt >= 1.0 / GUI_PREVIEW_FPS) {
                lastPreview = preview_now;
                ++preview_frame;

                cv::Mat bayerL(H, W, CV_8UC1, bufL.data());
                cv::Mat bayerR(H, W, CV_8UC1, bufR.data());
                cv::Mat bgrL, bgrR;
                cv::cvtColor(bayerL, bgrL, cv::COLOR_BayerBG2BGR);  // 海康BayerRG8 = OpenCV BayerBG
                cv::cvtColor(bayerR, bgrR, cv::COLOR_BayerBG2BGR);

                if (live_check_enabled && (preview_frame % LIVE_CHECK_INTERVAL == 1)) {
                    live_state.left_found = findPreviewCorners(bgrL, board_size);
                    live_state.right_found = findPreviewCorners(bgrR, board_size);
                }
                if (live_check_enabled) {
                    drawLiveCheck(bgrL, live_state.left_found, "L");
                    drawLiveCheck(bgrR, live_state.right_found, "R");
                } else {
                    char label[64];
                    snprintf(label, sizeof(label), "Captured: %d", captureCount);
                    cv::putText(bgrL, label, cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                }

                cv::Mat display;
                cv::hconcat(bgrL, bgrR, display);
                if (display.cols > 1920) {
                    double s = 1920.0 / display.cols;
                    cv::resize(display, display, cv::Size(), s, s);
                }
                cv::imshow("Stereo Capture", display);
            }
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

            const int64_t frame_number_delta =
                static_cast<int64_t>(resL.frame_number) -
                static_cast<int64_t>(resR.frame_number);
            const int64_t frame_counter_delta =
                static_cast<int64_t>(resL.frame_counter) -
                static_cast<int64_t>(resR.frame_counter);
            const int64_t trigger_delta =
                static_cast<int64_t>(resL.trigger_index) -
                static_cast<int64_t>(resR.trigger_index);
            const int64_t timestamp_delta =
                static_cast<int64_t>(resL.timestamp_us) -
                static_cast<int64_t>(resR.timestamp_us);
            metadata << captureCount << "," << name << ","
                     << resL.frame_number << "," << resR.frame_number << ","
                     << resL.frame_counter << "," << resR.frame_counter << ","
                     << resL.trigger_index << "," << resR.trigger_index << ","
                     << resL.timestamp_us << "," << resR.timestamp_us << ","
                     << frame_number_delta << "," << frame_counter_delta << ","
                     << trigger_delta << "," << timestamp_delta << "\n";
            metadata.flush();

            captureCount++;
            lastCapture = now;
            printf("\r[Saved] Pair #%d  %s  fc_delta=%ld fn_delta=%ld trig_delta=%ld ts_delta=%ldns\n",
                   captureCount, name,
                   static_cast<long>(frame_counter_delta),
                   static_cast<long>(frame_number_delta),
                   static_cast<long>(trigger_delta),
                   static_cast<long>(timestamp_delta));

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
            metadata.close();
            metadata.open(metadata_path.string(), std::ios::out | std::ios::trunc);
            if (!metadata.is_open()) {
                fprintf(stderr, "[ERROR] Failed to reset metadata file: %s\n",
                        metadata_path.string().c_str());
                break;
            }
            writeMetadataHeader(metadata);
            captureCount = 0;
            printf("\r[Cleared] All images\n");
        } else if (key == 'l') {
            live_check_enabled = !live_check_enabled;
            printf("\r[LiveCheck] %s\n", live_check_enabled ? "ON" : "OFF");
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
