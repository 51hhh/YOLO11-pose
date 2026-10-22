/**
 * @file capture_chessboard.cpp
 * @brief Stereo image pair capture tool for calibration
 *
 * Hardware-synchronized acquisition with lightweight preview for manual
 * positioning.
 *
 * Usage:
 *   ./capture_chessboard -o calibration_images -g 17.0 \
 *     --serial-left 00D39342665 --serial-right 00219471413
 *
 * Keys:
 *   SPACE  - save current frame pair
 *   q/ESC  - quit
 *   c      - clear all saved images
 */

#include "../capture/hikvision_camera.h"
#include "capture_chessboard_args.h"
#include "pwm_trigger.h"

#include <opencv2/opencv.hpp>
#include <csignal>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <memory>

namespace fs = std::filesystem;

static constexpr double PWM_FREQ     = 100.0;
static constexpr double PWM_DUTY     = 50.0;
static constexpr double GUI_PREVIEW_FPS = 60.0;
static constexpr unsigned int GUI_GRAB_TIMEOUT_MS = 40;
static constexpr unsigned int HEADLESS_GRAB_TIMEOUT_MS = 1000;
static const char*      GPIO_CHIP    = "gpiochip2";
static constexpr unsigned GPIO_LINE  = 7;

static std::atomic<bool> g_quit{false};
static void sigHandler(int) { g_quit.store(true); }

struct SyncInfo {
    int64_t frame_number_delta = 0;
    int64_t frame_counter_delta = 0;
    int64_t trigger_delta = 0;
    int64_t timestamp_delta_ns = 0;
};

static SyncInfo makeSyncInfo(const stereo3d::GrabResult& resL,
                             const stereo3d::GrabResult& resR) {
    SyncInfo sync;
    sync.frame_number_delta =
        static_cast<int64_t>(resL.frame_number) -
        static_cast<int64_t>(resR.frame_number);
    sync.frame_counter_delta =
        static_cast<int64_t>(resL.frame_counter) -
        static_cast<int64_t>(resR.frame_counter);
    sync.trigger_delta =
        static_cast<int64_t>(resL.trigger_index) -
        static_cast<int64_t>(resR.trigger_index);
    sync.timestamp_delta_ns =
        static_cast<int64_t>(resL.timestamp_us) -
        static_cast<int64_t>(resR.timestamp_us);
    return sync;
}

static void writeMetadataHeader(std::ofstream& metadata) {
    metadata << "pair,file,left_frame_number,right_frame_number,"
             << "left_frame_counter,right_frame_counter,left_trigger_index,right_trigger_index,"
             << "left_timestamp_ns,right_timestamp_ns,frame_number_delta,frame_counter_delta,"
             << "trigger_delta,timestamp_delta_ns\n";
}

static void writeMetadataRow(std::ofstream& metadata,
                             int pair_index,
                             const std::string& file_name,
                             const stereo3d::GrabResult& resL,
                             const stereo3d::GrabResult& resR,
                             const SyncInfo& sync) {
    metadata << pair_index << "," << file_name << ","
             << resL.frame_number << "," << resR.frame_number << ","
             << resL.frame_counter << "," << resR.frame_counter << ","
             << resL.trigger_index << "," << resR.trigger_index << ","
             << resL.timestamp_us << "," << resR.timestamp_us << ","
             << sync.frame_number_delta << "," << sync.frame_counter_delta << ","
             << sync.trigger_delta << "," << sync.timestamp_delta_ns << "\n";
}

static void drawStatus(cv::Mat& image,
                       int capture_count,
                       bool pair_ok) {
    char text[64];
    snprintf(text, sizeof(text), "Captured: %d", capture_count);
    cv::putText(image, text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, pair_ok ? "SYNC: OK" : "SYNC: WAIT", cv::Point(10, 64),
                cv::FONT_HERSHEY_SIMPLEX, 0.65,
                pair_ok ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 180, 255), 2);
}

int main(int argc, char* argv[]) {
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    CaptureChessboardArgs args;
    if (!parseCaptureChessboardArgs(argc, argv, args)) return 1;
    if (!validateCaptureChessboardArgs(args)) return 1;

    std::error_code ec;
    const fs::path left_dir = fs::path(args.output_dir) / "left";
    const fs::path right_dir = fs::path(args.output_dir) / "right";
    fs::create_directories(left_dir, ec);
    if (ec || !fs::is_directory(left_dir)) {
        fprintf(stderr, "[ERROR] Failed to create left output directory: %s\n",
                ec ? ec.message().c_str() : left_dir.string().c_str());
        return 1;
    }
    ec.clear();
    fs::create_directories(right_dir, ec);
    if (ec || !fs::is_directory(right_dir)) {
        fprintf(stderr, "[ERROR] Failed to create right output directory: %s\n",
                ec ? ec.message().c_str() : right_dir.string().c_str());
        return 1;
    }

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
        printf("Keys:   SPACE=save  q/ESC=quit  c=clear\n");
    printf("Trigger: %s\n",
           args.free_run ? "camera free-run" : "100.0 Hz PWM hardware trigger");
    if (!args.headless) {
        printf("Preview: display capped at %.0f fps\n", GUI_PREVIEW_FPS);
    }
    printf("==================================================\n");

    while (!g_quit.load()) {
        stereo3d::GrabResult resL, resR;
        const unsigned int grab_timeout_ms =
            args.headless ? HEADLESS_GRAB_TIMEOUT_MS : GUI_GRAB_TIMEOUT_MS;
        bool ok = camera.grabFramePair(
            bufL.data(), bufR.data(), 0, 0, grab_timeout_ms, resL, resR);

        if (!ok) {
            if (!args.headless) {
                printf("\rWaiting for trigger...");
                fflush(stdout);
            }
        }

        // Display is throttled independently from 100Hz capture/trigger.
        if (!args.headless) {
            auto preview_now = std::chrono::steady_clock::now();
            const double preview_dt =
                std::chrono::duration<double>(preview_now - lastPreview).count();
            if (ok && preview_dt >= 1.0 / GUI_PREVIEW_FPS) {
                lastPreview = preview_now;

                cv::Mat bayerL(H, W, CV_8UC1, bufL.data());
                cv::Mat bayerR(H, W, CV_8UC1, bufR.data());
                cv::Mat bgrL;
                cv::Mat bgrR;
                cv::cvtColor(bayerL, bgrL, cv::COLOR_BayerBG2BGR);  // 海康BayerRG8 = OpenCV BayerBG
                cv::cvtColor(bayerR, bgrR, cv::COLOR_BayerBG2BGR);
                drawStatus(bgrL, captureCount, ok);
                drawStatus(bgrR, captureCount, ok);

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

        if (!ok) {
            if (key == 'q' || key == 27) {
                break;
            }
            continue;
        }

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
            const SyncInfo sync = makeSyncInfo(resL, resR);

            cv::Mat bayerSaveL(H, W, CV_8UC1, bufL.data());
            cv::Mat bayerSaveR(H, W, CV_8UC1, bufR.data());
            cv::Mat saveBgrL, saveBgrR;
            cv::cvtColor(bayerSaveL, saveBgrL, cv::COLOR_BayerBG2BGR);
            cv::cvtColor(bayerSaveR, saveBgrR, cv::COLOR_BayerBG2BGR);
            const bool wroteL = cv::imwrite(pathL, saveBgrL);
            const bool wroteR = cv::imwrite(pathR, saveBgrR);
            if (!wroteL || !wroteR) {
                fprintf(stderr, "\n[ERROR] Failed to save pair %s (left=%d right=%d)\n",
                        name, wroteL ? 1 : 0, wroteR ? 1 : 0);
                lastCapture = now;
                continue;
            }

            writeMetadataRow(metadata, captureCount, name, resL, resR, sync);
            metadata.flush();

            captureCount++;
            lastCapture = now;
            printf("\r[Saved] Pair #%d  %s  fc_delta=%ld fn_delta=%ld trig_delta=%ld ts_delta=%ldns\n",
                   captureCount, name,
                   static_cast<long>(sync.frame_counter_delta),
                   static_cast<long>(sync.frame_number_delta),
                   static_cast<long>(sync.trigger_delta),
                   static_cast<long>(sync.timestamp_delta_ns));

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
        }
    }

    if (!args.headless) cv::destroyAllWindows();
    camera.stopGrabbing();
    camera.close();
    if (pwm) pwm->stop();

    printf("\nTotal: %d pairs in %s/\n", captureCount, args.output_dir.c_str());
    if (captureCount > 0) {
        printf("Next step: ./stereo_calibrate -s 26.0 --board-w 5 --board-h 8 -d %s\n",
               args.output_dir.c_str());
    }

    return 0;
}
