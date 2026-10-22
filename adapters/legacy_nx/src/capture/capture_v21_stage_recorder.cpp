#include "capture_v21_stage_recorder.h"

#include "capture_v21_stage_schema.h"

#include "../stereo/depth_match_contract.h"
#include "../utils/logger.h"
#include "../utils/vpi_image_copy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

namespace stereo3d {
namespace {

std::uint64_t realtimeNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string frameName(int frame_id) {
    std::ostringstream stream;
    stream << std::setw(8) << std::setfill('0') << frame_id << ".pgm";
    return stream.str();
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream stream;
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': stream << "\\\\"; break;
        case '"': stream << "\\\""; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (character < 0x20) {
                stream << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(character)
                       << std::dec;
            } else {
                stream << character;
            }
        }
    }
    return stream.str();
}

}  // namespace

bool CaptureV21StageRecorder::init(
    const CaptureV21StageRecorderConfig& config) {
    close();
    cfg_ = config;
    if (!cfg_.enabled) return true;
    if (cfg_.output_dir.empty() || cfg_.frame_limit < 0 ||
        cfg_.queue_capacity == 0 || cfg_.overflow_policy != "block" ||
        cfg_.loss_policy != "fail_trial" ||
        cfg_.exposure_timestamp_source != "nx_gpio_rising_edge" ||
        (cfg_.raw_frames != "none" && cfg_.raw_frames != "selected" &&
         cfg_.raw_frames != "all") ||
        cfg_.image_product != "rectified_gray8" ||
        cfg_.image_format != "pgm") {
        LOG_ERROR("CaptureV21: invalid configuration");
        cfg_.enabled = false;
        return false;
    }

    namespace fs = std::filesystem;
    const fs::path root(cfg_.output_dir);
    if (!fs::is_directory(root) || !fs::is_regular_file(root / "context.json")) {
        LOG_ERROR("CaptureV21: staging/context.json missing under %s",
                  cfg_.output_dir.c_str());
        cfg_.enabled = false;
        return false;
    }
    try {
        const fs::path outputs[] = {
            root / "frames.csv", root / "detections.csv",
            root / "pairs.csv", root / "status.json"};
        for (const auto& path : outputs) {
            if (fs::exists(path)) {
                LOG_ERROR("CaptureV21: refusing to overwrite %s",
                          path.string().c_str());
                cfg_.enabled = false;
                return false;
            }
        }
        const fs::path raw_root = root / "raw";
        if (cfg_.raw_frames != "none" && fs::exists(raw_root)) {
            LOG_ERROR("CaptureV21: refusing to reuse raw image directory %s",
                      raw_root.string().c_str());
            cfg_.enabled = false;
            return false;
        }
        if (cfg_.raw_frames != "none") {
            fs::create_directories(raw_root / "left");
            fs::create_directories(raw_root / "right");
        }
        frames_file_.open((root / "frames.csv").string(),
                          std::ios::out | std::ios::trunc);
        detections_file_.open((root / "detections.csv").string(),
                              std::ios::out | std::ios::trunc);
        pairs_file_.open((root / "pairs.csv").string(),
                         std::ios::out | std::ios::trunc);
    } catch (const std::exception& error) {
        LOG_ERROR("CaptureV21: staging initialization failed: %s", error.what());
        cfg_.enabled = false;
        return false;
    }
    if (!frames_file_ || !detections_file_ || !pairs_file_) {
        LOG_ERROR("CaptureV21: failed to open staging CSV files");
        cfg_.enabled = false;
        return false;
    }
    frames_file_.imbue(std::locale::classic());
    detections_file_.imbue(std::locale::classic());
    pairs_file_.imbue(std::locale::classic());
    writeCaptureV21FrameHeader(frames_file_);
    writeCaptureV21DetectionHeader(detections_file_);
    writeCaptureV21PairHeader(pairs_file_);
    if (!frames_file_ || !detections_file_ || !pairs_file_) {
        LOG_ERROR("CaptureV21: failed to write staging CSV headers");
        cfg_.enabled = false;
        return false;
    }

    queue_.clear();
    fatal_error_.clear();
    captured_frames_ = 0;
    written_frames_ = 0;
    dropped_frames_ = 0;
    copy_failures_ = 0;
    write_failures_ = 0;
    next_pair_id_ = 0;
    last_frame_id_ = -1;
    last_left_counter_ = 0;
    last_right_counter_ = 0;
    last_trigger_sequence_ = 0;
    have_previous_frame_ = false;
    capture_complete_ = false;
    status_publishable_ = false;
    running_ = true;
    try {
        writer_thread_ =
            std::thread(&CaptureV21StageRecorder::writerLoop, this);
    } catch (const std::exception& error) {
        running_ = false;
        ++write_failures_;
        LOG_ERROR("CaptureV21: failed to start writer thread: %s", error.what());
        setFatal("failed to start capture staging writer thread");
        writeStatus();
        return false;
    } catch (...) {
        running_ = false;
        ++write_failures_;
        LOG_ERROR("CaptureV21: unknown failure starting writer thread");
        setFatal("unknown failure starting capture staging writer thread");
        writeStatus();
        return false;
    }
    LOG_INFO("CaptureV21 armed: dir=%s frames=%d queue=%zu raw=%s",
             cfg_.output_dir.c_str(), cfg_.frame_limit, cfg_.queue_capacity,
             cfg_.raw_frames.c_str());
    return true;
}

bool CaptureV21StageRecorder::copyGrayImage(
    VPIImage image, cv::Mat& out) const {
    return copyVpiGray8ToHost(image, out, "CaptureV21 gray copy");
}

bool CaptureV21StageRecorder::writePgm(
    const std::string& path, const cv::Mat& image) const {
    if (image.empty() || image.type() != CV_8UC1) return false;
    std::error_code exists_error;
    if (std::filesystem::exists(path, exists_error) || exists_error) {
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) return false;
    output << "P5\n" << image.cols << ' ' << image.rows << "\n255\n";
    for (int row = 0; row < image.rows; ++row) {
        output.write(reinterpret_cast<const char*>(image.ptr(row)), image.cols);
        if (!output) return false;
    }
    output.flush();
    return static_cast<bool>(output);
}

void CaptureV21StageRecorder::setFatal(std::string message) {
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (fatal_error_.empty()) fatal_error_ = std::move(message);
    }
    capture_complete_ = true;
    queue_not_empty_.notify_all();
    queue_not_full_.notify_all();
}

std::string CaptureV21StageRecorder::fatalError() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return fatal_error_;
}

bool CaptureV21StageRecorder::shouldStop() const noexcept {
    return cfg_.enabled && capture_complete_.load();
}

void CaptureV21StageRecorder::record(const FrameCallbackData& frame) {
    if (!cfg_.enabled || !running_.load() || capture_complete_.load()) return;
    if (frame.frame_id < 0 || frame.metadata.grab_failed) {
        ++dropped_frames_;
        setFatal("invalid or failed stereo frame reached capture callback");
        return;
    }
    if (have_previous_frame_) {
        if (frame.frame_id != last_frame_id_ + 1) {
            ++dropped_frames_;
            setFatal("pipeline frame sequence contains a gap");
            return;
        }
        if (frame.metadata.left_frame_counter != last_left_counter_ + 1 ||
            frame.metadata.right_frame_counter != last_right_counter_ + 1) {
            ++dropped_frames_;
            setFatal("hardware frame counter contains a gap");
            return;
        }
    }
    if (frame.metadata.left_frame_counter !=
        frame.metadata.right_frame_counter) {
        ++dropped_frames_;
        setFatal("left/right hardware frame counters disagree");
        return;
    }
    if (frame.metadata.trigger_sequence == 0 ||
        frame.metadata.trigger_edge_monotonic_ns == 0 ||
        frame.metadata.trigger_edge_realtime_ns == 0 ||
        frame.metadata.trigger_scheduled_monotonic_ns == 0) {
        ++dropped_frames_;
        setFatal("frame has no NX GPIO trigger-edge timestamp");
        return;
    }
    if (static_cast<std::uint32_t>(frame.metadata.trigger_sequence) !=
        frame.metadata.left_frame_counter) {
        ++dropped_frames_;
        setFatal("GPIO trigger sequence does not match camera FrameCounter");
        return;
    }
    if (have_previous_frame_ &&
        frame.metadata.trigger_sequence != last_trigger_sequence_ + 1) {
        ++dropped_frames_;
        setFatal("GPIO trigger sequence contains a gap");
        return;
    }
    if (frame.metadata.trigger_edge_monotonic_ns <
        frame.metadata.trigger_scheduled_monotonic_ns) {
        ++dropped_frames_;
        setFatal("GPIO trigger edge predates its scheduled time");
        return;
    }
    if (frame.metadata.left_receive_monotonic_ns == 0 ||
        frame.metadata.right_receive_monotonic_ns == 0 ||
        frame.metadata.left_receive_realtime_ns == 0 ||
        frame.metadata.right_receive_realtime_ns == 0) {
        ++dropped_frames_;
        setFatal("frame has no per-camera USB receive timestamp");
        return;
    }
    if (frame.metadata.left_receive_monotonic_ns <
            frame.metadata.trigger_edge_monotonic_ns ||
        frame.metadata.right_receive_monotonic_ns <
            frame.metadata.trigger_edge_monotonic_ns ||
        frame.metadata.left_receive_realtime_ns <
            frame.metadata.trigger_edge_realtime_ns ||
        frame.metadata.right_receive_realtime_ns <
            frame.metadata.trigger_edge_realtime_ns) {
        ++dropped_frames_;
        setFatal("camera USB receive timestamp predates GPIO trigger edge");
        return;
    }

    QueuedFrame queued;
    queued.frame_id = frame.frame_id;
    queued.callback_timestamp_ns = realtimeNowNs();
    queued.metadata = frame.metadata;
    queued.candidates_left = frame.detector_candidates_left;
    queued.candidates_right = frame.detector_candidates_right;
    queued.pair_attempts = frame.stereo_pair_attempts;

    const bool accepted_pair =
        selectSingleBallObservationPair(queued.pair_attempts).has_value();
    for (const auto& attempt : queued.pair_attempts) {
        if (attempt.pair.left_index < 0 || attempt.pair.right_index < 0 ||
            attempt.pair.left_index >=
                static_cast<int>(queued.candidates_left.size()) ||
            attempt.pair.right_index >=
                static_cast<int>(queued.candidates_right.size())) {
            setFatal("stereo pair references a missing detector candidate");
            return;
        }
    }
    const bool copy_raw = cfg_.raw_frames == "all" ||
                          (cfg_.raw_frames == "selected" && accepted_pair);
    if (copy_raw) {
        if (!copyGrayImage(frame.rect_gray_left, queued.left_gray) ||
            !copyGrayImage(frame.rect_gray_right, queued.right_gray)) {
            ++copy_failures_;
            setFatal("failed to copy rectified gray frame from VPI");
            return;
        }
        queued.has_raw_images = true;
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_not_full_.wait(lock, [this] {
            return !running_.load() || !fatalError().empty() ||
                   queue_.size() < cfg_.queue_capacity;
        });
        if (!running_.load() || !fatalError().empty()) return;
        queue_.push_back(std::move(queued));
    }
    queue_not_empty_.notify_one();
    last_frame_id_ = frame.frame_id;
    last_left_counter_ = frame.metadata.left_frame_counter;
    last_right_counter_ = frame.metadata.right_frame_counter;
    last_trigger_sequence_ = frame.metadata.trigger_sequence;
    have_previous_frame_ = true;
    const int captured = ++captured_frames_;
    if (cfg_.frame_limit > 0 && captured >= cfg_.frame_limit) {
        capture_complete_ = true;
    }
}

void CaptureV21StageRecorder::writerLoop() noexcept {
    try {
        while (true) {
            QueuedFrame frame;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_not_empty_.wait(lock, [this] {
                    return !running_.load() || !queue_.empty();
                });
                if (queue_.empty()) {
                    if (!running_.load()) break;
                    continue;
                }
                frame = std::move(queue_.front());
                queue_.pop_front();
            }
            queue_not_full_.notify_one();
            if (!writeFrame(frame)) {
                ++write_failures_;
                setFatal("failed to write capture staging frame");
            } else {
                ++written_frames_;
            }
        }
    } catch (const std::exception& error) {
        running_ = false;
        ++write_failures_;
        LOG_ERROR("CaptureV21: writer thread failed: %s", error.what());
        setFatal("capture staging writer raised an exception");
    } catch (...) {
        running_ = false;
        ++write_failures_;
        LOG_ERROR("CaptureV21: writer thread failed with unknown exception");
        setFatal("capture staging writer raised an unknown exception");
    }
}

bool CaptureV21StageRecorder::writeFrame(const QueuedFrame& frame) {
    namespace fs = std::filesystem;
    std::string left_raw;
    std::string right_raw;
    if (frame.has_raw_images) {
        const std::string name = frameName(frame.frame_id);
        left_raw = "raw/left/" + name;
        right_raw = "raw/right/" + name;
        if (!writePgm((fs::path(cfg_.output_dir) / left_raw).string(),
                      frame.left_gray) ||
            !writePgm((fs::path(cfg_.output_dir) / right_raw).string(),
                      frame.right_gray)) {
            return false;
        }
    }
    CaptureV21FrameRow frame_row;
    frame_row.frame_id = frame.frame_id;
    frame_row.host_capture_timestamp_ns =
        frame.metadata.host_capture_timestamp_ns;
    frame_row.callback_timestamp_ns = frame.callback_timestamp_ns;
    frame_row.left_receive_monotonic_ns =
        frame.metadata.left_receive_monotonic_ns;
    frame_row.right_receive_monotonic_ns =
        frame.metadata.right_receive_monotonic_ns;
    frame_row.left_receive_realtime_ns =
        frame.metadata.left_receive_realtime_ns;
    frame_row.right_receive_realtime_ns =
        frame.metadata.right_receive_realtime_ns;
    frame_row.left_sdk_host_timestamp_raw =
        frame.metadata.left_sdk_host_timestamp_raw;
    frame_row.right_sdk_host_timestamp_raw =
        frame.metadata.right_sdk_host_timestamp_raw;
    frame_row.left_device_timestamp_ns = frame.metadata.left_device_timestamp_ns;
    frame_row.right_device_timestamp_ns =
        frame.metadata.right_device_timestamp_ns;
    frame_row.stereo_timestamp_residual_ns =
        frame.metadata.stereo_timestamp_residual_ns;
    frame_row.trigger_sequence = frame.metadata.trigger_sequence;
    frame_row.trigger_scheduled_monotonic_ns =
        frame.metadata.trigger_scheduled_monotonic_ns;
    frame_row.trigger_edge_monotonic_ns =
        frame.metadata.trigger_edge_monotonic_ns;
    frame_row.trigger_edge_realtime_ns =
        frame.metadata.trigger_edge_realtime_ns;
    frame_row.trigger_write_uncertainty_ns =
        frame.metadata.trigger_write_uncertainty_ns;
    frame_row.left_frame_number = frame.metadata.left_frame_number;
    frame_row.right_frame_number = frame.metadata.right_frame_number;
    frame_row.left_frame_counter = frame.metadata.left_frame_counter;
    frame_row.right_frame_counter = frame.metadata.right_frame_counter;
    frame_row.left_trigger_index = frame.metadata.left_trigger_index;
    frame_row.right_trigger_index = frame.metadata.right_trigger_index;
    frame_row.raw_left = std::move(left_raw);
    frame_row.raw_right = std::move(right_raw);
    if (!writeCaptureV21FrameRow(frames_file_, frame_row)) return false;

    const auto accepted_index =
        selectSingleBallObservationPair(frame.pair_attempts);
    std::vector<bool> selected_left(frame.candidates_left.size(), false);
    std::vector<bool> selected_right(frame.candidates_right.size(), false);
    if (accepted_index) {
        const auto& attempt = frame.pair_attempts[*accepted_index];
        selected_left[static_cast<std::size_t>(attempt.pair.left_index)] = true;
        selected_right[static_cast<std::size_t>(attempt.pair.right_index)] = true;
    }
    auto write_detections = [&](const char* role,
                                const std::vector<Detection>& candidates,
                                const std::vector<bool>& selected) {
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& detection = candidates[index];
            CaptureV21DetectionRow row;
            row.frame_id = frame.frame_id;
            row.camera_role = role;
            row.candidate_id = index;
            row.cx_px = detection.cx;
            row.cy_px = detection.cy;
            row.width_px = detection.width;
            row.height_px = detection.height;
            row.confidence = detection.confidence;
            row.class_id = detection.class_id;
            row.selected = selected[index];
            if (!writeCaptureV21DetectionRow(detections_file_, row)) {
                return false;
            }
        }
        return true;
    };
    if (!write_detections("left", frame.candidates_left, selected_left) ||
        !write_detections("right", frame.candidates_right, selected_right)) {
        return false;
    }

    for (std::size_t index = 0; index < frame.pair_attempts.size(); ++index) {
        const auto& attempt = frame.pair_attempts[index];
        const bool accepted = accepted_index && index == *accepted_index;
        StereoRoiPairRejectReason reason = attempt.reject_reason;
        if (!accepted && attempt.triangulated) {
            reason = StereoRoiPairRejectReason::NOT_SELECTED;
        }
        if (!accepted && reason == StereoRoiPairRejectReason::NONE) {
            reason = StereoRoiPairRejectReason::DEPTH_REJECTED;
        }
        CaptureV21PairRow row;
        row.pair_id = next_pair_id_++;
        row.frame_id = frame.frame_id;
        row.left_candidate_id = attempt.pair.left_index;
        row.right_candidate_id = attempt.pair.right_index;
        row.disparity_px = attempt.pair.initial_disparity;
        row.epipolar_error_px = attempt.pair.epipolar_dy;
        row.pair_score = attempt.pair.score;
        row.semantic_confidence = attempt.pair.semantic_confidence;
        row.accepted = accepted;
        if (accepted) {
            row.position_rig_m = std::array<float, 3>{
                attempt.x_rig_m, attempt.y_rig_m, attempt.z_rig_m};
        } else {
            row.reject_reason = stereoRoiPairRejectReasonName(reason);
        }
        if (!writeCaptureV21PairRow(pairs_file_, row)) return false;
    }
    frames_file_.flush();
    detections_file_.flush();
    pairs_file_.flush();
    return frames_file_.good() && detections_file_.good() && pairs_file_.good();
}

void CaptureV21StageRecorder::writeStatus() {
    if (!cfg_.enabled || cfg_.output_dir.empty()) return;
    const std::string fatal = fatalError();
    const bool reached_target = cfg_.frame_limit > 0
        ? captured_frames_.load() >= cfg_.frame_limit
        : captured_frames_.load() > 0;
    const bool complete = fatal.empty() && reached_target &&
        dropped_frames_.load() == 0 && copy_failures_.load() == 0 &&
        write_failures_.load() == 0 &&
        written_frames_.load() == captured_frames_.load();
    std::ofstream status(
        (std::filesystem::path(cfg_.output_dir) / "status.json").string(),
        std::ios::out | std::ios::trunc);
    if (!status) return;
    status << "{\n"
           << "  \"stage_schema_version\": "
           << kCaptureV21StageSchemaVersion << ",\n"
           << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
           << "  \"fatal_error\": \"" << jsonEscape(fatal) << "\",\n"
           << "  \"frames_captured\": " << captured_frames_.load() << ",\n"
           << "  \"frames_written\": " << written_frames_.load() << ",\n"
           << "  \"dropped_frames\": " << dropped_frames_.load() << ",\n"
           << "  \"copy_failures\": " << copy_failures_.load() << ",\n"
           << "  \"write_failures\": " << write_failures_.load() << "\n"
           << "}\n";
    status.flush();
    status_publishable_ = complete && status.good();
}

void CaptureV21StageRecorder::close() {
    if (running_.exchange(false)) {
        queue_not_empty_.notify_all();
        queue_not_full_.notify_all();
        if (writer_thread_.joinable()) writer_thread_.join();
    } else if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    if (frames_file_.is_open()) frames_file_.close();
    if (detections_file_.is_open()) detections_file_.close();
    if (pairs_file_.is_open()) pairs_file_.close();
    if (cfg_.enabled && !status_publishable_.load()) {
        writeStatus();
        LOG_INFO("CaptureV21 closed: captured=%d written=%d publishable=%d error=%s",
                 captured_frames_.load(), written_frames_.load(),
                 status_publishable_.load() ? 1 : 0,
                 fatalError().empty() ? "none" : fatalError().c_str());
    }
}

}  // namespace stereo3d

