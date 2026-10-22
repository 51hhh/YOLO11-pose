#include "volleyball/adapters/capture_v21/mainline_stage_trial_writer.hpp"

#include "volleyball/adapters/capture_v21/mainline_stage_schema.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace volleyball::adapters::capture_v21 {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kFrameFile = "frames_v6.csv";
constexpr std::string_view kDetectionFile = "detections_v6.csv";
constexpr std::string_view kAssociationFile = "associations_v6.csv";
constexpr std::string_view kObservationFile = "observations_v6.csv";
constexpr std::string_view kStatusFile = "status_v6.json";

struct FrameKey {
    std::string camera_id;
    core::CameraRole role{core::CameraRole::left};
    core::FrameIndex frame_index{0};

    friend bool operator<(const FrameKey& lhs, const FrameKey& rhs) noexcept {
        return std::tie(lhs.camera_id, lhs.role, lhs.frame_index) <
               std::tie(rhs.camera_id, rhs.role, rhs.frame_index);
    }

    friend bool operator==(const FrameKey& lhs, const FrameKey& rhs) noexcept {
        return std::tie(lhs.camera_id, lhs.role, lhs.frame_index) ==
               std::tie(rhs.camera_id, rhs.role, rhs.frame_index);
    }
};

struct CandidateKey {
    FrameKey frame;
    core::CandidateId candidate_id{0};

    friend bool operator<(
        const CandidateKey& lhs,
        const CandidateKey& rhs) noexcept {
        return std::tie(
                   lhs.frame.camera_id,
                   lhs.frame.role,
                   lhs.frame.frame_index,
                   lhs.candidate_id) <
               std::tie(
                   rhs.frame.camera_id,
                   rhs.frame.role,
                   rhs.frame.frame_index,
                   rhs.candidate_id);
    }

    friend bool operator==(
        const CandidateKey& lhs,
        const CandidateKey& rhs) noexcept {
        return lhs.frame == rhs.frame && lhs.candidate_id == rhs.candidate_id;
    }
};

struct AcceptedPair {
    CandidateKey left;
    CandidateKey right;
    core::Timestamp left_capture;
    core::Timestamp right_capture;
};

FrameKey frameKey(const core::FrameReference& frame) {
    return {frame.camera_id.value(), frame.role, frame.frame_index};
}

FrameKey frameKey(const core::CameraFrameRecord& frame) {
    return {frame.camera_id.value(), frame.role, frame.frame_index};
}

CandidateKey candidateKey(const core::Detection2D& value) {
    return {frameKey(value.frame), value.candidate_id};
}

std::string jsonString(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned int>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

bool imageLayoutIsWritable(const ports::CapturedFrame& frame) noexcept {
    if (!frame.image || frame.metadata.dropped ||
        frame.image->nativeHandle() == 0 ||
        (frame.image->memoryKind() != ports::MemoryKind::host &&
         frame.image->memoryKind() != ports::MemoryKind::host_pinned) ||
        (frame.image->pixelFormat() != core::PixelFormat::bayer8 &&
         frame.image->pixelFormat() != core::PixelFormat::gray8) ||
        frame.image->width() != frame.metadata.roi.width ||
        frame.image->height() != frame.metadata.roi.height ||
        frame.image->pixelFormat() != frame.metadata.pixel_format) {
        return false;
    }
    const std::size_t width = frame.image->width();
    const std::size_t height = frame.image->height();
    const std::size_t stride = frame.image->strideBytes();
    return stride >= width &&
           (stride == 0 || height <= std::numeric_limits<std::size_t>::max() / stride) &&
           frame.image->sizeBytes() >= stride * height &&
           width <= static_cast<std::size_t>(
               std::numeric_limits<std::streamsize>::max());
}

bool writePgm(
    const fs::path& final_path,
    const ports::CapturedFrame& frame,
    std::string& error) {
    const fs::path partial_path = final_path.string() + ".partial";
    if (fs::exists(final_path) || fs::exists(partial_path)) {
        error = "raw image path already exists: " + final_path.string();
        return false;
    }
    std::ofstream output(partial_path, std::ios::binary | std::ios::out);
    if (!output) {
        error = "cannot create raw image: " + partial_path.string();
        return false;
    }
    output << "P5\n" << frame.image->width() << ' ' << frame.image->height()
           << "\n255\n";
    const auto* bytes = reinterpret_cast<const char*>(
        frame.image->nativeHandle());
    const auto width = static_cast<std::streamsize>(frame.image->width());
    const auto stride = frame.image->strideBytes();
    for (std::uint32_t row = 0; row < frame.image->height(); ++row) {
        output.write(bytes + static_cast<std::size_t>(row) * stride, width);
    }
    output.flush();
    output.close();
    if (!output) {
        std::error_code ignored;
        fs::remove(partial_path, ignored);
        error = "cannot finish raw image: " + partial_path.string();
        return false;
    }
    std::error_code rename_error;
    fs::rename(partial_path, final_path, rename_error);
    if (rename_error) {
        std::error_code ignored;
        fs::remove(partial_path, ignored);
        error = "cannot publish raw image: " + rename_error.message();
        return false;
    }
    return true;
}

}  // namespace

struct MainlineStageTrialWriter::Impl {
    enum class State { created, open, finalized, aborted, failed };

    explicit Impl(MainlineStageTrialWriterConfig writer_config)
        : config(std::move(writer_config)) {}

    MainlineStageTrialWriterConfig config;
    State state{State::created};
    std::string failure;
    std::ofstream frames;
    std::ofstream detections;
    std::ofstream associations;
    std::ofstream observations;
    std::set<FrameKey> frame_keys;
    std::set<CandidateKey> candidate_keys;
    std::set<CandidateKey> selected_candidates;
    std::set<CandidateKey> accepted_candidates;
    std::set<core::PairId> pair_ids;
    std::map<core::PairId, AcceptedPair> accepted_pairs;
    std::set<core::PairId> observed_pairs;
    std::set<core::ObservationId> observation_ids;
    std::optional<std::string> left_camera_id;
    std::optional<std::string> right_camera_id;
    std::size_t stereo_frames_written{0};
    std::size_t camera_frames_written{0};
    std::size_t detections_written{0};
    std::size_t associations_written{0};
    std::size_t observations_written{0};
    std::size_t raw_images_written{0};

    [[nodiscard]] fs::path partial(std::string_view filename) const {
        return config.stage_directory /
               (std::string{filename} + ".partial");
    }

    [[nodiscard]] fs::path final(std::string_view filename) const {
        return config.stage_directory / filename;
    }

    void closeStreams() noexcept {
        frames.close();
        detections.close();
        associations.close();
        observations.close();
    }

    [[nodiscard]] bool streamsGood() const noexcept {
        return static_cast<bool>(frames) && static_cast<bool>(detections) &&
               static_cast<bool>(associations) && static_cast<bool>(observations);
    }

    [[nodiscard]] bool writeStatus(bool complete) noexcept {
        try {
            const fs::path target = final(kStatusFile);
            const fs::path temporary = partial(kStatusFile);
            if (fs::exists(target) || fs::exists(temporary)) return false;
            std::ofstream output(temporary, std::ios::out);
            if (!output) return false;
            output << "{\n"
                   << "  \"stage_schema_version\": "
                   << kCaptureV21MainlineStageSchemaVersion << ",\n"
                   << "  \"complete\": " << (complete ? "true" : "false")
                   << ",\n"
                   << "  \"fatal_error\": ";
            if (failure.empty()) {
                output << "null";
            } else {
                output << jsonString(failure);
            }
            output << ",\n"
                   << "  \"stereo_frames_written\": "
                   << stereo_frames_written << ",\n"
                   << "  \"camera_frames_written\": "
                   << camera_frames_written << ",\n"
                   << "  \"detections_written\": " << detections_written
                   << ",\n"
                   << "  \"associations_written\": "
                   << associations_written << ",\n"
                   << "  \"observations_written\": "
                   << observations_written << ",\n"
                   << "  \"raw_images_written\": " << raw_images_written
                   << "\n}\n";
            output.flush();
            output.close();
            if (!output) return false;
            std::error_code rename_error;
            fs::rename(temporary, target, rename_error);
            return !rename_error;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] ports::WriteResult fail(std::string message) noexcept {
        if (failure.empty()) {
            failure = message.empty() ? "mainline staging writer failed"
                                      : std::move(message);
        }
        closeStreams();
        state = State::failed;
        const bool status_written = writeStatus(false);
        if (!status_written) failure += "; cannot write failure status";
        return {ports::WriteStatus::failed, failure};
    }

    [[nodiscard]] ports::WriteResult requireOpen() const {
        if (state == State::open && failure.empty()) {
            return {ports::WriteStatus::accepted, {}};
        }
        return {
            ports::WriteStatus::failed,
            failure.empty() ? "mainline staging writer is not open" : failure,
        };
    }

    [[nodiscard]] bool publishCsvs(std::string& error) {
        frames.flush();
        detections.flush();
        associations.flush();
        observations.flush();
        if (!streamsGood()) {
            error = "cannot flush mainline staging tables";
            return false;
        }
        closeStreams();
        for (const auto filename : {
                 kFrameFile, kDetectionFile, kAssociationFile, kObservationFile}) {
            std::error_code rename_error;
            fs::rename(partial(filename), final(filename), rename_error);
            if (rename_error) {
                error = "cannot publish staging table " + std::string{filename} +
                        ": " + rename_error.message();
                return false;
            }
        }
        return true;
    }
};

MainlineStageTrialWriter::MainlineStageTrialWriter(
    MainlineStageTrialWriterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

MainlineStageTrialWriter::~MainlineStageTrialWriter() {
    if (impl_->state == Impl::State::open) {
        abort("mainline staging writer destroyed before finalize");
    }
}

bool MainlineStageTrialWriter::open(std::string& error) {
    error.clear();
    if (impl_->state != Impl::State::created) {
        error = "mainline staging writer can only be opened once";
        return false;
    }
    try {
        const fs::path& directory = impl_->config.stage_directory;
        if (directory.empty() || !fs::is_directory(directory) ||
            !fs::is_regular_file(directory / "context.json")) {
            error = "stage directory must exist and contain context.json";
            impl_->failure = error;
            impl_->state = Impl::State::failed;
            return false;
        }
        for (const auto filename : {
                 kFrameFile, kDetectionFile, kAssociationFile,
                 kObservationFile, kStatusFile}) {
            if (fs::exists(impl_->final(filename)) ||
                fs::exists(impl_->partial(filename))) {
                error = "staging output already exists: " +
                        impl_->final(filename).string();
                impl_->failure = error;
                impl_->state = Impl::State::failed;
                return false;
            }
        }
        const fs::path raw_root = directory / "raw" / "mainline";
        if (fs::exists(raw_root)) {
            error = "mainline raw image directory already exists";
            impl_->failure = error;
            impl_->state = Impl::State::failed;
            return false;
        }
        fs::create_directories(raw_root / "left");
        fs::create_directories(raw_root / "right");

        impl_->frames.open(impl_->partial(kFrameFile), std::ios::out);
        impl_->detections.open(impl_->partial(kDetectionFile), std::ios::out);
        impl_->associations.open(
            impl_->partial(kAssociationFile), std::ios::out);
        impl_->observations.open(
            impl_->partial(kObservationFile), std::ios::out);
        if (!impl_->streamsGood()) {
            return (error = impl_->fail("cannot create mainline staging tables").message,
                    false);
        }
        writeCaptureV21MainlineFrameHeader(impl_->frames);
        writeCaptureV21MainlineDetectionHeader(impl_->detections);
        writeCaptureV21MainlineAssociationHeader(impl_->associations);
        writeCaptureV21MainlineObservationHeader(impl_->observations);
        if (!impl_->streamsGood()) {
            return (error = impl_->fail("cannot write mainline staging headers").message,
                    false);
        }
        impl_->state = Impl::State::open;
        return true;
    } catch (const std::exception& exception) {
        error = impl_->fail(
            std::string{"cannot initialize mainline staging writer: "} +
            exception.what()).message;
        return false;
    } catch (...) {
        error = impl_->fail(
            "cannot initialize mainline staging writer: unknown error").message;
        return false;
    }
}

bool MainlineStageTrialWriter::isOpen() const noexcept {
    return impl_->state == Impl::State::open && impl_->failure.empty();
}

ports::WriteResult MainlineStageTrialWriter::append(
    const ports::StereoFrame& frame) {
    if (const auto state = impl_->requireOpen();
        state.status != ports::WriteStatus::accepted) {
        return state;
    }
    const auto left_valid = core::validate(frame.left.metadata);
    const auto right_valid = core::validate(frame.right.metadata);
    if (!left_valid || !right_valid || !imageLayoutIsWritable(frame.left) ||
        !imageLayoutIsWritable(frame.right) ||
        frame.left.metadata.role != core::CameraRole::left ||
        frame.right.metadata.role != core::CameraRole::right ||
        frame.left.metadata.camera_id == frame.right.metadata.camera_id ||
        frame.left.metadata.frame_index != frame.right.metadata.frame_index) {
        return impl_->fail("invalid or non-host-readable stereo frame");
    }
    if ((impl_->left_camera_id &&
         *impl_->left_camera_id != frame.left.metadata.camera_id.value()) ||
        (impl_->right_camera_id &&
         *impl_->right_camera_id != frame.right.metadata.camera_id.value())) {
        return impl_->fail("camera identity changed within one trial");
    }
    const FrameKey left_key = frameKey(frame.left.metadata);
    const FrameKey right_key = frameKey(frame.right.metadata);
    if (impl_->frame_keys.count(left_key) != 0 ||
        impl_->frame_keys.count(right_key) != 0) {
        return impl_->fail("duplicate camera frame in mainline staging");
    }

    const std::string stem = std::to_string(frame.left.metadata.frame_index) + ".pgm";
    const fs::path left_relative = fs::path("raw") / "mainline" / "left" / stem;
    const fs::path right_relative = fs::path("raw") / "mainline" / "right" / stem;
    std::string error;
    if (!writePgm(
            impl_->config.stage_directory / left_relative, frame.left, error)) {
        return impl_->fail(std::move(error));
    }
    if (!writePgm(
            impl_->config.stage_directory / right_relative, frame.right, error)) {
        std::error_code ignored;
        fs::remove(impl_->config.stage_directory / left_relative, ignored);
        return impl_->fail(std::move(error));
    }

    std::ostringstream rows;
    if (!writeCaptureV21MainlineFrameRow(
            rows, frame.left, left_relative.generic_string()) ||
        !writeCaptureV21MainlineFrameRow(
            rows, frame.right, right_relative.generic_string())) {
        std::error_code ignored;
        fs::remove(impl_->config.stage_directory / left_relative, ignored);
        fs::remove(impl_->config.stage_directory / right_relative, ignored);
        return impl_->fail("cannot serialize complete stereo frame transaction");
    }
    impl_->frames << rows.str();
    if (!impl_->frames) {
        return impl_->fail("cannot append stereo frame transaction");
    }
    impl_->left_camera_id = frame.left.metadata.camera_id.value();
    impl_->right_camera_id = frame.right.metadata.camera_id.value();
    impl_->frame_keys.insert(left_key);
    impl_->frame_keys.insert(right_key);
    ++impl_->stereo_frames_written;
    impl_->camera_frames_written += 2;
    impl_->raw_images_written += 2;
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult MainlineStageTrialWriter::append(
    const core::Detection2D& record) {
    if (const auto state = impl_->requireOpen();
        state.status != ports::WriteStatus::accepted) {
        return state;
    }
    const CandidateKey key = candidateKey(record);
    if (!core::validate(record) || impl_->frame_keys.count(key.frame) == 0 ||
        impl_->candidate_keys.count(key) != 0) {
        return impl_->fail("invalid, duplicate, or orphaned detection");
    }
    std::ostringstream row;
    if (!writeCaptureV21MainlineDetectionRow(row, record)) {
        return impl_->fail("cannot serialize detection");
    }
    impl_->detections << row.str();
    if (!impl_->detections) return impl_->fail("cannot append detection");
    impl_->candidate_keys.insert(key);
    if (record.selected) impl_->selected_candidates.insert(key);
    ++impl_->detections_written;
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult MainlineStageTrialWriter::append(
    const core::StereoAssociationAttempt& record) {
    if (const auto state = impl_->requireOpen();
        state.status != ports::WriteStatus::accepted) {
        return state;
    }
    if (!impl_->left_camera_id || !impl_->right_camera_id) {
        return impl_->fail("association arrived before stereo frames");
    }
    const CandidateKey left{
        {*impl_->left_camera_id, core::CameraRole::left, record.left_frame_index},
        record.left_candidate_id,
    };
    const CandidateKey right{
        {*impl_->right_camera_id, core::CameraRole::right, record.right_frame_index},
        record.right_candidate_id,
    };
    if (!core::validate(record) || impl_->pair_ids.count(record.pair_id) != 0 ||
        impl_->candidate_keys.count(left) == 0 ||
        impl_->candidate_keys.count(right) == 0) {
        return impl_->fail("invalid, duplicate, or orphaned stereo association");
    }
    std::ostringstream row;
    if (!writeCaptureV21MainlineAssociationRow(row, record)) {
        return impl_->fail("cannot serialize stereo association");
    }
    impl_->associations << row.str();
    if (!impl_->associations) {
        return impl_->fail("cannot append stereo association");
    }
    impl_->pair_ids.insert(record.pair_id);
    if (record.decision == core::AssociationDecision::accepted) {
        impl_->accepted_pairs.emplace(
            record.pair_id,
            AcceptedPair{left, right, record.left_capture, record.right_capture});
        impl_->accepted_candidates.insert(left);
        impl_->accepted_candidates.insert(right);
    }
    ++impl_->associations_written;
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult MainlineStageTrialWriter::append(
    const ports::EstimatedDirectObservation& record) {
    if (const auto state = impl_->requireOpen();
        state.status != ports::WriteStatus::accepted) {
        return state;
    }
    const auto& measurement = record.measurement;
    const auto pair = impl_->accepted_pairs.find(measurement.source_pair_id);
    if (!core::validate(measurement) || !ports::isValid(record.evidence) ||
        pair == impl_->accepted_pairs.end() ||
        impl_->observed_pairs.count(measurement.source_pair_id) != 0 ||
        impl_->observation_ids.count(measurement.observation_id) != 0 ||
        measurement.capture_reference.clock_domain !=
            pair->second.left_capture.clock_domain) {
        return impl_->fail("invalid, duplicate, or orphaned direct observation");
    }
    std::ostringstream row;
    if (!writeCaptureV21MainlineObservationRow(row, record)) {
        return impl_->fail("cannot serialize direct observation");
    }
    impl_->observations << row.str();
    if (!impl_->observations) {
        return impl_->fail("cannot append direct observation");
    }
    impl_->observed_pairs.insert(measurement.source_pair_id);
    impl_->observation_ids.insert(measurement.observation_id);
    ++impl_->observations_written;
    return {ports::WriteStatus::accepted, {}};
}

ports::WriteResult MainlineStageTrialWriter::finalize() {
    if (const auto state = impl_->requireOpen();
        state.status != ports::WriteStatus::accepted) {
        return state;
    }
    if (impl_->stereo_frames_written == 0) {
        return impl_->fail("cannot finalize an empty capture stage");
    }
    if (impl_->selected_candidates != impl_->accepted_candidates) {
        return impl_->fail(
            "selected detections do not match accepted associations");
    }
    std::set<core::PairId> accepted_pair_ids;
    for (const auto& entry : impl_->accepted_pairs) {
        accepted_pair_ids.insert(entry.first);
    }
    if (accepted_pair_ids != impl_->observed_pairs) {
        return impl_->fail(
            "accepted associations do not have exactly one direct observation");
    }
    std::string error;
    if (!impl_->publishCsvs(error)) return impl_->fail(std::move(error));
    if (!impl_->writeStatus(true)) {
        return impl_->fail("cannot publish complete mainline staging status");
    }
    impl_->state = Impl::State::finalized;
    return {ports::WriteStatus::accepted, {}};
}

void MainlineStageTrialWriter::abort(std::string reason) noexcept {
    if (impl_->state == Impl::State::finalized ||
        impl_->state == Impl::State::aborted) {
        return;
    }
    if (impl_->failure.empty()) {
        impl_->failure = reason.empty() ? "mainline staging writer aborted"
                                        : std::move(reason);
    }
    impl_->closeStreams();
    impl_->state = Impl::State::aborted;
    static_cast<void>(impl_->writeStatus(false));
}

}  // namespace volleyball::adapters::capture_v21
