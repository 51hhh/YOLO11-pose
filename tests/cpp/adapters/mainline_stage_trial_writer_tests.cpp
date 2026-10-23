#include "volleyball/adapters/capture_stage/mainline_stage_trial_writer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace capture_stage = volleyball::adapters::capture_stage;
namespace core = volleyball::core;
namespace ports = volleyball::ports;
namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

class TempStage {
public:
    TempStage() {
        const auto seed = std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = fs::temp_directory_path() /
                    ("volleyball-mainline-stage-" + std::to_string(seed) + "-" +
                     std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error)) break;
            path_.clear();
        }
        if (path_.empty()) throw std::runtime_error("cannot create test stage");
        std::ofstream context(path_ / "context.json");
        context << "{\"stage_schema_version\":6}\n";
    }

    ~TempStage() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

class HostImage final : public ports::ImageBuffer {
public:
    explicit HostImage(ports::MemoryKind memory_kind = ports::MemoryKind::host)
        : memory_kind_(memory_kind), bytes_(18) {
        for (std::size_t index = 0; index < bytes_.size(); ++index) {
            bytes_[index] = static_cast<std::uint8_t>(index + 1U);
        }
    }

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return memory_kind_;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override { return 4; }
    [[nodiscard]] std::uint32_t height() const noexcept override { return 3; }
    [[nodiscard]] std::size_t strideBytes() const noexcept override { return 6; }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::gray8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(bytes_.data());
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return bytes_.size();
    }

private:
    ports::MemoryKind memory_kind_;
    std::vector<std::uint8_t> bytes_;
};

core::CameraFrameRecord frame(
    core::CameraRole role,
    const char* camera_id,
    std::int64_t capture_ns) {
    core::CameraFrameRecord output;
    output.camera_id = core::CameraId{camera_id};
    output.role = role;
    output.frame_index = 42;
    output.hardware_timestamp = {
        100, core::ClockDomainId{std::string{camera_id} + "-device"}};
    output.exposure_start = {
        capture_ns - 50, core::ClockDomainId{"nx-realtime"}};
    output.capture_timestamp = {
        capture_ns, core::ClockDomainId{"nx-realtime"}};
    output.receive_timestamp = {
        capture_ns + 100, core::ClockDomainId{"nx-realtime"}};
    output.exposure_us = 0.1;
    output.gain_db = 1.0;
    output.roi = {0, 0, 4, 3};
    output.pixel_format = core::PixelFormat::gray8;
    output.acquisition.receive_monotonic_timestamp = {
        capture_ns - 10, core::ClockDomainId{"nx-monotonic"}};
    output.acquisition.sensor_frame_counter = 42;
    return output;
}

ports::StereoFrame stereoFrame(
    ports::MemoryKind memory_kind = ports::MemoryKind::host) {
    return {
        {frame(core::CameraRole::left, "left-camera", 1000),
         std::make_shared<HostImage>(memory_kind)},
        {frame(core::CameraRole::right, "right-camera", 1010),
         std::make_shared<HostImage>(memory_kind)},
    };
}

core::Detection2D detection(
    core::CameraRole role,
    const char* camera_id,
    core::CandidateId candidate_id) {
    core::Detection2D output;
    output.frame = {core::CameraId{camera_id}, role, 42};
    output.candidate_id = candidate_id;
    output.center_px = role == core::CameraRole::left
        ? core::Vec2{3.0, 1.0}
        : core::Vec2{1.0, 1.0};
    output.radius_px = 1.0;
    output.bounding_box = {0.0, 0.0, 4.0, 3.0};
    output.score = 0.9;
    output.selected = true;
    output.source_pipeline = core::PipelineId{"detector-a"};
    return output;
}

core::StereoAssociationAttempt acceptedAssociation() {
    core::StereoAssociationAttempt output;
    output.pair_id = 7;
    output.left_frame_index = 42;
    output.right_frame_index = 42;
    output.left_candidate_id = 1;
    output.right_candidate_id = 2;
    output.left_capture = {1000, core::ClockDomainId{"nx-realtime"}};
    output.right_capture = {1010, core::ClockDomainId{"nx-realtime"}};
    output.delta_t_lr_ns = 10;
    output.disparity_px = 2.0;
    output.epipolar_error_px = 0.0;
    output.decision = core::AssociationDecision::accepted;
    output.reject_reason = core::AssociationRejectReason::none;
    return output;
}

ports::EstimatedDirectObservation observation() {
    ports::EstimatedDirectObservation output;
    output.measurement.observation_id = 3;
    output.measurement.source_pair_id = 7;
    output.measurement.capture_reference = {
        1005, core::ClockDomainId{"nx-realtime"}};
    output.measurement.coordinate_frame =
        core::CoordinateFrameId{"stereo-rig"};
    output.measurement.position_m = {1.0, 2.0, 3.0};
    output.measurement.confidence = 0.9;
    output.measurement.reprojection_error_px = 0.0;
    output.evidence.method = core::StereoDepthMethodId{"circle_center"};
    output.evidence.disparity_px = 2.1;
    output.evidence.disparity_stddev_px = 0.1;
    output.evidence.support_points = 8;
    output.evidence.left_anchor_px = core::Vec2{100.0, 50.0};
    return output;
}

std::string readText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::size_t lineCount(const fs::path& path) {
    const std::string text = readText(path);
    std::size_t count = 0;
    for (const char character : text) {
        if (character == '\n') ++count;
    }
    return count;
}

void testCompleteTransactionPublishesMainlineStage() {
    TempStage stage;
    capture_stage::MainlineStageTrialWriter writer({stage.path()});
    std::string error;
    check(writer.open(error), "writer must open an exclusive prepared stage");
    check(writer.append(stereoFrame()).status == ports::WriteStatus::accepted,
          "writer must accept a complete host stereo pair");
    check(writer.append(detection(core::CameraRole::left, "left-camera", 1)).status ==
              ports::WriteStatus::accepted,
          "writer must accept the left selected detection");
    check(writer.append(detection(core::CameraRole::right, "right-camera", 2)).status ==
              ports::WriteStatus::accepted,
          "writer must accept the right selected detection");
    check(writer.append(acceptedAssociation()).status == ports::WriteStatus::accepted,
          "writer must accept an association referencing stored candidates");
    check(writer.append(observation()).status == ports::WriteStatus::accepted,
          "writer must accept the direct observation for the accepted pair");
    check(writer.finalize().status == ports::WriteStatus::accepted,
          "consistent evidence must finalize");
    check(!writer.isOpen(), "finalized writer must close its mutation boundary");

    check(fs::is_regular_file(stage.path() / "frames_v6.csv") &&
              lineCount(stage.path() / "frames_v6.csv") == 3,
          "published frame table must contain one header and two camera rows");
    check(fs::is_regular_file(stage.path() / "detections_v6.csv") &&
              lineCount(stage.path() / "detections_v6.csv") == 3,
          "published detection table must preserve both candidates");
    check(fs::is_regular_file(stage.path() / "associations_v6.csv") &&
              fs::is_regular_file(stage.path() / "observations_v6.csv"),
          "association and observation facts must remain separate tables");
    check(
        readText(stage.path() / "observations_v6.csv").find(
            "circle_center,2.1000000000000001,0.10000000000000001,8") !=
            std::string::npos,
        "observation table must preserve the selected depth evidence");
    check(
        readText(stage.path() / "stereo_measurements_v1.csv") ==
            "schema_version,pair_id,method,anchor_u_px,anchor_v_px,disparity_px\n"
            "1.0.0,7,circle_center,100,50,2.1000000000000001\n",
        "measurement extension must preserve the actual versioned UVD inputs");
    const std::string status = readText(stage.path() / "status_v6.json");
    check(status.find("\"complete\": true") != std::string::npos &&
              status.find("\"raw_images_written\": 2") != std::string::npos,
          "completion status must be written last with exact counts");
    check(!fs::exists(stage.path() / "frames_v6.csv.partial"),
          "successful finalization must leave no partial table");

    const std::string pgm = readText(
        stage.path() / "raw" / "mainline" / "left" / "42.pgm");
    const std::string header = "P5\n4 3\n255\n";
    check(pgm.size() == header.size() + 12 &&
              pgm.compare(0, header.size(), header) == 0,
          "PGM must strip host row padding and contain exactly width*height bytes");
    const std::vector<unsigned char> expected = {
        1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16};
    bool pixels_match = pgm.size() == header.size() + expected.size();
    for (std::size_t index = 0; pixels_match && index < expected.size(); ++index) {
        pixels_match = static_cast<unsigned char>(pgm[header.size() + index]) ==
                       expected[index];
    }
    check(pixels_match, "PGM rows must preserve pixels while dropping stride padding");
}

void testWriterRejectsNonHostImageWithoutHalfFrame() {
    TempStage stage;
    capture_stage::MainlineStageTrialWriter writer({stage.path()});
    std::string error;
    check(writer.open(error), "writer must open for failure-path test");
    check(writer.append(stereoFrame(ports::MemoryKind::cuda_device)).status ==
              ports::WriteStatus::failed,
          "writer must reject device memory without an explicit copy adapter");
    const std::string status = readText(stage.path() / "status_v6.json");
    check(status.find("\"complete\": false") != std::string::npos &&
              status.find("\"camera_frames_written\": 0") != std::string::npos,
          "failure status must prove that no half stereo frame was committed");
    check(!fs::exists(
              stage.path() / "raw" / "mainline" / "left" / "42.pgm"),
          "invalid pair must not publish a left-only raw image");
}

void testAcceptedPairRequiresDirectObservation() {
    TempStage stage;
    capture_stage::MainlineStageTrialWriter writer({stage.path()});
    std::string error;
    check(writer.open(error), "writer must open for relation test");
    check(writer.append(stereoFrame()).status == ports::WriteStatus::accepted,
          "relation test frame must be accepted");
    check(writer.append(detection(core::CameraRole::left, "left-camera", 1)).status ==
              ports::WriteStatus::accepted,
          "relation test left detection must be accepted");
    check(writer.append(detection(core::CameraRole::right, "right-camera", 2)).status ==
              ports::WriteStatus::accepted,
          "relation test right detection must be accepted");
    check(writer.append(acceptedAssociation()).status == ports::WriteStatus::accepted,
          "relation test association must be accepted");
    check(writer.finalize().status == ports::WriteStatus::failed,
          "accepted pair without direct observation must not finalize");
    check(readText(stage.path() / "status_v6.json").find(
              "\"complete\": false") != std::string::npos,
          "failed relation must leave an explicitly incomplete stage");
}

void testObservationRequiresReplayableInputAnchor() {
    TempStage stage;
    capture_stage::MainlineStageTrialWriter writer({stage.path()});
    std::string error;
    check(writer.open(error), "writer must open for measurement evidence test");
    check(writer.append(stereoFrame()).status == ports::WriteStatus::accepted,
          "measurement evidence frame must be accepted");
    check(writer.append(detection(core::CameraRole::left, "left-camera", 1)).status ==
              ports::WriteStatus::accepted,
          "measurement evidence left detection must be accepted");
    check(writer.append(detection(core::CameraRole::right, "right-camera", 2)).status ==
              ports::WriteStatus::accepted,
          "measurement evidence right detection must be accepted");
    check(writer.append(acceptedAssociation()).status ==
              ports::WriteStatus::accepted,
          "measurement evidence association must be accepted");
    auto missing_anchor = observation();
    missing_anchor.evidence.left_anchor_px.reset();
    check(writer.append(missing_anchor).status == ports::WriteStatus::failed,
          "writer must reject 3D output without its measured input anchor");
    check(readText(stage.path() / "status_v6.json").find(
              "\"complete\": false") != std::string::npos,
          "missing measurement input must fail the complete trial");
}

}  // namespace

int main() {
    testCompleteTransactionPublishesMainlineStage();
    testWriterRejectsNonHostImageWithoutHalfFrame();
    testAcceptedPairRequiresDirectObservation();
    testObservationRequiresReplayableInputAnchor();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "mainline stage trial writer tests passed\n";
    return EXIT_SUCCESS;
}
