#include "volleyball/services/stereo_observation_service.hpp"

#include "stereo_detector_executor.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace volleyball::services {
namespace {

bool writeAccepted(const ports::WriteResult& result, std::string& error) {
    if (result.status == ports::WriteStatus::accepted) {
        return true;
    }
    error = result.message.empty() ? "trial writer rejected a record" : result.message;
    return false;
}

core::Detection2D* findCandidate(
    std::vector<core::Detection2D>& candidates,
    core::CandidateId candidate_id) noexcept {
    for (auto& candidate : candidates) {
        if (candidate.candidate_id == candidate_id) {
            return &candidate;
        }
    }
    return nullptr;
}

std::size_t bytesPerPixel(core::PixelFormat format) noexcept {
    switch (format) {
    case core::PixelFormat::bayer8:
    case core::PixelFormat::gray8:
        return 1;
    case core::PixelFormat::bgr8:
    case core::PixelFormat::rgb8:
        return 3;
    }
    return 0;
}

bool imageMatchesContract(
    const ports::ImageBufferPtr& image,
    std::uint32_t width,
    std::uint32_t height,
    core::PixelFormat pixel_format) noexcept {
    if (!image || image->nativeHandle() == 0 || image->width() != width ||
        image->height() != height || image->pixelFormat() != pixel_format) {
        return false;
    }
    const std::size_t channels = bytesPerPixel(pixel_format);
    const std::size_t image_width = image->width();
    const std::size_t image_height = image->height();
    if (channels == 0 ||
        image_width > std::numeric_limits<std::size_t>::max() / channels) {
        return false;
    }
    const std::size_t minimum_stride = image_width * channels;
    const std::size_t stride = image->strideBytes();
    if (stride < minimum_stride ||
        (stride != 0 &&
         image_height > std::numeric_limits<std::size_t>::max() / stride)) {
        return false;
    }
    return image->sizeBytes() >= stride * image_height;
}

bool imageMatchesMetadata(const ports::CapturedFrame& frame) noexcept {
    return !frame.metadata.dropped && imageMatchesContract(
        frame.image,
        frame.metadata.roi.width,
        frame.metadata.roi.height,
        frame.metadata.pixel_format);
}

bool referencesFrame(
    const core::FrameReference& reference,
    const core::CameraFrameRecord& frame) noexcept {
    return reference.camera_id == frame.camera_id &&
           reference.role == frame.role &&
           reference.frame_index == frame.frame_index;
}

bool detectionInsideImage(
    const core::Detection2D& detection,
    const core::CameraFrameRecord& frame) noexcept {
    const double width = static_cast<double>(frame.roi.width);
    const double height = static_cast<double>(frame.roi.height);
    if (detection.center_px.x < 0.0 || detection.center_px.x >= width ||
        detection.center_px.y < 0.0 || detection.center_px.y >= height) {
        return false;
    }
    if (detection.radius_px &&
        *detection.radius_px > std::max(width, height)) {
        return false;
    }
    if (!detection.bounding_box) {
        return true;
    }
    const auto& box = *detection.bounding_box;
    return box.width <= width && box.height <= height && box.x < width &&
           box.y < height && box.x + box.width > 0.0 &&
           box.y + box.height > 0.0 && detection.center_px.x >= box.x &&
           detection.center_px.x <= box.x + box.width &&
           detection.center_px.y >= box.y &&
           detection.center_px.y <= box.y + box.height;
}

std::int64_t integerMidpoint(
    std::int64_t left,
    std::int64_t right) noexcept {
    return left / 2 + right / 2 + (left % 2 + right % 2) / 2;
}

bool validEstimationResult(
    const ports::DirectObservationResult& result,
    const core::StereoAssociationAttempt& association,
    core::ObservationId observation_id) noexcept {
    if (result.status == ports::DirectObservationStatus::rejected ||
        result.status == ports::DirectObservationStatus::failed) {
        return !result.observation.has_value() &&
               !result.attempted_method.empty();
    }
    if (result.status != ports::DirectObservationStatus::observation) {
        return false;
    }
    if (!result.observation ||
        !ports::isValid(result.observation->evidence) ||
        !core::validate(result.observation->measurement)) {
        return false;
    }
    const auto& observation = result.observation->measurement;
    const auto expected_capture = integerMidpoint(
        association.left_capture.nanoseconds,
        association.right_capture.nanoseconds);
    return observation.observation_id == observation_id &&
           observation.source_pair_id == association.pair_id &&
           observation.capture_reference.clock_domain ==
               association.left_capture.clock_domain &&
           observation.capture_reference.nanoseconds == expected_capture &&
           observation.reprojection_error_px == association.epipolar_error_px;
}

}  // namespace

StereoObservationService::StereoObservationService(
    StereoObservationServiceConfig config,
    ports::StereoPreprocessor& preprocessor,
    ports::Detector& left_detector,
    ports::Detector& right_detector,
    ports::DirectObservationEstimator& estimator,
    ports::TrialWriter& writer,
    ports::ObservationPublisher* publisher)
    : config_(std::move(config)),
      preprocessor_(preprocessor),
      detector_executor_(std::make_unique<StereoDetectorExecutor>(
          left_detector, right_detector)),
      estimator_(estimator),
      writer_(writer),
      publisher_(publisher) {
    if (!algorithms::stereo::isValidAssociationConfig(config_.association)) {
        throw std::invalid_argument("invalid stereo association configuration");
    }
    const auto& gate = config_.gate;
    if (!std::isfinite(gate.min_depth_m) || gate.min_depth_m < 0.0 ||
        !std::isfinite(gate.max_depth_m) ||
        gate.max_depth_m <= gate.min_depth_m ||
        !std::isfinite(gate.max_reprojection_error_px) ||
        gate.max_reprojection_error_px < 0.0) {
        throw std::invalid_argument("invalid stereo observation quality gate");
    }
}

StereoObservationService::~StereoObservationService() = default;

StereoProcessResult StereoObservationService::fail(
    ProcessStatus status,
    std::string message) noexcept {
    if (message.empty()) {
        message = "stereo observation service failed";
    }
    abort(message);
    return {status, std::nullopt, {}, std::move(message)};
}

bool StereoObservationService::appendDetections(
    const std::vector<core::Detection2D>& detections,
    std::string& error) {
    for (const auto& detection : detections) {
        if (!writeAccepted(writer_.append(detection), error)) {
            return false;
        }
    }
    return true;
}

bool StereoObservationService::appendAssociations(
    const std::vector<algorithms::stereo::ScoredAssociation>& associations,
    std::string& error) {
    for (const auto& scored : associations) {
        if (!writeAccepted(writer_.append(scored.attempt), error)) {
            return false;
        }
    }
    return true;
}

bool StereoObservationService::validateDetectionBatch(
    const ports::DetectionBatch& batch,
    const core::CameraFrameRecord& frame,
    std::string& error) const {
    const bool batch_matches_frame =
        batch.frame.camera_id == frame.camera_id &&
        batch.frame.role == frame.role &&
        batch.frame.frame_index == frame.frame_index;
    if (!batch_matches_frame || batch.pipeline_id.empty()) {
        error = "detector batch does not identify its input frame/pipeline";
        return false;
    }

    std::unordered_set<core::CandidateId> candidate_ids;
    for (const auto& candidate : batch.candidates) {
        if (!core::validate(candidate) ||
            candidate.frame.camera_id != batch.frame.camera_id ||
            candidate.frame.role != batch.frame.role ||
            candidate.frame.frame_index != batch.frame.frame_index ||
            candidate.source_pipeline != batch.pipeline_id ||
            !detectionInsideImage(candidate, frame)) {
            error =
                "detector candidate is invalid, out of image, or does not "
                "belong to its declared batch";
            return false;
        }
        if (!candidate_ids.insert(candidate.candidate_id).second) {
            error = "detector candidate IDs must be unique within one frame";
            return false;
        }
    }
    return true;
}

bool StereoObservationService::validateRectifiedFrame(
    const ports::RectifiedStereoFrame& rectified,
    const ports::StereoFrame& source,
    std::string& error) const {
    const auto valid_camera = [](
                                  const ports::RectifiedCameraFrame& output,
                                  const ports::CapturedFrame& input) {
        const auto& roi = input.metadata.roi;
        return referencesFrame(output.source, input.metadata) &&
               imageMatchesContract(
                   output.detector_input,
                   roi.width,
                   roi.height,
                   core::PixelFormat::bgr8) &&
               imageMatchesContract(
                   output.geometry_gray,
                   roi.width,
                   roi.height,
                   core::PixelFormat::gray8);
    };
    if (!valid_camera(rectified.left, source.left) ||
        !valid_camera(rectified.right, source.right)) {
        error =
            "preprocessor output must preserve frame identity and provide "
            "full-size rectified BGR8/Gray8 images";
        return false;
    }
    return true;
}

bool StereoObservationService::passesGate(
    const core::TriangulatedObservation& observation) const noexcept {
    const double depth = observation.position_m.z;
    const double reprojection = observation.reprojection_error_px.value_or(0.0);
    return std::isfinite(depth) && depth >= config_.gate.min_depth_m &&
           depth <= config_.gate.max_depth_m &&
           std::isfinite(reprojection) &&
           reprojection <= config_.gate.max_reprojection_error_px;
}

StereoProcessResult StereoObservationService::process(
    const ports::StereoFrame& frame) {
    if (closed_) {
        return {ProcessStatus::write_error, std::nullopt, {}, "service is closed"};
    }
    const auto left_valid = core::validate(frame.left.metadata);
    const auto right_valid = core::validate(frame.right.metadata);
    if (!left_valid || !right_valid ||
        !imageMatchesMetadata(frame.left) ||
        !imageMatchesMetadata(frame.right) ||
        frame.left.metadata.role != core::CameraRole::left ||
        frame.right.metadata.role != core::CameraRole::right ||
        frame.left.metadata.camera_id == frame.right.metadata.camera_id ||
        frame.left.metadata.frame_index != frame.right.metadata.frame_index) {
        return fail(ProcessStatus::invalid_input, "invalid left/right stereo frame");
    }
    if (frame.left.metadata.capture_timestamp.clock_domain !=
            frame.right.metadata.capture_timestamp.clock_domain) {
        return fail(
            ProcessStatus::invalid_input,
            "stereo frames must use one mapped capture clock domain");
    }

    std::string error;
    if (!writeAccepted(writer_.append(frame), error)) {
        return fail(ProcessStatus::write_error, std::move(error));
    }

    auto preprocess_result = preprocessor_.prepare(frame);
    if (!preprocess_result.success || !preprocess_result.frame) {
        return fail(
            ProcessStatus::preprocess_error,
            preprocess_result.message.empty()
                ? "stereo preprocessing failed"
                : std::move(preprocess_result.message));
    }
    if (!validateRectifiedFrame(*preprocess_result.frame, frame, error)) {
        return fail(ProcessStatus::preprocess_error, std::move(error));
    }

    auto [left_result, right_result] =
        detector_executor_->detect(*preprocess_result.frame);
    if (!left_result.success || !right_result.success) {
        const std::string message = !left_result.success
            ? left_result.message
            : right_result.message;
        return fail(
            ProcessStatus::detector_error,
            message.empty() ? "stereo detector failed" : message);
    }
    if (!validateDetectionBatch(
            left_result.batch, frame.left.metadata, error) ||
        !validateDetectionBatch(
            right_result.batch, frame.right.metadata, error)) {
        return fail(ProcessStatus::detector_error, std::move(error));
    }
    for (auto& candidate : left_result.batch.candidates) {
        candidate.selected = false;
    }
    for (auto& candidate : right_result.batch.candidates) {
        candidate.selected = false;
    }

    auto associations = algorithms::stereo::associateCandidates(
        left_result.batch.candidates,
        right_result.batch.candidates,
        frame.left.metadata.capture_timestamp,
        frame.right.metadata.capture_timestamp,
        config_.association,
        next_pair_id_);
    if (!associations.ok()) {
        return fail(ProcessStatus::invalid_input, std::move(associations.message));
    }
    next_pair_id_ += associations.attempts.size();
    auto appendEvidence = [&]() {
        return appendDetections(left_result.batch.candidates, error) &&
               appendDetections(right_result.batch.candidates, error) &&
               appendAssociations(associations.attempts, error);
    };
    if (!associations.best_accepted_index) {
        if (!appendEvidence()) {
            return fail(ProcessStatus::write_error, std::move(error));
        }
        return {
            ProcessStatus::no_match,
            std::nullopt,
            std::move(associations.attempts),
            "no stereo association passed all gates",
        };
    }

    const auto selected_index = *associations.best_accepted_index;
    for (std::size_t index = 0; index < associations.attempts.size(); ++index) {
        auto& attempt = associations.attempts[index].attempt;
        if (index != selected_index &&
            attempt.decision == core::AssociationDecision::accepted) {
            attempt.decision = core::AssociationDecision::rejected;
            attempt.reject_reason = core::AssociationRejectReason::not_selected;
        }
    }

    auto& selected = associations.attempts[selected_index].attempt;
    auto* left_candidate = findCandidate(
        left_result.batch.candidates, selected.left_candidate_id);
    auto* right_candidate = findCandidate(
        right_result.batch.candidates, selected.right_candidate_id);
    if (!left_candidate || !right_candidate) {
        return fail(ProcessStatus::invalid_input, "selected candidate disappeared");
    }
    auto estimated = estimator_.estimate({
        *preprocess_result.frame,
        *left_candidate,
        *right_candidate,
        selected,
        next_observation_id_,
    });
    if (!validEstimationResult(estimated, selected, next_observation_id_)) {
        return fail(
            ProcessStatus::estimator_error,
            "direct observation estimator returned an invalid result");
    }
    if (estimated.status == ports::DirectObservationStatus::failed) {
        return fail(
            ProcessStatus::estimator_error,
            estimated.message.empty()
                ? "direct observation estimator failed"
                : std::move(estimated.message));
    }
    if (estimated.status == ports::DirectObservationStatus::rejected) {
        selected.decision = core::AssociationDecision::rejected;
        selected.reject_reason =
            core::AssociationRejectReason::triangulation_failure;
        if (!appendEvidence()) {
            return fail(ProcessStatus::write_error, std::move(error));
        }
        return {
            ProcessStatus::no_match,
            std::nullopt,
            std::move(associations.attempts),
            estimated.message.empty()
                ? "direct observation estimator rejected the pair"
                : std::move(estimated.message),
        };
    }
    if (!passesGate(estimated.observation->measurement)) {
        selected.decision = core::AssociationDecision::rejected;
        selected.reject_reason = core::AssociationRejectReason::quality_gate;
        if (!appendEvidence()) {
            return fail(ProcessStatus::write_error, std::move(error));
        }
        return {
            ProcessStatus::no_match,
            std::nullopt,
            std::move(associations.attempts),
            "triangulated observation failed quality gate",
        };
    }
    left_candidate->selected = true;
    right_candidate->selected = true;
    if (!appendEvidence()) {
        return fail(ProcessStatus::write_error, std::move(error));
    }
    const auto& estimated_observation = *estimated.observation;
    const auto& observation = estimated_observation.measurement;
    if (!writeAccepted(writer_.append(estimated_observation), error)) {
        return fail(ProcessStatus::write_error, std::move(error));
    }
    ++next_observation_id_;
    if (publisher_) {
        ports::ObservationPublication publication;
        publication.observation.measurement = observation;
        publication.observation.status = core::ValidationStatus::accepted;
        publication.left_frame = frame.left.metadata;
        publication.right_frame = frame.right.metadata;
        publication.left_detection = *left_candidate;
        publication.right_detection = *right_candidate;
        publication.association = selected;
        publication.estimation = estimated_observation.evidence;
        const auto published = publisher_->publish(publication);
        if (!published.success) {
            return fail(
                ProcessStatus::publish_error,
                published.message.empty() ? "observation publish failed"
                                          : published.message);
        }
    }
    return {
        ProcessStatus::observation,
        observation,
        std::move(associations.attempts),
        {},
    };
}

ports::WriteResult StereoObservationService::finalize() {
    if (closed_) {
        return {ports::WriteStatus::failed, "service is closed"};
    }
    auto result = writer_.finalize();
    if (result.status == ports::WriteStatus::accepted) {
        closed_ = true;
        return result;
    }
    if (result.message.empty()) {
        result.message = "trial writer finalize failed";
    }
    abort(result.message);
    return result;
}

void StereoObservationService::abort(std::string reason) noexcept {
    if (closed_) {
        return;
    }
    if (reason.empty()) {
        reason = "stereo observation service aborted";
    }
    writer_.abort(std::move(reason));
    closed_ = true;
}

}  // namespace volleyball::services
