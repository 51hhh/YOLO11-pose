#include "assembly.hpp"

#include "volleyball/adapters/capture_stage/mainline_stage_trial_writer.hpp"
#include "volleyball/adapters/cuda_depth/direct_observation_estimator.hpp"
#include "volleyball/adapters/cuda_preprocess/stereo_calibration.hpp"
#include "volleyball/adapters/cuda_preprocess/stereo_preprocessor.hpp"
#include "volleyball/adapters/gpio/pwm_trigger.hpp"
#include "volleyball/adapters/gpio/trigger_events.hpp"
#include "volleyball/adapters/hik_mvs/hik_stereo_camera_source.hpp"
#include "volleyball/adapters/ros2/observation_publisher.hpp"
#include "volleyball/adapters/ros2/source_epoch.hpp"
#include "volleyball/adapters/tensorrt/detector_adapter.hpp"
#include "volleyball/services/managed_stereo_trial.hpp"
#include "volleyball/services/stereo_observation_service.hpp"

#include <openssl/evp.h>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace volleyball::apps::nx_mainline {
namespace {

namespace capture = adapters::capture_stage;
namespace cuda_depth = adapters::cuda_depth;
namespace cuda_preprocess = adapters::cuda_preprocess;
namespace gpio = adapters::gpio;
namespace hik = adapters::hik_mvs;
namespace ros2 = adapters::ros2;
namespace tensorrt = adapters::tensorrt;

tensorrt::DetectorOutputContract outputContract(const std::string& value) {
    if (value == "yolo_pre_nms") {
        return tensorrt::DetectorOutputContract::yolo_pre_nms;
    }
    if (value == "yolo_dfl") {
        return tensorrt::DetectorOutputContract::yolo_dfl;
    }
    throw std::runtime_error("unsupported detector output contract: " + value);
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot open asset for SHA-256: " + path.string());
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("cannot initialize SHA-256 verifier");
    }
    std::array<char, 64U * 1024U> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 && EVP_DigestUpdate(
                             context.get(),
                             buffer.data(),
                             static_cast<std::size_t>(count)) != 1) {
            throw std::runtime_error("cannot update model SHA-256 verifier");
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "cannot read asset for SHA-256: " + path.string());
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;
    if (EVP_DigestFinal_ex(
            context.get(), digest.data(), &digest_size) != 1 ||
        digest_size != 32U) {
        throw std::runtime_error("cannot finalize model SHA-256 verifier");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

double matrixValue(const cv::Mat& matrix, int row, int column) {
    if (matrix.channels() != 1) {
        throw std::runtime_error("calibration projection must be single-channel");
    }
    if (matrix.depth() == CV_64F) return matrix.at<double>(row, column);
    if (matrix.depth() == CV_32F) {
        return static_cast<double>(matrix.at<float>(row, column));
    }
    throw std::runtime_error("calibration projection must use float32/float64");
}

core::PinholeIntrinsics intrinsics(const cv::Mat& projection) {
    if (projection.rows != 3 || projection.cols != 4) {
        throw std::runtime_error("calibration projection must be 3x4");
    }
    return {
        matrixValue(projection, 0, 0),
        matrixValue(projection, 1, 1),
        matrixValue(projection, 0, 2),
        matrixValue(projection, 1, 2),
    };
}

core::RectifiedStereoCalibration loadCalibration(
    const MainlineRuntimeConfig& config,
    cuda_preprocess::StereoRemapMaps& maps) {
    cuda_preprocess::StereoCalibration file;
    if (!file.load(config.calibration.file.string())) {
        throw std::runtime_error(
            "cannot load stereo calibration: " +
            config.calibration.file.string());
    }
    if (!file.buildRemapMaps(
            maps,
            config.calibration.output_width,
            config.calibration.output_height)) {
        throw std::runtime_error("cannot build stereo rectification maps");
    }
    return {
        intrinsics(file.getProjectionLeft()),
        intrinsics(file.getProjectionRight()),
        static_cast<double>(file.getBaseline()),
        core::CoordinateFrameId{config.calibration.rig_frame},
    };
}

hik::CameraConfig cameraConfig(const MainlineRuntimeConfig& config) {
    const double midpoint_ns = config.camera.exposure_us * 500.0;
    if (!std::isfinite(midpoint_ns) || midpoint_ns <= 0.0 ||
        midpoint_ns > static_cast<double>(
            std::numeric_limits<long long>::max())) {
        throw std::runtime_error("camera exposure midpoint is out of range");
    }
    hik::CameraConfig output;
    output.camera_index_left = config.camera.left_index;
    output.camera_index_right = config.camera.right_index;
    output.serial_left = config.camera.left_serial;
    output.serial_right = config.camera.right_serial;
    output.expected_model_left = config.camera.left_model.value_or("");
    output.expected_model_right = config.camera.right_model.value_or("");
    output.expected_firmware_left =
        config.camera.left_firmware_version.value_or("");
    output.expected_firmware_right =
        config.camera.right_firmware_version.value_or("");
    output.exposure_us = static_cast<float>(config.camera.exposure_us);
    output.gain_db = static_cast<float>(config.camera.gain_db);
    output.use_trigger = true;
    output.trigger_source = config.camera.trigger_source;
    output.trigger_activation = config.camera.trigger_activation;
    output.trigger_frequency_hz =
        static_cast<int>(std::llround(config.runtime.nominal_fps));
    output.image_node_num = config.camera.image_node_num;
    output.embedded_info_clear_rows =
        config.camera.embedded_info_clear_rows;
    output.shutter_type = config.camera.shutter_type;
    output.exposure_timestamp_source =
        config.camera.exposure_timestamp_source;
    output.exposure_delay_status = config.camera.exposure_delay_status;
    output.exposure_delay_ns = config.camera.exposure_delay_ns;
    output.exposure_delay_uncertainty_known =
        config.camera.exposure_delay_uncertainty_ns.has_value();
    output.exposure_delay_uncertainty_ns =
        config.camera.exposure_delay_uncertainty_ns.value_or(0U);
    output.exposure_midpoint_offset_ns =
        static_cast<std::uint64_t>(std::llround(midpoint_ns));
    output.width = config.camera.width;
    output.height = config.camera.height;
    return output;
}

algorithms::stereo::AssociationConfig associationConfig(
    const MainlineRuntimeConfig& config,
    const core::RectifiedStereoCalibration& calibration) {
    algorithms::stereo::AssociationConfig output;
    output.max_capture_delta_ns = config.association.max_capture_delta_ns;
    output.epipolar_tolerance_px =
        config.association.epipolar_tolerance_px;
    output.adaptive_epipolar_height_ratio =
        config.association.adaptive_epipolar_height_ratio;
    output.max_size_ratio = config.association.max_size_ratio;
    output.min_shifted_iou = config.association.min_shifted_iou;
    output.max_disparity_px =
        static_cast<double>(config.association.max_disparity_px);
    output.baseline_m = calibration.baseline_m;
    output.ball_diameter_m = config.association.ball_diameter_m;
    output.bbox_scale = config.association.bbox_scale;
    output.bbox_disparity_consistency_ratio =
        config.association.bbox_disparity_consistency_ratio;
    output.bbox_disparity_consistency_min_px =
        config.association.bbox_disparity_consistency_min_px;
    output.bbox_disparity_penalty_scale =
        config.association.bbox_disparity_penalty_scale;
    output.require_bounding_box = true;
    return output;
}

cuda_depth::CudaDirectObservationEstimatorConfig estimatorConfig(
    const MainlineRuntimeConfig& config,
    const core::RectifiedStereoCalibration& calibration) {
    cuda_depth::CudaDirectObservationEstimatorConfig output;
    output.calibration = calibration;
    output.method_order.clear();
    output.method_order.reserve(config.depth.method_order.size());
    for (const auto& method : config.depth.method_order) {
        output.method_order.emplace_back(method);
    }
    output.max_disparity_px = config.association.max_disparity_px;
    output.circle_max_roi_pixels = config.depth.circle_max_roi_pixels;
    output.anchor_stddev_px = config.depth.anchor_stddev_px;
    output.disparity_stddev_px = config.depth.disparity_stddev_px;
    output.min_depth_m = config.depth.min_depth_m;
    output.max_depth_m = config.depth.max_depth_m;
    return output;
}

services::StereoObservationServiceConfig serviceConfig(
    const MainlineRuntimeConfig& config,
    const core::RectifiedStereoCalibration& calibration) {
    services::StereoObservationServiceConfig output;
    output.association = associationConfig(config, calibration);
    output.gate.min_depth_m = config.depth.min_depth_m;
    output.gate.max_depth_m = config.depth.max_depth_m;
    output.gate.max_reprojection_error_px = std::max(
        config.association.epipolar_tolerance_px,
        config.association.adaptive_epipolar_height_ratio *
            static_cast<double>(config.camera.height));
    return output;
}

services::SourceQueueOverflowPolicy overflowPolicy(
    const std::string& value) {
    return value == "drop_oldest"
        ? services::SourceQueueOverflowPolicy::drop_oldest
        : services::SourceQueueOverflowPolicy::block;
}

}  // namespace

services::RunSummary runMainlineApplication(
    const MainlineRuntimeConfig& config,
    std::function<bool()> stop_requested) {
    const std::string actual_engine_sha =
        sha256File(config.detector.engine_file);
    if (actual_engine_sha != config.detector.engine_sha256) {
        throw std::runtime_error(
            "TensorRT engine SHA-256 mismatch: expected " +
            config.detector.engine_sha256 + ", actual " + actual_engine_sha);
    }
    const std::string actual_calibration_sha =
        sha256File(config.calibration.file);
    if (actual_calibration_sha != config.calibration.sha256) {
        throw std::runtime_error(
            "stereo calibration SHA-256 mismatch: expected " +
            config.calibration.sha256 + ", actual " +
            actual_calibration_sha);
    }

    cuda_preprocess::StereoRemapMaps maps;
    const core::RectifiedStereoCalibration calibration =
        loadCalibration(config, maps);
    cuda_preprocess::CudaStereoPreprocessor preprocessor({
        static_cast<std::uint32_t>(config.camera.width),
        static_cast<std::uint32_t>(config.camera.height),
        3U,
    });
    std::string error;
    if (!preprocessor.initialize(maps, error)) {
        throw std::runtime_error(
            error.empty() ? "cannot initialize CUDA stereo preprocessor"
                          : std::move(error));
    }

    const tensorrt::TensorRtDetectorConfig detector_config{
        config.detector.engine_file.string(),
        core::PipelineId{config.detector.pipeline_id},
        config.detector.input_size,
        outputContract(config.detector.output_contract),
        config.detector.class_count,
        static_cast<float>(config.detector.confidence_threshold),
        static_cast<float>(config.detector.nms_iou_threshold),
        config.detector.max_detections,
        config.detector.class_id,
        config.detector.use_dla,
        config.detector.dla_core,
    };
    tensorrt::TensorRtDetector left_detector(detector_config);
    tensorrt::TensorRtDetector right_detector(detector_config);
    if (!left_detector.initialize(error)) {
        throw std::runtime_error(
            error.empty() ? "cannot initialize left TensorRT detector"
                          : std::move(error));
    }
    if (!right_detector.initialize(error)) {
        throw std::runtime_error(
            error.empty() ? "cannot initialize right TensorRT detector"
                          : std::move(error));
    }

    cuda_depth::CudaDirectObservationEstimator estimator(
        estimatorConfig(config, calibration));
    gpio::PWMTrigger trigger(
        config.trigger.gpio_chip,
        config.trigger.gpio_line,
        config.trigger.frequency_hz,
        config.trigger.duty_cycle);

    if (config.runtime.queue_capacity >
        std::numeric_limits<std::size_t>::max() - 2U) {
        throw std::runtime_error("camera buffer pool size overflows");
    }
    hik::HikStereoCameraSourceConfig source_config;
    source_config.camera = cameraConfig(config);
    source_config.left_camera_id = core::CameraId{config.camera.left_id};
    source_config.right_camera_id = core::CameraId{config.camera.right_id};
    source_config.left_device_clock =
        core::ClockDomainId{config.clocks.left_device};
    source_config.right_device_clock =
        core::ClockDomainId{config.clocks.right_device};
    source_config.capture_realtime_clock =
        core::ClockDomainId{config.clocks.capture_realtime};
    source_config.host_monotonic_clock =
        core::ClockDomainId{config.clocks.host_monotonic};
    source_config.timeout_ms = config.runtime.source_timeout_ms;
    source_config.buffer_pool_pairs = config.runtime.queue_capacity + 2U;
    source_config.frame_limit = config.runtime.frame_limit;
    hik::HikStereoCameraSource source(
        std::move(source_config),
        [&trigger](std::uint64_t sequence, hik::TriggerEdgeEvidence& output) {
            gpio::PWMTriggerEdgeEvent event;
            if (!trigger.eventForSequence(sequence, event)) return false;
            output = {
                event.sequence,
                event.scheduled_monotonic_ns,
                event.edge_monotonic_ns,
                event.edge_realtime_ns,
                event.write_uncertainty_ns,
            };
            return true;
        });

    std::unique_ptr<capture::MainlineStageTrialWriter> writer;
    if (config.runtime.capture_stage) {
        writer = std::make_unique<capture::MainlineStageTrialWriter>(
            capture::MainlineStageTrialWriterConfig{
                *config.runtime.capture_stage});
        if (!writer->open(error)) {
            throw std::runtime_error(
                error.empty() ? "cannot open capture staging writer"
                              : std::move(error));
        }
    }

    std::shared_ptr<rclcpp::Node> node;
    std::unique_ptr<ros2::Ros2ObservationPublisher> publisher;
    if (config.publisher.enabled) {
        if (!rclcpp::ok()) {
            throw std::runtime_error("ROS2 context is not initialized");
        }
        node = rclcpp::Node::make_shared("volleyball_nx_mainline");
        ros2::ObservationMessageConfig message_config;
        message_config.header_frame_id = config.publisher.frame_id;
        message_config.ros_time_clock_domain =
            core::ClockDomainId{config.clocks.capture_realtime};
        message_config.primary_depth_method =
            core::StereoDepthMethodId{config.depth.method_order.front()};
        message_config.source_epoch = ros2::createSourceEpoch(
            config.publisher.source_epoch_file.string());
        message_config.class_id = config.detector.class_id;
        publisher = std::make_unique<ros2::Ros2ObservationPublisher>(
            node, config.publisher.topic, std::move(message_config));
    }

    services::StereoObservationService service(
        serviceConfig(config, calibration),
        preprocessor,
        left_detector,
        right_detector,
        estimator,
        writer.get(),
        publisher.get());
    services::StereoPipelineRunnerConfig runner_config;
    runner_config.max_consecutive_timeouts =
        config.runtime.max_consecutive_timeouts;
    runner_config.stop_requested = std::move(stop_requested);
    runner_config.finalize_on_stop = config.runtime.mode == "runtime";
    runner_config.source_queue_capacity = config.runtime.queue_capacity;
    runner_config.source_queue_overflow =
        overflowPolicy(config.runtime.overflow_policy);
    runner_config.progress_interval_frames = config.metrics_interval_frames;
    runner_config.progress_callback = [](const services::RunSummary& summary) {
        std::clog << "mainline progress frames=" << summary.frames_received
                  << " observations=" << summary.observations
                  << " no_matches=" << summary.no_matches
                  << " dropped=" << summary.frames_dropped
                  << " queue_high_watermark="
                  << summary.queue_high_watermark << '\n';
    };
    return services::runManagedStereoTrial(
        std::move(runner_config), source, trigger, service);
}

}  // namespace volleyball::apps::nx_mainline
