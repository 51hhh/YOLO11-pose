#include "volleyball/adapters/hik_mvs/hik_stereo_camera_source.hpp"

#include "volleyball/core/observation.hpp"

#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace volleyball::adapters::hik_mvs {
namespace {

class HostBayerBuffer final : public ports::ImageBuffer {
public:
    HostBayerBuffer(std::uint32_t width, std::uint32_t height)
        : width_(width), height_(height), bytes_(checkedSize(width, height)) {}

    [[nodiscard]] ports::MemoryKind memoryKind() const noexcept override {
        return ports::MemoryKind::host;
    }
    [[nodiscard]] std::uint32_t width() const noexcept override {
        return width_;
    }
    [[nodiscard]] std::uint32_t height() const noexcept override {
        return height_;
    }
    [[nodiscard]] std::size_t strideBytes() const noexcept override {
        return width_;
    }
    [[nodiscard]] core::PixelFormat pixelFormat() const noexcept override {
        return core::PixelFormat::bayer8;
    }
    [[nodiscard]] std::uintptr_t nativeHandle() const noexcept override {
        return reinterpret_cast<std::uintptr_t>(bytes_.data());
    }
    [[nodiscard]] std::size_t sizeBytes() const noexcept override {
        return bytes_.size();
    }
    [[nodiscard]] std::uint8_t* mutableData() noexcept {
        return bytes_.data();
    }

private:
    static std::size_t checkedSize(
        std::uint32_t width,
        std::uint32_t height) {
        if (width == 0 || height == 0 ||
            static_cast<std::size_t>(height) >
                std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(width)) {
            throw std::bad_array_new_length{};
        }
        return static_cast<std::size_t>(width) * height;
    }

    std::uint32_t width_;
    std::uint32_t height_;
    std::vector<std::uint8_t> bytes_;
};

enum class SourceState { idle, started, ended, failed, stopped };

bool checkedAdd(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t& output) noexcept {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) return false;
    output = lhs + rhs;
    return true;
}

bool fitsTimestamp(std::uint64_t value) noexcept {
    return value <= static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max());
}

bool exposureMidpointNs(double exposure_us, std::uint64_t& output) noexcept {
    const double midpoint_ns = exposure_us * 500.0;
    if (!std::isfinite(midpoint_ns) || midpoint_ns <= 0.0 ||
        midpoint_ns > static_cast<double>(
            std::numeric_limits<std::int64_t>::max())) {
        return false;
    }
    output = static_cast<std::uint64_t>(std::llround(midpoint_ns));
    return output > 0;
}

core::Timestamp timestamp(
    std::uint64_t value,
    const core::ClockDomainId& clock) {
    return {static_cast<std::int64_t>(value), clock};
}

bool validateConfig(
    const HikStereoCameraSourceConfig& config,
    const TriggerEvidenceLookup& trigger_lookup,
    std::string& error) {
    const auto& camera = config.camera;
    if (config.left_camera_id.empty() || config.right_camera_id.empty() ||
        config.left_camera_id == config.right_camera_id ||
        config.left_device_clock.empty() || config.right_device_clock.empty() ||
        config.left_device_clock == config.right_device_clock ||
        config.capture_realtime_clock.empty() ||
        config.host_monotonic_clock.empty() ||
        config.capture_realtime_clock == config.host_monotonic_clock ||
        config.left_device_clock == config.capture_realtime_clock ||
        config.left_device_clock == config.host_monotonic_clock ||
        config.right_device_clock == config.capture_realtime_clock ||
        config.right_device_clock == config.host_monotonic_clock) {
        error = "camera and clock identities must be non-empty and unambiguous";
        return false;
    }
    if (camera.serial_left.empty() || camera.serial_right.empty() ||
        camera.serial_left == camera.serial_right) {
        error = "live stereo source requires two explicit camera serials";
        return false;
    }
    const bool gpio_line_trigger =
        camera.trigger_source == "Line0" || camera.trigger_source == "Line1";
    if (!camera.use_trigger || !gpio_line_trigger ||
        camera.trigger_activation != "RisingEdge" ||
        camera.shutter_type != "global" ||
        camera.exposure_timestamp_source != "nx_gpio_rising_edge" ||
        camera.exposure_midpoint_offset_ns == 0 || !trigger_lookup) {
        error = "live stereo source requires global-shutter GPIO rising-edge timing";
        return false;
    }
    const bool measured_delay =
        camera.exposure_delay_status == "measured" &&
        camera.exposure_delay_uncertainty_known;
    const bool explicit_unmeasured_delay =
        camera.exposure_delay_status == "assumed_zero_unmeasured" &&
        camera.exposure_delay_ns == 0 &&
        camera.exposure_delay_uncertainty_ns == 0 &&
        !camera.exposure_delay_uncertainty_known;
    if (!measured_delay && !explicit_unmeasured_delay) {
        error = "exposure delay must be measured or explicitly marked unmeasured";
        return false;
    }
    if (camera.width <= 0 || camera.height <= 0 ||
        camera.trigger_frequency_hz <= 0 || camera.image_node_num < 2 ||
        camera.image_node_num > 16 || camera.embedded_info_clear_rows <= 0 ||
        camera.embedded_info_clear_rows > camera.height ||
        config.timeout_ms == 0 || config.buffer_pool_pairs == 0 ||
        config.buffer_pool_pairs >
            std::numeric_limits<std::size_t>::max() / 2U) {
        error = "camera geometry, trigger transport, timeout, or buffer pool is invalid";
        return false;
    }
    std::uint64_t expected_midpoint = 0;
    if (!exposureMidpointNs(camera.exposure_us, expected_midpoint) ||
        expected_midpoint != camera.exposure_midpoint_offset_ns) {
        error = "exposure midpoint offset must equal half the configured exposure";
        return false;
    }
    return true;
}

bool looksLikeTimeout(
    const GrabResult& left,
    const GrabResult& right) noexcept {
    const auto invalid_after_payload = [](const GrabResult& result) {
        return !result.success &&
               (result.receive_monotonic_ns != 0 ||
                result.receive_realtime_ns != 0);
    };
    return (!left.success || !right.success) &&
           !invalid_after_payload(left) && !invalid_after_payload(right);
}

}  // namespace

struct HikStereoCameraSource::Impl {
    explicit Impl(
        HikStereoCameraSourceConfig source_config,
        TriggerEvidenceLookup lookup)
        : config(std::move(source_config)), trigger_lookup(std::move(lookup)) {}

    HikStereoCameraSourceConfig config;
    TriggerEvidenceLookup trigger_lookup;
    HikvisionCamera camera;
    std::vector<std::shared_ptr<HostBayerBuffer>> buffers;
    SourceState state{SourceState::idle};
    std::uint64_t emitted_frames{0};
    std::string terminal_message;

    void closeCamera() noexcept {
        camera.stopGrabbing();
        camera.close();
    }

    ports::SourceResult fail(std::string message) {
        terminal_message = std::move(message);
        closeCamera();
        state = SourceState::failed;
        return {ports::SourceStatus::error, std::nullopt, terminal_message};
    }

    std::shared_ptr<HostBayerBuffer> acquire(
        const HostBayerBuffer* excluded = nullptr) const {
        for (const auto& buffer : buffers) {
            if (buffer.get() != excluded && buffer.use_count() == 1) {
                return buffer;
            }
        }
        return {};
    }

    bool makeFrame(
        const GrabResult& result,
        core::CameraRole role,
        const TriggerEdgeEvidence& edge,
        core::CameraFrameRecord& output,
        std::string& error) const {
        if (config.camera.auto_exposure && !result.exposure_us) {
            error = "auto exposure frame lacks measured exposure metadata";
            return false;
        }
        const double exposure_us =
            result.exposure_us.value_or(config.camera.exposure_us);
        std::uint64_t midpoint_offset_ns = 0;
        if (!exposureMidpointNs(exposure_us, midpoint_offset_ns)) {
            error = "frame exposure cannot produce a finite midpoint timestamp";
            return false;
        }
        std::uint64_t exposure_start_ns = 0;
        std::uint64_t capture_ns = 0;
        if (!checkedAdd(edge.edge_realtime_ns, config.camera.exposure_delay_ns,
                        exposure_start_ns) ||
            !checkedAdd(exposure_start_ns, midpoint_offset_ns, capture_ns) ||
            !fitsTimestamp(result.timestamp_us) ||
            !fitsTimestamp(result.receive_monotonic_ns) ||
            !fitsTimestamp(result.receive_realtime_ns) ||
            !fitsTimestamp(edge.scheduled_monotonic_ns) ||
            !fitsTimestamp(edge.edge_monotonic_ns) ||
            !fitsTimestamp(edge.edge_realtime_ns) ||
            !fitsTimestamp(exposure_start_ns) || !fitsTimestamp(capture_ns)) {
            error = "camera or trigger timestamp exceeds the signed core range";
            return false;
        }

        output.camera_id = role == core::CameraRole::left
            ? config.left_camera_id
            : config.right_camera_id;
        output.role = role;
        output.frame_index = result.frame_counter;
        output.hardware_timestamp = timestamp(
            result.timestamp_us,
            role == core::CameraRole::left
                ? config.left_device_clock
                : config.right_device_clock);
        output.exposure_start = timestamp(
            exposure_start_ns, config.capture_realtime_clock);
        output.capture_timestamp = timestamp(
            capture_ns, config.capture_realtime_clock);
        output.receive_timestamp = timestamp(
            result.receive_realtime_ns, config.capture_realtime_clock);
        output.exposure_us = exposure_us;
        output.gain_db = result.gain_db.value_or(config.camera.gain_db);
        output.roi = {
            0,
            0,
            static_cast<std::uint32_t>(config.camera.width),
            static_cast<std::uint32_t>(config.camera.height),
        };
        output.pixel_format = core::PixelFormat::bayer8;

        auto& acquisition = output.acquisition;
        acquisition.receive_monotonic_timestamp = timestamp(
            result.receive_monotonic_ns, config.host_monotonic_clock);
        acquisition.sdk_host_timestamp_raw = result.host_timestamp;
        acquisition.sensor_frame_number = result.frame_number;
        acquisition.sensor_frame_counter = result.frame_counter;
        acquisition.sensor_trigger_index = result.trigger_index;
        acquisition.stereo_hardware_timestamp_residual_ns =
            result.stereo_timestamp_residual_ns;
        acquisition.trigger_sequence = edge.sequence;
        acquisition.trigger_scheduled_monotonic = timestamp(
            edge.scheduled_monotonic_ns, config.host_monotonic_clock);
        acquisition.trigger_edge_monotonic = timestamp(
            edge.edge_monotonic_ns, config.host_monotonic_clock);
        acquisition.trigger_edge_realtime = timestamp(
            edge.edge_realtime_ns, config.capture_realtime_clock);
        acquisition.trigger_write_uncertainty_ns = edge.write_uncertainty_ns;

        const bool measured_delay =
            config.camera.exposure_delay_status == "measured" &&
            config.camera.exposure_delay_uncertainty_known;
        acquisition.capture_timestamp_mapping_valid = measured_delay;
        if (measured_delay) {
            std::uint64_t uncertainty = 0;
            if (!checkedAdd(edge.write_uncertainty_ns,
                            config.camera.exposure_delay_uncertainty_ns,
                            uncertainty)) {
                error = "capture timestamp uncertainty overflow";
                return false;
            }
            acquisition.capture_timestamp_uncertainty_ns = uncertainty;
        }

        const auto validation = core::validate(output);
        if (!validation) {
            error = validation.message;
            return false;
        }
        return true;
    }
};

HikStereoCameraSource::HikStereoCameraSource(
    HikStereoCameraSourceConfig config,
    TriggerEvidenceLookup trigger_lookup)
    : impl_(std::make_unique<Impl>(
          std::move(config), std::move(trigger_lookup))) {}

HikStereoCameraSource::~HikStereoCameraSource() {
    stop();
}

bool HikStereoCameraSource::start(std::string& error) {
    error.clear();
    if (impl_->state != SourceState::idle) {
        error = "Hik stereo source can only be started once";
        return false;
    }
    if (!validateConfig(impl_->config, impl_->trigger_lookup, error)) {
        impl_->state = SourceState::failed;
        impl_->terminal_message = error;
        return false;
    }
    try {
        impl_->buffers.reserve(impl_->config.buffer_pool_pairs * 2U);
        for (std::size_t index = 0;
             index < impl_->config.buffer_pool_pairs * 2U;
             ++index) {
            impl_->buffers.push_back(std::make_shared<HostBayerBuffer>(
                static_cast<std::uint32_t>(impl_->config.camera.width),
                static_cast<std::uint32_t>(impl_->config.camera.height)));
        }
    } catch (const std::exception& exception) {
        error = std::string{"cannot allocate camera buffer pool: "} +
                exception.what();
        impl_->state = SourceState::failed;
        impl_->terminal_message = error;
        return false;
    }
    if (!impl_->camera.open(impl_->config.camera)) {
        error = "cannot open the configured Hik stereo cameras";
        impl_->state = SourceState::failed;
        impl_->terminal_message = error;
        return false;
    }
    if (!impl_->camera.startGrabbing()) {
        impl_->camera.close();
        error = "cannot start Hik stereo acquisition";
        impl_->state = SourceState::failed;
        impl_->terminal_message = error;
        return false;
    }
    impl_->state = SourceState::started;
    return true;
}

void HikStereoCameraSource::stop() noexcept {
    if (impl_->state == SourceState::started) {
        impl_->closeCamera();
        impl_->state = SourceState::stopped;
    }
}

bool HikStereoCameraSource::isStarted() const noexcept {
    return impl_->state == SourceState::started;
}

ports::SourceResult HikStereoCameraSource::next() {
    if (impl_->state == SourceState::ended) {
        return {ports::SourceStatus::end_of_stream, std::nullopt, {}};
    }
    if (impl_->state == SourceState::failed) {
        return {
            ports::SourceStatus::error,
            std::nullopt,
            impl_->terminal_message,
        };
    }
    if (impl_->state != SourceState::started) {
        return {
            ports::SourceStatus::error,
            std::nullopt,
            "Hik stereo source is not started",
        };
    }
    if (impl_->config.frame_limit &&
        impl_->emitted_frames >= *impl_->config.frame_limit) {
        impl_->closeCamera();
        impl_->state = SourceState::ended;
        return {ports::SourceStatus::end_of_stream, std::nullopt, {}};
    }

    auto left_buffer = impl_->acquire();
    auto right_buffer = impl_->acquire(left_buffer.get());
    if (!left_buffer || !right_buffer) {
        return impl_->fail(
            "Hik image buffer pool exhausted; downstream retained too many frames");
    }

    GrabResult left_result;
    GrabResult right_result;
    const bool grabbed = impl_->camera.grabFramePair(
        left_buffer->mutableData(),
        right_buffer->mutableData(),
        static_cast<int>(left_buffer->strideBytes()),
        static_cast<int>(right_buffer->strideBytes()),
        impl_->config.timeout_ms,
        left_result,
        right_result);
    if (!grabbed) {
        if (looksLikeTimeout(left_result, right_result)) {
            return {
                ports::SourceStatus::timeout,
                std::nullopt,
                "Hik stereo frame acquisition timed out",
            };
        }
        return impl_->fail(
            "Hik stereo frame failed geometry, synchronization, or SDK release checks");
    }
    if (!left_result.success || !right_result.success ||
        left_result.frame_counter == 0 ||
        left_result.frame_counter != right_result.frame_counter) {
        return impl_->fail("Hik adapter returned an invalid stereo identity");
    }

    TriggerEdgeEvidence edge;
    bool found_trigger = false;
    try {
        found_trigger = impl_->trigger_lookup(
            left_result.frame_counter, edge);
    } catch (const std::exception& exception) {
        return impl_->fail(
            std::string{"trigger evidence lookup threw: "} + exception.what());
    } catch (...) {
        return impl_->fail("trigger evidence lookup threw a non-standard exception");
    }
    if (!found_trigger || edge.sequence != left_result.frame_counter ||
        edge.scheduled_monotonic_ns == 0 || edge.edge_monotonic_ns == 0 ||
        edge.edge_realtime_ns == 0) {
        return impl_->fail(
            "no matching GPIO rising-edge evidence for Hik FrameCounter");
    }

    ports::StereoFrame stereo;
    std::string error;
    if (!impl_->makeFrame(
            left_result, core::CameraRole::left, edge,
            stereo.left.metadata, error) ||
        !impl_->makeFrame(
            right_result, core::CameraRole::right, edge,
            stereo.right.metadata, error)) {
        return impl_->fail(
            error.empty() ? "cannot construct audited camera metadata" : error);
    }
    stereo.left.image = std::move(left_buffer);
    stereo.right.image = std::move(right_buffer);
    ++impl_->emitted_frames;
    return {
        ports::SourceStatus::frame,
        std::move(stereo),
        {},
    };
}

}  // namespace volleyball::adapters::hik_mvs
