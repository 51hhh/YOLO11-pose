#include "args.h"
#include "camera_session.h"
#include "pair_audit.h"
#include "capture/pwm_trigger.h"
#include "../common/parallel_tasks.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

bool validateStereoRoles(
    const MV_CC_DEVICE_INFO_LIST& devices,
    const hik_metadata_probe::Args& args) {
    const MV_CC_DEVICE_INFO* left = hik_metadata_probe::resolveCameraInfo(
        devices, args.left_sn, args.left_index);
    const MV_CC_DEVICE_INFO* right = hik_metadata_probe::resolveCameraInfo(
        devices, args.right_sn, args.right_index);
    if (left == nullptr || right == nullptr) {
        std::fprintf(
            stderr,
            "ERROR: cannot resolve stereo roles: left sn='%s' index=%d, "
            "right sn='%s' index=%d\n",
            args.left_sn.c_str(),
            args.left_index,
            args.right_sn.c_str(),
            args.right_index);
        return false;
    }
    const std::string left_serial = hik_metadata_probe::serialOf(left);
    const std::string right_serial = hik_metadata_probe::serialOf(right);
    if (left == right || left_serial.empty() || right_serial.empty() ||
        left_serial == right_serial) {
        std::fprintf(
            stderr,
            "ERROR: left/right must resolve to two distinct cameras with "
            "non-empty serial numbers (left='%s', right='%s')\n",
            left_serial.c_str(),
            right_serial.c_str());
        return false;
    }
    return true;
}

void printDevices(const MV_CC_DEVICE_INFO_LIST& devices) {
    std::printf("Enumerated %u devices\n", devices.nDeviceNum);
    for (unsigned int index = 0; index < devices.nDeviceNum; ++index) {
        std::printf(
            "  [%u] SN=%s model=%s layer=%s\n",
            index,
            hik_metadata_probe::serialOf(devices.pDeviceInfo[index]).c_str(),
            hik_metadata_probe::modelOf(devices.pDeviceInfo[index]).c_str(),
            hik_metadata_probe::layerName(devices.pDeviceInfo[index]));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    hik_metadata_probe::Args args;
    if (!hik_metadata_probe::parseArgs(argc, argv, args)) return 2;
    if (args.help_requested) return 0;
    if (!hik_metadata_probe::validateArgs(args)) return 2;

    MV_CC_DEVICE_INFO_LIST devices {};
    const int enum_status = MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE, &devices);
    if (enum_status != MV_OK || devices.nDeviceNum == 0) {
        std::fprintf(
            stderr,
            "EnumDevices failed ret=0x%X count=%u\n",
            enum_status,
            devices.nDeviceNum);
        return 1;
    }
    printDevices(devices);
    if (devices.nDeviceNum < 2) {
        std::fprintf(
            stderr,
            "ERROR: stereo metadata probe requires two enumerated cameras, "
            "got %u.\nCheck USB connection/power, close MVS or any process "
            "holding a camera, then rerun the probe.\n",
            devices.nDeviceNum);
        return 1;
    }
    if (!validateStereoRoles(devices, args)) return 1;

    hik_metadata_probe::CameraSession left;
    hik_metadata_probe::CameraSession right;
    if (!hik_metadata_probe::openCamera(
            devices, args.left_sn, args.left_index, "LEFT", left) ||
        !hik_metadata_probe::openCamera(
            devices, args.right_sn, args.right_index, "RIGHT", right)) {
        return 1;
    }
    if (!hik_metadata_probe::configureCamera(
            left.handle(), args, "LEFT") ||
        !hik_metadata_probe::configureCamera(
            right.handle(), args, "RIGHT")) {
        return 1;
    }

    std::puts("\nNode readback after configuration:");
    hik_metadata_probe::printNodeReadback(left.handle(), "LEFT");
    hik_metadata_probe::printNodeReadback(right.handle(), "RIGHT");
    if (!left.startGrabbing("LEFT") || !right.startGrabbing("RIGHT")) {
        return 1;
    }

    std::unique_ptr<stereo3d::PWMTrigger> pwm;
    if (args.start_pwm) {
        pwm = std::make_unique<stereo3d::PWMTrigger>(
            args.trigger_chip, args.trigger_line, args.pwm_hz, 50.0);
        if (!pwm->start()) {
            std::fprintf(
                stderr,
                "PWM start failed; cannot verify FrameCounter-to-GPIO "
                "mapping\n");
            return 1;
        }
    }

    hik_metadata_probe::PairAudit audit;
    std::puts("\nFrame metadata:");
    for (int sample_index = 0;
         sample_index < args.frames;
         ++sample_index) {
        hik_metadata_probe::FrameMeta left_meta;
        hik_metadata_probe::FrameMeta right_meta;
        std::string worker_error;
        const bool workers_ok = legacy_nx_tools::runIndexedTasks(
            2,
            2,
            [&](std::size_t camera_index) {
                if (camera_index == 0) {
                    left_meta = hik_metadata_probe::grabMeta(
                        left.handle(), args.timeout_ms);
                } else {
                    right_meta = hik_metadata_probe::grabMeta(
                        right.handle(), args.timeout_ms);
                }
            },
            worker_error);
        if (!workers_ok) {
            std::fprintf(
                stderr,
                "Frame acquisition worker failed: %s\n",
                worker_error.c_str());
            audit.recordAcquisitionFailure();
            break;
        }
        if (!audit.recordPair(sample_index, left_meta, right_meta)) continue;

        if (pwm) {
            stereo3d::PWMTriggerEdgeEvent edge;
            const bool mapping_matches =
                left_meta.frame_counter == right_meta.frame_counter &&
                pwm->eventForSequence(left_meta.frame_counter, edge) &&
                edge.sequence == left_meta.frame_counter;
            audit.recordPwmMapping(mapping_matches);
            if (!mapping_matches) {
                std::printf(
                    "    GPIO mapping failure at %04d: Lfc=%u Rfc=%u "
                    "latest_gpio=%llu\n",
                    sample_index,
                    left_meta.frame_counter,
                    right_meta.frame_counter,
                    static_cast<unsigned long long>(
                        pwm->latestEventSequence()));
            }
        }
    }

    // 先停止产生触发边沿，再释放任一相机；提前返回也由 RAII 完成清理。
    if (pwm) pwm->stop();
    left.shutdown();
    right.shutdown();

    audit.printSummary(args.start_pwm);
    if (!audit.passed(args.start_pwm)) {
        std::fprintf(
            stderr,
            "FAIL: metadata continuity or GPIO mapping is invalid; "
            "do not start capture\n");
        return 1;
    }
    return 0;
}
