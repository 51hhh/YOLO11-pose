#include "zedx_camera.h"
#include <sl/Camera.hpp>
#include <cmath>
#include <cstdio>

namespace stereo3d {

struct ZedxCamera::Impl {
    sl::Camera camera;
    sl::Mat left_mat_;
    sl::Mat right_mat_;
    bool opened_ = false;
};

ZedxCamera::ZedxCamera() : impl_(new Impl) {}

ZedxCamera::~ZedxCamera() {
    close();
    delete impl_;
}

bool ZedxCamera::open(const ZedCameraConfig& cfg) {
    sl::InitParameters init;

    if (cfg.resolution == "HD1200")
        init.camera_resolution = sl::RESOLUTION::HD1200;
    else if (cfg.resolution == "HD1080")
        init.camera_resolution = sl::RESOLUTION::HD1080;
    else if (cfg.resolution == "HD720")
        init.camera_resolution = sl::RESOLUTION::HD720;
    else if (cfg.resolution == "VGA")
        init.camera_resolution = sl::RESOLUTION::VGA;
    else
        init.camera_resolution = sl::RESOLUTION::HD1200;

    init.camera_fps = cfg.fps;
    init.depth_mode = sl::DEPTH_MODE::NONE;
    init.enable_image_enhancement = cfg.enable_image_enhancement;

    auto err = impl_->camera.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::fprintf(stderr, "[ZedX] Failed to open: %s\n", sl::toString(err).c_str());
        return false;
    }

    if (cfg.manual_exposure) {
        impl_->camera.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, cfg.exposure_pct);
        impl_->camera.setCameraSettings(sl::VIDEO_SETTINGS::GAIN, cfg.gain);
    }

    auto res = impl_->camera.getCameraInformation().camera_configuration.resolution;
    std::printf("[ZedX] Opened: %dx%d @ %d, depth=%s\n",
               res.width, res.height, cfg.fps, cfg.depth_mode.c_str());

    impl_->opened_ = true;
    return true;
}

void ZedxCamera::close() {
    if (impl_->opened_) {
        impl_->camera.close();
        impl_->opened_ = false;
    }
}

bool ZedxCamera::grab() {
    sl::RuntimeParameters rt;
    return impl_->camera.grab(rt) == sl::ERROR_CODE::SUCCESS;
}

void* ZedxCamera::getLeftBGRA_GPU() {
    impl_->camera.retrieveImage(impl_->left_mat_, sl::VIEW::LEFT, sl::MEM::GPU);
    return impl_->left_mat_.getPtr<sl::uchar1>(sl::MEM::GPU);
}

void* ZedxCamera::getRightBGRA_GPU() {
    impl_->camera.retrieveImage(impl_->right_mat_, sl::VIEW::RIGHT, sl::MEM::GPU);
    return impl_->right_mat_.getPtr<sl::uchar1>(sl::MEM::GPU);
}

StereoIntrinsics ZedxCamera::getIntrinsics() const {
    auto info = impl_->camera.getCameraInformation();
    auto& left_cam = info.camera_configuration.calibration_parameters.left_cam;
    auto res = info.camera_configuration.resolution;

    StereoIntrinsics intr;
    intr.focal = left_cam.fx;
    intr.cx = left_cam.cx;
    intr.cy = left_cam.cy;
    intr.baseline = std::abs(info.camera_configuration.calibration_parameters.getCameraBaseline()) / 1000.0f;
    intr.width = res.width;
    intr.height = res.height;
    return intr;
}

int ZedxCamera::getWidth() const {
    return impl_->camera.getCameraInformation().camera_configuration.resolution.width;
}

int ZedxCamera::getHeight() const {
    return impl_->camera.getCameraInformation().camera_configuration.resolution.height;
}

bool ZedxCamera::isOpened() const {
    return impl_->opened_;
}

} // namespace stereo3d
