#pragma once
#ifndef STEREO_3D_PIPELINE_ZEDX_CAMERA_H_
#define STEREO_3D_PIPELINE_ZEDX_CAMERA_H_

#include <string>

namespace sl { class Camera; }

namespace stereo3d {

struct ZedCameraConfig {
    std::string resolution = "HD1200";
    int fps = 60;
    std::string depth_mode = "NONE";
    bool manual_exposure = true;
    int exposure_pct = 30;
    int gain = 70;
    bool enable_image_enhancement = false;
};

struct StereoIntrinsics {
    float focal = 0;
    float baseline = 0;
    float cx = 0, cy = 0;
    int width = 0, height = 0;
};

class ZedxCamera {
public:
    ZedxCamera();
    ~ZedxCamera();

    bool open(const ZedCameraConfig& cfg);
    void close();
    bool grab();

    void* getLeftBGRA_GPU();
    void* getRightBGRA_GPU();

    StereoIntrinsics getIntrinsics() const;
    int getWidth() const;
    int getHeight() const;
    bool isOpened() const;

private:
    struct Impl;
    Impl* impl_;
};

} // namespace stereo3d

#endif // STEREO_3D_PIPELINE_ZEDX_CAMERA_H_
