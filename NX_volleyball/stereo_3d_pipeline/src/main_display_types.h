#pragma once

#include "pipeline/pipeline.h"

#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>

struct ClickMeasureState {
    std::mutex mtx;
    int click_u = -1;
    int click_v = -1;
    float click_x = 0.0f;
    float click_y = 0.0f;
    float click_z = 0.0f;
    bool has_click = false;
    int display_frames = 0;
};

struct DisplayJob {
    cv::Mat frame;
    std::vector<stereo3d::Detection> detections;
    std::vector<stereo3d::Object3D> results;
    std::vector<stereo3d::LandingPrediction> preds;
    float fps = 0.0f;
    int frame_id = 0;
    int rec_frames = 0;
};
