#ifndef STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_
#define STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_

#include "pipeline/pipeline.h"
#include "capture/capture_v21_stage_recorder.h"
#include "diagnostics/realtime_debug_dump.h"

#include <string>

#ifdef HAS_ROS2
#include "ros/nx_observation_ros2_config.h"
#endif

stereo3d::PipelineConfig loadConfig(const std::string& path);
stereo3d::CaptureV21StageRecorderConfig loadCaptureV21StageRecorderConfig(
    const std::string& path);
RealtimeDebugDumpConfig loadRealtimeDebugDumpConfig(const std::string& path);

#ifdef HAS_ROS2
stereo3d::NxObservationRos2Config loadNxObservationRos2Config(
    const std::string& path);
#endif

#endif  // STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_
