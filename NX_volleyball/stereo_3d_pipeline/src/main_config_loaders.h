#ifndef STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_
#define STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_

#include "pipeline/pipeline.h"
#include "fusion/trajectory_predictor.h"
#include "utils/trajectory_recorder.h"
#include "utils/baseline_clip_recorder.h"
#include "main_realtime_debug_dump.h"

#include <string>

#ifdef HAS_ROS2
#include "ros/goal_pose_bridge.h"
#endif

stereo3d::PipelineConfig loadConfig(const std::string& path);
stereo3d::TrajectoryPredictorConfig loadPredictorConfig(const std::string& path);
stereo3d::TrajectoryRecorderConfig loadRecorderConfig(const std::string& path);
stereo3d::BaselineClipRecorderConfig loadBaselineClipRecorderConfig(const std::string& path);
RealtimeDebugDumpConfig loadRealtimeDebugDumpConfig(const std::string& path);

#ifdef HAS_ROS2
stereo3d::Ros2BridgeConfig loadRos2Config(const std::string& path);
#endif

#endif  // STEREO_3D_PIPELINE_MAIN_CONFIG_LOADERS_H_
