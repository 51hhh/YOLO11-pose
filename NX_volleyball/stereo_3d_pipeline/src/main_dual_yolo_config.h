#ifndef STEREO_3D_PIPELINE_MAIN_DUAL_YOLO_CONFIG_H_
#define STEREO_3D_PIPELINE_MAIN_DUAL_YOLO_CONFIG_H_

#include "pipeline/pipeline_config.h"

#include <yaml-cpp/yaml.h>

void loadDualYoloConfig(const YAML::Node& dual, stereo3d::PipelineConfig& cfg);

#endif  // STEREO_3D_PIPELINE_MAIN_DUAL_YOLO_CONFIG_H_
