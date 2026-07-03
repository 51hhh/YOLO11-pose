/**
 * @file stereo_depth_viewer_models.h
 * @brief Optional ONNX stereo model loading for stereo_depth_viewer.
 */

#ifndef STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_MODELS_H_
#define STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_MODELS_H_

#include "stereo/onnx_stereo.h"

#include <string>

void loadViewerOnnxModels(const std::string& crestereo_path,
                          const std::string& hitnet_path,
                          stereo3d::OnnxStereo& crestereo,
                          stereo3d::OnnxStereo& hitnet);

#endif  // STEREO_3D_PIPELINE_STEREO_DEPTH_VIEWER_MODELS_H_
