/**
 * @file stereo_depth_viewer_models.cpp
 * @brief Optional ONNX stereo model loading for stereo_depth_viewer.
 */

#include "stereo_depth_viewer_models.h"

#include "utils/logger.h"

#include <unistd.h>

void loadViewerOnnxModels(const std::string& crestereo_path,
                          const std::string& hitnet_path,
                          stereo3d::OnnxStereo& crestereo,
                          stereo3d::OnnxStereo& hitnet) {
    if (!crestereo_path.empty()) {
        if (crestereo.load(crestereo_path, stereo3d::OnnxStereo::Model::CREStereo)) {
            LOG_INFO("CREStereo ONNX 模型已加载: %s", crestereo_path.c_str());
        } else {
            LOG_WARN("CREStereo 加载失败: %s", crestereo_path.c_str());
        }
    } else {
        const char* defaultPaths[] = {
            "dl_models/crestereo_init_iter10_480x640.onnx",
            "../dl_models/crestereo_init_iter10_480x640.onnx",
            nullptr
        };
        for (const char** p = defaultPaths; *p; ++p) {
            if (access(*p, F_OK) == 0) {
                if (crestereo.load(*p, stereo3d::OnnxStereo::Model::CREStereo))
                    LOG_INFO("CREStereo 自动加载: %s", *p);
                break;
            }
        }
    }

    if (!hitnet_path.empty()) {
        if (hitnet.load(hitnet_path, stereo3d::OnnxStereo::Model::HITNet)) {
            LOG_INFO("HITNet ONNX 模型已加载: %s", hitnet_path.c_str());
        } else {
            LOG_WARN("HITNet 加载失败: %s", hitnet_path.c_str());
        }
    } else {
        const char* defaultPaths[] = {
            "dl_models/hitnet_eth3d_480x640.onnx",
            "../dl_models/hitnet_eth3d_480x640.onnx",
            nullptr
        };
        for (const char** p = defaultPaths; *p; ++p) {
            if (access(*p, F_OK) == 0) {
                if (hitnet.load(*p, stereo3d::OnnxStereo::Model::HITNet))
                    LOG_INFO("HITNet 自动加载: %s", *p);
                break;
            }
        }
    }
}
