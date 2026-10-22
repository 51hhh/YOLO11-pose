#ifndef STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_HELPERS_H_
#define STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_HELPERS_H_

#include "neural_feature_matcher.h"

#include "utils/logger.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace stereo3d {
namespace {

class NeuralFeatureTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) {
            LOG_WARN("[NeuralFeatureTRT] %s", msg);
        }
    }
};

NeuralFeatureTrtLogger gLogger;

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

size_t dataTypeBytes(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 0;
    }
}

bool hasDynamicDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return true;
    }
    return false;
}

size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) return 0;
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return 0;
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

int tensorChannels(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[1]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[0]);
    return 0;
}

int tensorHeight(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[2]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[1]);
    return 0;
}

int tensorWidth(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 4) return static_cast<int>(dims.d[3]);
    if (dims.nbDims == 3) return static_cast<int>(dims.d[2]);
    return 0;
}

int tensorLastDim(const nvinfer1::Dims& dims) {
    return dims.nbDims > 0 ? static_cast<int>(dims.d[dims.nbDims - 1]) : 0;
}

template <size_t N>
bool nameHasAny(const std::string& lname,
                const std::array<const char*, N>& needles) {
    for (const char* needle : needles) {
        if (lname.find(needle) != std::string::npos) return true;
    }
    return false;
}

bool isImageSizeTensorName(const std::string& lname) {
    return nameHasAny(lname, std::array<const char*, 5>{
        "image_size", "imagesize", "image_shape", "imageshape", "input_size"});
}

bool hasSideSeparator(const std::string& lname, char side) {
    const std::array<char, 5> separators{'_', '/', '.', ':', '-'};
    for (char sep : separators) {
        const std::string left{sep, side};
        const std::string right{side, sep};
        if (lname.find(left) != std::string::npos ||
            lname.find(right) != std::string::npos) {
            return true;
        }
    }
    return false;
}

int splitTensorSideFromName(const std::string& lname) {
    if (lname.find("left") != std::string::npos ||
        lname.find("query") != std::string::npos) {
        return 0;
    }
    if (lname.find("right") != std::string::npos ||
        lname.find("train") != std::string::npos) {
        return 1;
    }

    const bool semantic =
        nameHasAny(lname, std::array<const char*, 10>{
            "image", "img", "keypoint", "kpt", "point", "coord",
            "descriptor", "desc", "score", "size"});
    if (!semantic) return -1;

    if (nameHasAny(lname, std::array<const char*, 15>{
            "image0", "img0", "keypoints0", "keypoint0", "kpts0",
            "kpt0", "points0", "coords0", "descriptors0", "descriptor0",
            "descs0", "desc0", "scores0", "score0", "size0"}) ||
        hasSideSeparator(lname, '0')) {
        return 0;
    }
    if (nameHasAny(lname, std::array<const char*, 15>{
            "image1", "img1", "keypoints1", "keypoint1", "kpts1",
            "kpt1", "points1", "coords1", "descriptors1", "descriptor1",
            "descs1", "desc1", "scores1", "score1", "size1"}) ||
        hasSideSeparator(lname, '1')) {
        return 1;
    }
    return -1;
}

bool isLeftMatchIndexTensorName(const std::string& lname) {
    return nameHasAny(lname, std::array<const char*, 8>{
               "matches0", "match0", "indices0", "index0",
               "matching_indices0", "matching_index0",
               "assignment0", "assign0"}) ||
           (nameHasAny(lname, std::array<const char*, 3>{
                "match", "index", "indice"}) &&
            splitTensorSideFromName(lname) == 0);
}

bool isRightMatchIndexTensorName(const std::string& lname) {
    return nameHasAny(lname, std::array<const char*, 8>{
               "matches1", "match1", "indices1", "index1",
               "matching_indices1", "matching_index1",
               "assignment1", "assign1"}) ||
           (nameHasAny(lname, std::array<const char*, 3>{
                "match", "index", "indice"}) &&
            splitTensorSideFromName(lname) == 1);
}

bool isLeftScoreTensorName(const std::string& lname) {
    return nameHasAny(lname, std::array<const char*, 8>{
               "scores0", "score0", "mscores0", "mscore0",
               "matching_scores0", "matching_score0",
               "conf0", "prob0"}) ||
           (nameHasAny(lname, std::array<const char*, 3>{
                "score", "conf", "prob"}) &&
            splitTensorSideFromName(lname) == 0);
}

bool isRightScoreTensorName(const std::string& lname) {
    return nameHasAny(lname, std::array<const char*, 8>{
               "scores1", "score1", "mscores1", "mscore1",
               "matching_scores1", "matching_score1",
               "conf1", "prob1"}) ||
           (nameHasAny(lname, std::array<const char*, 3>{
                "score", "conf", "prob"}) &&
            splitTensorSideFromName(lname) == 1);
}

float medianOf(std::vector<float>& values) {
    if (values.empty()) return -1.0f;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n & 1u) ? values[n / 2] : 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

float bilinearChannelSample(const std::vector<float>& chw,
                            int channels, int height, int width,
                            int channel, float x, float y) {
    if (channels <= 0 || height <= 0 || width <= 0 ||
        channel < 0 || channel >= channels || chw.empty()) {
        return 0.0f;
    }
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) {
        const size_t idx = (static_cast<size_t>(channel) * height +
                            static_cast<size_t>(yy)) * width +
                           static_cast<size_t>(xx);
        return idx < chw.size() ? chw[idx] : 0.0f;
    };
    const float v00 = at(x0, y0);
    const float v10 = at(x1, y0);
    const float v01 = at(x0, y1);
    const float v11 = at(x1, y1);
    return v00 * (1.0f - fx) * (1.0f - fy) +
           v10 * fx * (1.0f - fy) +
           v01 * (1.0f - fx) * fy +
           v11 * fx * fy;
}

struct XFeatRawOutput {
    std::vector<float> feats;
    std::vector<float> keypoints;
    std::vector<float> heatmap;
    int feat_h = 0;
    int feat_w = 0;
};

struct XFeatFeature {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
    std::vector<float> descriptor;
};

struct XFeatCandidate {
    float x = 0.0f;
    float y = 0.0f;
    float feat_x = 0.0f;
    float feat_y = 0.0f;
    float score = 0.0f;
};

struct DirectFeature {
    float x = 0.0f;
    float y = 0.0f;
    float score = 1.0f;
    std::vector<float> descriptor;
};


}  // namespace
}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_NEURAL_FEATURE_MATCHER_HELPERS_H_
