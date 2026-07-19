#include "roi_feature_match_cpu.h"

namespace stereo3d {

std::string sparseFeatureModeName(SparseFeatureMode mode)
{
    switch (mode) {
    case SparseFeatureMode::CORNER: return "corner";
    case SparseFeatureMode::TEXTURE: return "texture";
    case SparseFeatureMode::BINARY: return "binary";
    }
    return "unknown";
}

const char* openCVFeatureModeName(OpenCVFeatureMode mode)
{
    switch (mode) {
    case OpenCVFeatureMode::ORB: return "ORB";
    case OpenCVFeatureMode::BRISK: return "BRISK";
    case OpenCVFeatureMode::AKAZE: return "AKAZE";
    case OpenCVFeatureMode::SIFT: return "SIFT";
    }
    return "UNKNOWN";
}

}  // namespace stereo3d
