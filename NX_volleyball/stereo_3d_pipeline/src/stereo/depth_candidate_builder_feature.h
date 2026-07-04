#ifndef STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_FEATURE_H_
#define STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_FEATURE_H_

#include "depth_candidate_builder.h"

namespace stereo3d {

void appendFeatureDepthCandidates(
    const DepthCandidateBuilderInput& input,
    float circle_anchor_x,
    float circle_anchor_y,
    std::vector<DepthCandidateObservation>* candidates);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_DEPTH_CANDIDATE_BUILDER_FEATURE_H_
