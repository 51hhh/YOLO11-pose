#pragma once

#include "../pipeline/object3d_types.h"

namespace stereo3d {

float selectedMatchConfidence(const Object3D& obj);
double depthSigmaFromObservation(const Object3D& obj, float confidence);

}  // namespace stereo3d
