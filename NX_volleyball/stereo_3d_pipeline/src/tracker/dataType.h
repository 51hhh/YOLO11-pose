#ifndef STEREO3D_DATA_TYPE_H
#define STEREO3D_DATA_TYPE_H

#include <vector>

namespace stereo3d {

struct Detection {
    float bbox[4];    // [x1, y1, x2, y2]
    float confidence;
    int class_id;
    int track_id;     // 用于跟踪
};

struct TrackResult {
    Detection detection;
    int track_id;
    float tracking_confidence;
};

} // namespace stereo3d

#endif // STEREO3D_DATA_TYPE_H
