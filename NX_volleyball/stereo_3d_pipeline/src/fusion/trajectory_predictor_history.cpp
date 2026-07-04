#include "trajectory_predictor.h"

#include <algorithm>

namespace stereo3d {

LandingPrediction TrajectoryPredictor::getPrediction(int track_id) const {
    for (const auto& h : histories_) {
        if (h.track_id == track_id) return h.last_pred;
    }
    return LandingPrediction{};
}

TrajectoryPredictor::TrackHistory* TrajectoryPredictor::findHistory(int track_id) {
    for (auto& h : histories_) {
        if (h.track_id == track_id) return &h;
    }
    return nullptr;
}

TrajectoryPredictor::TrackHistory& TrajectoryPredictor::getOrCreateHistory(int track_id) {
    auto* h = findHistory(track_id);
    if (h) return *h;

    histories_.push_back({});
    auto& nh = histories_.back();
    nh.track_id = track_id;
    nh.t_origin = elapsed_time_;
    return nh;
}

void TrajectoryPredictor::pruneHistories(const std::vector<Object3D>& results) {
    histories_.erase(
        std::remove_if(histories_.begin(), histories_.end(),
            [&](const TrackHistory& h) {
                for (const auto& r : results) {
                    if (r.track_id == h.track_id) return false;
                }
                return true;
            }),
        histories_.end());
}

void TrajectoryPredictor::reset() {
    histories_.clear();
    elapsed_time_ = 0.0;
}

}  // namespace stereo3d
