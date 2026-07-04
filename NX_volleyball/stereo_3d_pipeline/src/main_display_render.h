#pragma once

#include "main_display_types.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <mutex>

inline void updateClickMeasurement(
    ClickMeasureState& click,
    const DisplayJob& job)
{
    std::lock_guard<std::mutex> lock(click.mtx);
    if (!click.has_click) return;
    click.has_click = false;
    click.click_z = 0.0f;
    float min_dist = 1e9f;
    for (size_t i = 0; i < job.detections.size() && i < job.results.size(); ++i) {
        const float dx = job.detections[i].cx - click.click_u;
        const float dy = job.detections[i].cy - click.click_v;
        const float dist = dx * dx + dy * dy;
        if (dist < min_dist && job.results[i].z > 0.0f) {
            min_dist = dist;
            click.click_x = job.results[i].x;
            click.click_y = job.results[i].y;
            click.click_z = job.results[i].z;
        }
    }
}

inline void drawClickMeasurement(
    cv::Mat& frame,
    ClickMeasureState& click)
{
    std::lock_guard<std::mutex> lock(click.mtx);
    if (click.display_frames <= 0) return;

    cv::drawMarker(frame,
        cv::Point(click.click_u, click.click_v),
        cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 20, 2);
    if (click.click_z > 0.0f) {
        char click_text[128];
        snprintf(click_text, sizeof(click_text),
                 "(%.2f, %.2f, %.2f)m",
                 click.click_x, click.click_y, click.click_z);
        cv::putText(frame, click_text,
                    cv::Point(click.click_u + 12, click.click_v - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
    } else {
        cv::putText(frame, "No depth",
                    cv::Point(click.click_u + 12, click.click_v - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 2);
    }
    --click.display_frames;
}

inline void renderPipelineDisplayFrame(
    DisplayJob& job,
    ClickMeasureState& click)
{
    cv::Mat& frame = job.frame;
    updateClickMeasurement(click, job);

    for (size_t i = 0; i < job.detections.size(); ++i) {
        const auto& d = job.detections[i];
        const int x1 = static_cast<int>(d.cx - d.width / 2);
        const int y1 = static_cast<int>(d.cy - d.height / 2);
        const int bw = static_cast<int>(d.width);
        const int bh = static_cast<int>(d.height);
        const cv::Scalar color(0, 255, 0);
        cv::rectangle(frame, cv::Rect(x1, y1, bw, bh), color, 2);

        char label[128];
        if (i < job.results.size()) {
            snprintf(label, sizeof(label),
                     "%.2fm (%.0f%%)", job.results[i].z, d.confidence * 100);
        } else {
            snprintf(label, sizeof(label),
                     "conf=%.0f%%", d.confidence * 100);
        }
        cv::putText(frame, label, cv::Point(x1, y1 - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        if (i >= job.results.size() || job.results[i].z <= 0.0f) continue;

        const auto& r = job.results[i];
        const float speed = std::sqrt(r.vx * r.vx + r.vy * r.vy + r.vz * r.vz);
        char pos[160];
        snprintf(pos, sizeof(pos), "X=%.3f Y=%.3f Z=%.3f |v|=%.1f",
                 r.x, r.y, r.z, speed);
        cv::putText(frame, pos, cv::Point(x1, y1 + bh + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 200, 0), 1);

        char depth_info[160];
        const char* mstr = r.depth_method == 0 ? "M" :
                           r.depth_method == 1 ? "S" : "B";
        snprintf(depth_info, sizeof(depth_info),
                 "zm=%.3f zs=%.3f [%s]",
                 r.z_mono, r.z_stereo, mstr);
        cv::putText(frame, depth_info, cv::Point(x1, y1 + bh + 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);

        if (i < job.preds.size() && job.preds[i].valid) {
            const auto& p = job.preds[i];
            char pred_text[128];
            snprintf(pred_text, sizeof(pred_text),
                     "LAND(%.1f,%.1f) %.2fs %s",
                     p.x, p.y, p.time_to_land,
                     p.method == 0 ? "B" : "P");
            cv::putText(frame, pred_text,
                        cv::Point(x1, y1 + bh + 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 100, 255), 2);
        }
    }

    drawClickMeasurement(frame, click);

    char hud[128];
    if (job.rec_frames > 0) {
        snprintf(hud, sizeof(hud), "FPS: %.1f  Frame: %d  REC: %d",
                 job.fps, job.frame_id, job.rec_frames);
    } else {
        snprintf(hud, sizeof(hud), "FPS: %.1f  Frame: %d",
                 job.fps, job.frame_id);
    }
    cv::putText(frame, hud, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 200, 255), 2);
}
