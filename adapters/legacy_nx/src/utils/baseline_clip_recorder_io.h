/**
 * @file baseline_clip_recorder_io.h
 * @brief Baseline clip recorder file naming and CSV/metadata helpers.
 */

#ifndef STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_IO_H_
#define STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_IO_H_

#include <cstddef>
#include <iosfwd>
#include <string>

namespace stereo3d {

struct BaselineClipRecorderConfig;
struct Detection;

std::string baselineClipTimestampName();
std::string baselineClipFrameName(int frame_id, const std::string& ext);
std::string normalizeBaselineImageFormat(std::string fmt);
std::string normalizeBaselineImageMode(std::string mode);

void writeBaselineClipHeader(const std::string& clip_dir);

void writeBaselineClipMetadata(const std::string& clip_dir,
                               int clip_number,
                               const BaselineClipRecorderConfig& cfg,
                               const std::string& image_ext,
                               const std::string& image_mode,
                               int target_frames,
                               int gap_frames,
                               std::size_t effective_max_queue_frames);

void writeBaselineDetectionColumns(std::ostream& os, const Detection* det);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_BASELINE_CLIP_RECORDER_IO_H_
