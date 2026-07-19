#ifndef STEREO_3D_PIPELINE_MAIN_RUNTIME_OVERRIDES_H_
#define STEREO_3D_PIPELINE_MAIN_RUNTIME_OVERRIDES_H_

#include "main_cli_options.h"
#include "main_realtime_debug_dump.h"
#include "pipeline/pipeline_config.h"
#include "utils/baseline_clip_recorder.h"

void applyBaselineClipOverrides(
    const MainCliOptions& cli,
    stereo3d::PipelineConfig& pipeline_cfg,
    stereo3d::BaselineClipRecorderConfig& baseline_cfg);

void applyRealtimeDebugDumpOverrides(
    const MainCliOptions& cli,
    RealtimeDebugDumpConfig& realtime_dump_cfg);

#endif  // STEREO_3D_PIPELINE_MAIN_RUNTIME_OVERRIDES_H_
