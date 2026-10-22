"""把主线 profile 的功能启用意图投影为旧 NX 开关。

旧 YAML 仍保存尚未迁移的算法数值，但不能自行打开新的处理支路、写盘器或
调试输出。这个模块只控制“是否执行”，不调整候选算法的数值参数。
"""

from __future__ import annotations

from typing import Any

from .config import ConfigError, ResolvedConfig


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"旧 NX 基础配置 {key} 必须是对象")
    return value


def apply_legacy_activation_policy(
    payload: dict[str, Any],
    config: ResolvedConfig,
) -> None:
    """原地清除基础模板中未由主线 profile 授权的功能开关。"""

    runtime = config.data["runtime"]
    detector_config = config.data["detector"]
    triangulation = config.data["stereo"]["triangulation"]
    realtime = runtime["mode"] == "runtime"
    cuda_depth = triangulation["backend"] == "cuda"
    p2_realtime = "cuda_template_match" in triangulation["candidate_methods"]
    use_dla = config.data["platform"]["accelerator"] == "cuda_dla"

    _mapping(payload, "disparity_offset")["enabled"] = False
    detector = _mapping(payload, "detector")
    detector["use_dla"] = use_dla
    dual = _mapping(detector, "dual_yolo")
    dual.update(
        {
            "enabled": bool(detector_config["per_camera"]),
            "right_use_dla": use_dla,
            "use_for_depth": triangulation["backend"] != "recorded",
            "fallback_to_roi_match": False,
            "gpu_candidate_refine": cuda_depth,
            "roi_denoise": True,
            "log_matches": bool(config.data["diagnostics"]["processing_timing"]),
        }
    )

    _mapping(payload, "stereo")["strategy"] = "roi_only"
    for name in (
        "neural_feature_matching",
        "neural_feature_matching_xfeat",
        "neural_feature_matching_superpoint",
        "neural_feature_matching_aliked",
    ):
        _mapping(payload, name)["enabled"] = False

    period_ms = round(1000.0 / float(runtime["nominal_fps"]), 1)
    performance = _mapping(payload, "performance")
    performance.update(
        {
            "drop_stale_roi_frames": realtime,
            "async_roi_stage2": realtime,
            "async_roi_buffers": int(runtime["queue"]["capacity"]),
            "async_roi_deadline_ms": period_ms,
            "p2_feature_job_scaffold_enabled": p2_realtime,
            "p2_realtime_lane_decision_enabled": p2_realtime,
            "p2_diagnostic_lane_decision_enabled": False,
            "p2_selective_trigger": False,
            "p2_trigger_on_fallback": True,
            "p2_trigger_on_direct_pair": False,
            "p2_trigger_on_host_gray": False,
            "p2_trigger_on_bgr": False,
            "p2_trigger_on_pair_quality": False,
            "p2_trigger_on_no_valid_direct_pair": False,
            "p2_diagnostic_use_source_snapshot": False,
            "p2_diagnostic_results_enabled": False,
            "p2_diagnostic_results_path": "",
            "p2_diagnostic_point_debug_enabled": False,
            "p2_diagnostic_point_debug_path": "",
            "p2_diagnostic_artifacts_enabled": False,
            "p2_diagnostic_artifacts_dir": "",
            "p2_realtime_deadline_ms": period_ms,
        }
    )

    _mapping(payload, "display")["enable"] = False
