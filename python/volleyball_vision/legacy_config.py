"""把主线分层配置投影到可删除的旧 NX 流水线适配器。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, ResolvedConfig, find_project_root
from .model_manifest import resolve_model


def build_legacy_pipeline_config(
    config: ResolvedConfig,
    *,
    capture_stage: str | Path | None = None,
    frame_limit: int = 0,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root)
    adapter = config.data["adapters"]["legacy_pipeline"]
    base_path = (root / adapter["base_config"]).resolve()
    if not base_path.is_file():
        raise ConfigError(f"旧管线基础配置不存在: {base_path}")
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("旧管线基础配置顶层必须是对象")
    result = deepcopy(payload)
    cameras = config.data["cameras"]
    left = cameras["left"]
    right = cameras["right"]
    result["camera"].update(
        {
            "left_index": left["index"],
            "right_index": right["index"],
            "serial_left": left["serial"],
            "serial_right": right["serial"],
            "exposure_us": left["exposure_us"],
            "gain_db": left["gain_db"],
            "use_trigger": cameras["sync"]["method"] == "hardware_trigger",
            "trigger_source": cameras["sync"]["trigger_source"],
            "trigger_activation": cameras["sync"]["trigger_activation"],
            "trigger_frequency_hz": round(config.data["runtime"]["nominal_fps"]),
            "width": left["resolution"][0],
            "height": left["resolution"][1],
        }
    )
    result["calibration"]["file"] = str((root / config.data["calibration"]["file"]).resolve())
    result["rectify"].update(
        {
            "backend": config.data["rectification"]["backend"].upper(),
            "output_width": config.data["rectification"]["output_resolution"][0],
            "output_height": config.data["rectification"]["output_resolution"][1],
        }
    )
    _, artifact = resolve_model(config, project_root=root)
    detector = config.data["detector"]
    result["detector"].update(
        {
            "engine_path": str(artifact),
            "input_size": detector["input_resolution"][0],
            "input_format": detector["input_format"].removesuffix("8"),
            "confidence_threshold": detector["confidence_threshold"],
            "nms_threshold": detector["nms_iou_threshold"],
            "max_detections": detector["max_detections"],
        }
    )
    association = config.data["stereo"]["association"]
    triangulation = config.data["stereo"]["triangulation"]
    dual = result["detector"]["dual_yolo"]
    dual.update(
        {
            "epipolar_y_tolerance": association["epipolar_tolerance_px"],
            "max_size_ratio": association["max_size_ratio"],
            "min_shifted_iou": association["min_shifted_iou"],
            "subpixel_patch_radius": triangulation["subpixel"]["patch_radius"],
            "subpixel_search_radius_px": triangulation["subpixel"]["search_radius_px"],
            "subpixel_max_points": triangulation["subpixel"]["max_points"],
            "subpixel_min_points": triangulation["subpixel"]["min_points"],
            "subpixel_min_confidence": triangulation["subpixel"]["min_confidence"],
            "subpixel_max_stddev_px": triangulation["subpixel"]["max_stddev_px"],
            "subpixel_time_budget_ms": triangulation["subpixel"]["time_budget_ms"],
        }
    )
    result["stereo"]["max_disparity"] = int(association["max_disparity_px"])
    result["fusion"].update(
        {
            "min_depth": config.data["stereo"]["min_depth_m"],
            "max_depth": config.data["stereo"]["max_depth_m"],
            "object_diameter": config.data["stereo"]["ball_diameter_m"],
        }
    )
    result["performance"].update(
        {
            "pwm_frequency": config.data["runtime"]["nominal_fps"],
            "target_fps": round(config.data["runtime"]["nominal_fps"]),
        }
    )
    # 主线从不启用旧版 CSV 录制器；原始层只能通过 capture_v21 发布。
    result.setdefault("recording", {})["enabled"] = False
    result.setdefault("baseline_recording", {})["enabled"] = False
    prediction = config.data["prediction"]
    result.setdefault("prediction", {})["enable"] = bool(
        prediction["enabled"] and prediction["owner"] == "nx"
    )
    publisher = config.data["publisher"]
    ros2 = result.setdefault("ros2", {})
    ros2["enable"] = bool(
        publisher["enabled"] and publisher["backend"] == "ros2"
    )
    nx_observation = ros2.setdefault("nx_observation", {})
    nx_observation["enable"] = ros2["enable"]
    nx_observation["topic"] = publisher["topic"]
    nx_observation["frame_id"] = publisher["frame_id"]
    ros2.setdefault("control_goal", {})["enable"] = bool(
        ros2["enable"] and publisher["control_goal_enabled"]
    )
    diagnostics = config.data["diagnostics"]
    ros2.setdefault("diagnostic", {})["enable"] = bool(
        ros2["enable"] and diagnostics["ros2_depth_enabled"]
    )
    result.setdefault("debug_realtime_dump", {})["enabled"] = diagnostics[
        "realtime_dump_enabled"
    ]
    if capture_stage is not None:
        # 采集模式禁用会丢旧帧的异步路径和旧 CSV 录制器。
        result["performance"]["drop_stale_roi_frames"] = False
        result["performance"]["async_roi_stage2"] = False
        result.setdefault("prediction", {})["enable"] = False
        result.setdefault("ros2", {})["enable"] = False
        result["ros2"]["nx_observation"]["enable"] = False
        result["ros2"]["control_goal"]["enable"] = False
        result["ros2"]["diagnostic"]["enable"] = False
        result["debug_realtime_dump"]["enabled"] = False
        result["capture_v21"] = {
            "enabled": True,
            "output_dir": str(Path(capture_stage).resolve()),
            "frame_limit": max(0, int(frame_limit)),
            "queue_capacity": config.data["runtime"]["queue"]["capacity"],
            "overflow_policy": config.data["runtime"]["queue"]["overflow_policy"],
            "raw_frames": config.data["recording"]["raw_frames"],
            "image_product": config.data["recording"]["image_product"],
            "image_format": config.data["recording"]["image_format"],
            "left_camera_id": left["id"],
            "right_camera_id": right["id"],
            "pipeline_id": detector["model_id"],
            "loss_policy": config.data["recording"]["loss_policy"],
            "exposure_timestamp_source": cameras["sync"][
                "exposure_timestamp_source"
            ],
        }
    return result


def write_legacy_pipeline_config(
    destination: str | Path,
    config: ResolvedConfig,
    **kwargs: Any,
) -> Path:
    path = Path(destination)
    payload = build_legacy_pipeline_config(config, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return path

