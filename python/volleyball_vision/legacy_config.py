"""把主线分层配置投影到可删除的旧 NX 流水线适配器。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, ResolvedConfig, find_project_root
from .legacy_activation_policy import apply_legacy_activation_policy
from .model_manifest import resolve_model
from .triangulation_config import legacy_depth_projection


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
    apply_legacy_activation_policy(result, config)
    cameras = config.data["cameras"]
    transport = cameras["transport"]
    sync = cameras["sync"]
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
            "use_trigger": sync["method"] == "hardware_trigger",
            "trigger_source": sync["trigger_source"],
            "trigger_activation": sync["trigger_activation"],
            "shutter_type": sync["shutter_type"],
            "exposure_timestamp_source": sync["exposure_timestamp_source"],
            "exposure_delay_status": sync["exposure_delay_status"],
            "exposure_delay_ns": sync["exposure_delay_ns"],
            "exposure_delay_uncertainty_ns": sync[
                "exposure_delay_uncertainty_ns"
            ],
            "exposure_midpoint_offset_ns": round(
                float(left["exposure_us"]) * 1000.0 / 2.0
            ),
            "trigger_frequency_hz": round(config.data["runtime"]["nominal_fps"]),
            "trigger_chip": sync["gpio_chip"],
            "trigger_line": sync["gpio_line"],
            "image_node_num": transport["image_node_num"],
            "embedded_info_clear_rows": transport["embedded_info_clear_rows"],
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
    try:
        depth_projection = legacy_depth_projection(
            triangulation,
            dual["depth_modes"],
        )
    except (KeyError, ValueError) as error:
        raise ConfigError(f"无法投影旧 NX 深度候选: {error}") from error
    dual["depth_modes"] = depth_projection["depth_modes"]
    dual.update(
        {
            "depth_solver": depth_projection["depth_solver"],
            "subpixel_enabled": depth_projection["subpixel_enabled"],
            "fallback_epipolar_search": depth_projection[
                "fallback_epipolar_search"
            ],
            "center_refine": depth_projection["center_refine"],
            "epipolar_y_tolerance": association["epipolar_tolerance_px"],
            "max_size_ratio": association["max_size_ratio"],
            "min_shifted_iou": association["min_shifted_iou"],
            "bbox_disparity_consistency_ratio": association[
                "bbox_disparity_consistency_ratio"
            ],
            "bbox_disparity_consistency_min_px": association[
                "bbox_disparity_consistency_min_px"
            ],
            "bbox_disparity_penalty_scale": association[
                "bbox_disparity_penalty_scale"
            ],
            "subpixel_patch_radius": triangulation["subpixel"]["patch_radius"],
            "subpixel_search_radius_px": triangulation["subpixel"]["search_radius_px"],
            "subpixel_max_points": triangulation["subpixel"]["max_points"],
            "subpixel_min_points": triangulation["subpixel"]["min_points"],
            "subpixel_min_confidence": triangulation["subpixel"]["min_confidence"],
            "subpixel_max_disp_delta_px": triangulation["subpixel"][
                "max_disparity_delta_px"
            ],
            "subpixel_max_disp_delta_ratio": triangulation["subpixel"][
                "max_disparity_delta_ratio"
            ],
            "subpixel_max_depth_delta_m": triangulation["subpixel"][
                "max_depth_delta_m"
            ],
            "subpixel_max_stddev_px": triangulation["subpixel"]["max_stddev_px"],
            "subpixel_time_budget_ms": triangulation["subpixel"]["time_budget_ms"],
            "fallback_search_margin_px": triangulation["fallback_search"][
                "margin_px"
            ],
            "fallback_max_width_px": triangulation["fallback_search"][
                "max_width_px"
            ],
            "circle_max_roi_pixels": triangulation["circle_fit"][
                "max_roi_pixels"
            ],
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
            "pwm_duty_cycle": sync["pwm_duty_cycle"],
            "target_fps": round(config.data["runtime"]["nominal_fps"]),
        }
    )
    result.setdefault("capture_v21", {})["enabled"] = False
    publisher = config.data["publisher"]
    ros2 = result.setdefault("ros2", {})
    ros2["enable"] = bool(
        publisher["enabled"] and publisher["backend"] == "ros2"
    )
    nx_observation = ros2.setdefault("nx_observation", {})
    nx_observation["enable"] = ros2["enable"]
    nx_observation["topic"] = publisher["topic"]
    nx_observation["frame_id"] = publisher["frame_id"]
    diagnostics = config.data["diagnostics"]
    result["performance"]["log_interval"] = diagnostics[
        "metrics_interval_frames"
    ]
    result.setdefault("debug_realtime_dump", {})["enabled"] = diagnostics[
        "realtime_dump_enabled"
    ]
    if capture_stage is not None:
        # 采集模式禁用会丢旧帧的异步路径和所有非契约诊断输出。
        result["performance"]["drop_stale_roi_frames"] = False
        result["performance"]["async_roi_stage2"] = False
        result.setdefault("ros2", {})["enable"] = False
        result["ros2"]["nx_observation"]["enable"] = False
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

