"""把分层配置投影为原生 NX 主线进程的最小、可审计 YAML。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .calibration import resolve_calibration_source
from .config import ConfigError, ResolvedConfig, find_project_root
from .model_manifest import resolve_model
from .triangulation_config import SUPPORTED_LIVE_METHODS


MAINLINE_CONFIG_SCHEMA_VERSION = 4


def _selection_order(triangulation: dict[str, Any]) -> list[str]:
    primary = str(triangulation["primary_method"])
    order = [primary, *triangulation["fallback_methods"]]
    if len(order) != len(set(order)):
        raise ConfigError("原生主线深度选择顺序不得重复")
    candidates = set(triangulation["candidate_methods"])
    unsupported = sorted(candidates - SUPPORTED_LIVE_METHODS)
    if unsupported:
        raise ConfigError(
            "原生主线尚不支持这些深度候选: " + ", ".join(unsupported)
        )
    if candidates != set(order):
        missing = sorted(candidates - set(order))
        raise ConfigError(
            "原生主线不允许计算后丢弃证据的深度候选"
            + (": " + ", ".join(missing) if missing else "")
        )
    return order


def build_mainline_pipeline_config(
    config: ResolvedConfig,
    *,
    capture_stage: str | Path | None = None,
    frame_limit: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """生成只含原生进程实际消费字段的配置。

    路径全部在 Python 侧解析为绝对路径，避免进程工作目录改变含义。采集模式
    必须同时提供独占 staging 与有限帧数；实时模式两者都不得提供。
    """

    root = find_project_root(project_root)
    mode = str(config.data["runtime"]["mode"])
    is_capture = mode == "capture"
    if mode not in {"runtime", "capture"}:
        raise ConfigError("原生 NX 主线只接受 runtime 或 capture profile")
    if is_capture:
        if capture_stage is None or frame_limit is None or frame_limit <= 0:
            raise ConfigError("capture 主线配置必须提供 staging 和正 frame_limit")
    elif capture_stage is not None or frame_limit is not None:
        raise ConfigError("runtime 主线配置不得携带 capture staging/frame_limit")

    recording = config.data["recording"]
    if is_capture and (
        not recording["enabled"]
        or recording["loss_policy"] != "fail_trial"
        or recording["raw_frames"] != "all"
    ):
        raise ConfigError("原生 capture 必须启用 fail_trial 和全部原始帧")

    cameras = config.data["cameras"]
    sync = cameras["sync"]
    transport = cameras["transport"]
    left = cameras["left"]
    right = cameras["right"]
    detector = config.data["detector"]
    stereo = config.data["stereo"]
    association = stereo["association"]
    triangulation = stereo["triangulation"]
    max_disparity = float(association["max_disparity_px"])
    if not max_disparity.is_integer():
        raise ConfigError("原生 CUDA 深度要求 max_disparity_px 为整数")
    if not association["record_rejected"]:
        raise ConfigError("原生主线必须保存全部关联尝试")
    if not triangulation["propagate_covariance"]:
        raise ConfigError("原生主线必须传播直接观测协方差")
    sync_error_ns = sync["estimated_error_ns"]
    if (
        isinstance(sync_error_ns, bool)
        or not isinstance(sync_error_ns, (int, float))
        or not math.isfinite(sync_error_ns)
        or sync_error_ns < 0
        or not float(sync_error_ns).is_integer()
    ):
        raise ConfigError(
            "原生主线要求 cameras.sync.estimated_error_ns 为非负整数纳秒"
        )

    model, artifact = resolve_model(config, project_root=root)
    calibration_file, calibration_sha256 = resolve_calibration_source(
        config, project_root=root
    )
    source_epoch_file = Path(
        config.data["adapters"]["mainline_pipeline"]["source_epoch_file"]
    )
    stage_path = None if capture_stage is None else str(Path(capture_stage).resolve())
    publisher = config.data["publisher"]
    publish_enabled = bool(publisher["enabled"])
    if is_capture and publish_enabled:
        raise ConfigError("capture 主线不得同时发布 ROS2 观测")

    return {
        "schema_version": MAINLINE_CONFIG_SCHEMA_VERSION,
        "profile": config.profile,
        "config_sha256": config.fingerprint_sha256,
        "runtime": {
            "mode": mode,
            "nominal_fps": config.data["runtime"]["nominal_fps"],
            "source_timeout_ms": config.data["runtime"]["source_timeout_ms"],
            "max_consecutive_timeouts": config.data["runtime"][
                "max_consecutive_timeouts"
            ],
            "queue_capacity": config.data["runtime"]["queue"]["capacity"],
            "overflow_policy": config.data["runtime"]["queue"][
                "overflow_policy"
            ],
            "frame_limit": frame_limit,
            "capture_stage": stage_path,
        },
        "camera": {
            "left_id": left["id"],
            "right_id": right["id"],
            "left_serial": left["serial"],
            "right_serial": right["serial"],
            "left_index": left["index"],
            "right_index": right["index"],
            "width": left["resolution"][0],
            "height": left["resolution"][1],
            "exposure_us": left["exposure_us"],
            "gain_db": left["gain_db"],
            "image_node_num": transport["image_node_num"],
            "embedded_info_clear_rows": transport["embedded_info_clear_rows"],
            "trigger_source": sync["trigger_source"],
            "trigger_activation": sync["trigger_activation"],
            "exposure_timestamp_source": sync["exposure_timestamp_source"],
            "shutter_type": sync["shutter_type"],
            "exposure_delay_status": sync["exposure_delay_status"],
            "exposure_delay_ns": sync["exposure_delay_ns"],
            "exposure_delay_uncertainty_ns": sync[
                "exposure_delay_uncertainty_ns"
            ],
        },
        "clocks": {
            "reference": sync["clock_domain"],
            "left_device": f"{left['id']}-device-clock",
            "right_device": f"{right['id']}-device-clock",
            "capture_realtime": "nx-realtime",
            "host_monotonic": "nx-monotonic",
        },
        "trigger": {
            "gpio_chip": sync["gpio_chip"],
            "gpio_line": sync["gpio_line"],
            "frequency_hz": config.data["runtime"]["nominal_fps"],
            "duty_cycle": sync["pwm_duty_cycle"],
        },
        "calibration": {
            "file": str(calibration_file),
            "sha256": calibration_sha256,
            "rig_frame": config.data["calibration"]["rig_frame"],
            "output_width": config.data["rectification"]["output_resolution"][0],
            "output_height": config.data["rectification"]["output_resolution"][1],
        },
        "detector": {
            "engine_file": str(artifact),
            "engine_sha256": model.get("sha256"),
            "pipeline_id": detector["model_id"],
            "input_size": detector["input_resolution"][0],
            "output_contract": model["output_contract"],
            "class_count": len(model["class_names"]),
            "confidence_threshold": detector["confidence_threshold"],
            "nms_iou_threshold": detector["nms_iou_threshold"],
            "max_detections": detector["max_detections"],
            "class_id": detector["class_id"],
            "use_dla": config.data["platform"]["accelerator"] == "cuda_dla",
            "dla_core": 0,
        },
        "association": {
            "max_capture_delta_ns": int(sync_error_ns),
            "epipolar_tolerance_px": association["epipolar_tolerance_px"],
            "adaptive_epipolar_height_ratio": association[
                "adaptive_epipolar_height_ratio"
            ],
            "max_size_ratio": association["max_size_ratio"],
            "min_shifted_iou": association["min_shifted_iou"],
            "max_disparity_px": int(max_disparity),
            "ball_diameter_m": stereo["ball_diameter_m"],
            "bbox_scale": association["bbox_scale"],
            "bbox_disparity_consistency_ratio": association[
                "bbox_disparity_consistency_ratio"
            ],
            "bbox_disparity_consistency_min_px": association[
                "bbox_disparity_consistency_min_px"
            ],
            "bbox_disparity_penalty_scale": association[
                "bbox_disparity_penalty_scale"
            ],
        },
        "depth": {
            "method_order": _selection_order(dict(triangulation)),
            "circle_max_roi_pixels": triangulation["circle_fit"][
                "max_roi_pixels"
            ],
            "anchor_stddev_px": triangulation["measurement_uncertainty"][
                "anchor_stddev_px"
            ],
            "disparity_stddev_px": triangulation["measurement_uncertainty"][
                "disparity_stddev_px"
            ],
            "min_depth_m": stereo["min_depth_m"],
            "max_depth_m": stereo["max_depth_m"],
        },
        "publisher": {
            "enabled": publish_enabled,
            "topic": publisher["topic"],
            "frame_id": publisher["frame_id"],
            "source_epoch_file": str(source_epoch_file),
        },
        "diagnostics": {
            "metrics_interval_frames": config.data["diagnostics"][
                "metrics_interval_frames"
            ]
        },
    }


def write_mainline_pipeline_config(
    destination: str | Path,
    config: ResolvedConfig,
    **kwargs: Any,
) -> Path:
    path = Path(destination)
    payload = build_mainline_pipeline_config(config, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return path
