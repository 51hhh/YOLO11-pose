"""生成可审计的图像流水线启用报告。

报告只描述最终 profile 及其旧 NX 配置投影（L1 证据），不把资源能够分配、
目标机已经链接或某一帧实际执行过算子混为一谈。实机执行仍需 doctor、启动日志、
逐阶段计数和 Nsight 时间线共同证明。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ResolvedConfig
from .legacy_config import build_legacy_pipeline_config
from .triangulation_config import CANDIDATE_TO_LEGACY_DEPTH_MODE


def _stage(
    stage_id: str,
    *,
    enabled: bool,
    executor: str,
    input_product: str,
    output_product: str,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "enabled": enabled,
        "executor": executor,
        "input": input_product,
        "output": output_product,
    }


def _stream(
    stream_id: str,
    *,
    allocated: bool,
    per_frame_work: bool,
    backend: str,
    work: str,
) -> dict[str, Any]:
    return {
        "id": stream_id,
        "allocated": allocated,
        "per_frame_work": per_frame_work,
        "backend": backend,
        "work": work,
    }


def _candidate_report(
    triangulation: dict[str, Any],
    legacy_depth_modes: dict[str, bool],
) -> list[dict[str, Any]]:
    primary = triangulation["primary_method"]
    fallbacks = set(triangulation["fallback_methods"])
    out: list[dict[str, Any]] = []
    for method in triangulation["candidate_methods"]:
        legacy_mode = CANDIDATE_TO_LEGACY_DEPTH_MODE[method]
        if method == primary:
            role = "primary"
        elif method in fallbacks:
            role = "fallback"
        else:
            role = "computed_only"
        out.append(
            {
                "method": method,
                "legacy_depth_mode": legacy_mode,
                "compute_enabled": bool(legacy_depth_modes.get(legacy_mode, False)),
                "selection_role": role,
            }
        )
    return out


def build_pipeline_report(
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """返回 profile 的机器可读流水线启用摘要。"""

    data = config.data
    runtime = data["runtime"]
    cameras = data["cameras"]
    sync = cameras["sync"]
    transport = cameras["transport"]
    detector = data["detector"]
    stereo = data["stereo"]
    triangulation = stereo["triangulation"]
    live_pipeline = cameras["backend"] != "recorded"

    legacy: dict[str, Any] | None = None
    if live_pipeline:
        legacy = build_legacy_pipeline_config(
            config,
            project_root=project_root,
        )

    if legacy is None:
        legacy_camera: dict[str, Any] = {}
        legacy_detector: dict[str, Any] = {}
        legacy_dual: dict[str, Any] = {}
        legacy_performance: dict[str, Any] = {}
        legacy_stereo: dict[str, Any] = {}
        legacy_depth_modes: dict[str, bool] = {}
    else:
        legacy_camera = legacy["camera"]
        legacy_detector = legacy["detector"]
        legacy_dual = legacy_detector["dual_yolo"]
        legacy_performance = legacy["performance"]
        legacy_stereo = legacy["stereo"]
        legacy_depth_modes = legacy_dual["depth_modes"]

    trigger_enabled = live_pipeline and bool(legacy_camera.get("use_trigger", False))
    dual_yolo_enabled = live_pipeline and bool(legacy_dual.get("enabled", False))
    depth_enabled = dual_yolo_enabled and bool(legacy_dual.get("use_for_depth", False))
    roi_only = live_pipeline and legacy_stereo.get("strategy") == "roi_only"
    async_roi = live_pipeline and bool(
        legacy_performance.get("async_roi_stage2", False)
    )
    p2_scaffold = live_pipeline and bool(
        legacy_performance.get("p2_feature_job_scaffold_enabled", False)
    )
    p2_realtime = live_pipeline and bool(
        legacy_performance.get("p2_realtime_lane_decision_enabled", False)
    )
    p2_diagnostic = live_pipeline and bool(
        legacy_performance.get("p2_diagnostic_lane_decision_enabled", False)
    )
    cuda_template = "cuda_template_match" in triangulation["candidate_methods"]
    neural_xfeat = bool(
        legacy is not None
        and legacy["neural_feature_matching_xfeat"]["enabled"]
    )
    neural_superpoint = bool(
        legacy is not None
        and legacy["neural_feature_matching_superpoint"]["enabled"]
    )
    neural_aliked = bool(
        legacy is not None
        and legacy["neural_feature_matching_aliked"]["enabled"]
    )

    stages = [
        _stage(
            "gpio_trigger_and_exposure",
            enabled=trigger_enabled,
            executor="NX GPIO + Hikvision external trigger",
            input_product="scheduled_trigger_edge",
            output_product="stereo_exposure",
        ),
        _stage(
            "usb_capture_and_framecounter_pairing",
            enabled=live_pipeline,
            executor="MVS callbacks + grab thread",
            input_product="left/right camera payloads",
            output_product="paired_bayer_frame",
        ),
        _stage(
            "embedded_row_clear",
            enabled=live_pipeline
            and int(transport["embedded_info_clear_rows"]) > 0,
            executor="capture adapter",
            input_product="paired_bayer_frame",
            output_product="clean_bayer_frame",
        ),
        _stage(
            "bayer_to_bgr",
            enabled=live_pipeline,
            executor="custom CUDA kernel",
            input_product="clean_bayer_frame",
            output_product="full_resolution_bgr",
        ),
        _stage(
            "stereo_rectification",
            enabled=live_pipeline,
            executor=f"VPI {data['rectification']['backend'].upper()} remap",
            input_product="full_resolution_bgr",
            output_product="rectified_bgr",
        ),
        _stage(
            "gray_conversion",
            enabled=live_pipeline,
            executor=f"VPI {data['rectification']['backend'].upper()} convert",
            input_product="rectified_bgr",
            output_product="rectified_gray8",
        ),
        _stage(
            "dual_yolo",
            enabled=dual_yolo_enabled,
            executor="two TensorRT contexts on separate CUDA streams",
            input_product="rectified_bgr",
            output_product="left/right_detection_sets",
        ),
        _stage(
            "stereo_pair_gate",
            enabled=depth_enabled,
            executor="CPU pair enumeration and geometry gates",
            input_product="left/right_detection_sets",
            output_product="pair_attempts",
        ),
        _stage(
            "roi_depth_candidates",
            enabled=depth_enabled and any(legacy_depth_modes.values()),
            executor="CUDA ROI kernels + configured fallback helpers",
            input_product="rectified_gray8 + accepted_pairs",
            output_product="depth_candidate_set",
        ),
        _stage(
            "direct_triangulation",
            enabled=depth_enabled,
            executor="candidate selector + calibrated projection",
            input_product="depth_candidate_set",
            output_product="direct_3d_observation",
        ),
        _stage(
            "recorded_observation_replay",
            enabled=not live_pipeline,
            executor="deterministic replay adapter",
            input_product="v2.1 captured trial",
            output_product="recorded_direct_observation",
        ),
    ]

    # PipelineStreams 在旧实时进程初始化时统一创建；per_frame_work 单独说明当前
    # profile 是否会给该 stream 提交任务。辅助 ROI/诊断 stream 则按配置创建。
    streams = [
        _stream(
            "vpi_rectification",
            allocated=live_pipeline,
            per_frame_work=live_pipeline,
            backend=data["rectification"]["backend"],
            work="stereo remap and BGR-to-Gray conversion",
        ),
        _stream(
            "vpi_full_frame_disparity",
            allocated=live_pipeline,
            per_frame_work=live_pipeline and not roi_only,
            backend="cuda",
            work="full/half-resolution disparity (disabled by ROI_ONLY)",
        ),
        _stream(
            "cuda_bayer",
            allocated=live_pipeline,
            per_frame_work=live_pipeline,
            backend="cuda",
            work="left/right BayerRG8-to-BGR8",
        ),
        _stream(
            "cuda_yolo_left",
            allocated=live_pipeline,
            per_frame_work=dual_yolo_enabled,
            backend="tensorrt_cuda",
            work="left preprocess, inference, decode and D2H",
        ),
        _stream(
            "cuda_yolo_right",
            allocated=live_pipeline,
            per_frame_work=dual_yolo_enabled,
            backend="tensorrt_cuda",
            work="right preprocess, inference, decode and D2H",
        ),
        _stream(
            "cuda_pipeline_gpu",
            allocated=live_pipeline,
            per_frame_work=live_pipeline,
            backend="cuda",
            work="rectification completion event and optional tracker/full-frame coordination",
        ),
        _stream(
            "cuda_fuse",
            allocated=live_pipeline,
            per_frame_work=depth_enabled,
            backend="cuda",
            work="synchronous ROI/fusion coordination",
        ),
        _stream(
            "cuda_p2_ncc",
            allocated=live_pipeline,
            per_frame_work=depth_enabled
            and p2_scaffold
            and p2_realtime
            and cuda_template,
            backend="cuda",
            work="inline CUDA template/NCC for eligible pairs",
        ),
        _stream(
            "cuda_p2_xfeat",
            allocated=live_pipeline,
            per_frame_work=depth_enabled
            and p2_scaffold
            and p2_realtime
            and neural_xfeat,
            backend="cuda_tensorrt",
            work="XFeat inline worker",
        ),
        _stream(
            "cuda_p2_superpoint",
            allocated=live_pipeline,
            per_frame_work=depth_enabled
            and p2_scaffold
            and p2_realtime
            and neural_superpoint,
            backend="cuda_tensorrt",
            work="SuperPoint inline worker",
        ),
        _stream(
            "cuda_p2_aliked",
            allocated=live_pipeline,
            per_frame_work=depth_enabled
            and p2_scaffold
            and p2_realtime
            and neural_aliked,
            backend="cuda_tensorrt",
            work="ALIKED inline worker",
        ),
        _stream(
            "async_roi_copy",
            allocated=async_roi,
            per_frame_work=async_roi,
            backend="cuda",
            work="FrameSlot-to-worker snapshot copies",
        ),
        _stream(
            "async_roi_core",
            allocated=async_roi,
            per_frame_work=async_roi,
            backend="cuda",
            work="asynchronous ROI candidate computation",
        ),
        _stream(
            "p2_diagnostic_copy",
            allocated=p2_diagnostic,
            per_frame_work=p2_diagnostic,
            backend="cuda",
            work="diagnostic source snapshots",
        ),
        _stream(
            "p2_diagnostic_core",
            allocated=p2_diagnostic,
            per_frame_work=p2_diagnostic,
            backend="cuda",
            work="low-priority diagnostic candidates",
        ),
    ]

    publisher = data["publisher"]
    recording = data["recording"]
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "evidence": {
            "level": "L1_configuration_projection",
            "runtime_observed": False,
            "requires_target_runtime_confirmation": live_pipeline,
            "note": (
                "此报告不证明 SDK、二进制链接、资源初始化或逐帧算子实际执行"
            ),
        },
        "profile": config.profile,
        "config_sha256": config.fingerprint_sha256,
        "runtime": {
            "mode": runtime["mode"],
            "nominal_fps": runtime["nominal_fps"],
            "queue_capacity": runtime["queue"]["capacity"],
            "overflow_policy": runtime["queue"]["overflow_policy"],
            "live_pipeline": live_pipeline,
        },
        "frame_identity": {
            "sync_method": sync["method"],
            "trigger_source": sync["trigger_source"],
            "trigger_activation": sync["trigger_activation"],
            "frame_identity_source": sync["frame_identity_source"],
            "shutter_type": sync["shutter_type"],
            "exposure_timestamp_source": sync["exposure_timestamp_source"],
            "exposure_delay_status": sync["exposure_delay_status"],
            "exposure_delay_ns": sync["exposure_delay_ns"],
            "exposure_delay_uncertainty_ns": sync[
                "exposure_delay_uncertainty_ns"
            ],
            "observation_timestamp_semantics": (
                "exposure_midpoint"
                if sync["shutter_type"] == "global"
                else "recorded"
            ),
            "transport": transport["interface"],
            "grab_strategy": transport["grab_strategy"],
            "embedded_info_clear_rows": transport["embedded_info_clear_rows"],
        },
        "image_products": {
            "source_resolution": list(cameras["left"]["resolution"]),
            "source_pixel_format": cameras["left"]["pixel_format"],
            "rectified_resolution": list(data["rectification"]["output_resolution"]),
            "rectified_bgr": live_pipeline,
            "rectified_gray8": live_pipeline,
            "full_frame_disparity": live_pipeline and not roi_only,
            "black_white_iou_image": False,
        },
        "stages": stages,
        "detector": {
            "enabled": dual_yolo_enabled,
            "backend": detector["backend"],
            "per_camera": detector["per_camera"],
            "input_format": detector["input_format"],
            "input_resolution": list(detector["input_resolution"]),
            "confidence_threshold": detector["confidence_threshold"],
            "nms_iou_threshold": detector["nms_iou_threshold"],
            "max_detections_per_camera": detector["max_detections"],
        },
        "association": {
            "enabled": depth_enabled,
            "epipolar_tolerance_px": stereo["association"]["epipolar_tolerance_px"],
            "max_size_ratio": stereo["association"]["max_size_ratio"],
            "min_shifted_iou": stereo["association"]["min_shifted_iou"],
            "max_disparity_px": stereo["association"]["max_disparity_px"],
            "shifted_iou_is_image": False,
        },
        "triangulation": {
            "enabled": depth_enabled,
            "backend": triangulation["backend"],
            "primary_method": triangulation["primary_method"],
            "fallback_order": list(triangulation["fallback_methods"]),
            "candidates": _candidate_report(
                dict(triangulation), legacy_depth_modes
            ),
            "legacy_enabled_depth_modes": sorted(
                name for name, enabled in legacy_depth_modes.items() if enabled
            ),
        },
        "concurrency": {
            "frame_slots": 3 if live_pipeline else 0,
            "async_roi_stage2": async_roi,
            "async_roi_buffers": (
                int(legacy_performance.get("async_roi_buffers", 0))
                if async_roi
                else 0
            ),
            "async_roi_deadline_ms": (
                float(legacy_performance.get("async_roi_deadline_ms", 0.0))
                if async_roi
                else None
            ),
            "drop_stale_roi_frames": bool(
                legacy_performance.get("drop_stale_roi_frames", False)
            ),
            "p2_split_workers": p2_scaffold,
            "p2_realtime_lane": p2_realtime,
            "p2_diagnostic_lane": p2_diagnostic,
            "streams": streams,
        },
        "tracking": {
            "high_level_state_estimation": bool(data["tracking"]["enabled"]),
            "filter": data["tracking"]["filter"],
            "legacy_image_sot": bool(
                legacy is not None and legacy["tracker"]["enabled"]
            ),
        },
        "outputs": {
            "publisher_enabled": bool(publisher["enabled"]),
            "publisher_backend": publisher["backend"],
            "publisher_topic": publisher["topic"],
            "capture_profile_intent": bool(recording["enabled"]),
            "capture_v21_armed": bool(
                legacy is not None and legacy["capture_v21"]["enabled"]
            ),
            "capture_v21_note": (
                "capture 命令创建 trial staging 后才注入输出目录并置为 true"
                if recording["enabled"] and live_pipeline
                else "not requested by this profile"
            ),
            "image_product": recording["image_product"],
            "raw_frames": recording["raw_frames"],
        },
    }
    return report
