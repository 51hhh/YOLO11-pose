"""主线 C++ v6 staging 到冻结 capture v2.1 数据包的最终化适配。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .calibration import (
    build_capture_calibration,
    ensure_dataset_calibration,
    resolve_calibration_source,
)
from .capture_stage_io import (
    environment_record,
    parse_bool,
    parse_float,
    parse_int,
)
from .config import ResolvedConfig, find_project_root
from .dataset import DatasetError, TrialPackageWriter, with_pipeline_fingerprint
from .git_state import GitStateError, inspect_git_state
from .hashing import sha256_file
from .mainline_stage_contract import (
    COVARIANCE_3D_FIELDS,
    MAINLINE_DETECTION_COLUMNS,
    MAINLINE_STAGE_SCHEMA_VERSION,
)
from .mainline_stage_validation import (
    optional_float_group as _optional_group,
    optional_int as _optional_int,
    safe_stage_source as _safe_stage_source,
    validate_mainline_stage,
)
from .model_manifest import resolve_model
from .runtime import mainline_executable
from .stereo_measurements import (
    STEREO_MEASUREMENT_SCHEMA_VERSION,
    validate_stereo_measurements,
)
from .timebase import (
    fit_affine_clock_mapping,
    integer_midpoint_ns,
    map_timestamp_ns,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping_or_identity(
    source_times: list[int],
    reference_times: list[int],
    *,
    source_clock: str,
    reference_clock: str,
    mapping_id: str,
    source: str,
    validity_source_times_ns: Iterable[int] = (),
) -> dict[str, Any] | None:
    if source_clock == reference_clock:
        if source_times != reference_times:
            raise DatasetError(
                f"时钟 {source_clock!r} 已声明为参考时钟但数值并非恒等映射"
            )
        return None
    mapping = fit_affine_clock_mapping(
        source_times,
        reference_times,
        source_clock=source_clock,
        reference_clock=reference_clock,
        mapping_id=mapping_id,
        source=source,
        validity_source_times_ns=validity_source_times_ns,
    )
    if mapping is None:
        raise DatasetError(f"无法建立 {source_clock!r} 到参考时钟的映射")
    return mapping


def _mapped(value: int, mapping: Mapping[str, Any] | None) -> int:
    return value if mapping is None else map_timestamp_ns(value, mapping)


def build_mainline_capture_context(
    config: ResolvedConfig,
    *,
    trial_id: str,
    session_id: str,
    ball_id: str,
    trajectory_kind: str = "free_flight",
    conditions: Mapping[str, Any] | None = None,
    notes: str = "",
    verify_assets: bool = True,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """生成不可变的主线 v6 staging 身份与感知流水线指纹。"""

    root = find_project_root(project_root)
    try:
        git_state = inspect_git_state(root)
    except GitStateError as error:
        raise DatasetError(str(error)) from error
    if verify_assets and not git_state.clean:
        raise DatasetError(
            "真实采集要求干净 Git 工作树；请先提交、暂存到其他位置或使用 dry-run"
        )
    _, calibration_source_sha256 = resolve_calibration_source(
        config, project_root=root
    )
    entry, artifact = resolve_model(config, project_root=root)
    expected = entry.get("sha256")
    if artifact.is_file():
        weights_sha256 = sha256_file(artifact)
        if expected and expected != weights_sha256:
            raise DatasetError("模型文件 SHA-256 与 models/manifest.yaml 不一致")
    else:
        weights_sha256 = expected
    executable = mainline_executable(root, config)
    executable_sha256 = (
        sha256_file(executable) if executable.is_file() else None
    )
    if verify_assets:
        if not artifact.is_file():
            raise DatasetError(f"模型文件不存在: {artifact}")
        if not expected:
            raise DatasetError("模型清单必须先固定 weights SHA-256 才能发布真实数据")
        if expected != weights_sha256:
            raise DatasetError("模型文件 SHA-256 与 models/manifest.yaml 不一致")
        if executable_sha256 is None:
            raise DatasetError(f"原生主线可执行文件不存在: {executable}")

    detector = config.data["detector"]
    pipeline = with_pipeline_fingerprint(
        {
            "pipeline_id": detector["model_id"],
            "detector_family": entry["family"],
            "detector_version": str(entry.get("version", detector["model_id"])),
            "code_revision": git_state.revision_label,
            "config_sha256": config.fingerprint_sha256,
            "weights_sha256": weights_sha256,
            "input_size_px": list(detector["input_resolution"]),
            "output_contract": entry["output_contract"],
            "class_names": list(entry["class_names"]),
            "target_class_id": detector["class_id"],
            "target_class_name": entry["class_names"][detector["class_id"]],
            "resize_mode": detector["resize_mode"],
            "score_threshold": detector["confidence_threshold"],
            "nms_iou_threshold": detector["nms_iou_threshold"],
            "center_method": detector["center_method"],
        }
    )
    context = {
        "stage_schema_version": MAINLINE_STAGE_SCHEMA_VERSION,
        "stereo_measurement_schema_version": STEREO_MEASUREMENT_SCHEMA_VERSION,
        "trial_id": trial_id,
        "session_id": session_id,
        "ball_id": ball_id,
        "calibration_id": config.data["calibration"]["id"],
        "calibration_source_sha256": calibration_source_sha256,
        "captured_at_utc": _utc_now(),
        "trajectory_kind": trajectory_kind,
        "conditions": dict(conditions or {}),
        "notes": notes,
        "profile": config.profile,
        "config_sha256": config.fingerprint_sha256,
        "code_worktree_clean": git_state.clean,
        "runtime_executable_sha256": executable_sha256,
        "exposure_timestamp": {
            "shutter_type": config.data["cameras"]["sync"]["shutter_type"],
            "source": config.data["cameras"]["sync"][
                "exposure_timestamp_source"
            ],
            "delay_status": config.data["cameras"]["sync"][
                "exposure_delay_status"
            ],
            "delay_ns": config.data["cameras"]["sync"]["exposure_delay_ns"],
            "delay_uncertainty_ns": config.data["cameras"]["sync"][
                "exposure_delay_uncertainty_ns"
            ],
            "observation_timestamp": "exposure_midpoint",
        },
        "perception_pipeline": pipeline,
    }
    if not verify_assets:
        # 预演允许目标机 artifact 尚未安装，但该 staging 永远不能发布。
        context["planning_only"] = True
    return context


def finalize_mainline_capture_stage(
    stage: str | Path,
    dataset_root: str | Path,
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """严格校验 v6 逐事实 staging，并原子发布为 capture v2.1 trial。"""

    evidence = validate_mainline_stage(
        stage, config, project_root=project_root
    )
    directory = evidence.directory
    context = evidence.context
    expected_exposure = evidence.expected_exposure
    frames = evidence.frames
    detections = evidence.detections
    associations = evidence.associations
    observations = evidence.observations
    measurement_file = directory / "stereo_measurements_v1.csv"
    capture_calibration = build_capture_calibration(
        config,
        project_root=project_root,
    )
    has_measurement_inputs = validate_stereo_measurements(
        measurement_file,
        observations,
        capture_calibration,
    )
    declared_measurement_version = context.get(
        "stereo_measurement_schema_version"
    )
    if declared_measurement_version not in {
        None,
        STEREO_MEASUREMENT_SCHEMA_VERSION,
    }:
        raise DatasetError("mainline 双目测量扩展版本不兼容")
    if declared_measurement_version is not None and not has_measurement_inputs:
        raise DatasetError("mainline 声明了双目测量扩展但文件缺失")
    camera_ids = evidence.camera_ids
    frame_by_key = evidence.frame_by_key
    frame_pairs = evidence.frame_pairs
    indexes = evidence.indexes
    paired_rows = evidence.paired_rows
    edge_monotonic = evidence.edge_monotonic
    sync_limit = evidence.sync_limit_ns
    reference_clock = evidence.reference_clock
    left_device_clock = evidence.left_device_clock
    right_device_clock = evidence.right_device_clock
    capture_clock = evidence.capture_clock
    monotonic_clock = evidence.monotonic_clock

    left_hardware = [parse_int(left, "hardware_timestamp_ns") for left, _ in paired_rows]
    right_hardware = [parse_int(right, "hardware_timestamp_ns") for _, right in paired_rows]
    reference_left = list(left_hardware)
    reference_right = [
        left_value - parse_int(left, "stereo_hardware_timestamp_residual_ns")
        for left_value, (left, _) in zip(left_hardware, paired_rows)
    ]
    if any(value < 0 for value in reference_right):
        raise DatasetError("mainline 右目映射设备时间为负")
    reference_midpoints = [
        integer_midpoint_ns(left, right)
        for left, right in zip(reference_left, reference_right)
    ]
    capture_midpoints = [
        integer_midpoint_ns(
            parse_int(left, "capture_timestamp_ns"),
            parse_int(right, "capture_timestamp_ns"),
        )
        for left, right in paired_rows
    ]
    realtime_values = [
        parse_int(row, key)
        for row in frames
        for key in (
            "exposure_start_ns",
            "capture_timestamp_ns",
            "receive_timestamp_ns",
        )
    ]
    realtime_values.extend(
        value
        for row in frames
        if (value := _optional_int(row, "inference_done_timestamp_ns")) is not None
    )
    realtime_values.extend(
        parse_int(left, "trigger_edge_realtime_ns") for left, _ in paired_rows
    )
    monotonic_values = [
        parse_int(row, key)
        for row in frames
        for key in ("receive_monotonic_ns",)
    ]
    monotonic_values.extend(
        parse_int(left, key)
        for left, _ in paired_rows
        for key in ("trigger_scheduled_monotonic_ns", "trigger_edge_monotonic_ns")
    )
    monotonic_values.extend(
        edge + offset
        for left, _ in paired_rows
        for edge, uncertainty in (
            (
                parse_int(left, "trigger_edge_monotonic_ns"),
                parse_int(left, "trigger_write_uncertainty_ns"),
            ),
        )
        for offset in (-uncertainty // 2, uncertainty - uncertainty // 2)
    )
    mappings = [
        _mapping_or_identity(
            left_hardware,
            reference_left,
            source_clock=left_device_clock,
            reference_clock=reference_clock,
            mapping_id="mainline-left-device-to-reference-0001",
            source="paired-mainline-frame-affine-fit",
        ),
        _mapping_or_identity(
            right_hardware,
            reference_right,
            source_clock=right_device_clock,
            reference_clock=reference_clock,
            mapping_id="mainline-right-device-to-reference-0001",
            source="paired-mainline-frame-affine-fit",
        ),
        _mapping_or_identity(
            capture_midpoints,
            reference_midpoints,
            source_clock=capture_clock,
            reference_clock=reference_clock,
            mapping_id="mainline-realtime-to-reference-0001",
            source="gpio-capture-to-device-affine-fit",
            validity_source_times_ns=realtime_values,
        ),
    ]
    realtime_mapping = mappings[2]
    # 同一物理 GPIO 上升沿才可用于跨时钟拟合；不能把上升沿与曝光中心配对，
    # 否则所有软件时延会被平移半个曝光时长（以及触发到曝光延迟）。
    reference_edges = [
        _mapped(parse_int(left, "trigger_edge_realtime_ns"), realtime_mapping)
        for left, _ in paired_rows
    ]
    monotonic_mapping = _mapping_or_identity(
        edge_monotonic,
        reference_edges,
        source_clock=monotonic_clock,
        reference_clock=reference_clock,
        mapping_id="mainline-monotonic-to-reference-0001",
        source="gpio-edge-to-device-affine-fit",
        validity_source_times_ns=monotonic_values,
    )
    mappings.append(monotonic_mapping)
    clock_mappings = [mapping for mapping in mappings if mapping is not None]

    normalized_frames: dict[tuple[str, int], dict[str, Any]] = {}
    for role in ("left", "right"):
        for index in indexes:
            row = frame_pairs[index][role]
            exposure_us = float(row["exposure_us"])
            gpio_edge = _mapped(
                parse_int(row, "trigger_edge_realtime_ns"), realtime_mapping
            )
            exposure_start = gpio_edge + round(expected_exposure["delay_ns"])
            capture = exposure_start + round(exposure_us * 500.0)
            normalized_frames[(role, index)] = {
                "camera_id": row["camera_id"],
                "frame_index": index,
                # capture v2.1 将此字段定义为 GPIO 上升沿。相机设备原始时间
                # 不丢弃，完整保存在 raw/audit/frames_v6.csv 和时钟映射中。
                "hardware_timestamp_ns": gpio_edge,
                "clock_domain": reference_clock,
                "exposure_start_ns": exposure_start,
                "capture_timestamp_ns": capture,
                "receive_timestamp_ns": _mapped(
                    parse_int(row, "receive_timestamp_ns"), realtime_mapping
                ),
                "inference_done_timestamp_ns": (
                    None
                    if _optional_int(row, "inference_done_timestamp_ns") is None
                    else _mapped(
                        int(row["inference_done_timestamp_ns"]), realtime_mapping
                    )
                ),
                "exposure_us": exposure_us,
                "gain_db": float(row["gain_db"]),
                "roi_x": int(row["roi_x"]),
                "roi_y": int(row["roi_y"]),
                "roi_width": int(row["roi_width"]),
                "roi_height": int(row["roi_height"]),
                "pixel_format": row["pixel_format"],
                "dropped": False,
                "drop_reason": None,
            }

    candidate_keys: set[tuple[str, int, int]] = set()
    selected_keys: set[tuple[str, int, int]] = set()
    normalized_detections: list[dict[str, Any]] = []
    pipeline_id = context["perception_pipeline"]["pipeline_id"]
    for row in detections:
        role = row["camera_role"]
        frame_index = parse_int(row, "frame_index")
        candidate_id = parse_int(row, "candidate_id")
        key = (role, frame_index, candidate_id)
        if (
            role not in camera_ids
            or row["camera_id"] != camera_ids[role]
            or (role, frame_index) not in frame_by_key
            or key in candidate_keys
        ):
            raise DatasetError("mainline 检测引用错误、重复或角色不一致")
        if row["source"] != pipeline_id:
            raise DatasetError("mainline 检测 source 未引用冻结感知流水线")
        candidate_keys.add(key)
        selected = parse_bool(row, "selected")
        if selected:
            selected_keys.add(key)
        normalized_detections.append(
            {
                **{name: row[name] or None for name in MAINLINE_DETECTION_COLUMNS[4:17]},
                "camera_id": row["camera_id"],
                "frame_index": frame_index,
                "candidate_id": candidate_id,
                "selected": selected,
                "source": row["source"],
            }
        )

    observation_by_pair: dict[int, dict[str, str]] = {}
    observation_ids: set[int] = set()
    allowed_depth_methods = set(
        config.data["stereo"]["triangulation"]["candidate_methods"]
    )
    for row in observations:
        observation_id = parse_int(row, "observation_id")
        pair_id = parse_int(row, "source_pair_id")
        if observation_id in observation_ids or pair_id in observation_by_pair:
            raise DatasetError("mainline 直接观测 ID 或 source_pair_id 重复")
        observation_ids.add(observation_id)
        observation_by_pair[pair_id] = row
        if row["coordinate_frame"] != config.data["calibration"]["rig_frame"]:
            raise DatasetError("mainline 直接观测坐标系与标定 rig_frame 不一致")
        if parse_float(row, "z_m") is None or float(row["z_m"]) <= 0.0:
            raise DatasetError("mainline 直接观测 z_m 必须大于零")
        _optional_group(row, COVARIANCE_3D_FIELDS)
        if row["depth_method"] not in allowed_depth_methods:
            raise DatasetError("mainline 直接观测使用了配置外的深度方法")
        measured_disparity = parse_float(row, "measurement_disparity_px")
        if measured_disparity is None or measured_disparity <= 0.0:
            raise DatasetError("mainline 直接观测测量视差必须大于零")
        disparity_stddev = parse_float(
            row, "disparity_stddev_px", optional=True
        )
        if disparity_stddev is not None and disparity_stddev < 0.0:
            raise DatasetError("mainline 直接观测视差标准差不得为负")
        support_points = _optional_int(row, "support_points")
        if support_points is not None and support_points <= 0:
            raise DatasetError("mainline 直接观测支持点数必须为正整数")

    accepted_keys: set[tuple[str, int, int]] = set()
    pair_ids: set[int] = set()
    normalized_associations: list[dict[str, Any]] = []
    trajectory_records: list[tuple[int, dict[str, Any]]] = []
    for row in associations:
        pair_id = parse_int(row, "pair_id")
        if pair_id in pair_ids:
            raise DatasetError("mainline association pair_id 重复")
        pair_ids.add(pair_id)
        left_index = parse_int(row, "left_frame_index")
        right_index = parse_int(row, "right_frame_index")
        left_candidate = parse_int(row, "left_candidate_id")
        right_candidate = parse_int(row, "right_candidate_id")
        left_key = ("left", left_index, left_candidate)
        right_key = ("right", right_index, right_candidate)
        if left_key not in candidate_keys or right_key not in candidate_keys:
            raise DatasetError("mainline association 引用了不存在的候选")
        left_stage_frame = frame_by_key[("left", left_index)]
        right_stage_frame = frame_by_key[("right", right_index)]
        left_capture_raw = parse_int(row, "left_capture_ns")
        right_capture_raw = parse_int(row, "right_capture_ns")
        if (
            row["left_capture_clock_domain"] != left_stage_frame["capture_clock_domain"]
            or row["right_capture_clock_domain"] != right_stage_frame["capture_clock_domain"]
            or left_capture_raw != int(left_stage_frame["capture_timestamp_ns"])
            or right_capture_raw != int(right_stage_frame["capture_timestamp_ns"])
            or parse_int(row, "delta_t_lr_ns") != right_capture_raw - left_capture_raw
        ):
            raise DatasetError("mainline association 曝光时间证据与来源帧不一致")
        decision = row["decision"]
        if decision not in {"accepted", "rejected"}:
            raise DatasetError("mainline association decision 无效")
        accepted = decision == "accepted"
        observation = observation_by_pair.get(pair_id)
        if accepted:
            if row["reject_reason"] or observation is None:
                raise DatasetError("mainline accepted association 缺少唯一直接观测")
            disparity = parse_float(row, "disparity_px", optional=True)
            if disparity is None or disparity <= 0.0:
                raise DatasetError("mainline accepted association 必须具有正视差")
            expected_reference = integer_midpoint_ns(
                left_capture_raw, right_capture_raw
            )
            if (
                int(observation["capture_reference_ns"]) != expected_reference
                or observation["capture_reference_clock_domain"]
                != row["left_capture_clock_domain"]
            ):
                raise DatasetError("mainline 直接观测时间不是左右曝光中心的整数中点")
            accepted_keys.update((left_key, right_key))
        elif not row["reject_reason"] or observation is not None:
            raise DatasetError("mainline rejected association 的理由或观测关系无效")

        left_capture = normalized_frames[("left", left_index)][
            "capture_timestamp_ns"
        ]
        right_capture = normalized_frames[("right", right_index)][
            "capture_timestamp_ns"
        ]
        covariance = (
            None
            if observation is None
            else _optional_group(observation, COVARIANCE_3D_FIELDS)
        )
        position = (
            None
            if observation is None
            else tuple(float(observation[name]) for name in ("x_m", "y_m", "z_m"))
        )
        pair_record = {
            "pair_id": pair_id,
            "left_frame_index": left_index,
            "right_frame_index": right_index,
            "left_candidate_id": left_candidate,
            "right_candidate_id": right_candidate,
            "left_capture_ns": left_capture,
            "right_capture_ns": right_capture,
            "delta_t_lr_ns": right_capture - left_capture,
            # 正式 accepted pair 使用真正产生 3D 的最终测量视差；检测中心的
            # 初始门控视差仍完整保存在 raw/audit/associations_v6.csv。
            "disparity_px": (
                parse_float(row, "disparity_px", optional=True)
                if observation is None
                else parse_float(observation, "measurement_disparity_px")
            ),
            "epipolar_error_px": parse_float(
                row, "epipolar_error_px", optional=True
            ),
            "x_rig_m": None if position is None else position[0],
            "y_rig_m": None if position is None else position[1],
            "z_rig_m": None if position is None else position[2],
            **{
                name: None if covariance is None else covariance[index]
                for index, name in enumerate(COVARIANCE_3D_FIELDS)
            },
            "accepted": accepted,
            "reject_reason": None if accepted else row["reject_reason"],
        }
        normalized_associations.append(pair_record)
        if accepted and observation is not None:
            epipolar = pair_record["epipolar_error_px"]
            reprojection = parse_float(
                observation, "reprojection_error_px", optional=True
            )
            if epipolar != reprojection:
                raise DatasetError("mainline 直接观测重投影误差与配对极线误差不一致")
            timestamp = integer_midpoint_ns(left_capture, right_capture)
            trajectory_records.append(
                (
                    timestamp,
                    {
                        "left_frame_index": left_index,
                        "right_frame_index": right_index,
                        "x_m": position[0],
                        "y_m": position[1],
                        "z_m": position[2],
                        "confidence": parse_float(
                            observation, "confidence", optional=True
                        ),
                        "reprojection_error_px": reprojection,
                        **{
                            name: None if covariance is None else covariance[index]
                            for index, name in enumerate(COVARIANCE_3D_FIELDS)
                        },
                    },
                )
            )

    if set(observation_by_pair) != {
        pair_id
        for pair_id, row in zip(
            (parse_int(row, "pair_id") for row in associations), associations
        )
        if row["decision"] == "accepted"
    }:
        raise DatasetError("mainline 每个 accepted association 必须恰有一个直接观测")
    if selected_keys != accepted_keys:
        raise DatasetError("mainline selected 检测与 accepted association 引用不一致")

    trajectory_records.sort(key=lambda item: item[0])
    if any(
        right[0] <= left[0]
        for left, right in zip(trajectory_records, trajectory_records[1:])
    ):
        raise DatasetError("mainline 直接观测时间必须严格递增")
    first_left = normalized_frames[("left", indexes[0])]["capture_timestamp_ns"]
    first_right = normalized_frames[("right", indexes[0])]["capture_timestamp_ns"]
    origin_ns = integer_midpoint_ns(first_left, first_right)

    root = Path(dataset_root).resolve()
    calibration_contract = ensure_dataset_calibration(
        root, config, project_root=project_root
    )
    calibration_source, calibration_source_sha256 = resolve_calibration_source(
        config, project_root=project_root
    )
    mapping_uncertainties = [
        float(mapping["uncertainty_ns"]) for mapping in clock_mappings
    ]
    capture_uncertainties = [
        value
        for row in frames
        if (value := _optional_int(row, "capture_timestamp_uncertainty_ns"))
        is not None
    ]
    metadata = {
        "trial_id": context["trial_id"],
        "session_id": context["session_id"],
        "ball_id": context["ball_id"],
        "calibration_id": context["calibration_id"],
        "captured_at_utc": context["captured_at_utc"],
        "trajectory_kind": context["trajectory_kind"],
        "coordinate_frame": {
            "name": config.data["calibration"]["rig_frame"],
            "origin": "left rectified camera optical center",
            "x_axis": "image right",
            "y_axis": "image down",
            "z_axis": "optical forward",
            "handedness": "right",
            "position_unit": "m",
            "time_unit": "s",
        },
        "capture": {
            "nominal_fps": config.data["runtime"]["nominal_fps"],
            "sync_method": config.data["cameras"]["sync"]["method"],
            "estimated_sync_error_s": max(
                float(sync_limit or 0.0),
                *mapping_uncertainties,
                *capture_uncertainties,
            )
            * 1.0e-9,
            "exposure_timing": {
                "shutter_type": expected_exposure["shutter_type"],
                "timestamp_source": expected_exposure["source"],
                "delay_status": expected_exposure["delay_status"],
                "delay_ns": expected_exposure["delay_ns"],
                "delay_uncertainty_ns": expected_exposure[
                    "delay_uncertainty_ns"
                ],
                "observation_timestamp": expected_exposure[
                    "observation_timestamp"
                ],
            },
            "trajectory_timebase": {
                "clock_domain": reference_clock,
                "origin_ns": origin_ns,
                "timestamp_semantics": "stereo_midpoint_capture",
            },
        },
        "conditions": context.get("conditions", {}),
        "notes": context.get("notes", ""),
        "perception_pipelines": [context["perception_pipeline"]],
    }

    with TrialPackageWriter(root, metadata, project_root=project_root) as writer:
        for index in indexes:
            writer.append("camera_frames", normalized_frames[("left", index)])
            writer.append("camera_frames", normalized_frames[("right", index)])
        for record in normalized_detections:
            writer.append("detections_2d", record)
        for record in normalized_associations:
            writer.append("stereo_pairs", record)
        for sample_index, (timestamp, record) in enumerate(trajectory_records):
            writer.append(
                "trajectory",
                {
                    "sample_index": sample_index,
                    "timestamp_s": (timestamp - origin_ns) / 1.0e9,
                    **record,
                },
            )
        for mapping in clock_mappings:
            writer.append("clock_mappings", mapping)

        for index, (left, right) in zip(indexes, paired_rows):
            for role, row in (("left", left), ("right", right)):
                capture_reference = normalized_frames[(role, index)][
                    "capture_timestamp_ns"
                ]
                receive_mono = parse_int(row, "receive_monotonic_ns")
                writer.append(
                    "processing_timings",
                    {
                        "event_id": f"usb-receive-{role}-{index}",
                        "stage": "camera_frame_receive",
                        "input_type": "camera_frame",
                        "input_id": f"{camera_ids[role]}:{index}",
                        "capture_reference_ns": capture_reference,
                        "start_timestamp_ns": receive_mono,
                        "done_timestamp_ns": receive_mono,
                        "clock_domain": monotonic_clock,
                        "queue_depth": None,
                        "valid": True,
                        "source": "hik-mvs-usb",
                    },
                )
                inference_done = _optional_int(row, "inference_done_timestamp_ns")
                if inference_done is not None:
                    writer.append(
                        "processing_timings",
                        {
                            "event_id": f"inference-done-{role}-{index}",
                            "stage": "detector_complete",
                            "input_type": "camera_frame",
                            "input_id": f"{camera_ids[role]}:{index}",
                            "capture_reference_ns": capture_reference,
                            "start_timestamp_ns": inference_done,
                            "done_timestamp_ns": inference_done,
                            "clock_domain": capture_clock,
                            "queue_depth": None,
                            "valid": True,
                            "source": pipeline_id,
                        },
                    )
            edge = parse_int(left, "trigger_edge_monotonic_ns")
            uncertainty = parse_int(left, "trigger_write_uncertainty_ns")
            start = edge - uncertainty // 2
            writer.append(
                "processing_timings",
                {
                    "event_id": f"gpio-trigger-{parse_int(left, 'trigger_sequence')}",
                    "stage": "gpio_rising_edge_write",
                    "input_type": "trigger_event",
                    "input_id": left["trigger_sequence"],
                    "capture_reference_ns": normalized_frames[("left", index)][
                        "capture_timestamp_ns"
                    ],
                    "start_timestamp_ns": start,
                    "done_timestamp_ns": start + uncertainty,
                    "clock_domain": monotonic_clock,
                    "queue_depth": None,
                    "valid": True,
                    "source": "nx-gpio-pwm",
                },
            )

        environment = environment_record(
            context.get("conditions", {}), origin_ns, reference_clock
        )
        if environment is not None:
            writer.append("environment", environment)

        for label, source, filename in (
            (
                "calibration_source",
                calibration_source,
                "audit/calibration_source.yaml",
            ),
            (
                "calibration_contract",
                calibration_contract,
                "audit/calibration_contract.json",
            ),
            ("mainline_stage_frames", directory / "frames_v6.csv", "audit/frames_v6.csv"),
            (
                "mainline_stage_detections",
                directory / "detections_v6.csv",
                "audit/detections_v6.csv",
            ),
            (
                "mainline_stage_associations",
                directory / "associations_v6.csv",
                "audit/associations_v6.csv",
            ),
            (
                "mainline_stage_observations",
                directory / "observations_v6.csv",
                "audit/observations_v6.csv",
            ),
            ("mainline_stage_status", directory / "status_v6.json", "audit/status_v6.json"),
            ("capture_context", directory / "context.json", "audit/context.json"),
        ):
            archived = writer.add_raw_source(label, source, filename=filename)
            if (
                label == "calibration_source"
                and sha256_file(archived) != calibration_source_sha256
            ):
                raise DatasetError("归档标定源与本次采集固定哈希不一致")
        if has_measurement_inputs:
            writer.add_raw_source(
                "direct_stereo_measurements",
                measurement_file,
                filename="audit/stereo_measurements_v1.csv",
            )
        if config.data["recording"]["raw_frames"] != "none":
            for index in indexes:
                for role in ("left", "right"):
                    source = _safe_stage_source(
                        directory, frame_pairs[index][role]["raw_image"]
                    )
                    writer.add_raw_source(
                        f"{role}_{index:08d}",
                        source,
                        filename=f"mainline/{role}/{index:08d}.pgm",
                    )
        return writer.finalize()
