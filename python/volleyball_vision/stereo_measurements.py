"""独立版本的双目测量输入证据，供跨仓确定性回放使用。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capture_stage_io import parse_float, parse_int, read_csv_exact
from .dataset import DatasetError


STEREO_MEASUREMENT_SCHEMA_VERSION = "1.0.0"
STEREO_MEASUREMENT_COLUMNS = [
    "schema_version",
    "pair_id",
    "method",
    "anchor_u_px",
    "anchor_v_px",
    "disparity_px",
]


def _rectified_model(
    calibration: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    """提取左目校正针孔模型 `(fx, fy, cx, cy, baseline)`。"""

    try:
        left_id = calibration["camera_roles"]["left"]
        left = next(
            camera
            for camera in calibration["cameras"]
            if camera["camera_id"] == left_id
        )
        matrix = left["camera_matrix"]
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[0][2])
        cy = float(matrix[1][2])
        baseline = float(calibration["baseline_m"])
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        raise DatasetError(f"双目测量扩展缺少有效校正模型: {error}") from error
    values = (fx, fy, cx, cy, baseline)
    if not all(math.isfinite(value) for value in values) or min(
        fx, fy, baseline
    ) <= 0:
        raise DatasetError("双目测量扩展的校正焦距或基线无效")
    return values


def validate_stereo_measurements(
    path: Path,
    observations: Sequence[Mapping[str, str]],
    calibration: Mapping[str, Any],
) -> bool:
    """校验 UVD 输入与直接 3D；旧 staging 缺失扩展时返回 ``False``。"""

    if not path.is_file():
        return False

    expected: dict[int, Mapping[str, str]] = {}
    for observation in observations:
        pair_id = parse_int(observation, "source_pair_id")
        if pair_id < 0 or pair_id in expected:
            raise DatasetError("直接观测的配对引用必须非负且唯一")
        expected[pair_id] = observation

    fx, fy, cx, cy, baseline = _rectified_model(calibration)
    rows = read_csv_exact(path, STEREO_MEASUREMENT_COLUMNS)
    seen: set[int] = set()
    for row in rows:
        pair_id = parse_int(row, "pair_id")
        if (
            row["schema_version"] != STEREO_MEASUREMENT_SCHEMA_VERSION
            or pair_id < 0
            or pair_id in seen
            or pair_id not in expected
        ):
            raise DatasetError("双目测量扩展版本、配对引用或唯一性错误")
        observation = expected[pair_id]
        u = float(parse_float(row, "anchor_u_px"))
        v = float(parse_float(row, "anchor_v_px"))
        disparity = float(parse_float(row, "disparity_px"))
        observation_disparity = float(
            parse_float(observation, "measurement_disparity_px")
        )
        if (
            disparity <= 0
            or not row["method"]
            or row["method"] != observation["depth_method"]
            or not math.isclose(
                disparity,
                observation_disparity,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ):
            raise DatasetError("双目测量扩展与观测方法或视差不一致")

        z = fx * baseline / disparity
        reconstructed = (
            (u - cx) * z / fx,
            (v - cy) * z / fy,
            z,
        )
        recorded = tuple(
            float(parse_float(observation, name))
            for name in ("x_m", "y_m", "z_m")
        )
        if any(
            not math.isclose(
                actual,
                expected_value,
                rel_tol=1.0e-10,
                abs_tol=1.0e-8,
            )
            for actual, expected_value in zip(recorded, reconstructed)
        ):
            raise DatasetError("双目测量扩展的 UVD 与已发布 XYZ 不一致")
        seen.add(pair_id)

    if seen != set(expected):
        raise DatasetError("每个正式 3D 观测必须恰有一条双目测量输入")
    return True
