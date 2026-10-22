"""旧 NX v4 与主线 v5 staging 共用的严格解析原语。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .dataset import DatasetError


def read_csv_exact(path: Path, columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise DatasetError(
                    f"staging 表头不匹配 {path.name}: "
                    f"expected={columns}, actual={reader.fieldnames}"
                )
            rows = list(reader)
            for line_number, row in enumerate(rows, start=2):
                # DictReader 会把多余字段放入 None 键，把缺失字段填为 None。
                if None in row or any(value is None for value in row.values()):
                    raise DatasetError(
                        f"staging {path.name} 第 {line_number} 行列数不匹配"
                    )
            return rows
    except FileNotFoundError as error:
        raise DatasetError(f"staging 缺少文件: {path}") from error


def parse_int(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as error:
        raise DatasetError(f"staging 字段 {key} 必须是整数") from error


def parse_float(
    row: Mapping[str, str], key: str, *, optional: bool = False
) -> float | None:
    value = row.get(key, "")
    if value == "" and optional:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"staging 字段 {key} 必须是数字") from error
    if not math.isfinite(parsed):
        raise DatasetError(f"staging 字段 {key} 必须为有限数")
    return parsed


def parse_bool(row: Mapping[str, str], key: str) -> bool:
    value = row.get(key)
    if value in {"1", "true", "True"}:
        return True
    if value in {"0", "false", "False"}:
        return False
    raise DatasetError(f"staging 字段 {key} 必须是布尔值")


def uint32_is_next(previous: int, current: int) -> bool:
    """按相机 uint32 计数器语义判断连续性，允许自然回绕。"""

    if not 0 <= previous <= 0xFFFFFFFF or not 0 <= current <= 0xFFFFFFFF:
        return False
    return ((current - previous) & 0xFFFFFFFF) == 1


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"staging 缺少文件: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"staging JSON 无效 {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError(f"staging JSON 顶层必须是对象: {path}")
    return value


def environment_record(
    conditions: Mapping[str, Any],
    timestamp_ns: int,
    clock_domain: str,
) -> dict[str, Any] | None:
    value = conditions.get("environment")
    if not isinstance(value, Mapping):
        return None
    return {
        "timestamp_ns": timestamp_ns,
        "clock_domain": clock_domain,
        "air_temperature_c": value.get("air_temperature_c"),
        "relative_humidity": value.get("relative_humidity"),
        "static_pressure_pa": value.get("static_pressure_pa"),
        "wind_x_mps": value.get("wind_x_mps"),
        "wind_y_mps": value.get("wind_y_mps"),
        "wind_z_mps": value.get("wind_z_mps"),
        "hvac_state": value.get("hvac_state"),
        "source": value.get("source", "manual"),
    }
