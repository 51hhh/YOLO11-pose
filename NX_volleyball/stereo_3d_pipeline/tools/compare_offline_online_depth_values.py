"""CSV value helpers for offline/online depth comparison."""

from __future__ import annotations

import csv
import io
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    data = path.read_bytes().replace(b"\x00", b"")
    text = data.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def positive_float(value: object) -> float | None:
    result = parse_float(value)
    if result is None or result <= 0.0:
        return None
    return result


def choose_value(values: Sequence[float], mode: str) -> float | None:
    if not values:
        return None
    if mode == "first":
        return values[0]
    if mode == "last":
        return values[-1]
    return float(statistics.median(values))


def collect_column(rows: Iterable[dict[str, str]], col: str | None, mode: str, positive: bool) -> float | None:
    if not col:
        return None
    parser = positive_float if positive else parse_float
    values: list[float] = []
    for row in rows:
        value = parser(row.get(col))
        if value is not None:
            values.append(value)
    return choose_value(values, mode)


def load_offline_summary(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row.get("method", "").strip(): row for row in rows if row.get("method", "").strip()}


def all_online_columns(rows: Sequence[dict[str, str]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())
    return columns


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"
