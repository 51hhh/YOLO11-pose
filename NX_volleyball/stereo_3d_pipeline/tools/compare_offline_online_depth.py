#!/usr/bin/env python3
"""Compare offline probe depth candidates with online recorder columns.

The script intentionally compares only methods with a known semantic mapping.
Missing realtime implementations are reported as missing instead of being
matched to a different online heuristic.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

from compare_offline_online_depth_mappings import MAPPINGS


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


def build_report(
    offline_by_method: dict[str, dict[str, str]],
    online_rows: list[dict[str, str]],
    *,
    online_mode: str,
    max_disparity_delta_px: float,
    max_depth_delta_m: float,
) -> tuple[list[dict[str, str]], bool]:
    online_columns = all_online_columns(online_rows)
    report_rows: list[dict[str, str]] = []
    has_failure = False

    for mapping in MAPPINGS:
        offline = offline_by_method.get(mapping.offline_method)
        row = {
            "offline_method": mapping.offline_method,
            "online_method": mapping.online_method,
            "status": "",
            "offline_disparity_px": "",
            "online_disparity_px": "",
            "delta_disparity_px": "",
            "offline_depth_m": "",
            "online_depth_m": "",
            "delta_depth_m": "",
            "offline_validation": "",
            "online_support": "",
            "online_std_px": "",
            "online_confidence": "",
            "notes": "",
        }

        if offline is None:
            row["status"] = "missing_offline"
            row["notes"] = "offline summary has no row for this method"
            report_rows.append(row)
            has_failure = True
            continue

        offline_disp = positive_float(offline.get("disparity_px"))
        offline_depth = positive_float(offline.get("depth_m"))
        offline_validation = offline.get("validation_status", "").strip()
        row["offline_disparity_px"] = fmt(offline_disp)
        row["offline_depth_m"] = fmt(offline_depth)
        row["offline_validation"] = offline_validation

        online_cols = [c for c in (mapping.disparity_col, mapping.z_col) if c]
        if not online_cols:
            row["status"] = "missing_online"
            row["notes"] = "no realtime column is defined for this offline method"
            report_rows.append(row)
            has_failure = True
            continue
        missing_cols = [c for c in online_cols if c not in online_columns]
        if missing_cols:
            row["status"] = "missing_online"
            row["notes"] = "missing columns: " + ",".join(missing_cols)
            report_rows.append(row)
            has_failure = True
            continue

        online_disp = collect_column(online_rows, mapping.disparity_col, online_mode, positive=True)
        online_depth = collect_column(online_rows, mapping.z_col, online_mode, positive=True)
        support = collect_column(online_rows, mapping.support_col, online_mode, positive=False)
        std_px = collect_column(online_rows, mapping.std_col, online_mode, positive=False)
        confidence = collect_column(online_rows, mapping.confidence_col, online_mode, positive=False)

        row["online_disparity_px"] = fmt(online_disp)
        row["online_depth_m"] = fmt(online_depth)
        row["online_support"] = fmt(support, 1)
        row["online_std_px"] = fmt(std_px)
        row["online_confidence"] = fmt(confidence)

        comparisons: list[bool] = []
        notes: list[str] = []
        if offline_disp is not None and online_disp is not None:
            delta_disp = abs(offline_disp - online_disp)
            row["delta_disparity_px"] = fmt(delta_disp)
            comparisons.append(delta_disp <= max_disparity_delta_px)
        elif mapping.disparity_col:
            notes.append("no valid online disparity")

        if offline_depth is not None and online_depth is not None:
            delta_depth = abs(offline_depth - online_depth)
            row["delta_depth_m"] = fmt(delta_depth)
            comparisons.append(delta_depth <= max_depth_delta_m)
        elif mapping.z_col:
            notes.append("no valid online depth")

        if offline_validation and offline_validation != "pass":
            notes.append(f"offline validation is {offline_validation}")

        if not comparisons:
            row["status"] = "no_common_metric"
            has_failure = True
        elif all(comparisons):
            row["status"] = "pass"
        else:
            row["status"] = "fail"
            has_failure = True
        row["notes"] = "; ".join(notes)
        report_rows.append(row)

    return report_rows, has_failure


def render_markdown(rows: Sequence[dict[str, str]]) -> str:
    headers = (
        "offline_method",
        "online_method",
        "status",
        "offline_disparity_px",
        "online_disparity_px",
        "delta_disparity_px",
        "offline_depth_m",
        "online_depth_m",
        "delta_depth_m",
        "offline_validation",
        "online_support",
        "online_std_px",
        "online_confidence",
        "notes",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [row.get(header, "").replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_csv(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-summary", required=True, type=Path, help="offline probe summary.csv")
    parser.add_argument("--online-csv", required=True, type=Path, help="TrajectoryRecorder CSV")
    parser.add_argument("--out", type=Path, help="optional output file; .csv suffix writes CSV, otherwise Markdown")
    parser.add_argument("--online-rows", choices=("median", "first", "last"), default="median")
    parser.add_argument("--max-disparity-delta-px", type=float, default=0.5)
    parser.add_argument("--max-depth-delta-m", type=float, default=0.02)
    parser.add_argument("--fail-on-mismatch", action="store_true", help="exit 1 when any mapped method fails or is missing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    offline_by_method = load_offline_summary(args.offline_summary)
    online_rows = read_csv_rows(args.online_csv)
    rows, has_failure = build_report(
        offline_by_method,
        online_rows,
        online_mode=args.online_rows,
        max_disparity_delta_px=args.max_disparity_delta_px,
        max_depth_delta_m=args.max_depth_delta_m,
    )

    if args.out and args.out.suffix.lower() == ".csv":
        rendered = render_csv(rows)
    else:
        rendered = render_markdown(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.fail_on_mismatch and has_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
