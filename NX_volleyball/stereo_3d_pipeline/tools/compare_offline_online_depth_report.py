"""Report construction and rendering for offline/online depth comparison."""

from __future__ import annotations

import csv
import io
from typing import Sequence

from compare_offline_online_depth_mappings import MAPPINGS
from compare_offline_online_depth_values import (
    all_online_columns,
    collect_column,
    fmt,
    positive_float,
)


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

        online_cols = [col for col in (mapping.disparity_col, mapping.z_col) if col]
        if not online_cols:
            row["status"] = "missing_online"
            row["notes"] = "no realtime column is defined for this offline method"
            report_rows.append(row)
            has_failure = True
            continue
        missing_cols = [col for col in online_cols if col not in online_columns]
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
