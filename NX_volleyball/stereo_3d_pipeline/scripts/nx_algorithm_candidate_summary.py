"""Candidate CSV summary helpers for NX algorithm matrix tests."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

from nx_algorithm_cases import Case


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def count_frame_rows(frames_path: Path) -> int:
    if not frames_path.exists():
        return 0
    with frames_path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def summarize_candidate_csv(case: Case, csv_path: Path, frames_path: Path) -> dict[str, str]:
    empty = {
        "candidate_rows": "",
        "candidate_attempted": "",
        "candidate_not_attempted": "",
        "candidate_valid": "",
        "candidate_rate": "",
        "candidate_median_m": "",
        "candidate_mad_m": "",
        "support_median": "",
        "field_valids": "",
        "candidate_reject_reason": "",
        "target_rows": "",
    }
    frame_total = count_frame_rows(frames_path)
    if not case.candidate_fields or not csv_path.exists():
        return empty

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    target_total = len(rows)
    total = frame_total if frame_total > 0 else target_total
    if target_total == 0 and total == 0:
        return {**empty, "candidate_rows": "0", "target_rows": "0"}

    valid_depths: list[float] = []
    field_counts: dict[str, int] = {field: 0 for field in case.candidate_fields}
    supports: list[float] = []
    attempted = 0
    for row in rows:
        row_values = []
        for field in case.candidate_fields:
            value = parse_float(row.get(field))
            if value is not None and value > 0.0:
                field_counts[field] += 1
                row_values.append(value)
        row_attempted = False
        if case.support_field:
            support = parse_float(row.get(case.support_field))
            if support is not None and support >= 0.0:
                supports.append(support)
                row_attempted = True
        else:
            row_attempted = any(row.get(field) not in (None, "") for field in case.candidate_fields)
        if row_attempted:
            attempted += 1
        if row_values:
            valid_depths.append(row_values[0])

    valid = len(valid_depths)
    not_attempted = max(0, total - attempted)
    if target_total == 0:
        reject_reason = "no_candidate_rows"
    elif attempted == 0:
        reject_reason = "not_attempted"
    elif valid == 0:
        reject_reason = "attempted_no_valid_depth"
    elif valid < attempted:
        reject_reason = "partial_valid"
    else:
        reject_reason = "ok"
    if valid_depths:
        med = median(valid_depths)
        mad = median(abs(v - med) for v in valid_depths)
        med_s = f"{med:.4f}"
        mad_s = f"{mad:.4f}"
    else:
        med_s = ""
        mad_s = ""
    support_s = f"{median(supports):.1f}" if supports else ""
    field_valids = ";".join(f"{k}={v}/{total}" for k, v in field_counts.items())
    return {
        "candidate_rows": str(total),
        "candidate_attempted": str(attempted),
        "candidate_not_attempted": str(not_attempted),
        "candidate_valid": str(valid),
        "candidate_rate": f"{valid / total:.3f}" if total else "",
        "candidate_median_m": med_s,
        "candidate_mad_m": mad_s,
        "support_median": support_s,
        "field_valids": field_valids,
        "candidate_reject_reason": reject_reason,
        "target_rows": str(target_total),
    }
