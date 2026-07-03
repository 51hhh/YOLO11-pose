"""Diagnosis helpers for NX algorithm matrix result rows."""

from __future__ import annotations


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def row_int(row: dict[str, str], key: str) -> int:
    return parse_int(row.get(key))


def classify_case_result(row: dict[str, str]) -> str:
    status = row.get("status", "")
    if status.startswith("skipped"):
        return "skipped_missing_dependency"
    if status == "failed":
        return "pipeline_failed"

    stale_or_expired = (
        row_int(row, "async_drop_stale_count") +
        row_int(row, "async_drop_stale_ready_count") +
        row_int(row, "async_drop_expired_pending_count") +
        row_int(row, "stage2_drop_stale_roi_count")
    )
    infrastructure_drop = (
        row_int(row, "async_drop_pending_count") +
        row_int(row, "async_drop_no_buffer_count") +
        row_int(row, "async_submit_drop_count")
    )
    if row_int(row, "async_over_deadline_count") > 0 or stale_or_expired > 0:
        return "late_or_deadline_dropped"
    if infrastructure_drop > 0:
        return "async_queue_or_buffer_drop"
    if (
        row_int(row, "cpu_fallback_count") > 0
        or row_int(row, "async_need_host_gray_count") > 0
        or row_int(row, "async_host_gray_submit_count") > 0
    ):
        return "realtime_path_used_cpu_or_host_gray"
    if row_int(row, "async_no_detections_count") > 0 and row_int(row, "target_rows") == 0:
        return "no_detections"
    if row_int(row, "target_rows") == 0 and row_int(row, "async_accepted_count") == 0:
        return "no_accepted_results"
    if row_int(row, "candidate_rows") > 0 and row_int(row, "candidate_valid") == 0:
        return "ran_but_no_valid_candidate"
    if row_int(row, "candidate_valid") > 0:
        return "ok"
    return "needs_log_review"


def should_debug_case(row: dict[str, str]) -> bool:
    return classify_case_result(row) not in {
        "ok",
        "skipped_missing_dependency",
    }
