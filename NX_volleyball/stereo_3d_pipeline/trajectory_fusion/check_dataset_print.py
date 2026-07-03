"""Console reporting for trajectory dataset quality checks."""

from __future__ import annotations

from typing import Any, Dict

try:
    from .check_dataset_fields import DEPTH_KEYS
except ImportError:  # pragma: no cover - direct script execution
    from check_dataset_fields import DEPTH_KEYS


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_report(report: Dict[str, Any]) -> None:
    print(f"csv={report['csv']}")
    print(f"metadata={report['metadata']}")
    print(
        f"rows={report['rows']} duration={_fmt(report['duration_sec'], 3)}s "
        f"fps_rows={_fmt(report['fps_rows'], 3)} fps_intervals={_fmt(report['fps_intervals'], 3)} "
        f"timing_source={report['timing_source']}"
    )
    print(f"missing_fields={report['missing_fields']}")
    print(f"frame_gaps={report['frame_gaps']['count']} first={report['frame_gaps']['first']}")
    for key, stats in report["watermarks"].items():
        print(f"{key}: present={stats['present']} nonzero={stats.get('nonzero')} unique={stats.get('unique')}")
    source = report["source_breakdown"]
    print(f"match_source={source['match_source']}")
    print(
        "epipolar_fallback: "
        f"valid={source['epipolar_fallback']['valid']} "
        f"by_direction={source['epipolar_fallback']['by_direction']} "
        f"median={_fmt(source['epipolar_fallback']['median'])} "
        f"mad={_fmt(source['epipolar_fallback']['mad'])}"
    )
    for key in DEPTH_KEYS:
        stats = report["depth"][key]
        jumps = report["depth_jump"].get(key, {})
        print(
            f"{key}: valid={stats['valid']}/{stats['total']} "
            f"hit={stats['hit_rate'] * 100.0:.1f}% "
            f"median={_fmt(stats['median'])} mad={_fmt(stats['mad'])} "
            f"known_z_bias={_fmt(stats['known_z_bias'])} known_z_mad={_fmt(stats['known_z_mad'])} "
            f"jump_p95={_fmt(jumps.get('p95_abs_delta'))}"
        )
    frame_summary = report["frame_summary"]
    print(
        f"frame_summary: present={frame_summary['present']} "
        f"path={frame_summary['path']} rows={frame_summary.get('rows')}"
    )
    if frame_summary["present"]:
        print(f"frame_summary_totals={frame_summary['totals']}")
