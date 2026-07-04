"""Common helpers for NX full benchmark sections."""

from __future__ import annotations


def print_section(title: str) -> None:
    prefix = "\n" if title.startswith("[2/") or title.startswith("[3/") or title.startswith("[4/") else ""
    print(prefix + "=" * 60)
    print(title)
    print("=" * 60)


def summary_stats(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "avg": sum(times) / len(times),
        "min": min(times),
        "p50": ordered[len(ordered) // 2],
        "p99": ordered[int(0.99 * len(ordered))],
        "max": max(times),
    }
