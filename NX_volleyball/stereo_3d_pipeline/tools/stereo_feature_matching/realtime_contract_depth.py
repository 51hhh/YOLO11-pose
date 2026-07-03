"""Depth candidate selection helpers for the realtime contract mirror."""

from __future__ import annotations

from typing import Iterable, Optional

from .realtime_contract_types import DEPTH_CANDIDATE_PRIORITY, DepthCandidateObservation


def select_first_usable_depth_candidate(
    candidates: Iterable[DepthCandidateObservation],
) -> Optional[DepthCandidateObservation]:
    priority = {method: idx for idx, method in enumerate(DEPTH_CANDIDATE_PRIORITY)}
    ordered = sorted(candidates, key=lambda item: priority.get(item.method, 10_000))
    for candidate in ordered:
        if candidate.usable:
            return candidate
    return None
