"""Profile-stage parsing helpers for NX algorithm matrix results."""

from __future__ import annotations

import re

from nx_algorithm_cases import Case


def profile_stage_for_case(case: Case) -> str:
    if case.neural_backend:
        return "Stage2_NeuralFeatureMatch"
    if case.modes.get("roi_orb_points"):
        return "Stage2_OpenCVCudaORB"
    if case.modes.get("roi_brisk_points"):
        return "Stage2_CPUFeatureOpenCVBRISK"
    if case.modes.get("roi_akaze_points"):
        return "Stage2_CPUFeatureOpenCVAKAZE"
    if case.modes.get("roi_sift_points"):
        return "Stage2_CPUFeatureOpenCVSIFT"
    if case.modes.get("roi_cuda_template_match"):
        return "Stage2_OpenCVCudaTemplateMatch"
    if case.modes.get("roi_cuda_stereo_bm"):
        return "Stage2_OpenCVCudaStereoBM"
    if case.modes.get("roi_cuda_stereo_sgm"):
        return "Stage2_OpenCVCudaStereoSGM"
    if case.modes.get("roi_subpixel"):
        return "Stage2_SubpixelMatch"
    return "Stage2_DualYoloGpuCandidates"


def parse_profile_stage(log: str, name: str) -> tuple[str, str, str, str, str, str, str, str]:
    matches = re.findall(
        rf"^{re.escape(name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)"
        rf"(?:\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+))?",
        log,
        re.M,
    )
    if not matches:
        return ("", "", "", "", "", "", "", "")
    match = matches[-1]
    return (
        match[0], match[1], match[2], match[3],
        match[4] or "", match[5] or "", match[6] or "", match[7] or "",
    )
