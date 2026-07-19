#!/usr/bin/env python3
"""Recursive training for the causal 3D trajectory state estimator.

This trains ``CausalKalmanNet`` frame-by-frame exactly the way it runs at
inference: one ``step()`` per frame, carrying state and GRU hidden across the
sequence. Training and inference share the same code path, so there is no
train/inference mismatch.

Inputs come from ``dataset.build_metric_state_arrays`` (Stage-1 reprojected
metric XYZ per method, with the disparity zero-point ``d0`` already corrected).
The model never sees the legacy online ``x/y/z`` state, avoiding label leakage.

Loss terms (each gated by what the clip actually supports):

- observation consistency: fused/predicted position vs each valid per-method
  metric point (Student-t robust), the always-on self-supervised anchor.
- known_z: on static known-distance clips, pull filtered Z to the tape value.
- static jitter: on static clips, penalise frame-to-frame position change.
- continuity / ballistic: low-jerk prior; the gravity term only switches on when
  metadata carries a confirmed gravity axis+magnitude (else pure low-jerk).
- gain regulariser: keep gains from collapsing to 0 (ignore obs) or 1 (trust raw).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None

try:
    from .dataset import (
        build_metric_state_arrays,
        load_legacy_sequences,
        metric_feature_names,
        metric_state_method_names,
        resolve_method_allowlist,
    )
    from .manifest import DatasetClip, is_manifest_path, load_manifest
    from .reproject import load_reprojection_model
    from .state_estimator import CausalKalmanNet, FilterConfig
except ImportError:  # pragma: no cover - direct script execution
    from dataset import (
        build_metric_state_arrays,
        load_legacy_sequences,
        metric_feature_names,
        metric_state_method_names,
        resolve_method_allowlist,
    )
    from manifest import DatasetClip, is_manifest_path, load_manifest
    from reproject import load_reprojection_model
    from state_estimator import CausalKalmanNet, FilterConfig


def _require_torch() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for train_state_estimator.py")


def _log(message: str) -> None:
    print(message, flush=True)


def resolve_input_clips(inputs: List[str], metadata: str | None) -> List[DatasetClip]:
    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        if metadata:
            raise SystemExit("--metadata cannot be used with a manifest")
        return load_manifest(inputs[0])
    if metadata and len(inputs) != 1:
        raise SystemExit("--metadata can only be used with a single CSV input")
    metadata_path = Path(metadata) if metadata else None
    return [
        DatasetClip(
            csv=Path(item),
            metadata=metadata_path if len(inputs) == 1 else None,
            split="train",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def observation_consistency_loss(
    state_pos: torch.Tensor,  # [T, 3]
    obs_xyz: torch.Tensor,  # [T, M, 3]
    obs_valid: torch.Tensor,  # [T, M]
    df: float = 4.0,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Student-t robust distance between filtered position and each valid method."""

    residual = obs_xyz - state_pos.unsqueeze(1)  # [T, M, 3]
    valid = obs_valid.unsqueeze(-1)
    scaled = (residual * residual) / (df * sigma * sigma)
    nll = 0.5 * (df + 1.0) * torch.log1p(scaled)
    weighted = nll * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def known_z_loss(state_pos: torch.Tensor, known_z: float, valid: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    residual = state_pos[:, 2] - known_z
    loss = F.huber_loss(residual, torch.zeros_like(residual), delta=delta, reduction="none")
    weighted = loss * valid
    return weighted.sum() / valid.sum().clamp_min(1.0)


def static_jitter_loss(state_pos: torch.Tensor, valid: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    if state_pos.shape[0] < 2:
        return state_pos.new_tensor(0.0)
    dp = state_pos[1:, :] - state_pos[:-1, :]
    pair_valid = (valid[1:] * valid[:-1]).unsqueeze(-1)
    loss = F.huber_loss(dp, torch.zeros_like(dp), delta=delta, reduction="none") * pair_valid
    return loss.sum() / pair_valid.sum().clamp_min(1.0)


def continuity_loss(
    state_pos: torch.Tensor,  # [T, 3]
    dt: torch.Tensor,  # [T]
    gravity: float,
    gravity_axis: int,
    weight_jerk: float = 0.1,
) -> torch.Tensor:
    """Low-jerk / ballistic prior on the filtered trajectory.

    Second difference should be ~0 on all axes except the gravity axis, where it
    equals g*dt^2. When gravity==0 (axis unconfirmed) this is a pure low-jerk /
    constant-velocity prior, which is still correct for smooth flight.
    """

    if state_pos.shape[0] < 4:
        return state_pos.new_tensor(0.0)
    second = state_pos[2:, :] - 2.0 * state_pos[1:-1, :] + state_pos[:-2, :]
    target = torch.zeros_like(second)
    if gravity != 0.0:
        dt2 = dt[1:-1].pow(2)
        target[:, gravity_axis] = gravity * dt2
    accel_loss = F.huber_loss(second, target, delta=0.05, reduction="mean")
    jerk = second[1:, :] - second[:-1, :]
    jerk_loss = F.huber_loss(jerk, torch.zeros_like(jerk), delta=0.05, reduction="mean")
    return accel_loss + weight_jerk * jerk_loss


def gain_regularizer(gains: torch.Tensor, target: float = 0.3) -> torch.Tensor:
    """Keep learned gains away from 0 (ignore obs) and 1 (trust raw obs)."""

    return ((gains - target) ** 2).mean() * 0.01


def _detach_carry(carry: dict) -> dict:
    detached = {}
    for key, value in carry.items():
        detached[key] = value.detach() if torch.is_tensor(value) else value
    return detached


def _unroll_chunk(
    model: CausalKalmanNet,
    carry: dict,
    obs_xyz: torch.Tensor,
    obs_valid: torch.Tensor,
    quality: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    states = []
    logvars = []
    gains = []
    for idx in range(dt.shape[0]):
        state, logvar, carry = model.step(
            carry,
            obs_xyz[idx].unsqueeze(0),
            obs_valid[idx].unsqueeze(0),
            quality[idx].unsqueeze(0),
            dt[idx].unsqueeze(0),
        )
        states.append(state.squeeze(0))
        logvars.append(logvar.squeeze(0))
        gains.append(carry["last_gain"].squeeze(0))
    return torch.stack(states, dim=0), torch.stack(logvars, dim=0), torch.stack(gains, dim=0), carry


def _checkpoint_payload(
    model: CausalKalmanNet,
    cfg: FilterConfig,
    args: argparse.Namespace,
    method_allowlist,
    quality_names: List[str],
    *,
    epoch: int,
    metrics: Dict[str, float],
) -> dict:
    return {
        "model": model.state_dict(),
        "filter_config": vars(cfg),
        "method_allowlist": list(method_allowlist) if method_allowlist else None,
        "quality_feature_names": quality_names,
        "calib": args.calib,
        "offset_fit": args.offset_fit,
        "epoch": epoch,
        "metrics": metrics,
        "training_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "gravity": args.gravity,
            "gravity_axis": args.gravity_axis,
            "methods": args.methods,
            "seed": args.seed,
            "tbptt_steps": args.tbptt_steps,
            "max_frames_per_clip": args.max_frames_per_clip,
        },
        "note": "Causal KalmanNet-style 3D state estimator. Outputs [x,y,z,vx,vy,vz].",
    }


def _save_checkpoint(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="TrajectoryRecorder CSV(s) or one manifest")
    parser.add_argument("--metadata", help="metadata.yaml for a single CSV")
    parser.add_argument("--calib", required=True, help="stereo_calib.yaml for reprojection")
    parser.add_argument("--offset-fit", help="disparity_offset_fit.json (fitted fB/d0)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("-o", "--output", default="state_estimator.pt")
    parser.add_argument("--methods", default="p0p1_ncc_xfeat")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity", type=float, default=0.0, help="gravity magnitude on gravity axis (0 until confirmed)")
    parser.add_argument("--gravity-axis", type=int, default=1)
    parser.add_argument("--known-z-weight", type=float, default=1.0)
    parser.add_argument("--static-weight", type=float, default=1.0)
    parser.add_argument("--continuity-weight", type=float, default=0.25)
    parser.add_argument(
        "--tbptt-steps",
        type=int,
        default=512,
        help="Frames per truncated causal BPTT chunk; <=0 restores full-sequence backprop",
    )
    parser.add_argument(
        "--save-every-chunks",
        type=int,
        default=50,
        help="Write the latest checkpoint every N training chunks; <=0 disables intra-epoch saves",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=0,
        help="Debug/smoke option: keep only the first N frames from each clip",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _require_torch()
    torch.manual_seed(args.seed)

    method_allowlist = resolve_method_allowlist(args.methods)
    clips = resolve_input_clips(args.inputs, args.metadata)
    _log(f"resolved clips={len(clips)} train_split={args.train_split}")

    # Reprojection model (intrinsics + fitted d0) built once and shared.
    reprojection_model = load_reprojection_model(args.calib, args.offset_fit)

    quality_names = metric_feature_names(method_allowlist)
    method_names = metric_state_method_names(method_allowlist)
    num_methods = len(method_names)

    # weak_label_names() index order: known_z, known_z_valid, known_z_min,
    # known_z_max, known_z_range_valid, static, landing_frame, landing_valid.
    LBL_KNOWN_Z, LBL_KNOWN_Z_VALID, LBL_STATIC = 0, 1, 5

    # Load and pre-build all training sequences into tensors.
    sequences: List[dict] = []
    heldout = 0
    for clip_idx, clip in enumerate(clips, start=1):
        if clip.split != args.train_split:
            heldout += 1
            continue
        _log(f"[load {clip_idx}/{len(clips)}] {clip.name} split={clip.split} csv={clip.csv}")
        for seq in load_legacy_sequences(clip.csv, metadata_path=clip.metadata):
            if args.max_frames_per_clip > 0:
                seq.rows = seq.rows[: args.max_frames_per_clip]
            arrays = build_metric_state_arrays(
                seq,
                reprojection_model,
                method_names=method_allowlist,
            )
            if not arrays["dt"]:
                continue
            first_label = arrays["labels"][0]
            sequences.append(
                {
                    "clip": clip.name,
                    "points": arrays["points"],
                    "point_valid": arrays["point_valid"],
                    "features": arrays["features"],
                    "dt": arrays["dt"],
                    "known_z": float(first_label[LBL_KNOWN_Z]),
                    "known_z_valid": float(first_label[LBL_KNOWN_Z_VALID]),
                    "static": float(first_label[LBL_STATIC]),
                }
            )
            _log(
                f"  track={seq.track_id} frames={len(arrays['dt'])} "
                f"known_z={float(first_label[LBL_KNOWN_Z]):.3f} "
                f"known_valid={float(first_label[LBL_KNOWN_Z_VALID]):.0f} "
                f"static={float(first_label[LBL_STATIC]):.0f}"
            )

    if not sequences:
        raise SystemExit(f"no training sequences in split '{args.train_split}'")

    quality_dim = len(sequences[0]["features"][0]) if sequences[0]["features"] else 0
    cfg = FilterConfig(
        num_methods=num_methods,
        quality_dim=quality_dim,
        hidden_dim=args.hidden,
        gravity=args.gravity,
        gravity_axis=args.gravity_axis,
    )
    model = CausalKalmanNet(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    known_z_frames = sum(1 for s in sequences if s["known_z_valid"])
    static_frames = sum(1 for s in sequences if s["static"])
    _log(f"train sequences={len(sequences)} known_z_clips={known_z_frames} static_clips={static_frames} heldout={heldout}")
    _log(f"num_methods={num_methods} quality_dim={quality_dim} gravity={args.gravity} axis={args.gravity_axis}")
    _log(f"tbptt_steps={args.tbptt_steps} save_every_chunks={args.save_every_chunks} max_frames_per_clip={args.max_frames_per_clip}")

    tensors = []
    for s in sequences:
        tensors.append({
            "clip": s["clip"],
            "obs_xyz": torch.tensor(s["points"], dtype=torch.float32, device=args.device),
            "obs_valid": torch.tensor(s["point_valid"], dtype=torch.float32, device=args.device),
            "quality": torch.tensor(s["features"], dtype=torch.float32, device=args.device)
            if quality_dim
            else torch.zeros(len(s["dt"]), 0, dtype=torch.float32, device=args.device),
            "dt": torch.tensor([d[0] for d in s["dt"]], dtype=torch.float32, device=args.device),
            "known_z": float(s["known_z"]),
            "known_z_valid": float(s["known_z_valid"]),
            "static": float(s["static"]),
        })

    last_metrics: Dict[str, float] = {}
    global_chunks = 0
    chunk_size = args.tbptt_steps if args.tbptt_steps > 0 else 10**12
    for epoch in range(args.epochs):
        total = 0.0
        total_chunks = 0
        total_frames = 0
        for t in tensors:
            n_frames = t["dt"].shape[0]
            carry = model.init_state(1, t["obs_xyz"].device, t["obs_xyz"].dtype)
            for start in range(0, n_frames, chunk_size):
                end = min(n_frames, start + chunk_size)
                opt.zero_grad(set_to_none=True)
                states, logvars, gains, carry = _unroll_chunk(
                    model,
                    carry,
                    t["obs_xyz"][start:end],
                    t["obs_valid"][start:end],
                    t["quality"][start:end],
                    t["dt"][start:end],
                )
                state_pos = states[:, :3]
                any_valid = (t["obs_valid"][start:end].sum(dim=1) > 0.0).to(state_pos.dtype)

                loss_obs = observation_consistency_loss(
                    state_pos,
                    t["obs_xyz"][start:end],
                    t["obs_valid"][start:end],
                )
                loss = loss_obs
                loss = loss + args.continuity_weight * continuity_loss(
                    state_pos,
                    t["dt"][start:end],
                    args.gravity,
                    args.gravity_axis,
                )
                loss = loss + gain_regularizer(gains)
                if t["known_z_valid"] > 0.0:
                    loss = loss + args.known_z_weight * known_z_loss(state_pos, t["known_z"], any_valid)
                if t["static"] > 0.0:
                    loss = loss + args.static_weight * static_jitter_loss(state_pos, any_valid)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch + 1} clip={t['clip']} frames={start}:{end}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                carry = _detach_carry(carry)

                loss_value = float(loss.detach().cpu())
                total += loss_value
                total_chunks += 1
                total_frames += end - start
                global_chunks += 1

                if args.save_every_chunks > 0 and global_chunks % args.save_every_chunks == 0:
                    metrics = {
                        "loss": total / max(total_chunks, 1),
                        "chunks": float(total_chunks),
                        "frames": float(total_frames),
                    }
                    _save_checkpoint(
                        args.output,
                        _checkpoint_payload(
                            model,
                            cfg,
                            args,
                            method_allowlist,
                            quality_names,
                            epoch=epoch + 1,
                            metrics=metrics,
                        ),
                    )
                    _log(
                        f"checkpoint epoch={epoch + 1:04d} chunk={global_chunks} "
                        f"loss={metrics['loss']:.6f} -> {args.output}"
                    )

        mean_loss = total / max(total_chunks, 1)
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"non-finite epoch loss: {mean_loss}")
        last_metrics = {
            "loss": mean_loss,
            "chunks": float(total_chunks),
            "frames": float(total_frames),
        }
        _log(
            f"epoch={epoch + 1:04d} loss={mean_loss:.6f} "
            f"chunks={total_chunks} frames={total_frames}"
        )
        _save_checkpoint(
            args.output,
            _checkpoint_payload(
                model,
                cfg,
                args,
                method_allowlist,
                quality_names,
                epoch=epoch + 1,
                metrics=last_metrics,
            ),
        )
        _log(f"saved checkpoint epoch={epoch + 1:04d} -> {args.output}")

    _log(f"saved {args.output} final_loss={last_metrics.get('loss', 0.0):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
