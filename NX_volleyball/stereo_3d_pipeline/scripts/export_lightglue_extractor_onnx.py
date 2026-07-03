#!/usr/bin/env python3
"""Export LightGlue ALIKED/SuperPoint extractors to fixed-shape ONNX.

The realtime C++ path consumes TensorRT engines that output:
  keypoints   [1, top_k, 2]
  descriptors [1, top_k, descriptor_dim]
  scores      [1, top_k]

This script exports the real LightGlue extractor models only. Matching is still
performed by the realtime C++ direct-extractor fallback unless a separate
matcher/fused engine is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


class FixedSuperPoint(torch.nn.Module):
    def __init__(self, top_k: int, roi_size: int) -> None:
        super().__init__()
        from lightglue import SuperPoint

        self.extractor = SuperPoint(
            max_num_keypoints=top_k,
            detection_threshold=-1,
        ).eval()
        self.top_k = int(top_k)
        self.roi_size = int(roi_size)
        self.descriptor_dim = 256
        mask = torch.ones(1, 1, self.roi_size, self.roi_size, dtype=torch.float32)
        border = int(self.extractor.conf.remove_borders)
        if border > 0:
            mask[:, :, :border, :] = 0.0
            mask[:, :, -border:, :] = 0.0
            mask[:, :, :, :border] = 0.0
            mask[:, :, :, -border:] = 0.0
        self.register_buffer("border_mask", mask)

    @staticmethod
    def _simple_nms(scores: torch.Tensor, radius: int) -> torch.Tensor:
        zeros = torch.zeros_like(scores)
        max_mask = scores == F.max_pool2d(
            scores, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )
        for _ in range(2):
            supp_mask = F.max_pool2d(
                max_mask.float(),
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            ) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == F.max_pool2d(
                supp_scores,
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            )
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def _dense_scores_and_descriptors(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.extractor
        x = e.relu(e.conv1a(image))
        x = e.relu(e.conv1b(x))
        x = e.pool(x)
        x = e.relu(e.conv2a(x))
        x = e.relu(e.conv2b(x))
        x = e.pool(x)
        x = e.relu(e.conv3a(x))
        x = e.relu(e.conv3b(x))
        x = e.pool(x)
        x = e.relu(e.conv4a(x))
        x = e.relu(e.conv4b(x))

        c_pa = e.relu(e.convPa(x))
        scores = e.convPb(c_pa)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, 1, h * 8, w * 8)
        scores = self._simple_nms(scores, int(e.conf.nms_radius))
        scores = scores * self.border_mask - (1.0 - self.border_mask)

        c_da = e.relu(e.convDa(x))
        descriptors = e.convDb(c_da)
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return scores, descriptors

    def _sample_descriptors(
        self, keypoints: torch.Tensor, descriptors: torch.Tensor, stride: int = 8
    ) -> torch.Tensor:
        _, _, h, w = descriptors.shape
        points = keypoints - stride / 2 + 0.5
        scale = torch.tensor(
            [w * stride - stride / 2 - 0.5, h * stride - stride / 2 - 0.5],
            dtype=points.dtype,
            device=points.device,
        ).view(1, 1, 2)
        points = points / scale
        points = points * 2.0 - 1.0
        sampled = F.grid_sample(
            descriptors,
            points.view(points.shape[0], 1, self.top_k, 2),
            mode="bilinear",
            align_corners=True,
        )
        sampled = F.normalize(sampled.reshape(points.shape[0], self.descriptor_dim, -1),
                              p=2, dim=1)
        return sampled.transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_map, dense_desc = self._dense_scores_and_descriptors(images)
        flat = scores_map.reshape(images.shape[0], -1)
        scores, indices = torch.topk(flat, self.top_k, dim=1, sorted=True)
        width = float(self.roi_size)
        idx = indices.to(dtype=torch.float32)
        ys = torch.floor(idx / width)
        xs = idx - ys * width
        keypoints = torch.stack((xs, ys), dim=-1)
        descriptors = self._sample_descriptors(keypoints, dense_desc)
        return keypoints.contiguous(), descriptors, scores.contiguous()


class FixedExtractor(torch.nn.Module):
    def __init__(self, backend: str, top_k: int, aliked_model: str, roi_size: int) -> None:
        super().__init__()
        self.backend = backend
        self.top_k = int(top_k)
        if backend == "aliked":
            from lightglue import ALIKED

            self.extractor = ALIKED(
                max_num_keypoints=self.top_k,
                detection_threshold=-1,
                model_name=aliked_model,
            ).eval()
            self.descriptor_dim = 128 if aliked_model != "aliked-t16" else 64
        elif backend == "superpoint":
            self.extractor = FixedSuperPoint(self.top_k, roi_size)
            self.descriptor_dim = 256
        else:
            raise ValueError(f"unsupported backend: {backend}")

    def _fixed_count(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keypoints = keypoints[:, : self.top_k, :]
        descriptors = descriptors[:, : self.top_k, :]
        scores = scores[:, : self.top_k]
        count = keypoints.shape[1]
        if count < self.top_k:
            pad = self.top_k - count
            keypoints = F.pad(keypoints, (0, 0, 0, pad))
            descriptors = F.pad(descriptors, (0, 0, 0, pad))
            scores = F.pad(scores, (0, pad))
        return keypoints.contiguous(), descriptors.contiguous(), scores.contiguous()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backend == "superpoint":
            return self.extractor(images)
        pred = self.extractor({"image": images})
        scores = pred.get("keypoint_scores", pred.get("scores"))
        if scores is None:
            scores = torch.ones(
                pred["keypoints"].shape[:2],
                dtype=pred["keypoints"].dtype,
                device=pred["keypoints"].device,
            )
        keypoints, descriptors, scores = self._fixed_count(
            pred["keypoints"],
            pred["descriptors"],
            scores,
        )
        return keypoints, descriptors, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("aliked", "superpoint"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--aliked-model", default="aliked-n16")
    args = parser.parse_args()

    if args.roi_size % 32 != 0:
        raise ValueError("roi-size must be divisible by 32")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = FixedExtractor(args.backend, args.top_k, args.aliked_model, args.roi_size).eval()
    dummy = torch.rand(1, 1, args.roi_size, args.roi_size, dtype=torch.float32)
    with torch.no_grad():
        keypoints, descriptors, scores = model(dummy)
    print(
        f"{args.backend}: keypoints={tuple(keypoints.shape)} "
        f"descriptors={tuple(descriptors.shape)} scores={tuple(scores.shape)}"
    )

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
