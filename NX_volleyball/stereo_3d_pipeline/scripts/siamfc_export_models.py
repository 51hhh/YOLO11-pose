"""SiamFC model wrappers used by the ONNX export script."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFCBackbone(nn.Module):
    """AlexNet-based SiamFC backbone with native 1-channel grayscale input."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        return self.features(x)


class SiamFCHead(nn.Module):
    """Cross-correlation head with classification score output only."""

    def __init__(self):
        super().__init__()

    def forward(self, zf, xf):
        n, _, _, _ = zf.shape
        score = F.conv2d(xf, zf, groups=n)
        return score.sum(dim=1, keepdim=True)


class SiamFCBackboneFromPretrained(nn.Module):
    """Load backbone from a pretrained SiamFC checkpoint with 3ch -> 1ch conversion."""

    def __init__(self, backbone_3ch):
        super().__init__()
        self.features = nn.Sequential()
        first_done = False
        for i, layer in enumerate(backbone_3ch):
            if isinstance(layer, nn.Conv2d) and not first_done:
                w = layer.weight.data
                new_conv = nn.Conv2d(
                    1,
                    layer.out_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias is not None,
                )
                new_conv.weight.data = w.mean(dim=1, keepdim=True)
                if layer.bias is not None:
                    new_conv.bias.data = layer.bias.data
                self.features.add_module(str(i), new_conv)
                first_done = True
            else:
                self.features.add_module(str(i), layer)

    def forward(self, x):
        return self.features(x)


def load_siamfc_checkpoint(ckpt_path, backbone):
    """Attempt to load a SiamFC checkpoint into the export backbone."""

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]

    model_dict = backbone.state_dict()
    matched = {}
    for key, value in ckpt.items():
        for prefix in ["backbone.", "features.", "feature_extractor.", ""]:
            clean = key.replace(prefix, "", 1) if key.startswith(prefix) else None
            if clean and clean in model_dict:
                if value.shape == model_dict[clean].shape:
                    matched[clean] = value
                elif clean.endswith(".weight") and len(value.shape) == 4 and value.shape[1] == 3:
                    matched[clean] = value.mean(dim=1, keepdim=True)
                break

    if matched:
        model_dict.update(matched)
        backbone.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(matched)}/{len(model_dict)} parameters from checkpoint")
    else:
        print("  [WARN] No matching keys found in checkpoint, using random weights")

    return backbone
