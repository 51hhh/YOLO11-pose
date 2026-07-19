#!/usr/bin/env python3
"""Export official LightGlue ALIKED with DCN for TensorRT plugin builds.

This path is intentionally separate from export_lightglue_extractor_onnx.py:
the regular exporter is kept for SuperPoint and no-DCN experiments, while this
script preserves ALIKED deformable convolution and rewrites ONNX DeformConv
nodes to the custom dcn::DCNv2 nodes consumed by the TensorRT DCNv2 plugin.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_lightglue_extractor_onnx import FixedExtractor  # noqa: E402


def _pair(value: object) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def patch_aliked_dcn_forward(model: torch.nn.Module) -> int:
    """Make ALIKED DCN export explicit: offset, ones mask, weight, zero bias."""

    try:
        import torchvision.ops
    except Exception as exc:  # pragma: no cover - depends on NX env
        raise RuntimeError("torchvision.ops is required for ALIKED DCN export") from exc

    patched = 0
    for module in model.modules():
        if not (hasattr(module, "offset_conv") and hasattr(module, "regular_conv")):
            continue

        def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            out = self.offset_conv(x)
            if bool(getattr(self, "mask", False)):
                o1, o2, mask_logits = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask_logits)
            else:
                offset = out
                # TensorRT DCNv2 plugin expects a mask input. ALIKED's DCN
                # layers use mask=False, so a ones mask preserves semantics.
                mask_channels = offset.shape[1] // 2
                mask = torch.ones(
                    (x.shape[0], mask_channels, offset.shape[2], offset.shape[3]),
                    dtype=x.dtype,
                    device=x.device,
                )

            max_offset = getattr(self, "max_offset", None)
            if max_offset is not None:
                offset = torch.clamp(offset, -float(max_offset), float(max_offset))

            bias = self.regular_conv.bias
            if bias is None:
                bias = x.new_zeros((self.regular_conv.out_channels,))

            return torchvision.ops.deform_conv2d(
                input=x,
                offset=offset,
                weight=self.regular_conv.weight,
                bias=bias,
                stride=_pair(getattr(self, "stride", self.regular_conv.stride)),
                padding=_pair(getattr(self, "padding", self.regular_conv.padding)),
                dilation=_pair(getattr(self, "dilation", self.regular_conv.dilation)),
                mask=mask,
            )

        module.forward = types.MethodType(forward, module)
        patched += 1
    return patched


def rewrite_deformconv_to_dcnv2_plugin(path: Path) -> tuple[int, int]:
    import onnx
    from onnx import helper

    model = onnx.load(str(path))
    converted = 0
    stripped_scatternd_reduction = 0

    def get_ints(node, name: str, default: list[int]) -> list[int]:
        for attr in node.attribute:
            if attr.name != name:
                continue
            if attr.ints:
                return [int(v) for v in attr.ints]
            if attr.i:
                return [int(attr.i)]
        return list(default)

    for node in model.graph.node:
        if node.op_type != "DeformConv":
            continue
        if len(node.input) < 5 or not node.input[3] or not node.input[4]:
            raise RuntimeError(
                "DeformConv node does not have explicit bias and mask inputs; "
                "ALIKED DCN export must patch both before TensorRT plugin rewrite"
            )

        x_name = node.input[0]
        weight_name = node.input[1]
        offset_name = node.input[2]
        bias_name = node.input[3]
        mask_name = node.input[4]

        kernel = get_ints(node, "kernel_shape", [3, 3])
        stride = get_ints(node, "strides", [1, 1])
        dilation = get_ints(node, "dilations", [1, 1])
        pads = get_ints(node, "pads", [0, 0, 0, 0])
        if len(pads) >= 4:
            padding = [pads[0], pads[1]]
        else:
            padding = pads[:2]
        deformable_groups = get_ints(node, "offset_group", [1])[0]

        node.domain = "dcn"
        node.op_type = "DCNv2"
        node.ClearField("input")
        # flairziv/tensorrt-dcnv2-plugin schema:
        # input, offset, mask, weight, bias
        node.input.extend([x_name, offset_name, mask_name, weight_name, bias_name])
        node.ClearField("attribute")
        node.attribute.extend(
            [
                helper.make_attribute("stride", stride),
                helper.make_attribute("padding", padding),
                helper.make_attribute("dilation", dilation),
                helper.make_attribute("kernel", kernel),
                helper.make_attribute("deformable_groups", int(deformable_groups)),
            ]
        )
        converted += 1

    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue
        keep = []
        stripped = False
        for attr in node.attribute:
            if attr.name == "reduction" and attr.s == b"none":
                stripped = True
                continue
            keep.append(attr)
        if stripped:
            node.ClearField("attribute")
            node.attribute.extend(keep)
            stripped_scatternd_reduction += 1

    if converted == 0:
        raise RuntimeError("No ONNX DeformConv nodes found to rewrite")

    if not any(opset.domain == "dcn" for opset in model.opset_import):
        model.opset_import.extend([helper.make_operatorsetid("dcn", 1)])
    onnx.save(model, str(path))
    return converted, stripped_scatternd_reduction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--aliked-model", default="aliked-t16")
    parser.add_argument("--keep-standard-deformconv", action="store_true")
    args = parser.parse_args()

    if args.opset < 19:
        raise ValueError("ALIKED DCN export requires opset >= 19")
    try:
        import onnxscript  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "onnxscript is required for the PyTorch dynamo ONNX exporter path. "
            "Install it on NX before exporting official ALIKED DCN."
        ) from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = FixedExtractor("aliked", args.top_k, args.aliked_model, args.roi_size).eval()
    patched = patch_aliked_dcn_forward(model)
    if patched <= 0:
        raise RuntimeError("No ALIKED DeformableConv2d modules were patched")

    dummy = torch.rand(args.batch_size, 1, args.roi_size, args.roi_size)
    with torch.no_grad():
        keypoints, descriptors, scores = model(dummy)
    print(
        f"aliked-dcn: model={args.aliked_model} patched_dcn={patched} "
        f"keypoints={tuple(keypoints.shape)} descriptors={tuple(descriptors.shape)} "
        f"scores={tuple(scores.shape)}"
    )

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["images"],
        output_names=["keypoints", "descriptors", "scores"],
        opset_version=args.opset,
        dynamo=True,
    )
    if not args.keep_standard_deformconv:
        converted, stripped = rewrite_deformconv_to_dcnv2_plugin(out)
        print(
            f"rewrote DeformConv nodes to dcn::DCNv2: {converted}; "
            f"removed ScatterND reduction='none': {stripped}"
        )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
