#!/usr/bin/env python3
"""
export_mixformerv2_onnx.py
MixFormerV2-S -> single ONNX (grayscale 1ch, 2 inputs: template+search -> [score,cx,cy,w,h])

Setup:
  git clone https://github.com/MCG-NJU/MixFormerV2.git
  cd MixFormerV2
  pip install -r requirements.txt  # timm, einops, etc.
  # Download checkpoint: MixFormerV2-S from project page

Usage:
  python export_mixformerv2_onnx.py \
    --config experiments/mixformer2_vit/config_s.yaml \
    --checkpoint checkpoints/mixformerv2_s.pth \
    --output ./onnx_out/mixformerv2_s.onnx
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())

from lib.config.mixformer2_vit.config import cfg, update_config_from_file
from lib.models.mixformer2_vit.mixformer2_vit import build_mixformer_vit


def box_xyxy_to_cxcywh(boxes):
    """Pure tensor ops for ONNX tracing: xyxy -> cxcywh"""
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = x1 - x0
    h = y1 - y0
    return torch.stack([cx, cy, w, h], dim=-1)


def patch_first_conv_to_grayscale(model):
    """Patch PatchEmbed proj Conv2d from 3ch to 1ch"""
    patch_embed = model.backbone.patch_embed
    old_proj = patch_embed.proj
    if old_proj.in_channels != 3:
        print(f"[WARN] patch_embed.proj in_channels={old_proj.in_channels}, skip")
        return model

    new_proj = nn.Conv2d(
        1, old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=(old_proj.bias is not None),
    )
    with torch.no_grad():
        new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    patch_embed.proj = new_proj
    print("[INFO] Patched PatchEmbed: Conv2d(3,...) -> Conv2d(1,...)")
    return model


class MixFormerV2Wrapper(nn.Module):
    """
    Merge MixFormerV2 two-stage inference into single forward:
      forward(template, search) -> [score, cx_norm, cy_norm, w_norm, h_norm]

    Simplification: online_template = template (no online template update)
    """
    def __init__(self, mixformer_model, score_decoder=None):
        super().__init__()
        self.backbone = mixformer_model.backbone
        self.box_head = mixformer_model.box_head
        self.score_decoder = score_decoder

    def forward(self, template, search):
        """
        template: [1, 1, 128, 128]
        search:   [1, 1, 256, 256]
        returns:  [1, 5] -> [score, cx, cy, w, h] normalized [0,1]
        """
        t_feat, ot_feat, s_feat, reg_tokens, _ = self.backbone(
            template, template, search
        )

        pred_boxes_xyxy, prob_l, prob_t, prob_r, prob_b = self.box_head(
            reg_tokens, softmax=True
        )
        pred_boxes_cxcywh = box_xyxy_to_cxcywh(pred_boxes_xyxy)

        if self.score_decoder is not None:
            score = torch.sigmoid(self.score_decoder(reg_tokens))
        else:
            max_prob = torch.stack([
                prob_l.max(dim=-1).values,
                prob_t.max(dim=-1).values,
                prob_r.max(dim=-1).values,
                prob_b.max(dim=-1).values,
            ], dim=-1)
            score = max_prob.mean(dim=-1, keepdim=True)

        output = torch.cat([score, pred_boxes_cxcywh], dim=-1)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='MixFormerV2-S config yaml')
    parser.add_argument('--checkpoint', required=True,
                        help='MixFormerV2-S checkpoint .pth')
    parser.add_argument('--output', default='./onnx_out/mixformerv2_s.onnx')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    update_config_from_file(args.config)

    model = build_mixformer_vit(cfg, train=False)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'net' in ckpt:
        state = ckpt['net']
    elif 'model' in ckpt:
        state = ckpt['model']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    score_decoder = None
    try:
        from lib.models.mixformer2_vit.head import build_score_decoder
        score_decoder = build_score_decoder(cfg)
        sd_keys = {k.replace('score_decoder.', ''): v
                   for k, v in state.items() if 'score_decoder' in k}
        if sd_keys:
            score_decoder.load_state_dict(sd_keys, strict=False)
            print("[INFO] Score decoder loaded")
        else:
            print("[WARN] No score_decoder weights, using prob-based score")
            score_decoder = None
    except Exception as e:
        print(f"[WARN] Score decoder not available: {e}")

    model = patch_first_conv_to_grayscale(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper = MixFormerV2Wrapper(model, score_decoder).to(device).eval()

    template = torch.randn(1, 1, 128, 128, device=device)
    search = torch.randn(1, 1, 256, 256, device=device)

    torch.onnx.export(
        wrapper, (template, search), args.output,
        input_names=['template', 'search'],
        output_names=['output'],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[OK] Exported: {args.output}")

    with torch.no_grad():
        out = wrapper(template, search)
        print(f"  output shape: {list(out.shape)}")
        print(f"  sample: score={out[0,0]:.4f}, cx={out[0,1]:.4f}, cy={out[0,2]:.4f}, "
              f"w={out[0,3]:.4f}, h={out[0,4]:.4f}")

    try:
        import onnx
        from onnxsim import simplify as onnx_simplify
        m = onnx.load(args.output)
        m_sim, ok = onnx_simplify(m)
        if ok:
            onnx.save(m_sim, args.output)
            print(f"  [SIM] Simplified: {args.output}")
    except ImportError:
        print("  [SKIP] onnxsim not installed")

    print("\n=== MixFormerV2 export done ===")
    print(f"Engine: {args.output}")
    print("\ntrtexec command:")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={args.output} "
          f"--saveEngine=mixformerv2_fp16.engine --fp16 --workspace=512")


if __name__ == '__main__':
    main()
