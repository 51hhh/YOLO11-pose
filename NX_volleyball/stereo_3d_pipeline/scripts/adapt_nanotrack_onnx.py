#!/usr/bin/env python3
"""
adapt_nanotrack_onnx.py — Adapt existing NanoTrack 3ch ONNX to 1ch grayscale + dynamic shape

Takes the pre-existing NanoTrack ONNX models (from ZhangLi1210/NanoTrack_Tensorrt_Cpp)
and produces models compatible with our NanoTrackTRT C++ code:
  - Backbone: 1ch input, dynamic H/W (127 or 255)
  - Head: same tensor names as expected (template_feat/search_feat/cls_score/bbox_reg)

Usage:
  python adapt_nanotrack_onnx.py \
    --backbone_template nanotrack_backbone_template.onnx \
    --backbone_search nanotrack_backbone_exampler.onnx \
    --head nanotrack_head.onnx \
    --out_dir ./adapted
"""

import argparse
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("ERROR: pip install onnx")
    exit(1)


def conv3ch_to_1ch(model):
    """Modify first Conv node's weight from [out, 3, kH, kW] to [out, 1, kH, kW] via channel averaging."""
    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Conv":
            weight_name = node.input[1]
            if weight_name in initializer_map:
                w_tensor = initializer_map[weight_name]
                w = numpy_helper.to_array(w_tensor)
                if w.ndim == 4 and w.shape[1] == 3:
                    # Average over input channels: [out, 3, kH, kW] -> [out, 1, kH, kW]
                    w_1ch = w.mean(axis=1, keepdims=True).astype(np.float32)
                    new_tensor = numpy_helper.from_array(w_1ch, name=weight_name)

                    # Replace in initializer list
                    for i, init in enumerate(graph.initializer):
                        if init.name == weight_name:
                            graph.initializer[i].CopyFrom(new_tensor)
                            break

                    print(f"  Conv weight {weight_name}: {w.shape} -> {w_1ch.shape}")
                    return True
    return False


def set_input_channels(model, channels=1):
    """Change model input from 3ch to 1ch."""
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            old_ch = shape.dim[1].dim_value
            if old_ch == 3:
                shape.dim[1].dim_value = channels
                print(f"  Input {inp.name}: channel {old_ch} -> {channels}")
                return True
    return False


def make_dynamic_shape(model, h_name="height", w_name="width"):
    """Make H and W dimensions dynamic."""
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            shape.dim[2].dim_param = h_name
            shape.dim[2].ClearField("dim_value")
            shape.dim[3].dim_param = w_name
            shape.dim[3].ClearField("dim_value")
            print(f"  Input {inp.name}: H,W -> dynamic ({h_name}, {w_name})")

    for out in model.graph.output:
        shape = out.type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            shape.dim[2].dim_param = "feat_h"
            shape.dim[2].ClearField("dim_value")
            shape.dim[3].dim_param = "feat_w"
            shape.dim[3].ClearField("dim_value")
            print(f"  Output {out.name}: H,W -> dynamic (feat_h, feat_w)")


def rename_io(model, input_map, output_map):
    """Rename input/output tensor names."""
    for inp in model.graph.input:
        if inp.name in input_map:
            old = inp.name
            inp.name = input_map[old]
            # Update all node references
            for node in model.graph.node:
                for i, name in enumerate(node.input):
                    if name == old:
                        node.input[i] = input_map[old]
            print(f"  Renamed input: {old} -> {input_map[old]}")

    for out in model.graph.output:
        if out.name in output_map:
            old = out.name
            out.name = output_map[old]
            for node in model.graph.node:
                for i, name in enumerate(node.output):
                    if name == old:
                        node.output[i] = output_map[old]
            print(f"  Renamed output: {old} -> {output_map[old]}")


def merge_backbones_to_dynamic(template_path, search_path, out_path):
    """
    The two backbone ONNX files share the same weights (just different input shapes).
    Take one and make it dynamic shape so it works for both 127x127 and 255x255.
    """
    print("\n=== Merging backbones into single dynamic-shape model ===")
    model = onnx.load(template_path)

    # Verify weights are identical
    search_model = onnx.load(search_path)
    t_weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    s_weights = {init.name: numpy_helper.to_array(init) for init in search_model.graph.initializer}

    all_match = True
    for name in t_weights:
        if name in s_weights:
            if not np.allclose(t_weights[name], s_weights[name]):
                print(f"  [WARN] Weight mismatch: {name}")
                all_match = False
    if all_match:
        print("  Weights verified: template == search (same backbone)")

    # Convert to 1ch
    print("\n--- Converting backbone to 1ch ---")
    conv3ch_to_1ch(model)
    set_input_channels(model, 1)

    # Make dynamic
    make_dynamic_shape(model)

    # Rename to standard names
    rename_io(model, {"input": "input"}, {"output": "output"})

    # Simplify
    try:
        from onnxsim import simplify
        model_sim, ok = simplify(model)
        if ok:
            model = model_sim
            print("  [SIM] Simplified")
    except ImportError:
        pass

    onnx.save(model, out_path)
    print(f"  [OK] Saved: {out_path}")
    return model


def adapt_head(head_path, out_path):
    """Rename head I/O to match our C++ expectations."""
    print("\n=== Adapting head model ===")
    model = onnx.load(head_path)

    # Print original I/O
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Original input: {inp.name} {shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Original output: {out.name} {shape}")

    # Rename I/O
    rename_io(model,
              {"input1": "template_feat", "input2": "search_feat"},
              {"output1": "cls_score", "output2": "bbox_reg"})

    try:
        from onnxsim import simplify
        model_sim, ok = simplify(model)
        if ok:
            model = model_sim
            print("  [SIM] Simplified")
    except ImportError:
        pass

    onnx.save(model, out_path)
    print(f"  [OK] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_template", required=True)
    parser.add_argument("--backbone_search", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out_dir", default="./adapted")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    backbone_out = os.path.join(args.out_dir, "nanotrack_backbone_1ch.onnx")
    head_out = os.path.join(args.out_dir, "nanotrack_head_adapted.onnx")

    merge_backbones_to_dynamic(args.backbone_template, args.backbone_search, backbone_out)
    adapt_head(args.head, head_out)

    print("\n========================================")
    print("=== Adaptation complete ===")
    print(f"  Backbone: {backbone_out}  (1ch, dynamic H/W)")
    print(f"  Head: {head_out}  (renamed I/O)")
    print("\n  trtexec commands:")
    print(f"  trtexec --onnx={backbone_out} --saveEngine=nanotrack_backbone.engine \\")
    print(f"    --fp16 --minShapes=input:1x1x127x127 --optShapes=input:1x1x255x255 \\")
    print(f"    --maxShapes=input:1x1x255x255 --workspace=256")
    print(f"  trtexec --onnx={head_out} --saveEngine=nanotrack_head.engine \\")
    print(f"    --fp16 --workspace=256")


if __name__ == "__main__":
    main()
