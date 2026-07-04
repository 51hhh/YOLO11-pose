"""ONNX graph adapters for NanoTrack grayscale/dynamic-shape export."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper


def conv3ch_to_1ch(model):
    """Modify first Conv node's weight from [out, 3, kH, kW] to [out, 1, kH, kW]."""

    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Conv":
            weight_name = node.input[1]
            if weight_name in initializer_map:
                w_tensor = initializer_map[weight_name]
                w = numpy_helper.to_array(w_tensor)
                if w.ndim == 4 and w.shape[1] == 3:
                    w_1ch = w.mean(axis=1, keepdims=True).astype(np.float32)
                    new_tensor = numpy_helper.from_array(w_1ch, name=weight_name)

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
            print("  Output {name}: H,W -> dynamic (feat_h, feat_w)".format(name=out.name))


def rename_io(model, input_map, output_map):
    """Rename input/output tensor names."""

    for inp in model.graph.input:
        if inp.name in input_map:
            old = inp.name
            inp.name = input_map[old]
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


def simplify_if_available(model):
    try:
        from onnxsim import simplify
        model_sim, ok = simplify(model)
        if ok:
            print("  [SIM] Simplified")
            return model_sim
    except ImportError:
        pass
    return model


def merge_backbones_to_dynamic(template_path, search_path, out_path):
    """
    The two backbone ONNX files share the same weights.
    Take one and make it dynamic shape so it works for both 127x127 and 255x255.
    """

    print("\n=== Merging backbones into single dynamic-shape model ===")
    model = onnx.load(template_path)

    search_model = onnx.load(search_path)
    t_weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    s_weights = {init.name: numpy_helper.to_array(init) for init in search_model.graph.initializer}

    all_match = True
    for name in t_weights:
        if name in s_weights and not np.allclose(t_weights[name], s_weights[name]):
            print(f"  [WARN] Weight mismatch: {name}")
            all_match = False
    if all_match:
        print("  Weights verified: template == search (same backbone)")

    print("\n--- Converting backbone to 1ch ---")
    conv3ch_to_1ch(model)
    set_input_channels(model, 1)
    make_dynamic_shape(model)
    rename_io(model, {"input": "input"}, {"output": "output"})

    model = simplify_if_available(model)
    onnx.save(model, out_path)
    print(f"  [OK] Saved: {out_path}")
    return model


def adapt_head(head_path, out_path):
    """Rename head I/O to match the C++ NanoTrackTRT expectations."""

    print("\n=== Adapting head model ===")
    model = onnx.load(head_path)

    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Original input: {inp.name} {shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Original output: {out.name} {shape}")

    rename_io(
        model,
        {"input1": "template_feat", "input2": "search_feat"},
        {"output1": "cls_score", "output2": "bbox_reg"},
    )

    model = simplify_if_available(model)
    onnx.save(model, out_path)
    print(f"  [OK] Saved: {out_path}")
