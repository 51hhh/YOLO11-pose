#!/usr/bin/env python3
"""
Decompose attention Softmax in yolo26.onnx to prevent TRT ForeignNode fusion.

TRT 10.3 on Orin NX fuses Q*K^T -> Softmax -> *V into ForeignNode (flash attention)
but has no FP16 implementation for certain attention sizes, causing build failure.

This script replaces Softmax nodes in attention blocks with equivalent
Max-Sub-Exp-Sum-Div decomposition to break the fusion pattern while
preserving identical numerical behavior.

Input:  yolo26.onnx          [1,3,640,640] -> [1,5,8400]
Output: yolo26_noflash.onnx  [1,3,640,640] -> [1,5,8400]  (same I/O)
"""
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def decompose_attention_softmax(input_path, output_path):
    print(f"Loading {input_path}...")
    graph = gs.import_onnx(onnx.load(input_path))

    # Find all Softmax nodes in attention blocks
    softmax_nodes = [n for n in graph.nodes if n.op == "Softmax" and "attn" in n.name]
    print(f"Found {len(softmax_nodes)} Softmax nodes in attention blocks")

    if not softmax_nodes:
        # Fallback: find all Softmax nodes
        softmax_nodes = [n for n in graph.nodes if n.op == "Softmax"]
        print(f"Fallback: found {len(softmax_nodes)} total Softmax nodes")

    new_nodes = []
    for i, node in enumerate(softmax_nodes):
        axis = node.attrs.get("axis", -1)
        inp = node.inputs[0]
        out = node.outputs[0]

        prefix = f"decomposed_softmax_{i}"

        # Step 1: ReduceMax (for numerical stability)
        max_out = gs.Variable(f"{prefix}_max", dtype=np.float32)
        reduce_max = gs.Node(
            op="ReduceMax",
            name=f"{prefix}/ReduceMax",
            inputs=[inp],
            outputs=[max_out],
            attrs={"axes": [axis], "keepdims": 1}
        )

        # Step 2: Sub (x - max)
        sub_out = gs.Variable(f"{prefix}_sub", dtype=np.float32)
        sub_node = gs.Node(
            op="Sub",
            name=f"{prefix}/Sub",
            inputs=[inp, max_out],
            outputs=[sub_out]
        )

        # Step 3: Exp
        exp_out = gs.Variable(f"{prefix}_exp", dtype=np.float32)
        exp_node = gs.Node(
            op="Exp",
            name=f"{prefix}/Exp",
            inputs=[sub_out],
            outputs=[exp_out]
        )

        # Step 4: ReduceSum
        sum_out = gs.Variable(f"{prefix}_sum", dtype=np.float32)
        reduce_sum = gs.Node(
            op="ReduceSum",
            name=f"{prefix}/ReduceSum",
            inputs=[exp_out],
            outputs=[sum_out],
            attrs={"axes": [axis], "keepdims": 1}
        )

        # Step 5: Div (exp / sum) - reuse original output tensor
        div_node = gs.Node(
            op="Div",
            name=f"{prefix}/Div",
            inputs=[exp_out, sum_out],
            outputs=[out]
        )

        # Clear original node's outputs to disconnect it
        node.outputs.clear()
        new_nodes.extend([reduce_max, sub_node, exp_node, reduce_sum, div_node])
        print(f"  Decomposed: {node.name} (axis={axis})")

    graph.nodes.extend(new_nodes)
    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
    # Verify I/O shapes preserved
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input  '{inp.name}': {dims}")
    for out in model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output '{out.name}': {dims}")

    onnx.save(model, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "yolo26.onnx"
    out = sys.argv[2] if len(sys.argv) > 2 else "yolo26_noflash.onnx"
    decompose_attention_softmax(inp, out)
