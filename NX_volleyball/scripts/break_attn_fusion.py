#!/usr/bin/env python3
"""
Aggressively break attention ForeignNode fusion in yolo26.onnx.

TRT 10.3 fuses the entire attention pattern (Split->MatMul->Softmax->MatMul->Add)
into a ForeignNode with no working implementation on Orin NX.

This script inserts Identity nodes between every pair of consecutive operations
in the attention blocks to prevent ANY fusion pattern matching.

Input:  yolo26.onnx          [1,3,640,640] -> [1,5,8400]
Output: yolo26_nofusion.onnx [1,3,640,640] -> [1,5,8400]
"""
import sys
import numpy as np
import onnx
import onnx_graphsurgeon as gs

def break_attention_fusion(input_path, output_path):
    print(f"Loading {input_path}...")
    graph = gs.import_onnx(onnx.load(input_path))

    # Collect all attention nodes by block
    attn_blocks = {}
    for n in graph.nodes:
        if "attn" in n.name and n.op not in ("Constant",):
            # Extract block prefix: e.g. /model.10/m/m.0/attn
            parts = n.name.split("/")
            # Find the attn part and get everything up to it
            for i, p in enumerate(parts):
                if p == "attn":
                    block_key = "/".join(parts[:i+1])
                    if block_key not in attn_blocks:
                        attn_blocks[block_key] = []
                    attn_blocks[block_key].append(n)
                    break

    print(f"Found {len(attn_blocks)} attention blocks:")
    for k, nodes in attn_blocks.items():
        print(f"  {k}: {len(nodes)} non-constant nodes")

    # For each attention block, insert Identity nodes after MatMul and Softmax outputs
    # to break the MHA fusion pattern
    identity_count = 0
    new_nodes = []
    
    for block_key, nodes in attn_blocks.items():
        # Target nodes that participate in the ForeignNode fusion:
        # MatMul (Q*K^T), Mul (scale), Softmax, Transpose_1 (V), MatMul_1 (attn*V)
        for n in nodes:
            if n.op in ("MatMul", "Mul", "Softmax", "Transpose"):
                # Insert Identity after this node's output
                for j, out_var in enumerate(n.outputs):
                    # Check if this output feeds into another attn node
                    consumers = [c for c in graph.nodes if out_var in c.inputs]
                    if not consumers:
                        continue
                    
                    # Create Identity barrier
                    barrier_name = f"barrier_{identity_count}"
                    barrier_out = gs.Variable(
                        f"{barrier_name}_out",
                        dtype=out_var.dtype
                    )
                    barrier = gs.Node(
                        op="Identity",
                        name=f"{block_key}/{barrier_name}",
                        inputs=[out_var],
                        outputs=[barrier_out]
                    )
                    
                    # Rewire consumers to use barrier output
                    for consumer in consumers:
                        for k_idx, inp in enumerate(consumer.inputs):
                            if inp is out_var:
                                consumer.inputs[k_idx] = barrier_out
                    
                    new_nodes.append(barrier)
                    identity_count += 1

    print(f"Inserted {identity_count} Identity barrier nodes")
    graph.nodes.extend(new_nodes)
    graph.cleanup().toposort()

    model = gs.export_onnx(graph)
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
    out = sys.argv[2] if len(sys.argv) > 2 else "yolo26_nofusion.onnx"
    break_attention_fusion(inp, out)
