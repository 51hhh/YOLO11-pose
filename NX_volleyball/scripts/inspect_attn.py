import onnx
m = onnx.load('yolo26.onnx')
for n in m.graph.node:
    if 'model.10' in n.name and 'attn' in n.name:
        ins = ','.join(list(n.input)[:2])
        outs = ','.join(list(n.output)[:1])
        print(f'{n.op_type:12s} {n.name}  ->  {outs}')
print("--- model.22 attn ---")
for n in m.graph.node:
    if 'model.22' in n.name and 'attn' in n.name:
        ins = ','.join(list(n.input)[:2])
        outs = ','.join(list(n.output)[:1])
        print(f'{n.op_type:12s} {n.name}  ->  {outs}')
