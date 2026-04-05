#!/usr/bin/env python3
import sys, onnxruntime as ort
path = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/NX_volleyball/model/yolo26n_decomposed.onnx"
s = ort.InferenceSession(path)
for i in s.get_inputs():
    print(f"INPUT  {i.name}: {i.shape} {i.type}")
for o in s.get_outputs():
    print(f"OUTPUT {o.name}: {o.shape} {o.type}")
