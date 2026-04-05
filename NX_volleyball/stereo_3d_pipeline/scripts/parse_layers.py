import json, sys

with open('/home/nvidia/NX_volleyball/model/yolo26_layer_info.json') as f:
    data = json.load(f)

layers = data if isinstance(data, list) else data.get('Layers', data.get('layers', []))
print(f'Total layers: {len(layers)}')
for i, layer in enumerate(layers):
    name = layer.get('Name', layer.get('name', ''))
    dev = layer.get('Device', layer.get('device', '?'))
    tp = layer.get('LayerType', layer.get('type', ''))
    print(f'{i:3d}  [{dev:4s}] {tp:30s} {name[:90]}')
