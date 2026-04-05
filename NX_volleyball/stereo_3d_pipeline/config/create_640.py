import yaml
with open('pipeline_roi.yaml') as f:
    d = yaml.safe_load(f)
d['detector']['engine_path'] = '/home/nvidia/NX_volleyball/model/yolo26_dla_fp16.engine'
d['detector']['input_size'] = 640
with open('pipeline_roi_640.yaml', 'w') as f:
    yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
print('Created pipeline_roi_640.yaml')
