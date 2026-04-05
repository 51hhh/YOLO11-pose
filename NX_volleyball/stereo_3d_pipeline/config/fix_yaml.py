import yaml
with open('pipeline_roi.yaml') as f:
    d = yaml.safe_load(f)
d['detector']['engine_path'] = '/home/nvidia/NX_volleyball/model/yolo26n_dla0_int8.engine'
d['detector']['input_size'] = 320
with open('pipeline_roi.yaml', 'w') as f:
    yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
print('Fixed pipeline_roi.yaml')
