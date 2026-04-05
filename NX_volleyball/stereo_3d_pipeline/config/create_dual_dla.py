import yaml
with open('pipeline_roi.yaml') as f:
    d = yaml.safe_load(f)
d['detector']['dual_dla'] = True
d['detector']['engine_path_dla1'] = '/home/nvidia/NX_volleyball/model/yolo26n_dla1_int8.engine'
with open('pipeline_roi_dual_dla.yaml', 'w') as f:
    yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
print('Created pipeline_roi_dual_dla.yaml')
