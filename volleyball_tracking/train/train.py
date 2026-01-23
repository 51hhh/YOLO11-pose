#!/usr/bin/env python3
"""
排球检测模型训练脚本
基于 YOLOv11n-Pose 进行微调
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练排球检测模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--weights', type=str, help='预训练权重路径 (覆盖配置文件)')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def validate_config(cfg: dict):
    """验证配置"""
    required_keys = ['model', 'data', 'epochs', 'imgsz', 'batch']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"配置文件缺少必需字段: {key}")
    
    # 检查数据集文件
    data_path = Path(cfg['data'])
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")
    
    print("✅ 配置验证通过")

def main():
    args = parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    cfg = load_config(args.config)
    
    # 验证配置
    validate_config(cfg)
    
    # 覆盖权重路径
    if args.weights:
        cfg['model'] = args.weights
    
    # 设置随机种子
    if 'seed' in cfg:
        torch.manual_seed(cfg['seed'])
        print(f"设置随机种子: {cfg['seed']}")
    
    # 加载模型
    print(f"\n{'='*60}")
    print(f"加载模型: {cfg['model']}")
    print(f"{'='*60}")
    
    if args.resume:
        # 恢复训练
        model = YOLO(cfg.get('resume_from', 'runs/train/volleyball/weights/last.pt'))
        print("恢复训练模式")
    else:
        # 新训练
        model = YOLO(cfg['model'])
    
    # 打印模型信息
    print(f"参数量: {sum(p.numel() for p in model.model.parameters()) / 1e6:.2f}M")
    print(f"设备: {cfg.get('device', 0)}")
    
    # 开始训练
    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}\n")
    
    results = model.train(
        # 数据配置
        data=cfg['data'],
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=cfg.get('batch', 32),
        device=cfg.get('device', 0),
        workers=cfg.get('workers', 8),
        
        # 优化器
        optimizer=cfg.get('optimizer', 'AdamW'),
        lr0=cfg.get('lr0', 0.001),
        lrf=cfg.get('lrf', 0.01),
        momentum=cfg.get('momentum', 0.937),
        weight_decay=cfg.get('weight_decay', 0.0005),
        
        # 学习率调度
        cos_lr=cfg.get('cos_lr', True),
        warmup_epochs=cfg.get('warmup_epochs', 3),
        warmup_momentum=cfg.get('warmup_momentum', 0.8),
        warmup_bias_lr=cfg.get('warmup_bias_lr', 0.1),
        
        # 损失权重
        box=cfg.get('box', 7.5),
        cls=cfg.get('cls', 0.5),
        pose=cfg.get('pose', 12.0),
        kobj=cfg.get('kobj', 1.0),
        
        # 数据增强
        augment=cfg.get('augment', True),
        degrees=cfg.get('degrees', 10.0),
        translate=cfg.get('translate', 0.1),
        scale=cfg.get('scale', 0.5),
        shear=cfg.get('shear', 0.0),
        perspective=cfg.get('perspective', 0.0),
        flipud=cfg.get('flipud', 0.0),
        fliplr=cfg.get('fliplr', 0.5),
        mosaic=cfg.get('mosaic', 1.0),
        mixup=cfg.get('mixup', 0.1),
        copy_paste=cfg.get('copy_paste', 0.0),
        hsv_h=cfg.get('hsv_h', 0.015),
        hsv_s=cfg.get('hsv_s', 0.7),
        hsv_v=cfg.get('hsv_v', 0.4),
        
        # 验证和保存
        val=cfg.get('val', True),
        save=cfg.get('save', True),
        save_period=cfg.get('save_period', 10),
        plots=cfg.get('plots', True),
        
        # 其他
        patience=cfg.get('patience', 50),
        verbose=cfg.get('verbose', True),
        seed=cfg.get('seed', 42),
        deterministic=cfg.get('deterministic', True),
        
        # 项目配置
        project='runs/train',
        name=cfg.get('name', 'volleyball'),
        exist_ok=cfg.get('exist_ok', False),
        resume=args.resume,
    )
    
    # 打印训练结果
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"最佳模型: {results.save_dir / 'weights/best.pt'}")
    print(f"最终模型: {results.save_dir / 'weights/last.pt'}")
    
    # 打印关键指标
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n关键指标:")
        print(f"  Bbox mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  关键点 mAP@0.5: {metrics.get('metrics/mAP50(P)', 0):.4f}")
        print(f"  Bbox mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  关键点 mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(P)', 0):.4f}")
    
    # 复制最佳模型到 models 目录
    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)
    
    import shutil
    best_model = results.save_dir / 'weights/best.pt'
    target_model = models_dir / 'volleyball_best.pt'
    shutil.copy(best_model, target_model)
    print(f"\n✅ 最佳模型已复制到: {target_model}")

if __name__ == '__main__':
    main()
