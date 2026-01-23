#!/usr/bin/env python3
"""
TensorRT 引擎导出脚本
将 PyTorch 模型转换为 TensorRT 引擎
"""
import sys
sys.path.append('../train')

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='导出 TensorRT 引擎')
    parser.add_argument('--weights', type=str, required=True, help='PyTorch 模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=int, default=0, help='GPU 设备 ID')
    parser.add_argument('--fp16', action='store_true', help='使用 FP16 精度')
    parser.add_argument('--int8', action='store_true', help='使用 INT8 精度')
    parser.add_argument('--workspace', type=int, default=4, help='工作空间大小 (GB)')
    parser.add_argument('--simplify', action='store_true', default=True, help='简化 ONNX')
    parser.add_argument('--output', type=str, help='输出引擎路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查模型文件
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ 模型文件不存在: {weights_path}")
        return
    
    print(f"{'='*60}")
    print(f"导出 TensorRT 引擎")
    print(f"{'='*60}")
    print(f"模型: {weights_path}")
    print(f"输入尺寸: {args.imgsz}")
    print(f"精度: {'INT8' if args.int8 else 'FP16' if args.fp16 else 'FP32'}")
    print(f"工作空间: {args.workspace} GB")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("加载模型...")
    model = YOLO(str(weights_path))
    
    # 导出 TensorRT
    print("开始导出...")
    try:
        model.export(
            format='engine',        # TensorRT 格式
            imgsz=args.imgsz,      # 输入尺寸
            device=args.device,    # GPU 设备
            half=args.fp16,        # FP16 精度
            int8=args.int8,        # INT8 精度
            workspace=args.workspace,  # 工作空间
            simplify=args.simplify,    # 简化 ONNX
            dynamic=False,         # 静态 batch
            batch=1,               # batch size
            verbose=True,          # 详细输出
        )
        
        # 确定输出路径
        if args.output:
            engine_path = Path(args.output)
        else:
            engine_path = weights_path.with_suffix('.engine')
        
        # 检查导出结果
        if engine_path.exists():
            print(f"\n{'='*60}")
            print(f"✅ 导出成功!")
            print(f"{'='*60}")
            print(f"引擎路径: {engine_path}")
            print(f"文件大小: {engine_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # 复制到 models 目录
            models_dir = Path('../models')
            models_dir.mkdir(exist_ok=True)
            
            import shutil
            target_path = models_dir / engine_path.name
            shutil.copy(engine_path, target_path)
            print(f"已复制到: {target_path}")
        else:
            print(f"\n❌ 导出失败: 未找到输出文件")
    
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
