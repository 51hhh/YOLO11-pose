#!/usr/bin/env python3
"""
YOLOv11n 模型转换工具
将 PyTorch .pt 模型转换为 TensorRT Engine

使用方法:
    python3 convert_yolo_to_tensorrt.py --model best.pt --output yolo11n.engine
"""

import argparse
import os
import sys

def convert_pt_to_onnx(pt_model, onnx_model, imgsz=640):
    """
    步骤 1: 将 .pt 转换为 ONNX
    """
    print(f"📦 步骤 1: 转换 {pt_model} → {onnx_model}")
    
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(pt_model)
        
        # 导出为 ONNX
        model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=True,
            opset=12,
            dynamic=False  # 固定输入尺寸，TensorRT 优化更好
        )
        
        # YOLO 会自动生成 ONNX 文件
        auto_onnx = pt_model.replace('.pt', '.onnx')
        if os.path.exists(auto_onnx) and auto_onnx != onnx_model:
            os.rename(auto_onnx, onnx_model)
        
        print(f"✅ ONNX 模型已生成: {onnx_model}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX 转换失败: {e}")
        return False

def convert_onnx_to_tensorrt(onnx_model, engine_model, fp16=True):
    """
    步骤 2: 将 ONNX 转换为 TensorRT Engine
    """
    print(f"\n🚀 步骤 2: 转换 {onnx_model} → {engine_model}")
    
    # 使用 trtexec 命令行工具
    cmd = f"trtexec --onnx={onnx_model} --saveEngine={engine_model}"
    
    if fp16:
        cmd += " --fp16"  # 使用 FP16 精度加速
    
    # TensorRT 10.x 使用 --memPoolSize 替代 --workspace
    # 格式: --memPoolSize=workspace:4096 (4GB)
    cmd += " --memPoolSize=workspace:4096"
    
    # 其他有用的参数
    cmd += " --verbose"  # 详细输出
    cmd += " --dumpLayerInfo"  # 输出层信息
    cmd += " --exportLayerInfo=layer_info.json"  # 导出层信息
    
    print(f"执行命令: {cmd}")
    print("⏳ 转换中，这可能需要几分钟...")
    
    ret = os.system(cmd)
    
    if ret == 0 and os.path.exists(engine_model):
        print(f"✅ TensorRT Engine 已生成: {engine_model}")
        
        # 显示文件大小
        size_mb = os.path.getsize(engine_model) / (1024 * 1024)
        print(f"   文件大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ TensorRT 转换失败")
        print(f"   提示: 检查 TensorRT 版本和 ONNX 模型是否兼容")
        return False

def main():
    parser = argparse.ArgumentParser(description='YOLOv11n 模型转换工具')
    parser.add_argument('--model', type=str, required=True, help='输入 .pt 模型路径')
    parser.add_argument('--output', type=str, default='yolo11n.engine', help='输出 TensorRT Engine 路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸 (默认 640)')
    parser.add_argument('--fp16', action='store_true', default=True, help='使用 FP16 精度')
    parser.add_argument('--keep-onnx', action='store_true', help='保留中间 ONNX 文件')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    print("=" * 60)
    print("🏐 YOLOv11n 模型转换工具")
    print("=" * 60)
    print(f"输入模型: {args.model}")
    print(f"输出引擎: {args.output}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"FP16 精度: {args.fp16}")
    print("=" * 60)
    
    # 步骤 1: PT → ONNX
    onnx_model = args.model.replace('.pt', '.onnx')
    if not convert_pt_to_onnx(args.model, onnx_model, args.imgsz):
        sys.exit(1)
    
    # 步骤 2: ONNX → TensorRT
    if not convert_onnx_to_tensorrt(onnx_model, args.output, args.fp16):
        sys.exit(1)
    
    # 清理中间文件
    if not args.keep_onnx and os.path.exists(onnx_model):
        os.remove(onnx_model)
        print(f"🗑️  已删除中间文件: {onnx_model}")
    
    print("\n" + "=" * 60)
    print("🎉 转换完成！")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 将 {args.output} 复制到 NX:")
    print(f"   scp {args.output} nvidia@10.42.0.148:~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/model/")
    print(f"\n2. 在 NX 上编译和运行:")
    print(f"   cd ~/NX_volleyball/ros2_ws")
    print(f"   colcon build --packages-select volleyball_stereo_driver")
    print(f"   source install/setup.bash")
    print(f"   ros2 run volleyball_stereo_driver volleyball_tracker_node")
    print()

if __name__ == "__main__":
    main()
