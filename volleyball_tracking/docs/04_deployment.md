# 🚀 Orin NX 部署指南

## 部署架构

```
全局快门相机
    ↓
图像采集 (V4L2/GStreamer)
    ↓
预处理 (CUDA)
    ↓
TensorRT 推理
    ↓
几何拟合 (CPU/CUDA)
    ↓
ByteTrack 追踪
    ↓
卡尔曼滤波
    ↓
输出结果 (可视化/ROS/网络)
```

---

## Orin NX 环境配置

### 系统要求

| 组件 | 版本 |
|------|------|
| JetPack | 5.1.2+ |
| Ubuntu | 20.04 |
| Python | 3.8+ |
| CUDA | 11.4+ |
| TensorRT | 8.5+ |
| OpenCV | 4.5+ (with CUDA) |

### 性能模式设置

```bash
# 查看当前模式
sudo nvpmodel -q

# 设置为最大性能模式 (MODE_15W)
sudo nvpmodel -m 0

# 锁定最大频率
sudo jetson_clocks

# 查看实时功耗和频率
sudo tegrastats
```

### 依赖安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装系统依赖
sudo apt install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad

# 安装 Python 依赖
cd deploy
pip3 install -r requirements_nx.txt
```

**requirements_nx.txt**:
```
numpy==1.24.3
opencv-python==4.8.1.78
scipy==1.11.3
filterpy==1.4.5
pycuda==2022.2.2
```

---

## 实时相机演示

### 相机配置

```bash
# 查看可用相机
v4l2-ctl --list-devices

# 查看相机支持的格式
v4l2-ctl -d /dev/video0 --list-formats-ext

# 设置相机参数 (全局快门)
v4l2-ctl -d /dev/video0 \
    --set-fmt-video=width=1920,height=1080,pixelformat=YUYV \
    --set-parm=30
```

### 演示程序 `demo/demo_camera.py`

```python
#!/usr/bin/env python3
"""
实时相机演示程序
"""
import sys
sys.path.append('../deploy')

import cv2
import numpy as np
import argparse
import time
from inference import VolleyballDetector
from tracker import VolleyballTracker
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='TensorRT 引擎路径')
    parser.add_argument('--camera', type=int, default=0, help='相机设备 ID')
    parser.add_argument('--width', type=int, default=1920, help='相机宽度')
    parser.add_argument('--height', type=int, default=1080, help='相机高度')
    parser.add_argument('--fps', type=int, default=60, help='相机帧率')
    parser.add_argument('--show', action='store_true', help='显示可视化')
    parser.add_argument('--save', type=str, help='保存视频路径')
    args = parser.parse_args()
    
    # 初始化检测器和追踪器
    print("初始化检测器...")
    detector = VolleyballDetector(args.engine)
    
    print("初始化追踪器...")
    tracker = VolleyballTracker(frame_rate=args.fps)
    
    # 初始化可视化器
    visualizer = Visualizer()
    
    # 打开相机
    print(f"打开相机 {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print("❌ 无法打开相机")
        return
    
    # 视频写入器
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))
    
    # 性能统计
    fps_list = []
    frame_count = 0
    
    print("开始处理...")
    print("按 'q' 退出")
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            detections = detector.detect(frame)
            
            # 追踪
            tracks = tracker.update(detections)
            
            # 可视化
            if args.show or args.save:
                vis_frame = visualizer.draw(frame, detections, tracks)
                
                # 显示 FPS
                if len(fps_list) > 0:
                    fps = np.mean(fps_list[-30:])
                    cv2.putText(
                        vis_frame, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2
                    )
                
                if args.show:
                    cv2.imshow('Volleyball Tracking', vis_frame)
                
                if writer:
                    writer.write(vis_frame)
            
            # 计算 FPS
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            fps_list.append(fps)
            
            frame_count += 1
            
            # 每 30 帧打印一次统计
            if frame_count % 30 == 0:
                avg_fps = np.mean(fps_list[-30:])
                print(f"帧 {frame_count}: FPS={avg_fps:.1f}, 检测={len(detections)}, 追踪={len(tracks)}")
            
            # 按键处理
            if args.show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    finally:
        # 清理
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # 打印最终统计
        print("\n" + "="*50)
        print("统计信息:")
        print("="*50)
        print(f"总帧数: {frame_count}")
        print(f"平均 FPS: {np.mean(fps_list):.1f}")
        print(f"中位数 FPS: {np.median(fps_list):.1f}")
        print(f"最小 FPS: {np.min(fps_list):.1f}")
        print(f"最大 FPS: {np.max(fps_list):.1f}")

if __name__ == '__main__':
    main()
```

---

## 性能优化

### CUDA 加速预处理

```python
import cv2.cuda as cuda_cv

class CUDAPreprocessor:
    """CUDA 加速的预处理器"""
    
    def __init__(self):
        self.stream = cuda_cv.Stream()
    
    def preprocess(self, image: np.ndarray):
        """CUDA 加速预处理"""
        # 上传到 GPU
        gpu_img = cuda_cv.GpuMat()
        gpu_img.upload(image, self.stream)
        
        # Resize
        gpu_resized = cuda_cv.resize(gpu_img, (640, 640), stream=self.stream)
        
        # 归一化
        gpu_float = gpu_resized.convertTo(cv2.CV_32F, 1.0/255.0, stream=self.stream)
        
        # 下载
        result = gpu_float.download(self.stream)
        self.stream.waitForCompletion()
        
        return result
```

### 零拷贝内存

```python
# 使用 pinned memory
input_buffer = cuda.pagelocked_empty((3, 640, 640), dtype=np.float32)

# 直接写入 pinned memory
np.copyto(input_buffer, preprocessed_image)
```

---

## 常见问题

### Q1: 帧率达不到 150 FPS?

**检查清单**:
```bash
# 1. 确认性能模式
sudo nvpmodel -q  # 应该是 MODE_0

# 2. 锁定频率
sudo jetson_clocks

# 3. 关闭 GUI
sudo systemctl stop gdm3

# 4. 检查 CPU/GPU 频率
sudo tegrastats
```

### Q2: 相机延迟高?

**优化**:
```python
# 使用 GStreamer 管道
pipeline = (
    f"v4l2src device=/dev/video0 ! "
    f"video/x-raw,width=1920,height=1080,framerate=60/1 ! "
    f"videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

---

## 下一步

部署完成后，可以进行实际场景测试和性能调优。
