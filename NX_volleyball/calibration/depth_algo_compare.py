#!/usr/bin/env python3
"""
depth_algo_compare.py - 多算法立体深度对比测试

在 Jetson Orin NX 上对比以下立体深度算法:
  1. VPI CUDA  — NVIDIA VPI GPU加速 SGM (Census+Hamming)
  2. VPI OFA   — NVIDIA VPI OFA硬件加速 (如可用)
  3. OpenCV SGBM — CPU半全局块匹配
  4. OpenCV CUDA SGM — GPU加速SGM (如可用)
  5. OpenCV CUDA BM  — GPU加速块匹配 (如可用)

功能:
  - 自动加载标定文件，校正图像
  - 每个算法输出: 视差图、深度图、耗时
  - 支持鼠标点击测量距离 (多算法对比视图)
  - 输出基准测试结果 (延迟/吞吐/有效像素率/一致性)
  - 将结果保存为 PNG 和 JSON

用法:
  python3 depth_algo_compare.py -c stereo_calib.yaml
  python3 depth_algo_compare.py -c stereo_calib.yaml --left l.png --right r.png
  python3 depth_algo_compare.py -c stereo_calib.yaml --benchmark --iterations 50
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# VPI is optional — may not be available on non-Jetson
try:
    import vpi
    HAS_VPI = True
except ImportError:
    HAS_VPI = False

# Check CUDA stereo availability
HAS_CUDA_STEREO = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CUDA_STEREO = True
except Exception:
    pass


# ============================================================
#  标定加载 (复用 stereo_depth_test 逻辑)
# ============================================================

def load_calibration_yaml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开 {path}")

    def _read(name):
        node = fs.getNode(name)
        return node.mat() if not node.empty() else None

    cal = {
        'camera_matrix_left': _read("camera_matrix_left"),
        'distortion_left': _read("distortion_coefficients_left"),
        'camera_matrix_right': _read("camera_matrix_right"),
        'distortion_right': _read("distortion_coefficients_right"),
        'R': _read("rotation"),
        'T': _read("translation"),
        'R1': _read("rectification_left"),
        'R2': _read("rectification_right"),
        'P1': _read("projection_left"),
        'P2': _read("projection_right"),
        'Q': _read("disparity_to_depth_map"),
    }

    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    cal['image_size'] = (w, h)

    bl_node = fs.getNode("baseline")
    cal['baseline'] = (float(bl_node.real()) if not bl_node.empty()
                       else float(np.linalg.norm(cal['T'])))
    fs.release()

    cal['map_lx'], cal['map_ly'] = cv2.initUndistortRectifyMap(
        cal['camera_matrix_left'], cal['distortion_left'],
        cal['R1'], cal['P1'], cal['image_size'], cv2.CV_32FC1)
    cal['map_rx'], cal['map_ry'] = cv2.initUndistortRectifyMap(
        cal['camera_matrix_right'], cal['distortion_right'],
        cal['R2'], cal['P2'], cal['image_size'], cv2.CV_32FC1)
    return cal


def load_calibration_npz(path):
    d = np.load(str(path))
    return {
        'camera_matrix_left': d['camera_matrix_left'],
        'distortion_left': d['distortion_left'],
        'camera_matrix_right': d['camera_matrix_right'],
        'distortion_right': d['distortion_right'],
        'R': d['R'], 'T': d['T'],
        'R1': d['R1'], 'R2': d['R2'],
        'P1': d['P1'], 'P2': d['P2'],
        'Q': d['Q'],
        'map_lx': d['map_lx'], 'map_ly': d['map_ly'],
        'map_rx': d['map_rx'], 'map_ry': d['map_ry'],
        'baseline': float(d['baseline']),
    }


def load_calibration(path):
    p = str(path)
    if p.endswith('.npz'):
        return load_calibration_npz(p)
    npz = p.replace('.yaml', '.npz').replace('.yml', '.npz')
    if Path(npz).exists():
        return load_calibration_npz(npz)
    return load_calibration_yaml(p)


# ============================================================
#  图像读取 & 校正
# ============================================================

def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        return img
    try:
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    except cv2.error:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def rectify_pair(img_l, img_r, cal):
    rect_l = cv2.remap(img_l, cal['map_lx'], cal['map_ly'], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, cal['map_rx'], cal['map_ry'], cv2.INTER_LINEAR)
    return rect_l, rect_r


# ============================================================
#  深度算法基类
# ============================================================

class StereoAlgorithm:
    """所有立体深度算法的基类"""

    def __init__(self, name, max_disp=128, win_size=5):
        self.name = name
        self.max_disp = max_disp
        self.win_size = win_size
        self.available = False

    def compute_disparity(self, gray_l, gray_r):
        """返回 float32 视差图 (像素单位)"""
        raise NotImplementedError

    def warmup(self, gray_l, gray_r, n=3):
        """预热 (GPU算法需要)"""
        for _ in range(n):
            self.compute_disparity(gray_l, gray_r)


# ============================================================
#  算法1: VPI CUDA Stereo Disparity
# ============================================================

class VPICudaStereo(StereoAlgorithm):
    def __init__(self, max_disp=128, win_size=5, target_size=None):
        super().__init__("VPI_CUDA", max_disp, win_size)
        self.target_size = target_size  # (w, h) to resize before stereo
        if HAS_VPI:
            self.available = True

    def compute_disparity(self, gray_l, gray_r):
        if self.target_size:
            gl = cv2.resize(gray_l, self.target_size)
            gr = cv2.resize(gray_r, self.target_size)
        else:
            gl, gr = gray_l, gray_r

        vpi_left = vpi.asimage(gl, format=vpi.Format.Y8_ER)
        vpi_right = vpi.asimage(gr, format=vpi.Format.Y8_ER)

        # VPI stereodisp: output is Q10.5 (divide by 32)
        output = vpi.stereodisp(
            vpi_left, vpi_right,
            backend=vpi.Backend.CUDA,
            window=self.win_size,
            maxdisp=self.max_disp,
        )

        with output.rlock_cpu() as data:
            disp_raw = np.array(data, copy=True).astype(np.float32) / 32.0

        if self.target_size:
            h, w = gray_l.shape[:2]
            scale_x = w / self.target_size[0]
            disp_raw = cv2.resize(disp_raw, (w, h)) * scale_x

        return disp_raw


# ============================================================
#  算法2: VPI OFA Stereo Disparity (硬件加速)
# ============================================================

class VPIOFAStereo(StereoAlgorithm):
    def __init__(self, max_disp=128, win_size=5, downscale=4):
        super().__init__("VPI_OFA_ds4", max_disp, win_size)
        self.downscale = downscale
        self._stream = None
        self._out = None
        if HAS_VPI:
            try:
                self._stream = vpi.Stream()
                # Test image must be large enough: width >= maxdisp * downscale
                tw = max(512, max_disp * downscale)
                th = 256
                with vpi.Backend.VIC:
                    tl = vpi.asimage(np.zeros((th, tw), dtype=np.uint16),
                                     format=vpi.Format.Y16_ER).convert(
                                         vpi.Format.Y16_ER_BL)
                    tr = vpi.asimage(np.zeros((th, tw), dtype=np.uint16),
                                     format=vpi.Format.Y16_ER).convert(
                                         vpi.Format.Y16_ER_BL)
                with self._stream, vpi.Backend.OFA:
                    vpi.stereodisp(tl, tr, maxdisp=max_disp, window=5,
                                   downscale=downscale)
                self._stream.sync()
                self.available = True
            except Exception as e:
                print(f"  VPI OFA 不可用: {e}")
                self.available = False

    def compute_disparity(self, gray_l, gray_r):
        gray16_l = gray_l.astype(np.uint16) * 256
        gray16_r = gray_r.astype(np.uint16) * 256

        with vpi.Backend.VIC:
            bl_left = vpi.asimage(gray16_l, format=vpi.Format.Y16_ER).convert(
                vpi.Format.Y16_ER_BL)
            bl_right = vpi.asimage(gray16_r, format=vpi.Format.Y16_ER).convert(
                vpi.Format.Y16_ER_BL)

        with self._stream, vpi.Backend.OFA:
            self._out = vpi.stereodisp(
                bl_left, bl_right,
                out=self._out,
                window=self.win_size,
                maxdisp=self.max_disp,
                downscale=self.downscale,
            )
        self._stream.sync()

        with self._out.rlock_cpu() as data:
            disp_raw = np.array(data, copy=True).astype(np.float32) / 32.0

        h, w = gray_l.shape[:2]
        if disp_raw.shape[0] != h or disp_raw.shape[1] != w:
            scale_x = w / disp_raw.shape[1]
            disp_raw = cv2.resize(disp_raw, (w, h)) * scale_x

        return disp_raw


# ============================================================
#  算法3: OpenCV StereoSGBM (CPU)
# ============================================================

class OpenCVSGBM(StereoAlgorithm):
    def __init__(self, max_disp=128, block_size=5):
        super().__init__("OpenCV_SGBM_CPU", max_disp, block_size)
        self.available = True
        num_disp = (max_disp + 15) & ~0xF  # round up to 16
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(self, gray_l, gray_r):
        disp16 = self.sgbm.compute(gray_l, gray_r)
        return disp16.astype(np.float32) / 16.0

    def warmup(self, gray_l, gray_r, n=1):
        self.compute_disparity(gray_l, gray_r)


# ============================================================
#  算法4: OpenCV CUDA SGM
# ============================================================

class OpenCVCudaSGM(StereoAlgorithm):
    def __init__(self, max_disp=128):
        super().__init__("OpenCV_CUDA_SGM", max_disp)
        if not HAS_CUDA_STEREO:
            return
        try:
            num_disp = (max_disp + 15) & ~0xF
            self.matcher = cv2.cuda.createStereoSGM(
                minDisparity=0,
                numDisparities=num_disp,
                P1=10,
                P2=120,
                uniquenessRatio=5,
            )
            self.available = True
        except Exception as e:
            print(f"  OpenCV CUDA SGM 不可用: {e}")

    def compute_disparity(self, gray_l, gray_r):
        gpu_l = cv2.cuda_GpuMat(gray_l)
        gpu_r = cv2.cuda_GpuMat(gray_r)
        gpu_disp = self.matcher.compute(gpu_l, gpu_r)
        disp = gpu_disp.download()
        # cuda::StereoSGM 输出与 StereoSGBM 相同, 为定点 disp*16
        return disp.astype(np.float32) / 16.0


# ============================================================
#  算法5: OpenCV CUDA BM
# ============================================================

class OpenCVCudaBM(StereoAlgorithm):
    def __init__(self, max_disp=128, block_size=19):
        super().__init__("OpenCV_CUDA_BM", max_disp, block_size)
        if not HAS_CUDA_STEREO:
            return
        try:
            num_disp = (max_disp + 15) & ~0xF
            self.matcher = cv2.cuda.createStereoBM(
                numDisparities=num_disp,
                blockSize=block_size,
            )
            self.available = True
        except Exception as e:
            print(f"  OpenCV CUDA BM 不可用: {e}")

    def compute_disparity(self, gray_l, gray_r):
        gpu_l = cv2.cuda_GpuMat(gray_l)
        gpu_r = cv2.cuda_GpuMat(gray_r)
        stream = cv2.cuda.Stream()
        gpu_disp = self.matcher.compute(gpu_l, gpu_r, stream)
        stream.waitForCompletion()
        disp = gpu_disp.download()
        return disp.astype(np.float32)


# ============================================================
#  可视化与工具函数
# ============================================================

def disparity_to_depth(disp, baseline_mm, focal_px):
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0.5  # 至少半个像素
    depth[valid] = baseline_mm * focal_px / disp[valid]
    return depth


def colorize_disparity(disp, max_disp=None):
    valid = disp > 0.5
    if not np.any(valid):
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo = 0
    hi = max_disp if max_disp else np.percentile(disp[valid], 99)
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def colorize_depth(depth_mm, max_mm=15000):
    valid = (depth_mm > 0) & (depth_mm < max_mm)
    if not np.any(valid):
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    norm = np.clip(depth_mm / max_mm, 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((255 - norm * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)


def compute_valid_ratio(disp):
    """有效视差像素比例"""
    return float(np.sum(disp > 0.5)) / max(disp.size, 1)


def compute_depth_stats(depth_mm):
    """计算深度统计"""
    valid = depth_mm > 0
    if not np.any(valid):
        return {}
    d = depth_mm[valid]
    return {
        'min_mm': float(np.min(d)),
        'max_mm': float(np.max(d)),
        'mean_mm': float(np.mean(d)),
        'median_mm': float(np.median(d)),
        'std_mm': float(np.std(d)),
    }


# ============================================================
#  基准测试
# ============================================================

def benchmark_algorithm(algo, gray_l, gray_r, iterations=30):
    """基准测试单个算法"""
    print(f"  基准测试 {algo.name}: ", end='', flush=True)

    # 预热
    algo.warmup(gray_l, gray_r, n=5)

    # 计时
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        disp = algo.compute_disparity(gray_l, gray_r)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    result = {
        'name': algo.name,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times)),
        'valid_ratio': compute_valid_ratio(disp),
        'iterations': iterations,
    }

    print(f"mean={result['mean_ms']:.2f}ms, "
          f"p99={result['p99_ms']:.2f}ms, "
          f"fps={result['fps']:.1f}, "
          f"有效={result['valid_ratio']*100:.1f}%")

    return result, disp


# ============================================================
#  交互式对比查看器
# ============================================================

class CompareViewer:
    """多算法对比 + 鼠标点击测量"""

    def __init__(self, rect_left, results, cal):
        self.rect_left = rect_left
        self.results = results  # list of (name, disp, depth)
        self.cal = cal
        self.points = []
        self.h, self.w = rect_left.shape[:2]

    def _draw(self):
        """绘制对比图"""
        # 缩放因子
        max_w = 640
        scale = min(1.0, max_w / self.w)
        sw = int(self.w * scale)
        sh = int(self.h * scale)

        # 第一行: 原图 + 各算法视差
        panels = [cv2.resize(self.rect_left, (sw, sh))]
        for name, disp, _ in self.results:
            vis = colorize_disparity(disp)
            vis = cv2.resize(vis, (sw, sh))
            cv2.putText(vis, name, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(vis)

        # 第二行: 深度图
        panels2 = [np.zeros((sh, sw, 3), dtype=np.uint8)]
        for name, _, depth in self.results:
            vis = colorize_depth(depth)
            vis = cv2.resize(vis, (sw, sh))
            cv2.putText(vis, f"{name} depth", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels2.append(vis)

        # 补齐行
        while len(panels2) < len(panels):
            panels2.append(np.zeros((sh, sw, 3), dtype=np.uint8))

        row1 = np.hstack(panels[:len(panels)])
        row2 = np.hstack(panels2[:len(panels)])
        canvas = np.vstack([row1, row2])

        # 绘制点击标记
        for px, py, labels in self.points:
            sx, sy = int(px * scale), int(py * scale)
            cv2.circle(canvas, (sx, sy), 4, (0, 0, 255), -1)
            for i, lbl in enumerate(labels):
                cv2.putText(canvas, lbl, (sx + 8, sy - 8 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0), 1)

        return canvas

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        max_w = 640
        scale = min(1.0, max_w / self.w)
        sw = int(self.w * scale)

        # 只响应第一行第一个panel (原图)
        if x >= sw or y >= int(self.h * scale):
            return

        # 逆映射到原图坐标
        ox = int(x / scale)
        oy = int(y / scale)

        if not (0 <= ox < self.w and 0 <= oy < self.h):
            return

        labels = []
        print(f"\n  像素 ({ox}, {oy}):")
        for name, disp, depth in self.results:
            d = disp[oy, ox]
            z = depth[oy, ox]
            if d > 0.5 and z > 0:
                lbl = f"{name}: d={d:.1f} z={z/1000:.2f}m"
            else:
                lbl = f"{name}: 无效"
            labels.append(lbl)
            print(f"    {lbl}")

        self.points.append((ox, oy, labels))
        canvas = self._draw()
        cv2.imshow("算法对比", canvas)

    def show(self):
        canvas = self._draw()
        cv2.imshow("算法对比", canvas)
        cv2.setMouseCallback("算法对比", self.on_mouse)


# ============================================================
#  主处理流程
# ============================================================

def process_pair(img_l_path, img_r_path, cal, algorithms, save_dir=None):
    """处理一对图像: 校正 → 多算法视差 → 深度 → 对比"""
    img_l = read_image(img_l_path)
    img_r = read_image(img_r_path)
    if img_l is None or img_r is None:
        print(f"无法读取: {img_l_path} / {img_r_path}")
        return True

    rect_l, rect_r = rectify_pair(img_l, img_r, cal)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    focal = cal['P1'][0, 0]
    baseline = cal['baseline']

    results = []
    for algo in algorithms:
        if not algo.available:
            continue
        print(f"  运行 {algo.name}...", end=' ', flush=True)
        t0 = time.perf_counter()
        disp = algo.compute_disparity(gray_l, gray_r)
        elapsed = (time.perf_counter() - t0) * 1000
        depth = disparity_to_depth(disp, baseline, focal)
        vr = compute_valid_ratio(disp)
        stats = compute_depth_stats(depth)
        results.append((algo.name, disp, depth))
        print(f"{elapsed:.1f}ms, 有效={vr*100:.1f}%, "
              f"深度中位={stats.get('median_mm', 0):.0f}mm")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = Path(img_l_path).stem
        for name, disp, depth in results:
            cv2.imwrite(str(Path(save_dir) / f"{stem}_{name}_disp.png"),
                        colorize_disparity(disp))
            cv2.imwrite(str(Path(save_dir) / f"{stem}_{name}_depth.png"),
                        colorize_depth(depth))

    # 交互式查看
    viewer = CompareViewer(rect_l, results, cal)
    viewer.show()

    print("\n点击原图(左上)测量距离，按任意键下一对，ESC退出")
    return cv2.waitKey(0) != 27


def run_benchmark(cal, algorithms, img_l_path, img_r_path,
                  iterations=30, save_path=None):
    """完整基准测试"""
    img_l = read_image(img_l_path)
    img_r = read_image(img_r_path)
    if img_l is None or img_r is None:
        print("无法读取测试图像")
        return

    rect_l, rect_r = rectify_pair(img_l, img_r, cal)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    focal = cal['P1'][0, 0]
    baseline = cal['baseline']
    img_size = f"{gray_l.shape[1]}x{gray_l.shape[0]}"

    print(f"\n{'='*60}")
    print(f"  立体深度算法基准测试")
    print(f"  图像: {img_size}, 基线: {baseline:.1f}mm, 焦距: {focal:.1f}px")
    print(f"  迭代: {iterations}")
    print(f"{'='*60}\n")

    all_results = []
    disp_maps = {}

    for algo in algorithms:
        if not algo.available:
            print(f"  跳过 {algo.name} (不可用)")
            continue
        result, disp = benchmark_algorithm(algo, gray_l, gray_r, iterations)
        result['image_size'] = img_size
        result['max_disparity'] = algo.max_disp

        # 深度统计
        depth = disparity_to_depth(disp, baseline, focal)
        result['depth_stats'] = compute_depth_stats(depth)

        all_results.append(result)
        disp_maps[algo.name] = (disp, depth)

    # 一致性分析: 与VPI CUDA对比
    ref_name = "VPI_CUDA"
    if ref_name in disp_maps:
        ref_disp = disp_maps[ref_name][0]
        ref_valid = ref_disp > 0.5
        for r in all_results:
            if r['name'] == ref_name:
                r['consistency_vs_ref'] = 1.0
                continue
            other_disp = disp_maps[r['name']][0]
            both_valid = ref_valid & (other_disp > 0.5)
            if np.any(both_valid):
                diff = np.abs(ref_disp[both_valid] - other_disp[both_valid])
                r['consistency_vs_ref'] = float(np.mean(diff < 2.0))
                r['mean_diff_px'] = float(np.mean(diff))
            else:
                r['consistency_vs_ref'] = 0.0
                r['mean_diff_px'] = float('inf')

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  基准测试结果汇总")
    print(f"{'='*60}")
    print(f"{'算法':<22} {'均值ms':>8} {'P99ms':>8} {'FPS':>8} {'有效%':>8} {'一致%':>8}")
    print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        cons = r.get('consistency_vs_ref', 0) * 100
        print(f"{r['name']:<22} {r['mean_ms']:>8.2f} {r['p99_ms']:>8.2f} "
              f"{r['fps']:>8.1f} {r['valid_ratio']*100:>8.1f} {cons:>8.1f}")

    # 保存结果
    if save_path:
        # 保存 JSON
        json_path = str(save_path).replace('.json', '') + '.json'
        with open(json_path, 'w') as f:
            json.dump({
                'benchmark': all_results,
                'calibration': {
                    'image_size': img_size,
                    'baseline_mm': baseline,
                    'focal_px': focal,
                },
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {json_path}")

        # 保存视差图对比
        panels = []
        for name, (disp, depth) in disp_maps.items():
            vis = colorize_disparity(disp)
            cv2.putText(vis, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            panels.append(vis)
        if panels:
            # 确保尺寸一致
            target_h, target_w = panels[0].shape[:2]
            panels = [cv2.resize(p, (target_w, target_h)) for p in panels]
            compare_img = np.hstack(panels) if len(panels) <= 3 else \
                np.vstack([np.hstack(panels[:3]),
                           np.hstack(panels[3:] + [np.zeros_like(panels[0])] *
                                     (3 - len(panels[3:])))])
            img_path = str(save_path).replace('.json', '') + '_compare.png'
            cv2.imwrite(img_path, compare_img)
            print(f"对比图已保存: {img_path}")

    return all_results


# ============================================================
#  不同分辨率测试
# ============================================================

def run_resolution_sweep(cal, gray_l, gray_r, iterations=20):
    """测试不同分辨率下的性能 (所有可用算法)"""
    h, w = gray_l.shape[:2]
    resolutions = [
        ("full", w, h),
        ("720p", 1280, 720),
        ("540p", 960, 540),
        ("360p", 640, 360),
        ("270p", 480, 270),
    ]

    algo_factories = [
        ("VPI_CUDA", lambda: VPICudaStereo(max_disp=128, win_size=5)),
        ("VPI_OFA_ds4", lambda: VPIOFAStereo(max_disp=128, win_size=5, downscale=4)),
        ("SGBM_CPU", lambda: OpenCVSGBM(max_disp=128, block_size=5)),
        ("CUDA_SGM", lambda: OpenCVCudaSGM(max_disp=128)),
        ("CUDA_BM", lambda: OpenCVCudaBM(max_disp=128, block_size=19)),
    ]

    print(f"\n{'='*60}")
    print(f"  分辨率性能扫描 (maxdisp=128, 迭代={iterations})")
    print(f"{'='*60}")

    for label, tw, th in resolutions:
        gl = cv2.resize(gray_l, (tw, th))
        gr = cv2.resize(gray_r, (tw, th))
        print(f"\n  --- {label} ({tw}x{th}) ---")

        for algo_name, factory in algo_factories:
            try:
                algo = factory()
                if not algo.available:
                    continue
                algo.warmup(gl, gr, n=3)
                times = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    algo.compute_disparity(gl, gr)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)
                times = np.array(times)
                print(f"    {algo_name:<14}: "
                      f"mean={np.mean(times):.2f}ms, "
                      f"p99={np.percentile(times, 99):.2f}ms, "
                      f"fps={1000/np.mean(times):.1f}")
            except Exception as e:
                print(f"    {algo_name:<14}: 失败 - {e}")


# ============================================================
#  main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="多算法立体深度对比测试")
    ap.add_argument('-c', '--calibration', required=True,
                    help='标定文件 (.yaml/.npz)')
    ap.add_argument('--left', help='左图路径')
    ap.add_argument('--right', help='右图路径')
    ap.add_argument('-d', '--images-dir', default='calibration_images',
                    help='图像目录 (默认: calibration_images)')
    ap.add_argument('--benchmark', action='store_true',
                    help='运行基准测试')
    ap.add_argument('--iterations', type=int, default=30,
                    help='基准测试迭代次数')
    ap.add_argument('--max-disp', type=int, default=256,
                    help='最大视差 (大基线相机建议256)')
    ap.add_argument('--save-dir', default='depth_results',
                    help='结果保存目录')
    ap.add_argument('--resolution-sweep', action='store_true',
                    help='分辨率性能扫描')
    ap.add_argument('--headless', action='store_true',
                    help='无头模式 (不显示窗口)')
    args = ap.parse_args()

    # 加载标定
    cal = load_calibration(args.calibration)
    print(f"基线 = {cal['baseline']:.2f} mm")
    print(f"焦距 = {cal['P1'][0,0]:.2f} px")
    print(f"VPI 可用: {HAS_VPI}")
    print(f"CUDA Stereo 可用: {HAS_CUDA_STEREO}")

    # 创建算法列表
    md = args.max_disp
    algorithms = [
        VPICudaStereo(max_disp=md, win_size=5),
        VPIOFAStereo(max_disp=md, win_size=5, downscale=4),
        OpenCVSGBM(max_disp=md, block_size=5),
        OpenCVCudaSGM(max_disp=md),
        OpenCVCudaBM(max_disp=md, block_size=19),
    ]

    available = [a for a in algorithms if a.available]
    print(f"\n可用算法: {', '.join(a.name for a in available)}")

    # 确定测试图像
    if args.left and args.right:
        left_path, right_path = args.left, args.right
    else:
        left_dir = Path(args.images_dir) / "left"
        right_dir = Path(args.images_dir) / "right"
        left_files = sorted(glob.glob(str(left_dir / "*.png")))
        right_files = sorted(glob.glob(str(right_dir / "*.png")))
        if not left_files:
            print(f"未找到图像: {left_dir}")
            return
        left_path = left_files[0]
        right_path = right_files[0]
        print(f"使用第一对图像: {Path(left_path).name}")

    # 分辨率扫描
    if args.resolution_sweep and HAS_VPI:
        img_l = read_image(left_path)
        img_r = read_image(right_path)
        rect_l, rect_r = rectify_pair(img_l, img_r, cal)
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        run_resolution_sweep(cal, gray_l, gray_r, args.iterations)

    # 基准测试
    if args.benchmark:
        save_path = str(Path(args.save_dir) / "benchmark")
        os.makedirs(args.save_dir, exist_ok=True)
        run_benchmark(cal, available, left_path, right_path,
                      args.iterations, save_path)
        return

    # 交互模式
    if args.headless:
        # 无头模式: 处理一对并保存
        img_l = read_image(left_path)
        img_r = read_image(right_path)
        rect_l, rect_r = rectify_pair(img_l, img_r, cal)
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
        focal = cal['P1'][0, 0]
        baseline = cal['baseline']

        os.makedirs(args.save_dir, exist_ok=True)
        for algo in available:
            print(f"  {algo.name}: ", end='', flush=True)
            t0 = time.perf_counter()
            disp = algo.compute_disparity(gray_l, gray_r)
            elapsed = (time.perf_counter() - t0) * 1000
            depth = disparity_to_depth(disp, baseline, focal)
            vr = compute_valid_ratio(disp)
            print(f"{elapsed:.1f}ms, 有效={vr*100:.1f}%")

            cv2.imwrite(str(Path(args.save_dir) / f"{algo.name}_disp.png"),
                        colorize_disparity(disp))
            cv2.imwrite(str(Path(args.save_dir) / f"{algo.name}_depth.png"),
                        colorize_depth(depth))
        print(f"结果已保存到 {args.save_dir}/")
        return

    # 单对交互
    if args.left and args.right:
        process_pair(args.left, args.right, cal, available, args.save_dir)
        cv2.destroyAllWindows()
        return

    # 浏览模式
    left_dir = Path(args.images_dir) / "left"
    right_dir = Path(args.images_dir) / "right"
    left_files = sorted(glob.glob(str(left_dir / "*.png")))
    right_files = sorted(glob.glob(str(right_dir / "*.png")))
    n = min(len(left_files), len(right_files))
    if n == 0:
        print(f"未找到图像: {args.images_dir}")
        return

    print(f"\n共 {n} 对图像")
    for i in range(n):
        print(f"\n--- 第 {i+1}/{n} 对: {Path(left_files[i]).name} ---")
        if not process_pair(left_files[i], right_files[i], cal, available,
                            args.save_dir):
            break

    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    main()
