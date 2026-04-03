#!/usr/bin/env python3
"""
stereo_depth_test.py - 双目深度测试工具

加载标定文件，对图像对进行校正、计算视差图和深度图，
支持鼠标点击测量距离。

用法:
  python3 stereo_depth_test.py -c stereo_calib.yaml
  python3 stereo_depth_test.py -c stereo_calib.yaml --left l.png --right r.png
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


# ============================================================
#  加载标定
# ============================================================

def load_calibration_npz(path):
    """从 npz 文件加载标定(快速路径)"""
    d = np.load(path)
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


def load_calibration_yaml(path):
    """从 OpenCV FileStorage YAML 加载标定"""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开 {path}")

    def _read(name):
        node = fs.getNode(name)
        if node.empty():
            return None
        return node.mat()

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

    # 生成校正映射表
    cal['map_lx'], cal['map_ly'] = cv2.initUndistortRectifyMap(
        cal['camera_matrix_left'], cal['distortion_left'],
        cal['R1'], cal['P1'], cal['image_size'], cv2.CV_32FC1)
    cal['map_rx'], cal['map_ry'] = cv2.initUndistortRectifyMap(
        cal['camera_matrix_right'], cal['distortion_right'],
        cal['R2'], cal['P2'], cal['image_size'], cv2.CV_32FC1)
    return cal


def load_calibration(path):
    """自动检测 npz/yaml 并加载"""
    p = str(path)
    if p.endswith('.npz'):
        return load_calibration_npz(p)
    npz = p.replace('.yaml', '.npz').replace('.yml', '.npz')
    if Path(npz).exists():
        print(f"加载: {npz}")
        return load_calibration_npz(npz)
    print(f"加载: {p}")
    return load_calibration_yaml(p)


# ============================================================
#  图像读取
# ============================================================

def _read_image(path):
    """读取图像，支持 Bayer 编码 PNG"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        return img
    try:
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)  # 海康BayerRG8 = OpenCV BayerBG
    except cv2.error:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _glob_images(directory):
    """搜索目录下的 png 和 jpg"""
    files = sorted(glob.glob(str(Path(directory) / "*.png")))
    files += sorted(glob.glob(str(Path(directory) / "*.jpg")))
    return sorted(set(files))


# ============================================================
#  StereoSGBM 匹配器
# ============================================================

def create_sgbm(img_width):
    """创建 StereoSGBM 匹配器，参数自动适配"""
    # numDisparities 必须为16的倍数
    # 大基线+高焦距需要更大的视差范围 (327mm baseline, ~1930px focal)
    # 3m 距离: disp ≈ 210px, 需要 numDisp >= 224
    num_disp = max(16, (img_width // 4) & ~0xF)
    num_disp = min(num_disp, 384)

    block = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ============================================================
#  深度计算
# ============================================================

def compute_disparity(gray_l, gray_r, sgbm):
    """计算视差图 (StereoSGBM 返回值为 disp*16，需除以 16)"""
    disp16 = sgbm.compute(gray_l, gray_r)
    return disp16.astype(np.float32) / 16.0


def disparity_to_depth(disp, baseline_mm, focal_px):
    """视差转深度: depth = baseline * focal / disp (mm)"""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = baseline_mm * focal_px / disp[valid]
    return depth


def colorize_disparity(disp):
    """视差可视化: 归一化 + JET 色图"""
    valid = disp > 0
    if not np.any(valid):
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo = np.percentile(disp[valid], 1)
    hi = np.percentile(disp[valid], 99)
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def colorize_depth(depth_mm, max_mm=20000):
    """深度可视化: 近=暖色, 远=冷色"""
    valid = (depth_mm > 0) & (depth_mm < max_mm)
    if not np.any(valid):
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    norm = np.clip(depth_mm / max_mm, 0, 1)
    norm[~valid] = 0
    return cv2.applyColorMap((255 - norm * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)


# ============================================================
#  交互式深度查看器
# ============================================================

class DepthViewer:
    """鼠标点击测量距离"""

    def __init__(self, rect_left, disp, depth_mm):
        self.rect_left = rect_left.copy()
        self.disp = disp
        self.depth = depth_mm
        self.display = rect_left.copy()
        self.points = []

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        h, w = self.depth.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        d = self.disp[y, x]
        z = self.depth[y, x]

        if d <= 0 or z <= 0:
            label = f"({x},{y}) 无效视差"
        else:
            label = f"({x},{y}) d={d:.1f}px z={z:.0f}mm ({z/1000:.2f}m)"

        self.points.append((x, y, label))
        print(f"  {label}")

        # 重绘
        self.display = self.rect_left.copy()
        for px, py, lbl in self.points:
            cv2.circle(self.display, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(self.display, lbl, (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("DepthViewer", self.display)


# ============================================================
#  处理图像对
# ============================================================

def process_pair(img_l_path, img_r_path, cal, sgbm):
    """校正 → 视差 → 深度 → 交互显示，返回是否继续"""
    img_l = _read_image(img_l_path)
    img_r = _read_image(img_r_path)
    if img_l is None or img_r is None:
        print(f"无法读取: {img_l_path} / {img_r_path}")
        return True

    # 校正
    rect_l = cv2.remap(img_l, cal['map_lx'], cal['map_ly'], cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, cal['map_rx'], cal['map_ry'], cv2.INTER_LINEAR)

    # 灰度
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # 视差 & 深度
    disp = compute_disparity(gray_l, gray_r, sgbm)
    focal = cal['P1'][0, 0]
    baseline = cal['baseline']
    depth = disparity_to_depth(disp, baseline, focal)

    # 统计
    valid = depth > 0
    if np.any(valid):
        print(f"  深度: min={depth[valid].min():.0f}mm  "
              f"max={depth[valid].max():.0f}mm  "
              f"中位={np.median(depth[valid]):.0f}mm")

    # --- 可视化 ---
    def _fit(img, max_w=960):
        s = min(1.0, max_w / img.shape[1])
        return cv2.resize(img, None, fx=s, fy=s) if s < 1.0 else img

    # 校正对比 + 极线
    pair = np.hstack([rect_l, rect_r])
    for yy in range(0, pair.shape[0], 30):
        cv2.line(pair, (0, yy), (pair.shape[1], yy), (0, 255, 0), 1)

    cv2.imshow("校正对比", _fit(pair, 1920))
    cv2.imshow("视差图", _fit(colorize_disparity(disp)))
    cv2.imshow("深度图", _fit(colorize_depth(depth)))

    # 点击测距
    viewer = DepthViewer(rect_l, disp, depth)
    cv2.imshow("DepthViewer", _fit(viewer.display))
    cv2.setMouseCallback("DepthViewer", viewer.on_mouse)

    print("点击 DepthViewer 窗口测量距离，按任意键下一对，ESC退出")
    return cv2.waitKey(0) != 27


# ============================================================
#  命令行入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="双目深度测试")
    ap.add_argument('-c', '--calibration', required=True,
                    help='标定文件(.yaml/.npz)')
    ap.add_argument('--left', help='左图路径(可选)')
    ap.add_argument('--right', help='右图路径(可选)')
    ap.add_argument('-d', '--images-dir', default='calibration_images',
                    help='图像目录(默认: calibration_images)')
    args = ap.parse_args()

    # 加载标定
    cal = load_calibration(args.calibration)
    print(f"基线 = {cal['baseline']:.2f} mm")
    print(f"焦距 = {cal['P1'][0,0]:.2f} px")

    # 创建匹配器
    w = cal.get('image_size', (cal['map_lx'].shape[1],))[0]
    sgbm = create_sgbm(w)

    # 单对模式
    if args.left and args.right:
        process_pair(args.left, args.right, cal, sgbm)
        cv2.destroyAllWindows()
        return

    # 浏览模式
    left_files = _glob_images(Path(args.images_dir) / "left")
    right_files = _glob_images(Path(args.images_dir) / "right")
    n = min(len(left_files), len(right_files))
    if n == 0:
        print(f"未找到图像: {args.images_dir}/left/ 或 right/")
        return

    print(f"\n共 {n} 对图像，交互浏览...\n")
    for i in range(n):
        print(f"--- 第 {i+1}/{n} 对: {Path(left_files[i]).name} ---")
        if not process_pair(left_files[i], right_files[i], cal, sgbm):
            break

    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    main()
