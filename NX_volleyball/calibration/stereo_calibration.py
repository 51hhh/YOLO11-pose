#!/usr/bin/env python3
"""
stereo_calibration.py - 双目相机标定

流程: 检测角点 → 单目标定 → 异常剔除 → 立体标定 → 立体校正 → 保存

参考: https://zhuanlan.zhihu.com/p/685376354
关键修正:
  1. findChessboardCorners 使用 FILTER_QUADS 过滤假四边形
  2. stereoCalibrate 使用 CALIB_USE_INTRINSIC_GUESS (非 flags=0)
  3. 按重投影误差自动剔除异常图像对
  4. 焦距合理性检查 (不应超过 ~4000)
  5. 图像使用无损 PNG 格式

用法:
  python3 stereo_calibration.py -s 30.0
  python3 stereo_calibration.py -s 25.0 -o my_calib.yaml
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

# ================== 默认配置 ==================
BOARD_WIDTH = 9         # 内角点列数
BOARD_HEIGHT = 6        # 内角点行数
IMAGES_DIR = "calibration_images"
OUTPUT_FILE = "stereo_calib.yaml"
# ==============================================


# ============================================================
#  工具函数
# ============================================================

def _to_gray(image_path):
    """读取图像并返回 (灰度图, BGR图)，支持 Bayer 编码"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img
    # 单通道 → Bayer 解码 (海康BayerRG8 = OpenCV BayerBG)
    try:
        bgr = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr
    except cv2.error:
        return img, None


def _glob_images(directory):
    """搜索目录下的 png 和 jpg 文件"""
    files = sorted(glob.glob(str(Path(directory) / "*.png")))
    files += sorted(glob.glob(str(Path(directory) / "*.jpg")))
    return sorted(set(files))


# ============================================================
#  角点检测
# ============================================================

# 从最严格到最宽松的检测标志组合
_CB_FLAGS_LIST = [
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK,
    0,
]

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)


def find_corners(gray, board_size):
    """检测棋盘格角点，带亚像素精化，返回角点或 None"""
    for flags in _CB_FLAGS_LIST:
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
            return corners
    return None


def make_object_points(board_size, square_size):
    """生成棋盘格 3D 模板点 (mm)"""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                            0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


# ============================================================
#  数据采集
# ============================================================

def collect_corners(images_dir, board_size):
    """扫描 left/ right/ 子目录，检测角点
    返回 (objpoints, imgpoints_l, imgpoints_r, img_size)
    """
    left_files = _glob_images(Path(images_dir) / "left")
    right_files = _glob_images(Path(images_dir) / "right")

    if not left_files or not right_files:
        raise FileNotFoundError(
            f"在 {images_dir}/left/ 或 right/ 中未找到图像(png/jpg)")

    if len(left_files) != len(right_files):
        print(f"[警告] 左右图像数量不匹配: L={len(left_files)} R={len(right_files)}")

    n = min(len(left_files), len(right_files))
    objp_unit = make_object_points(board_size, 1.0)  # 单位模板，后续再乘 square_size

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    img_size = None

    print(f"\n共 {n} 对图像，开始检测角点...\n")

    for i, (lp, rp) in enumerate(zip(left_files[:n], right_files[:n])):
        tag = f"[{i+1:3d}/{n}]"

        gl, _ = _to_gray(lp)
        gr, _ = _to_gray(rp)
        if gl is None or gr is None:
            print(f"{tag} 跳过(无法读取)")
            continue

        if img_size is None:
            img_size = (gl.shape[1], gl.shape[0])
            print(f"图像尺寸: {img_size[0]}x{img_size[1]}")

        cl = find_corners(gl, board_size)
        cr = find_corners(gr, board_size)

        if cl is None or cr is None:
            side = "左" if cl is None else "右"
            print(f"{tag} {Path(lp).stem} -- {side}图未检测到")
            continue

        objpoints.append(objp_unit)
        imgpoints_l.append(cl)
        imgpoints_r.append(cr)
        print(f"{tag} {Path(lp).stem} OK")

    print(f"\n成功检测 {len(objpoints)}/{n} 对")
    return objpoints, imgpoints_l, imgpoints_r, img_size


# ============================================================
#  单目标定
# ============================================================

def calibrate_single(objpoints, imgpoints, img_size, name="相机"):
    """单目标定，返回 (rms, 内参, 畸变, 每张图误差列表)"""
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # 逐图重投影误差
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        errors.append(err)

    print(f"\n  [{name}]")
    print(f"  RMS = {rms:.4f} px")
    print(f"  fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f}  "
          f"cx={mtx[0,2]:.1f}  cy={mtx[1,2]:.1f}")
    print(f"  畸变 = {dist.ravel()}")

    # 焦距异常检查
    if mtx[0, 0] > 4000 or mtx[1, 1] > 4000:
        print(f"  [!] 焦距异常偏大，请检查标定数据")

    return rms, mtx, dist, errors


# ============================================================
#  异常图像剔除
# ============================================================

def reject_outliers(objpoints, imgpoints_l, imgpoints_r,
                    errors_l, errors_r, sigma=2.0):
    """剔除重投影误差超过 mean + sigma*std 的图像对"""
    max_err = np.maximum(errors_l, errors_r)
    threshold = np.mean(max_err) + sigma * np.std(max_err)
    keep = [i for i, e in enumerate(max_err) if e < threshold]

    n_removed = len(max_err) - len(keep)
    if n_removed == 0:
        return objpoints, imgpoints_l, imgpoints_r, 0

    if len(keep) < 5:
        print(f"[警告] 剔除后仅剩 {len(keep)} 对，跳过剔除")
        return objpoints, imgpoints_l, imgpoints_r, 0

    print(f"\n异常剔除: 移除 {n_removed} 对 (阈值={threshold:.3f})，"
          f"保留 {len(keep)} 对")

    return ([objpoints[i] for i in keep],
            [imgpoints_l[i] for i in keep],
            [imgpoints_r[i] for i in keep],
            n_removed)


# ============================================================
#  立体标定 + 校正
# ============================================================

def calibrate_stereo(objpoints, imgpoints_l, imgpoints_r,
                     img_size, square_size):
    """完整流水线: 单目 → 剔除 → 立体标定 → 校正，返回结果字典"""
    # 将单位模板乘以实际方格尺寸(mm)
    scaled_obj = [o * square_size for o in objpoints]

    # --- 单目标定 ---
    print("\n" + "=" * 50)
    print("单目标定")
    print("=" * 50)

    _, mtx_l, dist_l, err_l = calibrate_single(
        scaled_obj, imgpoints_l, img_size, "左相机")
    _, mtx_r, dist_r, err_r = calibrate_single(
        scaled_obj, imgpoints_r, img_size, "右相机")

    # --- 异常剔除 ---
    scaled_obj, imgpoints_l, imgpoints_r, n_removed = reject_outliers(
        scaled_obj, imgpoints_l, imgpoints_r, err_l, err_r)

    if n_removed > 0:
        print("筛选后重新单目标定...")
        _, mtx_l, dist_l, _ = calibrate_single(
            scaled_obj, imgpoints_l, img_size, "左相机(筛选后)")
        _, mtx_r, dist_r, _ = calibrate_single(
            scaled_obj, imgpoints_r, img_size, "右相机(筛选后)")

    # --- 立体标定 ---
    print("\n" + "=" * 50)
    print("立体标定")
    print("=" * 50)

    # 关键: CALIB_USE_INTRINSIC_GUESS 以单目内参为起点优化
    # flags=0 会忽略单目结果从头算，导致 RMS 偏高
    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    stereo_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                       100, 1e-6)

    rms, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        scaled_obj, imgpoints_l, imgpoints_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_size,
        criteria=stereo_criteria,
        flags=stereo_flags)

    baseline = float(np.linalg.norm(T))

    print(f"\n  立体RMS = {rms:.4f} px")
    if rms > 1.0:
        print("  [!] RMS > 1.0，建议重新采集或检查棋盘格参数")
    print(f"  R =\n{R}")
    print(f"  T = {T.ravel()} mm")
    print(f"  基线 = {baseline:.2f} mm ({baseline/10:.2f} cm)")

    # --- 立体校正 ---
    # alpha=0: 裁剪无效像素; alpha=1: 保留所有像素
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0)

    print(f"  ROI左 = {roi_l}")
    print(f"  ROI右 = {roi_r}")

    # 畸变校正映射表
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    return {
        'camera_matrix_left': mtx_l,
        'distortion_left': dist_l,
        'camera_matrix_right': mtx_r,
        'distortion_right': dist_r,
        'R': R, 'T': T, 'E': E, 'F': F,
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'roi_left': roi_l, 'roi_right': roi_r,
        'image_size': img_size,
        'baseline': baseline,
        'rms_error': rms,
        'map_lx': map_lx, 'map_ly': map_ly,
        'map_rx': map_rx, 'map_ry': map_ry,
    }


# ============================================================
#  保存 / 可视化
# ============================================================

def save_calibration(results, output_path):
    """保存标定结果到 OpenCV YAML + NumPy npz"""
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    w, h = results['image_size']

    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("baseline", results['baseline'])
    fs.write("rms_error", results['rms_error'])

    fs.write("camera_matrix_left", results['camera_matrix_left'])
    fs.write("distortion_coefficients_left", results['distortion_left'])
    fs.write("rectification_left", results['R1'])
    fs.write("projection_left", results['P1'])

    fs.write("camera_matrix_right", results['camera_matrix_right'])
    fs.write("distortion_coefficients_right", results['distortion_right'])
    fs.write("rectification_right", results['R2'])
    fs.write("projection_right", results['P2'])

    fs.write("rotation", results['R'])
    fs.write("translation", results['T'])
    fs.write("essential_matrix", results['E'])
    fs.write("fundamental_matrix", results['F'])
    fs.write("disparity_to_depth_map", results['Q'])
    fs.release()

    npz_path = output_path.replace('.yaml', '.npz')
    np.savez(npz_path,
             camera_matrix_left=results['camera_matrix_left'],
             distortion_left=results['distortion_left'],
             camera_matrix_right=results['camera_matrix_right'],
             distortion_right=results['distortion_right'],
             R=results['R'], T=results['T'],
             R1=results['R1'], R2=results['R2'],
             P1=results['P1'], P2=results['P2'],
             Q=results['Q'],
             map_lx=results['map_lx'], map_ly=results['map_ly'],
             map_rx=results['map_rx'], map_ry=results['map_ry'],
             baseline=results['baseline'])

    print(f"\n已保存: {output_path}")
    print(f"已保存: {npz_path}")


def visualize_rectification(results, images_dir, n_samples=3):
    """可视化校正效果: 水平极线应对齐"""
    left_files = _glob_images(Path(images_dir) / "left")
    right_files = _glob_images(Path(images_dir) / "right")
    n = min(n_samples, len(left_files), len(right_files))
    if n == 0:
        return

    print("\n校正预览 (按任意键/ESC跳过)")

    indices = np.random.choice(
        min(len(left_files), len(right_files)), size=n, replace=False)

    for idx in indices:
        img_l = cv2.imread(left_files[idx], cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(right_files[idx], cv2.IMREAD_UNCHANGED)
        if img_l is None or img_r is None:
            continue
        # 海康 BayerRG8 sensor → OpenCV BayerBG convention
        if img_l.ndim == 2:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BayerBG2BGR)
        if img_r.ndim == 2:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BayerBG2BGR)

        rect_l = cv2.remap(img_l, results['map_lx'], results['map_ly'],
                           cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, results['map_rx'], results['map_ry'],
                           cv2.INTER_LINEAR)

        # 绘制水平线，校正后应对齐
        for y in range(0, rect_l.shape[0], 30):
            cv2.line(rect_l, (0, y), (rect_l.shape[1], y), (0, 255, 0), 1)
            cv2.line(rect_r, (0, y), (rect_r.shape[1], y), (0, 255, 0), 1)

        combined = np.hstack([rect_l, rect_r])
        scale = min(1.0, 1920 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
        cv2.imshow("校正效果(绿线应水平对齐)", combined)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


def print_depth_accuracy(results):
    """打印不同距离下的深度精度估算"""
    baseline = results['baseline']    # mm
    focal = results['P1'][0, 0]       # px

    print(f"\n基线 = {baseline:.2f} mm, 焦距 = {focal:.2f} px")
    print("深度精度估算 (假设视差精度 0.5 px):")

    for d_mm in [3000, 5000, 9000, 15000]:
        disp = baseline * focal / d_mm
        delta_z = (d_mm ** 2) / (baseline * focal) * 0.5
        print(f"  {d_mm/1000:.0f}m: 视差={disp:.2f}px, "
              f"误差=±{delta_z:.1f}mm (±{delta_z/10:.2f}cm)")


# ============================================================
#  命令行入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="双目相机标定")
    ap.add_argument('-s', '--square-size', type=float, required=True,
                    help='方格边长(mm)')
    ap.add_argument('-d', '--images-dir', default=IMAGES_DIR,
                    help='图像目录(默认: calibration_images)')
    ap.add_argument('-o', '--output', default=OUTPUT_FILE,
                    help='输出文件(默认: stereo_calib.yaml)')
    ap.add_argument('--no-vis', action='store_true',
                    help='跳过可视化')
    args = ap.parse_args()

    board_size = (BOARD_WIDTH, BOARD_HEIGHT)

    print("=" * 50)
    print("双目相机标定")
    print("=" * 50)
    print(f"棋盘格: {board_size[0]}x{board_size[1]} 内角点")
    print(f"方格:   {args.square_size} mm")
    print(f"图像:   {args.images_dir}/")
    print(f"输出:   {args.output}")
    print("=" * 50)

    # 1) 采集角点
    objpts, ipts_l, ipts_r, img_size = collect_corners(
        args.images_dir, board_size)

    if len(objpts) < 5:
        print(f"\n仅 {len(objpts)} 对有效图像，至少需要 5 对")
        return

    # 2) 标定
    results = calibrate_stereo(
        objpts, ipts_l, ipts_r, img_size, args.square_size)

    # 3) 保存
    save_calibration(results, args.output)

    # 4) 可视化
    if not args.no_vis:
        visualize_rectification(results, args.images_dir)

    # 5) 精度报告
    print_depth_accuracy(results)

    print(f"\n标定完成！将文件复制到 ROS2 工作空间:")
    print(f"  cp {args.output} ~/NX_volleyball/ros2_ws/src/"
          f"volleyball_stereo_driver/calibration/")


if __name__ == "__main__":
    main()
