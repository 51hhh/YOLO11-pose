#!/usr/bin/env python3
"""
stereo_calibration.py - 海康双目相机标定脚本

使用方法:
1. 先运行 capture_stereo_images 采集棋盘格图像
2. 测量实际棋盘格尺寸
3. 运行本脚本: python3 stereo_calibration.py --square-size 25.0 --pattern-width 9 --pattern-height 6

参数说明:
  --square-size: 棋盘格单个方格的边长 (毫米)
  --pattern-width: 棋盘格内部角点数 (宽度方向，列数-1)
  --pattern-height: 棋盘格内部角点数 (高度方向，行数-1)
  --images-dir: 图像目录 (默认: calibration_images)
  --output: 输出标定文件 (默认: stereo_calib.yaml)
"""

import numpy as np
import cv2
import glob
import os
import yaml
import argparse
from pathlib import Path


class StereoCalibrator:
    def __init__(self, square_size, pattern_size, images_dir="calibration_images"):
        """
        初始化标定器
        
        Args:
            square_size: 棋盘格方格边长 (毫米)
            pattern_size: (宽, 高) 内部角点数
            images_dir: 图像目录路径
        """
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.images_dir = Path(images_dir)
        
        # 准备3D点 (世界坐标系)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 转换为实际尺寸 (毫米)
        
        # 存储所有图像的点
        self.objpoints = []  # 3D点
        self.imgpoints_left = []  # 左相机2D点
        self.imgpoints_right = []  # 右相机2D点
        
        self.img_size = None
        
    def find_chessboard_corners(self, image_path, show=False):
        """
        在图像中查找棋盘格角点
        
        Returns:
            corners: 角点坐标，如果未找到返回None
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️  无法读取图像: {image_path}")
            return None
        
        # 🔍 调试：显示图像信息
        if show and len(self.objpoints) == 0:  # 只在第一张图显示调试信息
            print(f"\n   [图像信息] shape: {img.shape}, dtype: {img.dtype}")
        
        # 处理不同格式
        if img.ndim == 2:
            # 单通道图像，可能是灰度或Bayer
            # 尝试Bayer解码
            try:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            except:
                # 如果Bayer解码失败，直接当灰度图
                gray = img
        elif img.ndim == 3:
            # 彩色图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print(f"❌ 不支持的图像格式")
            return None
            
        # 更新图像尺寸
        if self.img_size is None:
            self.img_size = gray.shape[::-1]
            print(f"   [图像尺寸] {self.img_size[0]}x{self.img_size[1]}")
        
        # 查找棋盘格角点（尝试多种flags组合）
        flags_list = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            0  # 默认
        ]
        
        ret = False
        corners = None
        for flags in flags_list:
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
            if ret:
                break
        
        if ret:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            if show:
                # 绘制角点
                if img.ndim == 2:
                    img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    img_draw = img_bgr if img.ndim == 2 else img.copy()
                cv2.drawChessboardCorners(img_draw, self.pattern_size, corners, ret)
                cv2.imshow('Chessboard Detection', img_draw)
                cv2.waitKey(500)
        
        return corners if ret else None
    
    def collect_calibration_data(self, visualize=True, debug_first=False):
        """
        从左右相机图像中收集标定数据
        
        Args:
            visualize: 是否显示检测过程
            debug_first: 是否在第一张图上暂停调试
        """
        left_images = sorted(glob.glob(str(self.images_dir / "left" / "*.png")))
        right_images = sorted(glob.glob(str(self.images_dir / "right" / "*.png")))
        
        if len(left_images) == 0 or len(right_images) == 0:
            raise ValueError(f"❌ 未找到图像！请检查目录: {self.images_dir}")
        
        if len(left_images) != len(right_images):
            print(f"⚠️  警告: 左右图像数量不匹配 (L={len(left_images)}, R={len(right_images)})")
        
        num_pairs = min(len(left_images), len(right_images))
        print(f"\n📸 找到 {num_pairs} 对图像")
        print("🔍 开始检测棋盘格角点...\n")
        
        success_count = 0
        
        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            print(f"处理 [{i+1}/{num_pairs}]: {Path(left_path).name}", end=" ")
            
            # 🔍 第一张图调试模式
            if debug_first and i == 0:
                print("\n\n⚠️  调试模式：显示第一张图像")
                print("   请检查棋盘格是否清晰可见")
                print("   按任意键继续，ESC退出\n")
                
                img = cv2.imread(str(left_path))
                if img is not None:
                    # 显示原图
                    cv2.imshow('Left Image - Original', img)
                    
                    # 显示灰度图
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('Left Image - Grayscale', gray)
                    
                    # 尝试检测并显示
                    ret, corners = cv2.findChessboardCorners(
                        gray, self.pattern_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                    
                    img_draw = img.copy()
                    if ret:
                        cv2.drawChessboardCorners(img_draw, self.pattern_size, corners, ret)
                        cv2.putText(img_draw, f"Found {self.pattern_size[0]}x{self.pattern_size[1]} corners!", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(img_draw, f"NOT FOUND! Try different pattern size?", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(img_draw, f"Looking for: {self.pattern_size[0]}x{self.pattern_size[1]} inner corners", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.imshow('Left Image - Detection Result', img_draw)
                    
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    if key == 27:  # ESC
                        print("\n用户中止")
                        return 0
            
            # 检测左图角点
            corners_left = self.find_chessboard_corners(left_path, show=visualize)
            if corners_left is None:
                print("❌ 左图未检测到")
                continue
            
            # 检测右图角点
            corners_right = self.find_chessboard_corners(right_path, show=visualize)
            if corners_right is None:
                print("❌ 右图未检测到")
                continue
            
            # 两个图像都成功检测到角点
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            success_count += 1
            
            print(f"✅ 成功")
        
        if visualize:
            cv2.destroyAllWindows()
        
        print(f"\n✅ 成功检测 {success_count}/{num_pairs} 对图像")
        
        if success_count < 10:
            print("⚠️  警告: 成功图像对少于10对，标定精度可能不足")
            print("   建议: 重新采集更多图像或调整棋盘格放置")
        
        return success_count
    
    def calibrate_single_camera(self, imgpoints, camera_name="相机"):
        """
        单目标定
        """
        print(f"\n🔧 标定{camera_name}...")
        
        # 使用标准5参数模型（k1,k2,p1,p2,k3），避免过拟合
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, imgpoints, self.img_size, None, None
        )
        
        # 计算重投影误差
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(self.objpoints)
        
        print(f"   重投影误差: {mean_error:.4f} 像素")
        print(f"   内参矩阵:\n{mtx}")
        print(f"   畸变系数: {dist.ravel()}")
        
        return mtx, dist, rvecs, tvecs
    
    def calibrate_stereo(self):
        """
        双目标定
        """
        print("\n" + "=" * 60)
        print("🎯 开始双目立体标定")
        print("=" * 60)
        
        # 先进行单目标定
        mtx_left, dist_left, _, _ = self.calibrate_single_camera(
            self.imgpoints_left, "左相机"
        )
        mtx_right, dist_right, _, _ = self.calibrate_single_camera(
            self.imgpoints_right, "右相机"
        )
        
        # 双目标定
        print("\n🔧 双目立体标定...")
        
        # 使用单目标定结果作为初始值，但允许stereoCalibrate重新优化
        # 不使用CALIB_FIX_INTRINSIC，让双目标定重新优化所有参数
        flags = 0  # 默认flags，允许优化所有参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
        
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = \
            cv2.stereoCalibrate(
                self.objpoints,
                self.imgpoints_left,
                self.imgpoints_right,
                mtx_left, dist_left,
                mtx_right, dist_right,
                self.img_size,
                criteria=criteria,
                flags=flags
            )
        
        print(f"   立体标定RMS误差: {ret:.4f}")
        print(f"   旋转矩阵 R:\n{R}")
        print(f"   平移向量 T:\n{T.ravel()} (毫米)")
        
        # 计算基线长度
        baseline = np.linalg.norm(T)
        print(f"   基线长度: {baseline:.2f} 毫米 = {baseline/10:.2f} 厘米")
        
        # 立体校正
        print("\n🔧 立体校正...")
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            self.img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0  # 0=裁剪所有无效像素, 1=保留所有像素
        )
        
        print(f"   左相机ROI: {roi_left}")
        print(f"   右相机ROI: {roi_right}")
        
        # 计算去畸变映射
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, self.img_size, cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, self.img_size, cv2.CV_32FC1
        )
        
        results = {
            'camera_matrix_left': mtx_left,
            'distortion_left': dist_left,
            'camera_matrix_right': mtx_right,
            'distortion_right': dist_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'roi_left': roi_left,
            'roi_right': roi_right,
            'image_size': self.img_size,
            'baseline': baseline,
            'rms_error': ret,
            'map_left_x': map_left_x,
            'map_left_y': map_left_y,
            'map_right_x': map_right_x,
            'map_right_y': map_right_y
        }
        
        return results
    
    def save_calibration(self, results, output_path="stereo_calib.yaml"):
        """
        保存标定结果到YAML文件 (OpenCV格式)
        """
        print(f"\n💾 保存标定结果到: {output_path}")
        
        # 转换为OpenCV FileStorage格式
        fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
        
        fs.write("image_width", int(results['image_size'][0]))
        fs.write("image_height", int(results['image_size'][1]))
        fs.write("baseline", float(results['baseline']))
        fs.write("rms_error", float(results['rms_error']))
        
        # 左相机参数
        fs.write("camera_matrix_left", results['camera_matrix_left'])
        fs.write("distortion_coefficients_left", results['distortion_left'])
        fs.write("rectification_left", results['R1'])
        fs.write("projection_left", results['P1'])
        
        # 右相机参数
        fs.write("camera_matrix_right", results['camera_matrix_right'])
        fs.write("distortion_coefficients_right", results['distortion_right'])
        fs.write("rectification_right", results['R2'])
        fs.write("projection_right", results['P2'])
        
        # 立体参数
        fs.write("rotation", results['R'])
        fs.write("translation", results['T'])
        fs.write("essential_matrix", results['E'])
        fs.write("fundamental_matrix", results['F'])
        fs.write("disparity_to_depth_map", results['Q'])
        
        fs.release()
        
        # 同时保存NumPy格式（用于Python快速加载）
        np_output = output_path.replace('.yaml', '.npz')
        np.savez(
            np_output,
            camera_matrix_left=results['camera_matrix_left'],
            distortion_left=results['distortion_left'],
            camera_matrix_right=results['camera_matrix_right'],
            distortion_right=results['distortion_right'],
            R=results['R'],
            T=results['T'],
            R1=results['R1'],
            R2=results['R2'],
            P1=results['P1'],
            P2=results['P2'],
            Q=results['Q'],
            map_left_x=results['map_left_x'],
            map_left_y=results['map_left_y'],
            map_right_x=results['map_right_x'],
            map_right_y=results['map_right_y'],
            baseline=results['baseline']
        )
        
        print(f"   YAML格式: {output_path}")
        print(f"   NumPy格式: {np_output}")
        print("\n✅ 标定完成！")
    
    def visualize_rectification(self, results, num_samples=3):
        """
        可视化校正效果
        """
        print("\n👁️  可视化立体校正效果...")
        print("   (按任意键查看下一对图像，ESC退出)")
        
        left_images = sorted(glob.glob(str(self.images_dir / "left" / "*.png")))
        right_images = sorted(glob.glob(str(self.images_dir / "right" / "*.png")))
        
        num_samples = min(num_samples, len(left_images))
        
        for i in range(num_samples):
            # 随机选择一对图像
            idx = np.random.randint(0, len(left_images))
            
            img_left = cv2.imread(left_images[idx])
            img_right = cv2.imread(right_images[idx])
            
            # 应用去畸变和校正
            rectified_left = cv2.remap(
                img_left, 
                results['map_left_x'], 
                results['map_left_y'], 
                cv2.INTER_LINEAR
            )
            rectified_right = cv2.remap(
                img_right,
                results['map_right_x'],
                results['map_right_y'],
                cv2.INTER_LINEAR
            )
            
            # 绘制极线（横向线条，校正后应该对齐）
            for y in range(0, rectified_left.shape[0], 30):
                cv2.line(rectified_left, (0, y), (rectified_left.shape[1], y), (0, 255, 0), 1)
                cv2.line(rectified_right, (0, y), (rectified_right.shape[1], y), (0, 255, 0), 1)
            
            # 拼接显示
            combined = np.hstack([rectified_left, rectified_right])
            
            # 缩放以适应屏幕
            scale = min(1.0, 1920 / combined.shape[1])
            if scale < 1.0:
                combined = cv2.resize(combined, None, fx=scale, fy=scale)
            
            cv2.imshow('立体校正效果 (左 | 右) - 绿线应水平对齐', combined)
            
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="海康双目相机标定工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  1. 标准棋盘格 (9x6, 25mm):
     python3 stereo_calibration.py --square-size 25.0 --pattern-width 9 --pattern-height 6
  
  2. 自定义棋盘格:
     python3 stereo_calibration.py --square-size 30.0 --pattern-width 11 --pattern-height 8
  
  3. 指定输出文件:
     python3 stereo_calibration.py --square-size 25.0 -w 9 -h 6 --output my_calib.yaml

注意:
  • pattern_width/height 是内部角点数（列数-1, 行数-1）
  • square_size 必须准确测量，单位为毫米
  • 建议采集 15-20 对图像以获得良好精度
        """
    )
    
    parser.add_argument(
        '--square-size', '-s',
        type=float,
        required=True,
        help='棋盘格方格边长 (毫米)'
    )
    parser.add_argument(
        '--pattern-width', '-w',
        type=int,
        required=True,
        help='棋盘格内部角点数 (宽度，列数-1)'
    )
    parser.add_argument(
        '--pattern-height',
        type=int,
        required=True,
        help='棋盘格内部角点数 (高度，行数-1)'
    )
    parser.add_argument(
        '--images-dir', '-d',
        type=str,
        default='calibration_images',
        help='图像目录 (默认: calibration_images)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='stereo_calib.yaml',
        help='输出标定文件路径 (默认: stereo_calib.yaml)'
    )
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='不显示可视化窗口'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式：显示第一张图像并暂停'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🎯 海康双目相机标定工具")
    print("=" * 60)
    print(f"棋盘格参数:")
    print(f"  方格尺寸: {args.square_size} mm")
    print(f"  角点数: {args.pattern_width} x {args.pattern_height}")
    print(f"图像目录: {args.images_dir}")
    print(f"输出文件: {args.output}")
    if args.debug:
        print(f"调试模式: 已启用")
    print("=" * 60)
    
    # 创建标定器
    calibrator = StereoCalibrator(
        square_size=args.square_size,
        pattern_size=(args.pattern_width, args.pattern_height),
        images_dir=args.images_dir
    )
    
    # 收集标定数据
    num_success = calibrator.collect_calibration_data(
        visualize=not args.no_visualization,
        debug_first=args.debug
    )
    
    if num_success < 5:
        print("\n❌ 错误: 成功检测的图像对少于5对，无法进行标定")
        print("   请检查:")
        print("   1. 棋盘格参数是否正确")
        print("   2. 图像质量是否足够好（清晰、无模糊）")
        print("   3. 棋盘格是否完整出现在图像中")
        return
    
    # 执行标定
    results = calibrator.calibrate_stereo()
    
    # 保存结果
    calibrator.save_calibration(results, args.output)
    
    # 可视化校正效果
    if not args.no_visualization:
        calibrator.visualize_rectification(results)
    
    # 打印深度精度估算
    print("\n" + "=" * 60)
    print("📏 深度测量精度估算 (基于标定结果)")
    print("=" * 60)
    baseline = results['baseline']  # 毫米
    focal_length = results['P1'][0, 0]  # 像素
    
    print(f"基线长度: {baseline:.2f} mm = {baseline/10:.2f} cm")
    print(f"焦距: {focal_length:.2f} 像素")
    print("\n深度精度 (假设视差精度为 0.5 像素):")
    
    for distance in [3000, 5000, 9000, 15000]:  # 毫米
        # ΔZ = (Z²/bf) * Δd
        disparity = baseline * focal_length / distance
        depth_error = (distance ** 2) / (baseline * focal_length) * 0.5
        
        print(f"  距离 {distance/1000:.1f}m: 视差={disparity:.2f}px, "
              f"深度误差=±{depth_error:.1f}mm (±{depth_error/10:.2f}cm)")
    
    print("\n✅ 标定流程全部完成！")
    print(f"📁 标定文件已保存: {args.output}")
    print("\n下一步: 将标定文件复制到 ROS2 工作空间:")
    print(f"  cp {args.output} ~/NX_volleyball/ros2_ws/src/volleyball_stereo_driver/calibration/")


if __name__ == "__main__":
    main()
