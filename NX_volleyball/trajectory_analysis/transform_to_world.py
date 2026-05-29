"""
将CSV中的obs坐标从相机坐标系转换到世界坐标系(消除相机下倾).

原理:
  相机下倾θ度, 导致obs_y = sin(θ)*z + h*cos(θ), 随距离线性增长.
  通过绕x轴旋转-θ, 得到水平世界坐标系:
    x_world = obs_x
    y_world = obs_y * cos(θ) - obs_z * sin(θ)
    z_world = obs_y * sin(θ) + obs_z * cos(θ)

标定方法:
  用静止数据(球放地面)做 obs_y vs obs_z 线性拟合,
  斜率 = sin(θ)/cos(θ) = tan(θ) → θ = arctan(slope)
"""
import numpy as np
import glob
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def calibrate_tilt(data_dir: str) -> float:
    """从静态数据标定相机下倾角(弧度)."""
    files = sorted(glob.glob(os.path.join(data_dir, 'raw_observation_data_0_*.csv')))
    all_z, all_y = [], []
    for f in files:
        try:
            data = np.genfromtxt(f, delimiter=',', skip_header=1, filling_values=np.nan)
        except Exception:
            continue
        mask = data[:, 2] > 0  # has_detection
        if mask.sum() < 10:
            continue
        obs_y = data[mask, 14]
        obs_z = data[mask, 15]
        # 每段取中位数(抗噪声)
        all_z.append(np.median(obs_z))
        all_y.append(np.median(obs_y))

    if len(all_z) < 2:
        print("ERROR: 至少需要2段静态数据进行标定")
        sys.exit(1)

    z = np.array(all_z)
    y = np.array(all_y)
    slope, intercept = np.polyfit(z, y, 1)
    theta = np.arctan(slope)
    r2 = np.corrcoef(z, y)[0, 1] ** 2

    print(f"标定结果:")
    print(f"  线性拟合: obs_y = {slope:.5f} * obs_z + {intercept:.4f}  (R²={r2:.4f})")
    print(f"  相机下倾角: θ = {np.degrees(theta):.2f}°")
    print(f"  相机高于地面球: h = {intercept / np.cos(theta):.3f}m")
    return theta


def transform_file(filepath: str, theta: float) -> int:
    """转换单个CSV文件, 返回转换的行数."""
    try:
        with open(filepath, 'r') as f:
            header = f.readline()
            lines = f.readlines()
    except Exception as e:
        print(f"  跳过 {filepath}: {e}")
        return 0

    if not lines:
        return 0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    new_lines = [header]
    count = 0
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 16:
            new_lines.append(line)
            continue

        has_det = parts[2].strip()
        if has_det != '1':
            new_lines.append(line)
            continue

        try:
            obs_x = float(parts[13])
            obs_y = float(parts[14])
            obs_z = float(parts[15])
        except (ValueError, IndexError):
            new_lines.append(line)
            continue

        # 旋转: 绕x轴转-θ
        x_w = obs_x
        y_w = obs_y * cos_t - obs_z * sin_t
        z_w = obs_y * sin_t + obs_z * cos_t

        parts[13] = f"{x_w:.4f}"
        parts[14] = f"{y_w:.4f}"
        parts[15] = f"{z_w:.4f}"
        new_lines.append(','.join(parts) + '\n')
        count += 1

    with open(filepath, 'w') as f:
        f.writelines(new_lines)

    return count


def main():
    data_dir = DATA_DIR
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"数据目录: {os.path.abspath(data_dir)}")
    print()

    # 标定
    theta = calibrate_tilt(data_dir)
    print()

    # 转换所有CSV
    files = sorted(glob.glob(os.path.join(data_dir, 'raw_observation_data_*.csv')))
    print(f"待转换文件: {len(files)}")
    print()

    total = 0
    for f in files:
        n = transform_file(f, theta)
        if n > 0:
            print(f"  {os.path.basename(f)}: {n} 帧已转换")
            total += n

    print(f"\n完成: 共转换 {total} 帧, θ={np.degrees(theta):.2f}°")
    print(f"重力向量(世界系): [0, 9.81, 0]")

    # 验证: 重新读取静态数据检查y一致性
    print("\n=== 验证(转换后静态数据 obs_y) ===")
    files_static = sorted(glob.glob(os.path.join(data_dir, 'raw_observation_data_0_*.csv')))
    for f in files_static:
        try:
            data = np.genfromtxt(f, delimiter=',', skip_header=1, filling_values=np.nan)
        except Exception:
            continue
        mask = data[:, 2] > 0
        if mask.sum() < 10:
            continue
        obs_y = data[mask, 14]
        obs_z = data[mask, 15]
        print(f"  {os.path.basename(f)}: z={np.median(obs_z):.3f}m, y={np.median(obs_y):.4f}m (std={np.std(obs_y)*1000:.1f}mm)")


if __name__ == '__main__':
    main()
