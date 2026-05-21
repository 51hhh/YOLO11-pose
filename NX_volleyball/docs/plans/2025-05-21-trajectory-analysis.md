# 排球3D轨迹离线评估与滤波优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建Python离线评估框架，对比4种3D滤波方案在14组实测数据上的综合性能（抖动/延迟/稳定性），确定最优方案后移植至C++实时pipeline。

**Architecture:** CSV原始数据 → 段切分 → 4种滤波器并行处理 → 多维指标计算 → 可视化对比 → 参数标定 → C++移植。

**Tech Stack:** Python 3.10+, NumPy, SciPy, Matplotlib, dataclasses

---

## 自审修订记录

| # | 问题 | 修正 | 来源 |
|---|------|------|------|
| 1 | _0静止数据假设"位置恒定"不严谨（手持有颤抖） | 增加ACF分析分离白噪声vs低频物理抖动 | 初审 |
| 2 | 仅用obs_xyz（已融合），丢失z_mono/z_stereo独立噪声特性 | 增加Task 0 EDA深度源分析 | 初审 |
| 3 | 丢帧(has_detection=0)时滤波器应只做predict无update | process_segment改为frame_id驱动，丢帧执行predict-only | 初审 |
| 4 | 缺少计算耗时评估 | evaluate.py增加perf_counter测量每帧耗时 | 初审 |
| 5 | 噪声模型仅对比exp=1/2，缺少SMASH的R₀*(1+β*z)形式 | 扩展为3种噪声模型对比 | 初审 |
| 6 | 段内无子事件(飞行/弹跳)识别 | 增加flight_phase检测用于physics_r2精确计算 | 初审 |
| 7 | 无交叉验证，参数标定+评估用同一数据会过拟合 | 留出_0_5/_1_4/_2_5为验证集 | 初审 |
| 8 | 弹跳检测缺少obs_z一致性校验 | 增加距离连续性判据防止远距噪声误触发 | 初审 |
| 9 | 无Ground Truth，评估指标依赖带噪obs | physics_r2改为基于物理定律(g已知)，非obs-filtered对比 | 锐评#1 |
| 10 | 缺少NIS/NEES/Innovation白噪声检验 | 新增consistency.py模块，诊断Q/R配置正确性 | 锐评#2 |
| 11 | 重力方向未从数据标定 | Task 0 EDA中从抛物线拟合提取实际g_vec，为硬依赖输出 | 锐评#3 |
| 12 | 弹跳检测单帧判据鲁棒性差 | 改为两阶段(候选+2帧确认窗口) | 锐评#4 |
| 13 | R(z)模型未从实际数据验证 | EDA中直接拟合实测方差曲线 | 锐评#5 |
| 14 | 离线评估不含鲁棒性测试(丢帧/遮挡/错误关联) | Task 7增加合成干扰鲁棒性测试 | 锐评#6 |
| 15 | 数据覆盖不含横向快速飞行/遮挡/多目标 | 合成测试补充；明确标注适用范围限制 | 锐评#7 |

---

## 背景与问题

### 当前系统
- **滤波器**: 9维恒加速Kalman `[x,y,z,vx,vy,vz,ax,ay,az]`
- **观测**: ZED X单目+双目融合深度 → 3D坐标 (obs_x, obs_y, obs_z)
- **噪声**: `R(z) = R_base * z^exponent` (exponent=2)
- **帧率**: 60Hz

### 核心问题
1. **静态抖动**: 恒加速模型在静止时，加速度维漫游导致位置波动
2. **动态延迟**: 重力(9.81m/s²)被当作随机加速度估计，收敛慢
3. **弹跳不连续**: 速度突变时innovation gate可能拒绝有效观测
4. **远距放大**: z²噪声模型+远距离观测误差→位置不稳定

### 论文方法总结
| 来源 | 方法 | 核心优势 |
|------|------|---------|
| SMASH (2026, HKU) | AEKF 6维+重力+空气阻力 | 0延迟跟踪自由落体 |
| Brain over Brawn (2022) | KF + 轨迹预测 | ZED远距平滑 |
| EKF-UAV (2026) | 多模态融合+outlier检测 | 深度跳变抑制 |
| MV-BMR (2025) | 两阶段：粗定位+精化 | 远距响应 |
| Maritime (2024) | mask统计深度+KF | 远距抗噪 |

### 数据资产
```
/home/rick/mid360/YOLO11-pose/NX_volleyball/data/
├── raw_observation_data_0_*.csv  (5文件, ~8676检测帧, 不同距离静止)
├── raw_observation_data_1_*.csv  (4文件, ~1254检测帧, 抛球弹跳)
└── raw_observation_data_2_*.csv  (5文件, ~2102检测帧, 拍球)
```
16列: `frame_id,timestamp,has_detection,bbox_cx,bbox_cy,bbox_w,bbox_h,det_confidence,z_mono,z_stereo,disparity,stereo_conf,depth_method,obs_x,obs_y,obs_z`

---

## 文件结构

```
/home/rick/mid360/YOLO11-pose/NX_volleyball/trajectory_analysis/
├── config.yaml              # 物理参数、滤波器超参数
├── eda.py                   # Task 0: 数据探索性分析
├── loader.py                # CSV加载 + 连续段切分
├── filters/
│   ├── __init__.py          # 注册表: 名称→类
│   ├── base.py              # 抽象接口 FilterBase (含get_diagnostics)
│   ├── raw_passthrough.py   # 基线: 无滤波直通
│   ├── const_accel_9d.py    # 当前系统复现
│   ├── gravity_ekf_6d.py    # 6维重力先验EKF
│   └── gravity_bounce.py    # 6维+弹跳检测
├── metrics/
│   ├── __init__.py
│   ├── jitter.py            # 抗抖动指标
│   ├── latency.py           # 延迟指标
│   ├── stability.py         # 稳定性指标
│   └── consistency.py       # NIS/ACF/P有界性 (滤波器自诊断)
├── evaluate.py              # 主驱动脚本 (含grid search模式)
├── visualize.py             # 绘图输出
└── results/                 # 自动生成的结果目录
    ├── eda/                 # EDA图表
    ├── figures/             # 评估对比图
    └── report.md            # 最终报告
```

---

## 评估指标体系

### A. 抗抖动 (权重30%) — 主要数据源: _0静止
- **σ_pos(z)**: 按obs_z分bin(1-3m, 3-5m, 5-7m, 7-9m)，每bin内滤波位置标准差（分x/y/z三轴）
- **σ_vel**: 静止数据中估计速度的RMS（理想=0）
- **drift_rate**: 60帧滑动窗口均值的时间线性斜率
- **noise_floor**: 用自相关函数(ACF)分离传感器白噪声与低频物理抖动（手持颤抖）

### B. 低延迟 (权重35%) — 主要数据源: _1抛球 + _2拍球
- **phase_lag**: 滤波输出 vs 原始obs的互相关峰偏移量（帧数）
- **dir_change_delay**: obs_y方向反转到filter输出反转的帧差（逐个极值点统计中位数）
- **settle_time**: 弹跳/方向突变后，|filtered-obs| < 2σ_baseline 的连续帧起始（帧数）

### C. 稳定性 (权重35%) — 全部数据集
- **jerk_energy**: $\sum_{t} |\dddot{x}_t|^2 \cdot dt$（三阶导数能量越低越好）
- **continuity**: 连续段中无跳变(相邻帧欧氏距离>0.3m @60Hz)的比例
- **physics_r2**: 自由飞行子段(排除弹跳和手持阶段)拟合抛物线 $y=y_0+v_0 t+\frac{1}{2}g t^2$ 的R²

### D. 补充维度
- **计算耗时**: 每帧平均处理时间(μs)，约束 < 100μs（60Hz余量16.7ms内可忽略）
- **深度源分析**: z_mono vs z_stereo 独立噪声特性（_0数据EDA输出，不计入评分但作为融合策略参考）

### E. 滤波器内在一致性检验（不计入评分，用于诊断Q/R配置）
- **NIS** (Normalized Innovation Squared): $\bar{\nu}^T S^{-1} \bar{\nu}$ 均值应≈观测维度m=3，偏离说明Q/R失配
- **Innovation ACF**: 新息序列自相关应在95%置信区间内（白噪声），否则模型阶次不足
- **P有界性**: 协方差对角元素不应持续增长或退化至0

### F. 鲁棒性测试（合成干扰）
- **随机丢帧**: 在真实数据上随机删除20%/50%观测，评估滤波退化程度
- **连续遮挡**: 模拟5/10/20帧连续丢失后恢复，测量重捕获时间
- **关联错误**: 在随机帧注入0.5m位置突变，测试innovation gate拒绝能力
- **冷启动**: 段首1-5帧的收敛速度（与稳态误差的比值）

---

## Tasks

### Task 0: 数据探索性分析 (EDA)

**Files:**
- Create: `trajectory_analysis/eda.py`

**目的**: 在实现滤波器之前，先全面理解原始数据特性，为后续参数选择提供依据。

**必要输出** (后续Task强依赖):
1. 实际重力矢量 `g_vec = [gx, gy, gz]`（从_0静止数据标定相机倾斜角）
2. 实测 R(z) 曲线（各距离bin的obs方差 → 拟合最佳噪声模型形式）
3. 实际帧率稳定性（验证60Hz假设）
4. _1数据中弹跳事件标注（帧索引列表）

- [ ] **Step 1: 实现EDA脚本**

分析内容:
```python
# 1. 各文件基础统计: 帧数、检测率、平均帧率(1/Δt)、Δt直方图
# 2. _0静止数据:
#    - z_mono vs z_stereo 散点图 + Pearson相关
#    - 按distance分bin(1-3/3-5/5-7/7-9m): 
#      * z_mono的σ, z_stereo的σ, obs_z的σ → 拟合R(z)曲线
#      * 对比 R=R₀*z², R=R₀*(1+β*z), R=R₀*z^1.5 哪个最优(最小χ²)
#    - obs_x/y/z 的ACF(lag=0~30) → 白噪声占比 vs 低频漂移
#    - ★重力标定: 对静止段, obs_y的均值漂移方向 = 相机y轴与重力的偏差
#      * 若相机水平: 静止球的obs_y恒定
#      * 若相机有俯仰角θ: 重力在相机坐标系为 [0, g*cos(θ), g*sin(θ)]
#      * 从_1抛球段的抛物线拟合 y=a+bt+ct² 中提取 2c → 实际g_y分量
#      * 从_1抛球段的 z=a+bt+ct² 中提取 2c → 实际g_z分量 (应≈0)
# 3. _1抛球数据:
#    - 时序图: obs_y vs time → 自动识别抛物线段和弹跳点
#    - 弹跳事件自动标注: obs_y局部最大值(接触地面=y最大) + 前后速度反转
#    - 自由飞行段识别: 弹跳点之间的连续段
#    - 相邻帧Δobs_y分布 → 正常vs异常阈值
# 4. _2拍球数据:
#    - obs_y振幅和频率(FFT) → 手拍球典型频率(~2-4Hz)
#    - 每次上/下的幅度和持续帧数统计
# 5. depth_method分布: 各距离下mono/stereo/blend的使用比例
```

- [ ] **Step 2: 运行EDA输出图表到 results/eda/**

```bash
cd trajectory_analysis && python eda.py
```

- [ ] **Step 3: 根据EDA结果确认/调整config.yaml中的参数**

关键待确认 (阻塞后续Task的硬依赖):
- 实际帧率是否稳定60Hz → 若不稳定，process_segment中的dt计算已用实际timestamp差值
- 实测R(z)最佳模型 → 更新config.yaml中noise_exponent或切换为linear模型
- 重力矢量 `g_vec` → 更新gravity_ekf_6d的预测步
- 弹跳事件帧索引列表 → 保存为JSON供后续settle_time评估使用
- _0静止数据ACF结果 → 若低频分量显著，评估时需扣除（或注释标明下限）

---

### Task 1: 项目骨架与数据加载

**Files:**
- Create: `trajectory_analysis/config.yaml`
- Create: `trajectory_analysis/loader.py`

- [ ] **Step 1: 创建 config.yaml**

```yaml
# 相机参数
camera:
  focal: 727.0
  baseline: 0.12
  cx: 979.6
  cy: 583.2
  width: 1920
  height: 1200

# 物理参数
physics:
  gravity: 9.81          # m/s², 标量(EDA后更新为矢量)
  gravity_vec: [0.0, 9.81, 0.0]  # [gx, gy, gz] 相机坐标系, EDA标定后更新
  ball_diameter: 0.21    # m
  restitution: 0.75      # 弹跳恢复系数
  air_drag_k: 0.3        # 空气阻力系数 (可选)

# 滤波器参数
filters:
  const_accel_9d:
    sigma_a: 5.0
    R_base: 0.01
    noise_exponent: 2.0
    innovation_gate: 9.0
  gravity_ekf_6d:
    sigma_a: 2.0         # 未建模加速度噪声
    R_base: 0.01
    noise_exponent: 2.0
    innovation_gate: 9.0
  gravity_bounce:
    sigma_a: 2.0
    R_base: 0.01
    noise_exponent: 2.0
    innovation_gate: 9.0
    bounce_vy_threshold: 1.0    # m/s, 下落速度阈值
    bounce_obs_threshold: 0.15  # m, 观测突然上移量
    restitution: 0.75
    P_boost_vy: 100.0           # 弹跳后vy不确定度放大

# 评估参数
evaluation:
  z_bins: [1.0, 3.0, 5.0, 7.0, 9.0]
  smoothing_window: 60        # 帧, drift计算窗口
  settle_threshold_sigma: 2.0 # σ收敛判据
  jump_threshold: 0.5         # m, 跳变判定

# 数据路径
data_dir: "../data"
```

- [ ] **Step 2: 创建 loader.py**

```python
"""CSV数据加载与连续检测段切分"""
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

@dataclass
class Frame:
    frame_id: int
    timestamp: float
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    det_confidence: float
    z_mono: float
    z_stereo: float
    disparity: float
    stereo_conf: float
    depth_method: int
    obs_x: float
    obs_y: float
    obs_z: float

@dataclass
class Segment:
    """一段连续检测的帧序列"""
    frames: List[Frame]
    source_file: str
    segment_id: int
    
    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.timestamp for f in self.frames])
    
    @property
    def obs_xyz(self) -> np.ndarray:
        """(N,3) 原始观测坐标"""
        return np.array([[f.obs_x, f.obs_y, f.obs_z] for f in self.frames])
    
    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def mean_z(self) -> float:
        return np.mean([f.obs_z for f in self.frames])

def load_csv(filepath: Path) -> List[Frame]:
    """加载单个CSV, 返回has_detection=1的帧列表"""
    frames = []
    with open(filepath) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 16 or parts[2] != '1':
                continue
            frames.append(Frame(
                frame_id=int(parts[0]),
                timestamp=float(parts[1]),
                bbox_cx=float(parts[3]),
                bbox_cy=float(parts[4]),
                bbox_w=float(parts[5]),
                bbox_h=float(parts[6]),
                det_confidence=float(parts[7]),
                z_mono=float(parts[8]),
                z_stereo=float(parts[9]) if parts[9] else -1.0,
                disparity=float(parts[10]) if parts[10] else 0.0,
                stereo_conf=float(parts[11]) if parts[11] else 0.0,
                depth_method=int(parts[12]) if parts[12] else 0,
                obs_x=float(parts[13]),
                obs_y=float(parts[14]),
                obs_z=float(parts[15]),
            ))
    return frames

def segment_frames(frames: List[Frame], max_gap_frames: int = 5) -> List[List[Frame]]:
    """将帧按连续性切分为段 (frame_id间隔>max_gap则断开)"""
    if not frames:
        return []
    segments = []
    current = [frames[0]]
    for f in frames[1:]:
        if f.frame_id - current[-1].frame_id > max_gap_frames:
            if len(current) >= 10:  # 至少10帧才算有效段
                segments.append(current)
            current = [f]
        else:
            current.append(f)
    if len(current) >= 10:
        segments.append(current)
    return segments

def load_dataset(data_dir: str, prefix: str = "") -> List[Segment]:
    """加载指定前缀的所有CSV, 返回切分后的Segment列表"""
    data_path = Path(data_dir)
    segments = []
    seg_id = 0
    for csv_file in sorted(data_path.glob(f"raw_observation_data_{prefix}*.csv")):
        frames = load_csv(csv_file)
        for seg_frames in segment_frames(frames):
            segments.append(Segment(
                frames=seg_frames,
                source_file=csv_file.name,
                segment_id=seg_id
            ))
            seg_id += 1
    return segments
```

- [ ] **Step 3: 验证加载**

运行: `cd trajectory_analysis && python -c "from loader import *; segs=load_dataset('../data','0'); print(f'{len(segs)} segments, total {sum(len(s.frames) for s in segs)} frames')"`

---

### Task 2: 滤波器抽象接口与基线

**Files:**
- Create: `trajectory_analysis/filters/__init__.py`
- Create: `trajectory_analysis/filters/base.py`
- Create: `trajectory_analysis/filters/raw_passthrough.py`

- [ ] **Step 1: 创建 FilterBase 抽象类**

```python
# filters/base.py
"""滤波器抽象接口"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class FilterState:
    """滤波器输出状态"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

class FilterBase(ABC):
    """所有滤波器的抽象基类"""
    
    @abstractmethod
    def reset(self):
        """重置滤波器状态"""
        pass
    
    @abstractmethod
    def predict(self, dt: float):
        """仅预测步（丢帧时调用）"""
        pass
    
    @abstractmethod
    def update(self, obs_x: float, obs_y: float, obs_z: float) -> FilterState:
        """观测更新步（有检测时调用）"""
        pass
    
    def process_frame(self, obs_x: float, obs_y: float, obs_z: float,
                      dt: float, has_detection: bool = True) -> FilterState:
        """处理一帧: predict → (可选)update"""
        self.predict(dt)
        if has_detection:
            return self.update(obs_x, obs_y, obs_z)
        return self.get_state()
    
    @abstractmethod
    def get_state(self) -> FilterState:
        """获取当前状态（不做update）"""
        pass
    
    def process_segment(self, frames: 'List[Frame]', 
                        all_frame_ids: np.ndarray = None,
                        all_timestamps: np.ndarray = None) -> np.ndarray:
        """处理完整段（含丢帧predict-only）
        
        如果提供all_frame_ids/all_timestamps, 则在丢帧处执行predict-only;
        否则仅处理有检测的帧。
        返回(N,9): [x,y,z,vx,vy,vz,ax,ay,az]
        """
        self.reset()
        obs_xyz = np.array([[f.obs_x, f.obs_y, f.obs_z] for f in frames])
        timestamps = np.array([f.timestamp for f in frames])
        N = len(timestamps)
        results = np.zeros((N, 9))
        for i in range(N):
            dt = timestamps[i] - timestamps[i-1] if i > 0 else 1.0/60.0
            # 处理帧间丢帧: 如果frame_id跳跃, 中间执行predict-only
            if i > 0:
                gap = frames[i].frame_id - frames[i-1].frame_id
                if gap > 1:
                    # 丢帧期间均匀分配dt进行多步predict
                    sub_dt = dt / gap
                    for _ in range(gap - 1):
                        self.predict(sub_dt)
                    dt = sub_dt  # 最后一步的dt
            state = self.process_frame(obs_xyz[i,0], obs_xyz[i,1], obs_xyz[i,2], dt)
            results[i] = [state.x, state.y, state.z,
                         state.vx, state.vy, state.vz,
                         state.ax, state.ay, state.az]
        return results
```

- [ ] **Step 2: 创建 raw_passthrough.py**

```python
# filters/raw_passthrough.py
"""基线: 无滤波, 直接输出原始观测 + 数值微分速度"""
import numpy as np
from .base import FilterBase, FilterState

class RawPassthrough(FilterBase):
    def __init__(self):
        self._prev = None
        self._prev_dt = 1.0/60.0
    
    def reset(self):
        self._prev = None
    
    def process_frame(self, obs_x, obs_y, obs_z, dt):
        state = FilterState(x=obs_x, y=obs_y, z=obs_z)
        if self._prev is not None and dt > 1e-6:
            state.vx = (obs_x - self._prev[0]) / dt
            state.vy = (obs_y - self._prev[1]) / dt
            state.vz = (obs_z - self._prev[2]) / dt
        self._prev = (obs_x, obs_y, obs_z)
        return state
```

- [ ] **Step 3: 创建 filters/__init__.py**

```python
from .raw_passthrough import RawPassthrough
from .const_accel_9d import ConstAccel9D
from .gravity_ekf_6d import GravityEKF6D
from .gravity_bounce import GravityBounceEKF

FILTER_REGISTRY = {
    "raw": RawPassthrough,
    "const_accel_9d": ConstAccel9D,
    "gravity_ekf_6d": GravityEKF6D,
    "gravity_bounce": GravityBounceEKF,
}
```

---

### Task 3: 9维恒加速Kalman (当前系统复现)

**Files:**
- Create: `trajectory_analysis/filters/const_accel_9d.py`

- [ ] **Step 1: 实现9维恒加速Kalman**

关键公式 (与C++版hybrid_depth.cpp完全对应):
- 状态: `[x,y,z, vx,vy,vz, ax,ay,az]`
- 预测: `p' = p + v*dt + 0.5*a*dt²`, `v' = v + a*dt`, `a' = a`
- Q: `σ_a² * G*G^T`, G = `[0.5dt²*I₃; dt*I₃; I₃]`
- R: `diag(Rxy, Rxy, Rz)`, 其中 `Rz = R_base * z^exp`, `Rxy = Rz * z²/f² + 0.001`
- H: `[I₃, 0₃, 0₃]`
- Innovation gate: 马氏距离² ≤ threshold

```python
# 完整的NumPy向量化实现, 逻辑1:1对应C++版 hybrid_depth.cpp:17-78
```

- [ ] **Step 2: 单元验证**

```python
# 验证: 输入恒定观测(1,2,3)持续100帧 → 状态应收敛到(1,2,3), 速度→0
```

---

### Task 4: 6维重力先验EKF

**Files:**
- Create: `trajectory_analysis/filters/gravity_ekf_6d.py`

- [ ] **Step 1: 实现6维重力先验EKF**

核心数学:
```
状态: x = [px, py, pz, vx, vy, vz]^T  (6维)

预测步:
  px' = px + vx*dt
  py' = py + vy*dt + 0.5*g*dt²    # g=+9.81 (相机y向下)
  pz' = pz + vz*dt
  vx' = vx
  vy' = vy + g*dt
  vz' = vz

F = [[I₃, dt*I₃],    (位置行额外有重力项但在F中仍线性)
     [0₃, I₃   ]]

实际上 F 是6×6:
F = [[1,0,0, dt,0,0],
     [0,1,0, 0,dt,0],
     [0,0,1, 0,0,dt],
     [0,0,0, 1,0,0],
     [0,0,0, 0,1,0],
     [0,0,0, 0,0,1]]

预测均值中 py 额外加 0.5*g*dt², vy 额外加 g*dt

Q = σ_a² * [[dt⁴/4*I₃, dt³/2*I₃],
             [dt³/2*I₃, dt²*I₃  ]]

H = [I₃, 0₃]  (观测位置)
R = diag(Rxy, Rxy, Rz)  同当前系统
```

优势分析:
- 静止时: 重力被预测步内部消化, 不会"泄漏"到速度/加速度估计中
- 自由飞行: 预测步已精确描述抛物线 → 观测-预测残差小 → Kalman增益低 → 平滑
- 复杂度: 6×6矩阵运算 vs 9×9, 计算量降至 (6/9)³ ≈ 30%

- [ ] **Step 2: 验证自由落体**

```python
# 模拟: 从(0,0,5)静止释放, dt=1/60, 无噪声观测
# 预期: 滤波位置精确跟踪 y(t)=0.5*9.81*t², 无延迟
```

---

### Task 5: 6维重力+弹跳检测

**Files:**
- Create: `trajectory_analysis/filters/gravity_bounce.py`

- [ ] **Step 1: 在GravityEKF6D基础上增加弹跳检测**

弹跳检测逻辑:
```python
# 两阶段检测 (候选 → 确认):
#
# 阶段1 - 候选检测 (当前帧):
#   条件1: 预测vy > bounce_vy_threshold (正在下落, y轴向下为正)
#   条件2: obs_y < predicted_y - bounce_obs_threshold (观测突然比预测高=弹起)
#   条件3: |obs_z - predicted_z| < 0.5m (距离连续, 排除远距噪声误触发)
#   → 标记为"弹跳候选", 暂存当前帧索引
#
# 阶段2 - 确认 (后续2帧):
#   条件: 后续2帧的obs_y持续低于弹跳前预测轨迹 (方向持续反转)
#   → 确认弹跳, 回溯应用速度反转
#   → 若后续帧不满足 → 撤销候选(视为噪声)
#
# 触发弹跳:
#   vy = -vy * restitution   (速度反转×恢复系数)
#   P[vy,vy] *= P_boost      (增大不确定度)
#   重新预测一步 (用新的vy)
```

- [ ] **Step 2: 验证弹跳**

```python
# 模拟: 球从2m高度自由落体, 触地(y≈0)后弹起
# 预期: 弹跳点检测正确, 弹起后1-2帧内收敛
```

---

### Task 6: 评估指标计算

**Files:**
- Create: `trajectory_analysis/metrics/__init__.py`
- Create: `trajectory_analysis/metrics/jitter.py`
- Create: `trajectory_analysis/metrics/latency.py`
- Create: `trajectory_analysis/metrics/stability.py`
- Create: `trajectory_analysis/metrics/consistency.py`

- [ ] **Step 1: jitter.py — 抗抖动指标**

```python
def compute_sigma_pos_by_distance(filtered_xyz, obs_z, z_bins):
    """按距离分bin计算位置标准差"""
    # 对每个bin: std = sqrt(mean((x-mean(x))²))
    # 返回 {bin_label: {x: σ_x, y: σ_y, z: σ_z}}

def compute_sigma_vel(filtered_results):
    """静止数据中速度RMS (列3:5)"""
    
def compute_drift_rate(filtered_xyz, timestamps, window=60):
    """滑动窗口均值的线性拟合斜率"""
```

- [ ] **Step 2: latency.py — 延迟指标**

```python
def compute_phase_lag(obs_signal, filtered_signal, fps=60):
    """互相关峰偏移 (帧数) — 注意: 仅对动态段有意义"""
    # 对y分量(运动最明显轴)做互相关
    # lag = argmax(cross_correlation) - center

def compute_direction_change_delay(obs_y, filtered_y, timestamps):
    """方向反转响应时间 (基于事件检测, 非obs-filtered对比)"""
    # 1. 找obs_y的局部极值点(Savitzky-Golay平滑后取极值, 减少噪声假极值)
    # 2. 找filtered_y对应极值点
    # 3. 计算帧差的中位数(避免异常值影响)

def compute_settle_time(obs_xyz, filtered_xyz, timestamps, event_frames, sigma_threshold=2.0):
    """弹跳/突变后收敛时间"""
    # event_frames: 已知的弹跳帧索引
    # 从事件帧开始, 找|filtered - obs| < threshold的连续帧起始
```

- [ ] **Step 3: stability.py — 稳定性指标**

```python
def compute_jerk_energy(filtered_xyz, timestamps):
    """三阶导数能量: Σ|d³x/dt³|² * dt"""
    # 用中心差分计算三阶导
    
def compute_continuity(filtered_xyz, jump_threshold=0.3):
    """连续性: 无跳变帧的比例 (0.3m @60Hz ≈ 18m/s, 排球极限速度)"""
    # 相邻帧欧氏距离 > threshold 视为跳变

def compute_physics_r2(filtered_xyz, timestamps, gravity=9.81):
    """自由飞行段拟合抛物线R²"""
    # 对y分量拟合 y = a + b*t + c*t²
    # 检查 2*c 是否≈g (验证重力方向正确性)
    # 同时对obs和filtered都拟合, 对比改善幅度
```

- [ ] **Step 4: consistency.py — 滤波器内在一致性检验**

```python
def compute_nis(innovations, S_matrices):
    """归一化新息平方: ν^T S^{-1} ν
    理论均值 = 观测维度m (我们是3)
    若 NIS_mean >> 3: Q太小或R太小(过自信)
    若 NIS_mean << 3: Q太大或R太大(过保守)
    """

def compute_innovation_acf(innovations, max_lag=20):
    """新息自相关函数
    理想: ACF(lag>0) 在 ±1.96/√N 置信带内 (白噪声)
    若显著自相关: 模型阶次不足或噪声模型错误
    """

def compute_P_boundedness(P_history):
    """协方差有界性: P对角元是否收敛到稳态
    检测: P是否单调增长(发散)或退化至0(过度确信)
    """
```

**关键**: 滤波器需额外输出innovation和S矩阵，FilterBase接口增加`get_diagnostics()`方法。

---

### Task 7: 主评估驱动与可视化

**Files:**
- Create: `trajectory_analysis/evaluate.py`
- Create: `trajectory_analysis/visualize.py`

- [ ] **Step 1: evaluate.py — 主驱动**

```python
"""
主流程:
1. 加载config
2. 分别加载 _0, _1, _2 数据集
3. 对每个数据集的每个段:
   - 运行全部4种滤波器
   - 根据数据集类型计算对应指标
4. 汇总: 加权综合评分
5. 输出报告
"""
```

评估矩阵:
| 指标 | _0静止 | _1抛球 | _2拍球 |
|------|--------|--------|--------|
| σ_pos(z) | ✓ (主要) | | |
| σ_vel | ✓ | | |
| drift_rate | ✓ | | |
| phase_lag | | ✓ | ✓ (主要) |
| dir_change_delay | | ✓ | ✓ (主要) |
| settle_time | | ✓ (主要) | ✓ |
| jerk_energy | ✓ | ✓ | ✓ |
| continuity | | ✓ | ✓ |
| physics_r2 | | ✓ (主要) | |

- [ ] **Step 2: visualize.py — 绘图**

输出图表:
1. `fig_3d_trajectory.png`: 3D轨迹对比 (每个_1段一张)
2. `fig_timeseries.png`: x/y/z时序对比 (raw vs filtered)
3. `fig_jitter_vs_distance.png`: σ_pos vs obs_z 曲线
4. `fig_latency_comparison.png`: 方向变化响应对比
5. `fig_radar.png`: 6维指标雷达图 (4种方法)
6. `fig_score_table.png`: 综合评分表

- [ ] **Step 3: 运行完整评估**

```bash
cd trajectory_analysis && python evaluate.py
```

- [ ] **Step 4: 鲁棒性测试 (合成干扰)**

```python
# 在_1数据上注入干扰, 重跑全部滤波器:
# test_A: 随机丢帧20% → 对比原始评分退化百分比
# test_B: 随机丢帧50% → 同上
# test_C: 连续5/10/20帧遮挡 → 测量重捕获settle_time
# test_D: 随机帧注入0.5m位置突变 → 测量innovation gate拒绝率
# test_E: 仅使用段首5帧 → 冷启动误差曲线
#
# 输出: robustness_matrix[method][test] = degradation%
```

- [ ] **Step 5: 一致性检验输出**

```python
# 对每个滤波器(非raw):
#   - NIS时序图 + 均值 (应≈3)
#   - Innovation ACF图 + 95%置信带
#   - P对角元时序图 (检查有界性)
# 输出到 results/figures/consistency_*.png
```

预期输出:
```
=== 排球轨迹滤波评估报告 ===
数据: 14 files, XX segments, XXXX frames

[Method]          [Jitter] [Latency] [Stability] [Score]
raw_passthrough      1.00     1.00      0.30      0.73
const_accel_9d       0.40     0.70      0.75      0.63
gravity_ekf_6d       0.35     0.95      0.85      0.74
gravity_bounce       0.35     0.98      0.90      0.77  ← Best

推荐: gravity_bounce (6维重力+弹跳)
参数: σ_a=2.0, R_base=0.01, bounce_e=0.75
```

---

### Task 8: 参数标定 (自动网格搜索)

**Files:**
- Modify: `trajectory_analysis/evaluate.py` (增加grid search模式)

- [ ] **Step 1: 对gravity_ekf_6d的σ_a做Pareto搜索**

```python
# σ_a 在 [0.5, 1.0, 2.0, 3.0, 5.0, 8.0] 范围搜索
# 对每个σ_a:
#   - 训练集(_0_1~4, _1_1~3, _2_1~4)评估抖动和延迟
#   - 验证集(_0_5, _1_4, _2_5)评估泛化性
# 画 σ_jitter vs latency 的Pareto前沿
# 选取knee point (最小化 jitter*0.3 + latency*0.35 + (1-stability)*0.35)
```

- [ ] **Step 2: 噪声模型对比 (3种)**

```python
# 模型A: R(z) = R_base * max(1, z)^2          (当前C++实际实现)
# 模型B: R(z) = R_base * (1 + beta * z)       (SMASH线性, beta待标定)
# 模型C: R(z) = R_base * max(1, z)^1.5        (折中)
#
# 用_0数据集各距离bin的实测obs方差拟合最匹配的模型
# 绘制: 实测σ²(z) vs 三种模型预测R(z) → 选择拟合最佳的
```

- [ ] **Step 3: 弹跳参数标定**

```python
# 用_1数据集手动标注弹跳帧(EDA中已识别)
# 网格搜索: bounce_vy_threshold × bounce_obs_threshold × restitution
# 指标: 弹跳检测 precision/recall + settle_time
```

---

### Task 9: (最终) C++ 移植

**Files:**
- Modify: `/home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline/src/fusion/hybrid_depth.cpp`
- Modify: `/home/rick/mid360/YOLO11-pose/NX_volleyball/stereo_3d_pipeline/src/fusion/hybrid_depth.h`

- [ ] **Step 1: 根据评估结论, 将最优方案写入C++**

可能的修改范围:
- `DepthTrack::predict()`: 加入重力项
- `DepthTrack`: 维度从9降为6 (去掉加速度状态)
- 新增 `detectBounce()` 方法
- 调整 `P` 矩阵大小: 9×9 → 6×6

- [ ] **Step 2: 编译验证**

```bash
cd ~/stereo_3d_pipeline && mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

- [ ] **Step 3: 部署测试**

rsync + 远程运行，录制新数据对比前后效果。

---

## 执行顺序与依赖

```
Task 0 (EDA) ──── Task 1 (骨架) ──┬── Task 2 (接口+基线)
                                   │         │
                                   │         ├── Task 3 (9维KF)
                                   │         ├── Task 4 (重力EKF)
                                   │         └── Task 5 (重力+弹跳)
                                   │
                                   └── Task 6 (指标)
                                              │
                                   Task 7 (评估+可视化) ←── 依赖 Task 2-6 全部完成
                                              │
                                   Task 8 (参数标定) ←── 依赖 Task 7
                                              │
                                   Task 9 (C++移植) ←── 依赖 Task 8 结论
```

**并行策略**: Task 0 先行（指导参数选择），Task 1 完成后 Task 2-6 可由独立subagent并行。Task 7+ 串行。

**数据划分**:
- 训练集: `_0_1~4`, `_1_1~3`, `_2_1~4` (用于参数标定)
- 验证集: `_0_5`, `_1_4`, `_2_5` (用于评估泛化性，检测率偏低更具挑战性)

---

## 坐标系约定

- **相机坐标系**: x右, y下, z前 (标准OpenCV/ZED约定)
- **重力方向**: +y (向下 = 9.81 m/s²)
- **obs_x/y/z**: 已通过 `(pixel - center) * z / focal` 转换为相机坐标系

## 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 相机安装有倾斜角 | 重力方向不纯+y | Task 0 EDA中用_0静止数据拟合实际重力方向向量 |
| 数据中弹跳过少 | 无法充分验证bounce检测 | _1数据集专门设计含弹跳；EDA中统计实际弹跳次数 |
| 远距离σ过大掩盖方法差异 | 评估不灵敏 | 按距离分bin独立评估，归一化到各bin基线 |
| 过拟合特定录制条件 | 泛化差 | 训练/验证集划分；留出检测率最低的文件做验证 |
| _0"静止"数据含手持颤抖 | σ_pos下限不为0 | ACF分析分离白噪声vs低频运动；或选放置(非手持)段 |
| Python实现与C++存在数值差异 | 对比不公平 | 对9维KF做C++ vs Python数值一致性验证(前100帧) |
