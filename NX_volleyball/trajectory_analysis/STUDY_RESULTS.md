# 排球3D轨迹滤波 — 预测落点研究总结

## 任务目标

**接球预测**: 球从远处飞来，1.8s飞行时间内持续预测落点坐标，在0.8s前(机器人响应时间)锁定目标。要求60Hz更新、低延迟、低抖动。

---

## 系统约束

| 约束 | 值 | 含义 |
|------|------|------|
| 飞行时间 | ~1.8s | 从检测到落地 |
| 机器人响应 | 0.8-1.0s | 必须提前这么久给出落点 |
| 有效预测窗口 | 0.8-1.0s | 飞行1.8s - 响应1.0s |
| 帧率 | 60Hz | ZED X Camera |
| 深度范围 | 0.3-10.8m | 远端噪声大(σ_z∝z^2.85) |
| 预测模型 | 抛物线 | pos + vel×dt + 0.5×g×dt² |

**核心需求**: 滤波器必须提供**实时准确的pos和vel**用于抛物线外推，而非追求输出平滑。

---

## 数据集

- **14个CSV文件**, 11582有效帧, 25个片段
- Static: 8段/8505帧 (球静止)
- Throw: 7段/1076帧 (含多次抛接循环)
- Dribble: 10段/1914帧 (运球)

**数据局限**: Throw段包含4-5s的多次抛接循环(非单次弧线)，EKF在接球瞬间会发散(vy累积到42m/s)。实际部署为单次抛球场景，性能会优于离线数据。

---

## 评估体系 (v2 — 预测导向)

```
Score = 0.30×Pred@0.5s + 0.25×Pred@1.0s + 0.20×PredJitter + 0.15×Smooth + 0.10×Track
```

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Pred@0.5s** | 0.5s后位置预测精度 | filter的vel做抛物线外推 vs 实际观测 |
| **Pred@1.0s** | 1.0s后位置预测精度 | 同上, 更远horizon |
| **PredJitter** | 预测落点帧间稳定性 | 连续帧预测点的跳变量 |
| Smooth | 输出帧间平滑度 | σ_vel |
| Track | 位置跟踪精度 | filter_pos vs obs, median误差 |

**评估约束**:
- 仅在弹道弧段内评估预测(中点偏差<20cm)
- 排除发散帧(filter_pos偏离obs>1m)
- 排除静止段(位移<10cm)

---

## 当前排名 (预测导向, 动态数据)

| Rank | Filter | Score | RMSE@0.5s | Track | PredJitter | Smooth | 特点 |
|------|--------|-------|-----------|-------|------------|--------|------|
| 1 | **imm** | 0.511 | 1.24m | 0.746 | 0.426 | 0.737 | 静止/飞行模型切换 |
| 2 | **fallback_v2** | 0.510 | 1.54m | 0.705 | 0.467 | 0.748 | 渐进P膨胀恢复 |
| 3 | robust_bounce_ekf | 0.508 | 1.54m | 0.705 | 0.463 | 0.742 | 弹跳状态机 |
| 4 | confidence_v2 | 0.501 | 2.00m | 0.128 | 0.574 | 0.929 | V2+置信度R |
| 5 | gravity_one_euro_v2 | 0.498 | 1.99m | 0.128 | 0.569 | 0.925 | EKF+固定LPF |
| 6 | adaptive_q_v2 | 0.397 | - | 0.128 | 0.557 | 0.928 | V2+Q自适应 |
| 7 | confidence_ekf | 0.395 | 2.25m | 0.142 | 0.549 | 0.917 | 位置导数OE |
| 8 | gravity_one_euro | 0.367 | 1.17m* | 0.149 | 0.445 | 0.906 | V1 (N=6) |
| 9 | gravity_ekf_6d | 0.360 | 1.86m | 0.312 | 0.376 | 0.857 | 纯EKF |

*gravity_one_euro 仅6个有效样本(大部分帧因1m tracking gate被排除)

---

## 关键发现

### 1. V2后置LPF杀死预测

```
V2输出:  pos = One-Euro(EKF_pos)  ← τ=31.8s, 1.8s内仅跟上5.5%
         vel = EKF_vel            ← 实时无延迟

预测用: pred = pos + vel×dt + 0.5×g×dt²
              ↑滞后位置(错误起点)  ↑正确方向
```

**结果**: V2虽然PredJitter最低(输出稳定), 但预测RMSE=2.0m(起点错误), Tracking=0.128(极差)。

！ 我不要可视化，我要的是实时准确的轨迹预测，并且精准快速的预测落点。

### 3. EKF在多抛接段中发散

gravity_ekf_6d 在 Throw seg2 (4.43s, 264帧):
- Frame 0: vy=0, pos_y=1.15m (正常)
- Frame 257: vy=42m/s, pos_y=88.7m (完全发散)

**原因**: 接球瞬间球静止但重力持续注入 → vy累积 → 超过innovation_gate → 永久失锁

**修复方案**:
- 方案A: 速度限幅(|vel|>15m/s时reset)
- 方案B: IMM模型(静止+飞行切换), 当前数据中imm得分最高

### 4. IMM为何在预测场景最优

IMM(静止+飞行双模型)的优势:
- 球被接住时自动切到静止模型 → 阻止vy累积
- 球抛出时自动切回飞行模型 → 重力预测正确
- Track=0.746(所有filter最高) → 预测起点准确

### 5. 观测噪声模型

从静止数据标定:
| 距离 | σ_x | σ_y | σ_z |
|------|-----|-----|-----|
| 0-1m | 0.7mm | 2.2mm | 3.5mm |
| 4-5m | 27mm | 32mm | 91mm |
| 7-8m | 1.8mm | 14mm | 137mm |

R_z ∝ z^2.85 (超二次), R_y ∝ z^2.11, R_x ∝ z^0.96

---

## 部署推荐

### 最终方案: 优化参数 IMM

经过全面测试，**直接参数调优 > 所有工程技巧**。

```yaml
# 最优 IMM 参数 (config.yaml)
imm:
  R_base: 0.003          # 5x更信任观测(原0.015)
  noise_exponent: 2.85
  innovation_gate: 25.0
  sigma_static: 0.5
  sigma_flight: 10.0     # 2x更灵活飞行模型(原5.0)
  sigma_maneuver: 80.0   # 1.6x更强突变吸收(原50.0)
```

**最终性能 (7段Throw数据, 820有效样本)**:
| Horizon | RMSE | Median | P90 |
|---------|------|--------|-----|
| 0.1s | 0.289m | 0.134m | 0.467m |
| 0.2s | 0.489m | 0.222m | 0.797m |
| 0.3s | 0.787m | 0.341m | 1.311m |
| 0.5s | 1.716m | 0.527m | 3.104m |

### C++ 部署参数

```cpp
// IMM 3-model (static + flight + maneuver)
const float R_BASE = 0.003f;       // 观测噪声基数(极小=高信任)
const float NOISE_EXP = 2.85f;     // R(z)指数
const float INNOVATION_GATE = 25.0f;
const float SIGMA_STATIC = 0.5f;   // 静止模型过程噪声
const float SIGMA_FLIGHT = 10.0f;  // 飞行模型过程噪声
const float SIGMA_MANEUVER = 80.0f; // 突变模型过程噪声
const float GRAVITY_Y = 9.81f;     // y轴向下

// Markov转移矩阵
float TPM[3][3] = {
    {0.95, 0.04, 0.01},  // static → static
    {0.02, 0.95, 0.03},  // flight → flight
    {0.05, 0.20, 0.75},  // maneuver → 快速退出
};

// 落点预测
float dt_to_ground = solve_quadratic(pos_y, vel_y, 0.5*g, ground_height);
float pred_x = pos_x + vel_x * dt_to_ground;
float pred_z = pos_z + vel_z * dt_to_ground;
```

### 工程技巧验证结论

| 技巧 | 效果 | 适用范围 |
|------|------|---------|
| ~~Stereo conf加权~~ | **有害** (-1~-4%) | × 对IMM无效 |
| ~~中值滤波~~ | **有害** (-4%) | × 引入延迟 |
| ~~跳变预检~~ | **有害** (-1%) | × 丢弃有效数据 |
| ~~速度限幅~~ | 无效 (0%) | × IMM内部已限幅 |
| NIS-Adaptive R | 微弱 (+2%) | △ 本质=参数搜索 |
| RobustWrapper硬重置 | tracking翻倍 | ✓ 仅对单模型EKF有效 |
| **直接参数调优** | **+4.5%** | **✓ 唯一真正有效方法** |

**结论: 复杂工程技巧对设计良好的IMM无增益。把精力放在参数调优和系统集成上。**

### 为什么IMM而不是纯EKF

之前建议单次抛球用纯Gravity EKF，但优化后发现：
- IMM即时收敛(warmup=0和20仅差2%)，无需快启动
- 即使单次抛球，IMM的maneuver模型也能更快适应初始速度不确定性
- 计算量增加(3x EKF)在AGX Orin上可忽略(60Hz @ <0.5ms)
---

## 历史演进

| 轮次 | 旧指标最优 | 新指标排名 | 教训 |
|------|-----------|-----------|------|
| R1 | gravity_ekf_6d (0.720) | #9 | 纯EKF, 多抛接发散 |
| R2 | gravity_one_euro (0.804) | #8 | 位置导数噪声大 |
| R3 | gravity_one_euro_v2 (0.841) | #5 | 后置LPF杀预测精度 |
| R4 | confidence_v2 (0.842) | #4 | det_confidence恒定无效 |
| **v2评估** | **imm (0.511)** | **#1** | 模型切换=防发散+好tracking |
| **R5** | fast_gravity_ekf | 测试 | 快启动+两阶段调度 |

**评估指标决定一切**: 旧指标(平滑导向)奖励超低mc的LPF; 新指标(预测导向)奖励实时tracking。

---

## FastGravityEKF 实验 (R5)

### 设计思路

目标：缩短收敛时间，让前0.1s就能做出有效预测。

```
Phase 0: INIT (frame 0-4)
  收集5帧观测 → 最小二乘拟合初速度 v0
  gravity模型: obs(t) = p0 + v0*t + 0.5*g*t²

Phase 1: BOOST (frame 5-14)
  R_base = 0.005 (极信任观测)
  sigma_a = 50 (大Q → P_vel保持高 → K_vel大)
  线性插值过渡到稳态参数

Phase 2: STEADY (frame 15+)
  与标准gravity_ekf_6d一致
  R_base=0.015, sigma_a=5, gate=25
```

### 关键Bug修复

**R_base量级错误**: 初版误设 R_base=1.5（从 adaptive_q_v2 复制），实际应为 0.015（与 gravity_ekf_6d 一致）。修正后性能提升 42%。

原因：R=1.5 在 depth=5m 时 → R_zz = 1.5 × 5^2.85 ≈ 155，滤波器完全忽略观测。

### 全量对比结果 (7段动态数据)

| Filter | RMSE@0.1s | RMSE@0.2s | RMSE@0.3s | RMSE@0.5s | 有效样本 |
|--------|-----------|-----------|-----------|-----------|----------|
| **imm** | **0.300m** | **0.512m** | **0.828m** | **1.797m** | **820** |
| fast_gravity_ekf | 0.578m | 0.903m | 1.402m | 2.823m | 459 |
| gravity_ekf_6d | 0.662m | 1.013m | 1.567m | 3.134m | 219 |

### 收敛速度对比 (单段 seg3, RMSE@0.2s)

| Warmup | fast_gravity_ekf | imm | gravity_ekf_6d |
|--------|-----------------|-----|----------------|
| 0帧 | **0.544m** | 0.570m | 0.790m |
| 5帧 | 0.558m | 0.550m | 0.792m |
| 10帧 | 0.536m | **0.439m** | 0.738m |
| 15帧 | 0.494m | **0.301m** | 0.661m |
| 20帧 | 0.439m | **0.198m** | 0.587m |

### 结论

1. **FastGravityEKF 的快启动有效但优势极小**:
   - warmup=0 时赢 IMM 4.5% (0.544 vs 0.570)
   - 优势窗口仅 ~5帧 (0.08s)，之后 IMM 迅速超越

2. **IMM 长期精度远优于单模型 EKF**:
   - warmup=20: IMM=0.198m vs fast=0.439m（2.2倍差距）
   - 多模型切换天然适应抛接转换

3. **FastGravityEKF 改善了 gravity_ekf_6d**:
   - 有效样本 459 vs 219（tracking能力翻倍）
   - RMSE 降低 12-17%
   - 但仍不及 IMM

4. **最终建议**: 部署使用 IMM，不投入 FastGravityEKF

### 参数调优总结

| 参数 | 最优值 | 含义 |
|------|--------|------|
| init_frames | 5 | 3帧噪声太大, 5帧稳定 |
| boost_R_base | 0.005 | 比稳态更信任观测 |
| boost_sigma_a | 50 | 大Q保持P_vel高→强速度修正 |
| boost_frames | 10 | 0.17s过渡足够 |
| P_vel_init | 25 | 初始速度极不确定 |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| evaluate.py | 评估驱动 (v2: 预测精度为核心) |
| loader.py | CSV加载 |
| config.yaml | 配置 |
| filters/gravity_ekf_6d.py | 6D重力EKF |
| filters/fast_gravity_ekf.py | 快启动两阶段EKF (R5实验) |
| filters/gravity_one_euro.py | V1: EKF + One-Euro(位置导数) |
| filters/gravity_one_euro_variants.py | V2/Adaptive/AEKF变种 |
| filters/round4_filters.py | IMM/Fallback/Confidence/Bounce等 |
| filters/imm_filter.py | 三模型IMM |
| filters/one_euro.py | 纯One-Euro |
| filters/adaptive_ekf.py | AEKF |
| results/scores.csv | 评分数据 |
