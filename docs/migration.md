# 旧 NX 管线迁移说明

## 当前状态

`NX_volleyball/stereo_3d_pipeline` 已有成熟的海康 MVS、GPIO、CUDA 校正、TensorRT、ROI 深度和 ROS2 实现。重构不会一次性重写这些高风险硬件代码，而是先冻结外部契约，再逐步把实现迁到明确的 adapters/apps 边界。

根工程通过 `nx-legacy-release` 预设将旧工程作为 ExternalProject 编译。这样可以复用实机能力，同时防止旧工程的全局 CUDA、VPI、MVS、OpenCV 和 ROS2 依赖进入 `libs/core`、`ports`、`algorithms` 或 `services`。

## 配置所有权

`configs/` 是唯一主线配置源。启动时将合并后的配置投影为旧管线 YAML：

```text
分层主线配置 → schema/语义校验 → 配置 SHA-256
             → legacy YAML 生成 → 旧 NX 可执行文件
```

旧目录中的生产 YAML 现阶段仍是投影所需的基础模板，但不得直接作为新的部署入口。深度候选由主线 `candidate_methods` 完整覆盖旧 `depth_modes`；`legacy_activation_policy.py` 还会统一覆盖 TNR、DLA、双路检测、全帧视差、神经特征、异步 ROI、P2 诊断、图像 SOT 和显示等启用开关。尚未提升的数值调参仍需逐项决定保留、删除或加入 schema，但基础模板已不能通过修改布尔开关偷偷改变正式 profile 的执行支路。

## 迁移阶段

1. **冻结契约**：v2.1 数据、强类型核心、配置和 CI 已建立；
2. **包裹旧实现**：主线 CLI 负责 doctor、配置生成、采集 staging 和最终发布；
3. **提取算法**：纯关联/三角化已进入 `algorithms/stereo`，逐项用录制数据对比旧实现；
4. **提取硬件适配器**：把 MVS、GPIO、TensorRT/CUDA 和 ROS2 分别迁入 `adapters`；
5. **替换可执行入口**：在 `apps` 中组装新服务，旧管线只保留回归基准；
6. **删除兼容层**：只有在同一批实机数据、性能和输出契约均通过时执行。

## 兼容性门禁

每次迁移必须固定同一组录制输入并比较：

- 帧与候选数量；
- 接受/拒绝配对及理由；
- 3D 差异和协方差；
- 端到端延迟 p50/p95/p99；
- GPU/CPU 占用与队列水位；
- 输出 schema、配置指纹和原始文件哈希。

数值更接近旧实现并不自动代表更正确；标定靶、静止球、已知距离和独立参考轨迹是绝对精度判据。

## 不允许的迁移方式

- 新核心层直接包含 MVS、CUDA、VPI、TensorRT、OpenCV 或 ROS2 头文件；
- 为赶进度从主线反向依赖旧目录内部全局状态；
- 同时修改算法、数据 schema 和硬件时序而没有可重放基线；
- 直接编辑生成的 legacy YAML 后把结果当作可复现实验；
- 删除拒绝候选或用滤波轨迹覆盖直接观测。

