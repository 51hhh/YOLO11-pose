# 并发、数据流和启用矩阵

## 执行实体

| 实体 | 责任 | 当前主线 |
|---|---|---|
| PWM 线程 | 90 Hz GPIO 上升沿与边沿日志 | 实时/采集开启 |
| grab 线程 | 等待左右 USB payload、FrameCounter 对齐、装入 slot | 实时/采集开启 |
| pipeline 线程 | Stage 0/1 编排、结果 drain、融合和 callback | 实时/采集开启 |
| 左 TensorRT stream | 左图预处理、推理、输出 D2H | 开启 |
| 右 TensorRT stream | 右图预处理、推理、输出 D2H | 开启 |
| Bayer CUDA stream | 左右 Bayer→BGR | 开启，但当前有显式同步 |
| VPI remap stream | BGR 校正和 Gray 转换 | CUDA backend 开启 |
| Fuse CUDA stream | detection event 等待、部分融合任务 | 开启 |
| Async ROI copy stream | Gray D2D、必要时 Gray D2H/BGR D2D | 仅实时异步开启 |
| Async ROI worker | 关联、ROI 候选、深度计算 | 仅实时异步开启 |
| P2 inline worker/stream | NCC、XFeat、SuperPoint、ALIKED 作业框架 | 四组 worker/stream 会创建；当前只有 NCC 候选有逐帧任务，神经候选关闭 |
| P2 diagnostic worker/stream | 独立低优先级候选和 artifact/CSV | 当前 diagnostic lane 关闭，因此不创建 |
| capture writer | 写 v2.1 staging | 仅真实采集开启 |

pipeline 与 grab 线程之间不是无界队列，而是一个可测试的 single-flight 状态机：同一时刻最多一个 active 请求；重叠请求被拒绝；停止后拒绝新请求，但已接受请求必须写完并发出 completion。completion 才转移 raw 图像所有权，`running` 标志不承担图像完成语义。

线程生命周期另由 `PipelineRunState` 管理三种正交事实：是否仍运行、是否发生失败、是否仍有一次 cleanup 责任。worker 异常可以立即停止新调度，但不能消费 cleanup token；只有 owner 调用 `stop()` 才执行 join 和资源排空。若回调从 pipeline 线程内误调 `stop()`，实现只发停止信号并把 join 延后给 owner，避免线程自我 join。

主 stream 与辅助 stream 的退出结果会合并：异步 ROI core/copy、P2 diagnostic
core/copy 和 `PipelineStreams` 任一同步失败，都不算正常停止。辅助 stream 不能因
不在 `PipelineStreams` 结构体里而漏过最终 drain。

异步 ROI worker 的 snapshot completion 也承担所有权转移语义。worker 必须收到有效 buffer 和已记录 event，等待 event 成功后才可读取快照；CUDA event 错误或 worker 异常会产生显式失败结果。失败路径先排空 ROI core/copy stream，再归还 buffer，防止异常发生在 kernel 提交之后时复用仍被 GPU 读取的内存。`roi_postprocess_mutex_` 只包围共享 matcher/core 调用，artifact/CSV 写入不持有该锁，避免诊断 I/O 反向阻塞同步候选。

异步 buffer 只有在对应 copy event 或源 stream 被证明同步成功后才能回到空闲池。event/stream 同步失败时，该 buffer 会进入 `quarantined` 状态并一直保留到管线销毁；它不能被 pending-replace、expire 或下一帧重新取得。快照提交、slot completion event 或复用前等待属于基础设施边界，失败帧必须设置 `grab_failed`，不能降级成正常的 `predict_only` 观测。

## 实时异步 ROI 数据流

异步 Stage 2 只有同时满足以下条件才可启动：

- `async_roi_stage2=true`；
- `stereo.strategy=roi_only`；
- 不是 detection-only。

当前 `nx-rdk` 满足这些条件。提交时：

1. 主线程先 collect 左右 YOLO；
2. 把左右整帧校正 Gray8 从 FrameSlot D2D 复制到异步缓冲；
3. 只有 CPU fallback 可能需要时才附加 D2H Gray；
4. 只有已启用彩色/神经候选需要 BGR 时才附加 D2D BGR；当前不需要；
5. 主线程等待槽位快照 event，随后释放 FrameSlot 的 VPI CUDA read lock；
6. worker 只读取自己的快照，等待 copy event 后运行 ROI Stage 2；
7. 主线程 drain 完成结果并执行融合、ResultCallback/FrameCallback。

缓冲池当前为 3。pending 队列以新代旧，正在运行的任务不被抢占；没有空闲缓冲、结果过期或超过帧截止线时均有独立计数。被替换的 pending buffer 只有在快照复制完成且 P2 诊断引用为零时才允许原地交给新帧；否则进入 `WAITING_DIAGNOSTIC`，当前新帧按容量不足丢弃，不能覆盖诊断线程正在读取的 source snapshot。

## 三个 profile 的最终行为

| 能力 | `nx-rdk` 实时 | `free-flight` 经 `capture` 启动 | `replay` |
|---|---|---|---|
| 海康/GPIO | 开 | 开 | 不启动 |
| CUDA Bayer/VPI 校正 | 开 | 开 | 不启动 |
| 双路 TensorRT YOLO | 开 | 开 | 使用已录数据 |
| ROI_ONLY 深度 | 开 | 开 | 使用已录观测 |
| async ROI Stage 2 | 开 | 强制关 | 无 |
| 丢 stale ROI | 允许 | 强制关 | 无 |
| 主队列 | 3，`drop_oldest` | 256，`block` | 64，`block` |
| NX 对外轨迹预测/控制 | 不存在；由 RDK 负责 | 不存在 | 离线派生 |
| ROS2 直接观测 | 开 | 关 | 关 |
| v2.1 capture | 关 | 开 | 读取 |
| raw Gray8 | 不保存 | 全部保存 | 读取 |
| 契约破坏 | 记录/降级 | fail trial | 拒绝数据 |

注意：直接执行 `config render-legacy --profile free-flight` 时已经显示同步、无丢帧
执行策略，但 `capture_v21=false`，因为尚无 trial staging。真正 `volleyball capture`
会再调用 `capture_stage` 分支，注入输出目录、帧数、队列和采集身份。

## 当前算法启用矩阵

| 模块 | 分配/初始化 | 每帧计算 | 可直接成为最终 z | 对外保留 |
|---|---|---|---|---|
| Rectified BGR | 是 | 是 | 否 | FrameCallback 内可见 |
| Rectified Gray | 是 | 是 | 间接 | 采集可保存 |
| Full-frame disparity | 仅保留字段，ROI_ONLY 不分配图像 | 否 | 否 | 否 |
| 左右 YOLO | 是 | 每帧 | 提供关联框 | 全候选保存 |
| shifted-IoU | 无独立图像 | 每 pair | 仅 gate | 配对字段保存 |
| ROI multi-point | 是 | pair 通过后 | 首选 | 候选字段保存 |
| circle/radial/edge/bbox | 是 | pair 通过后 | legacy 回退 | 候选字段保存 |
| center patch | 是 | pair 通过后 | 当前否 | 候选字段保存 |
| CUDA template/NCC | 是 | pair 通过后 | 当前否 | 候选字段保存 |
| ORB/BRISK/AKAZE/SIFT | 否或惰性 | 否 | 否 | 字段可为空 |
| XFeat/SuperPoint/ALIKED | 配置对象存在 | 否 | 否 | 字段可为空 |

## 配置投影边界

根配置不是逐键一一映射旧 C++ 配置，目前需要特别关注：

- 根层 `tracking.enabled=true` 只表示 3D `kalman_ca` 状态估计；已删除的旧图像 SOT 不再拥有配置入口；
- 根层 `candidate_methods` 会完整重写旧 `depth_modes`，不再继承模板中的隐藏候选；
- `primary_method/fallback_methods` 会在投影前校验旧选择器是否能忠实实现；当前只允许两个实时首选和固定相对回退顺序；
- 根层 subpixel、关联、相机、模型、校正和发布参数会明确投影；
- 采集专用的同步/no-drop 覆盖只在带 `capture_stage` 构建配置时发生；
- 适配器模板中的默认值不代表主线状态，必须查看 profile 投影和运行时证据。

任何新候选仍应先加入主配置 schema、候选到旧 mode 的唯一映射及 adapter 测试，再修改旧 YAML；否则正式 profile 不会启用它。

## 推荐的数据流边界

后续重构应维持以下边界：

```text
FrameSource
  -> PreprocessedStereoFrame
  -> PerCameraDetections
  -> StereoPairAttempts
  -> DepthCandidateSet
  -> DirectTriangulatedObservation
  -> OptionalTrackEstimate
```

每层保存自己的输入身份和质量字段，禁止用下游平滑结果回写上游事实。CUDA/VPI 只负责图像存储与算子执行，不应承担 trial 数据契约或 RL 状态语义。

