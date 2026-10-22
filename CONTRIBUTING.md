# 贡献与提交规范

## 分支

`main` 是唯一新主线。`master` 只保留重构前历史，不再接收新功能。实验分支必须通过可重放数据和 profile 接入，不得将一次性阈值直接写进生产管线。

## 提交

提交使用 Conventional Commits，类型和 scope 使用英文，描述使用中文：

```text
feat(capture): 增加原始帧哈希清单
fix(clock): 修正设备时钟映射有效区间
refactor(stereo): 拆分配对与三角化边界
test(replay): 增加录制数据确定性回放
docs(architecture): 记录坐标系所有权
chore(ci): 增加 CPU 契约门禁
```

每个提交只做一类可独立回退的事情。不把大量格式化、算法变更和契约变更混在同一提交。

## 分层规则

- `libs/core` 只包含与平台无关的强类型事实、时钟和坐标语义。
- `ports` 定义用例需要的接口，不包含 Hik/TensorRT/CUDA/VPI/ROS2 头文件。
- `adapters` 实现硬件、推理、存储和中间件边界。
- 用户流程统一由 `volleyball_vision.cli` 组装；若未来增加原生 C++ 应用，再建立 `apps/`，且其中不得实现立体匹配算法。
- `python/volleyball_vision` 承担配置、数据契约、回放和离线评估；训练依赖必须可选。
- `legacy` 只允许兼容转换和可删除的过渡代码，不得反向被 core 依赖。
- `tests/cpp/mainline` 不得包含旧 NX 头文件；旧实现及 SDK 桩测试统一放入 `tests/cpp/legacy_nx`。
- `tools` 只保留仓库级维护门禁；可复用用户功能必须成为 `volleyball` 子命令。

## 必跑检查

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python3 -m unittest discover -s tests -v

cmake --preset host-core-debug
cmake --build --preset host-core-debug --parallel 2
ctest --preset host-core-debug

python3 tools/check_repository.py
```

涉及实时管线时，还必须在 NX 上完成 GPU 构建、录制回放和延迟分位数对比。不允许用“本机没有 CUDA”推导实机通过。

