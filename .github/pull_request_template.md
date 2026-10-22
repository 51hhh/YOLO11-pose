## 改动

- 目标：
- 影响的层/端口：
- 配置或 schema 变化：

## 验证

- [ ] `python -m unittest discover -s tests -v`
- [ ] `cmake --preset host-core-debug && cmake --build --preset host-core-debug`
- [ ] `ctest --preset host-core-debug`
- [ ] 如涉及 CUDA/相机/ROS2，已记录 NX 实机 profile、指纹和回放证据
- [ ] 未把滤波/预测数据写入 `trajectory.csv`
- [ ] 未丢弃全部 2D 候选或被拒绝的双目配对

## 风险与回退

- 性能/实时性风险：
- 数据契约风险：
- 可回退到的提交：

