# 🔧 CUDA 编译问题修复

## 问题描述
编译时出现错误：
```
fatal error: cuda_runtime_api.h: No such file or directory
```

## 解决方案

已在 `CMakeLists.txt` 中添加 CUDA 支持。

### 在 NX 上重新编译

```bash
cd ~/NX_volleyball/ros2_ws

# 清理旧的编译文件
rm -rf build/volleyball_stereo_driver install/volleyball_stereo_driver

# 重新编译
colcon build --packages-select volleyball_stereo_driver

# 加载环境
source install/setup.bash
```

### 验证编译结果

```bash
# 检查节点是否生成
ls install/volleyball_stereo_driver/lib/volleyball_stereo_driver/

# 应该看到:
# - stereo_system_node
# - volleyball_tracker_node (如果 TensorRT 可用)
```

### 如果仍然失败

检查 CUDA 是否安装：
```bash
# 检查 CUDA 路径
ls /usr/local/cuda/include/cuda_runtime_api.h

# 检查 TensorRT
dpkg -l | grep tensorrt
```

如果 CUDA 头文件不存在，volleyball_tracker_node 将不会编译，但 stereo_system_node 应该可以正常编译。

---

**修复完成！请在 NX 上重新编译测试。** ✅
