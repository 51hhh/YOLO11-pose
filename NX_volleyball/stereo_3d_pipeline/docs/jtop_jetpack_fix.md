# jtop JetPack 版本识别修复指南

## 问题描述

在 Jetson Orin NX（JetPack 6.2, L4T R36.4.7）上运行 `jtop` 时，JetPack 版本显示为 **MISSING**。
系统实际已正确安装 `nvidia-jetpack 6.2.1+b38`。

## 根因分析

`jtop`（jetson-stats 4.3.2）使用内部映射表将 L4T 版本号映射到 JetPack 版本。
映射表位于：
```
/usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py
```

该表包含 `"36.4.3": "6.2"` 等条目，但 **缺少 `"36.4.7"` 对应条目**。
当 L4T 版本为 R36.4.7 时，查表失败，显示 MISSING。

## 验证步骤

```bash
# 1. 确认 L4T 版本
cat /etc/nv_tegra_release
# 应输出: # R36 (release), REVISION: 4.7, ...

# 2. 确认 JetPack 已安装
dpkg -l | grep nvidia-jetpack
# 应显示: nvidia-jetpack 6.2.1+b38

# 3. 检查 jtop 映射表
grep "36.4" /usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py
# 如果只有 "36.4.3": "6.2"，没有 "36.4.7"，则确认问题
```

## 修复方法

向映射表中添加 `"36.4.7": "6.2"` 条目：

```bash
# 备份原文件
sudo cp /usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py \
       /usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py.bak

# 在 "36.4.3": "6.2" 之前插入 "36.4.7": "6.2"
sudo python3 -c "
path = '/usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py'
with open(path, 'r') as f:
    content = f.read()

old = '\"36.4.3\": \"6.2\"'
new = '\"36.4.7\": \"6.2\",\n    \"36.4.3\": \"6.2\"'
content = content.replace(old, new)

with open(path, 'w') as f:
    f.write(content)
print('Patched successfully')
"

# 重启 jtop 服务
sudo systemctl restart jtop.service
```

## 验证修复

```bash
# 检查映射是否已添加
grep "36.4.7" /usr/local/lib/python3.10/dist-packages/jtop/core/jetson_variables.py
# 应输出包含 "36.4.7": "6.2" 的行

# 运行 jtop 确认
jtop
# JetPack 应显示为 6.2 而非 MISSING
```

## 适用范围

| 项目 | 版本 |
|------|------|
| Jetson 平台 | Orin NX Super 16GB |
| L4T | R36.4.7 |
| JetPack | 6.2 (nvidia-jetpack 6.2.1+b38) |
| jetson-stats (jtop) | 4.3.2 (PyPI 最新) |
| Python | 3.10 |

## 备注

- 此问题在 jetson-stats 后续版本更新映射表后会自动修复
- 如果升级 jetson-stats (`pip3 install --upgrade jetson-stats`)，需要检查新版本是否包含 36.4.7 映射
- 修改仅影响版本号显示，不影响系统功能
