#pragma once

// 旧 NX 管线的兼容入口。实现与 CUDA 内核已迁入正式适配器，旧代码只保留
// 该转发头，避免两套 ROI 深度算法继续分叉。
#include "volleyball/adapters/cuda_depth/roi_matcher.hpp"
