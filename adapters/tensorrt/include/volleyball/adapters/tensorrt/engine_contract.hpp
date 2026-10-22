/**
 * @file engine_contract.hpp
 * @brief 当前排球检测器能够安全执行的 TensorRT I/O 契约。
 *
 * 这个文件故意不依赖 TensorRT/CUDA，便于在无 Jetson 的主机上单测。
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace volleyball::adapters::tensorrt {

struct DetectorTensorSpec {
    std::string name;
    bool is_input = false;
    bool is_float32 = false;
    std::vector<std::int64_t> dimensions;
};

struct DetectorEngineContract {
    std::string input_name;
    std::string single_output_name;
    int input_size = 0;
    int output_count = 0;
    int single_output_elements = 0;
    bool multi_scale_output = false;
};

/**
 * @brief 验证 engine 是否能由当前固定形状检测器安全执行。
 *
 * 支持：
 * - 一个 float32 NCHW 输入 `[1,3,S,S]`；
 * - 一个 `[1,C,N]` / `[1,N,C]` pre-NMS 或 `[1,N,6]` post-NMS 输出；或
 * - 六个 NHWC float32 DFL 输出（每个尺度一组 `[1,H,W,1]` 分类和
 *   `[1,H,W,64]` 回归）。
 *
 * 动态维度、额外输入/输出和其他布局必须明确拒绝，不能按默认值分配缓冲。
 */
bool validateDetectorEngineContract(
    const std::vector<DetectorTensorSpec>& tensors,
    DetectorEngineContract& contract,
    std::string& error);

}  // namespace volleyball::adapters::tensorrt
