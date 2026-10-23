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

enum class SingleOutputLayout {
    none,
    pre_nms_channels_first,
    pre_nms_rows,
    post_nms_xyxy_rows,
};

/** 模型输出的高层语义；用于把 manifest 期望与 engine 实际布局对齐。 */
enum class DetectorOutputContract {
    none,
    yolo_pre_nms,
    yolo_post_nms,
    yolo_dfl,
};

[[nodiscard]] const char* detectorOutputContractName(
    DetectorOutputContract contract) noexcept;

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
    int single_output_class_count = 0;
    SingleOutputLayout single_output_layout = SingleOutputLayout::none;
    DetectorOutputContract output_contract = DetectorOutputContract::none;
    int class_count = 0;
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
