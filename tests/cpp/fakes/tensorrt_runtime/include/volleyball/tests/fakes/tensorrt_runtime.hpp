#pragma once

#include "volleyball/adapters/tensorrt/engine_contract.hpp"
#include "volleyball/adapters/tensorrt/postprocess.hpp"

#include <cstdint>

namespace volleyball::tests::tensorrt_runtime_fake {

inline bool initialize_success = true;
inline int input_size = 640;
inline adapters::tensorrt::DetectorOutputContract output_contract =
    adapters::tensorrt::DetectorOutputContract::yolo_pre_nms;
inline int class_count = 1;
inline adapters::tensorrt::PostprocessResult detection_result{true, {}, {}};
inline int detection_calls = 0;
inline std::uintptr_t last_pointer = 0;
inline int last_pitch = 0;
inline int last_width = 0;
inline int last_height = 0;

inline void reset() {
    initialize_success = true;
    input_size = 640;
    output_contract =
        adapters::tensorrt::DetectorOutputContract::yolo_pre_nms;
    class_count = 1;
    detection_result = {true, {}, {}};
    detection_calls = 0;
    last_pointer = 0;
    last_pitch = 0;
    last_width = 0;
    last_height = 0;
}

}  // namespace volleyball::tests::tensorrt_runtime_fake
