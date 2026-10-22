#pragma once

#include <cstdint>
#include <string>

namespace volleyball::adapters::ros2 {

[[nodiscard]] std::uint32_t createSourceEpoch(const std::string& path);

}  // namespace volleyball::adapters::ros2
