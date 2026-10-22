#ifndef VOLLEYBALL_TOOLS_HARDWARE_PARALLEL_TASKS_H_
#define VOLLEYBALL_TOOLS_HARDWARE_PARALLEL_TASKS_H_

#include <cstddef>
#include <functional>
#include <string>

namespace volleyball::tools::hardware {

using IndexedTask = std::function<void(std::size_t index)>;

/**
 * 用固定数量工作线程恰好处理一次 [0, item_count) 中的每个索引。
 * 任务或线程创建异常被转换为 false 和错误文本，不允许越过线程入口导致
 * std::terminate。发生首个任务错误后停止领取新任务。
 */
[[nodiscard]] bool runIndexedTasks(
    std::size_t item_count,
    std::size_t worker_count,
    const IndexedTask& task,
    std::string& error_message);

}  // namespace volleyball::tools::hardware

#endif  // VOLLEYBALL_TOOLS_HARDWARE_PARALLEL_TASKS_H_
