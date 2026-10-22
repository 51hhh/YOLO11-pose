#include "parallel_tasks.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace legacy_nx_tools {

bool runIndexedTasks(
    std::size_t item_count,
    std::size_t worker_count,
    const IndexedTask& task,
    std::string& error_message) {
    error_message.clear();
    if (!task) {
        error_message = "并行任务回调不能为空";
        return false;
    }
    if (item_count == 0) return true;
    if (worker_count == 0) {
        error_message = "并行任务工作线程数必须大于零";
        return false;
    }
    worker_count = std::min(worker_count, item_count);

    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    auto record_error = [&](std::string message) {
        stop.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (error_message.empty()) error_message = std::move(message);
    };
    auto worker = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t index = next.fetch_add(1);
            if (index >= item_count) return;
            try {
                task(index);
            } catch (const std::exception& error) {
                record_error(
                    "并行任务索引 " + std::to_string(index) +
                    " 失败: " + error.what());
                return;
            } catch (...) {
                record_error(
                    "并行任务索引 " + std::to_string(index) +
                    " 发生未知异常");
                return;
            }
        }
    };

    std::vector<std::thread> workers;
    try {
        workers.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            workers.emplace_back(worker);
        }
    } catch (const std::exception& error) {
        record_error("无法创建并行工作线程: " + std::string(error.what()));
    } catch (...) {
        record_error("无法创建并行工作线程: 未知异常");
    }

    for (auto& thread : workers) {
        if (thread.joinable()) thread.join();
    }
    return error_message.empty();
}

}  // namespace legacy_nx_tools
