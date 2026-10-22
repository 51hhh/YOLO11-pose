/**
 * @file log.hpp
 * @brief 系统统一日志（printf 格式）。
 *
 * 所有 LOG_xxx 宏使用 printf 格式说明符（%s、%d、%f 等）。
 */

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <ctime>

#include <sys/syscall.h>
#include <unistd.h>

namespace stereo3d {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

namespace detail {

inline std::size_t clampedWritten(int result, std::size_t capacity) {
    if (result <= 0 || capacity == 0) return 0;
    const auto requested = static_cast<std::size_t>(result);
    return requested < capacity ? requested : capacity - 1;
}

}  // namespace detail

inline void logMsg(LogLevel level, const char* fmt, ...) {
    static const char* prefixes[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);

    char buf[2048];
    buf[0] = '\0';
    const int prefix_result = std::snprintf(
        buf,
        sizeof(buf),
        "[%ld.%03ld][T%ld][%s] ",
        ts.tv_sec % 10000,
        ts.tv_nsec / 1000000,
        static_cast<long>(syscall(SYS_gettid)),
        prefixes[static_cast<int>(level)]);
    std::size_t used = detail::clampedWritten(prefix_result, sizeof(buf));

    va_list args;
    va_start(args, fmt);
    const std::size_t remaining = sizeof(buf) - used;
    const int body_result = std::vsnprintf(
        buf + used,
        remaining,
        fmt != nullptr ? fmt : "(null log format)",
        args);
    va_end(args);
    used += detail::clampedWritten(body_result, remaining);

    if (used >= sizeof(buf) - 1) {
        buf[sizeof(buf) - 2] = '\n';
        buf[sizeof(buf) - 1] = '\0';
    } else {
        buf[used++] = '\n';
        buf[used] = '\0';
    }
    std::fputs(buf, stderr);
}

#define LOG_DEBUG(...) stereo3d::logMsg(stereo3d::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  stereo3d::logMsg(stereo3d::LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  stereo3d::logMsg(stereo3d::LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERROR(...) stereo3d::logMsg(stereo3d::LogLevel::ERROR, __VA_ARGS__)

}  // namespace stereo3d
