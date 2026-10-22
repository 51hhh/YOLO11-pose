/**
 * @file log.hpp
 * @brief 系统统一日志（printf 格式）。
 *
 * 所有 LOG_xxx 宏使用 printf 格式说明符（%s、%d、%f 等）。
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>

#include <sys/syscall.h>
#include <unistd.h>

namespace stereo3d {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

inline void logMsg(LogLevel level, const char* fmt, ...) {
    static const char* prefixes[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);

    char buf[2048];
    int off = std::snprintf(
        buf,
        sizeof(buf),
        "[%ld.%03ld][T%ld][%s] ",
        ts.tv_sec % 10000,
        ts.tv_nsec / 1000000,
        static_cast<long>(syscall(SYS_gettid)),
        prefixes[static_cast<int>(level)]);
    va_list args;
    va_start(args, fmt);
    off += std::vsnprintf(buf + off, sizeof(buf) - off, fmt, args);
    va_end(args);
    if (off < static_cast<int>(sizeof(buf)) - 1) {
        buf[off++] = '\n';
        buf[off] = '\0';
    }
    std::fputs(buf, stderr);
}

#define LOG_DEBUG(...) stereo3d::logMsg(stereo3d::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  stereo3d::logMsg(stereo3d::LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  stereo3d::logMsg(stereo3d::LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERROR(...) stereo3d::logMsg(stereo3d::LogLevel::ERROR, __VA_ARGS__)

}  // namespace stereo3d
