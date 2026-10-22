/**
 * @file logger.h
 * @brief 系统统一日志 (printf 格式)
 *
 * 所有 LOG_xxx 宏使用 printf 格式说明符 (%s, %d, %f, %.2f 等)。
 */

#ifndef STEREO_3D_PIPELINE_LOGGER_H_
#define STEREO_3D_PIPELINE_LOGGER_H_

#include <cstdio>
#include <cstdarg>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace stereo3d {

// ===================== 系统日志 =====================
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

inline void logMsg(LogLevel level, const char* fmt, ...) {
    static const char* prefixes[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    char buf[2048];
    int off = snprintf(buf, sizeof(buf), "[%ld.%03ld][T%ld][%s] ",
                       ts.tv_sec % 10000, ts.tv_nsec / 1000000,
                       (long)syscall(SYS_gettid),
                       prefixes[static_cast<int>(level)]);
    va_list args;
    va_start(args, fmt);
    off += vsnprintf(buf + off, sizeof(buf) - off, fmt, args);
    va_end(args);
    if (off < (int)sizeof(buf) - 1) { buf[off++] = '\n'; buf[off] = '\0'; }
    fputs(buf, stderr);
}

#define LOG_DEBUG(fmt, ...) stereo3d::logMsg(stereo3d::LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  stereo3d::logMsg(stereo3d::LogLevel::INFO,  fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  stereo3d::logMsg(stereo3d::LogLevel::WARN,  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) stereo3d::logMsg(stereo3d::LogLevel::ERROR, fmt, ##__VA_ARGS__)

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_LOGGER_H_
