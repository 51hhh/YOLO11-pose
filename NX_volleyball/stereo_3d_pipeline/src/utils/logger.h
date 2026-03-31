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

namespace stereo3d {

// ===================== 系统日志 =====================
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

inline void logMsg(LogLevel level, const char* fmt, ...) {
    const char* prefix = "";
    switch (level) {
        case LogLevel::DEBUG: prefix = "[DEBUG]"; break;
        case LogLevel::INFO:  prefix = "[INFO] "; break;
        case LogLevel::WARN:  prefix = "[WARN] "; break;
        case LogLevel::ERROR: prefix = "[ERROR]"; break;
    }
    fprintf(stderr, "%s ", prefix);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define LOG_DEBUG(fmt, ...) stereo3d::logMsg(stereo3d::LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  stereo3d::logMsg(stereo3d::LogLevel::INFO,  fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  stereo3d::logMsg(stereo3d::LogLevel::WARN,  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) stereo3d::logMsg(stereo3d::LogLevel::ERROR, fmt, ##__VA_ARGS__)

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_LOGGER_H_
