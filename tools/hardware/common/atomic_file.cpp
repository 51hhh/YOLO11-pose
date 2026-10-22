#include "atomic_file.h"

#include <cerrno>
#include <cstring>
#include <system_error>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace volleyball::tools::hardware {

bool publishFileNoReplace(
    const std::filesystem::path& temporary_path,
    const std::filesystem::path& final_path,
    std::string& error_message) {
    error_message.clear();
    if (temporary_path.empty() || final_path.empty() ||
        temporary_path == final_path) {
        error_message = "临时路径和最终路径必须非空且不同";
        return false;
    }
#if defined(__linux__) && defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
    if (::syscall(
            SYS_renameat2,
            AT_FDCWD,
            temporary_path.c_str(),
            AT_FDCWD,
            final_path.c_str(),
            RENAME_NOREPLACE) == 0) {
        return true;
    }
    error_message = "无法无覆盖发布 " + final_path.string() + ": " +
                    std::strerror(errno);
    return false;
#else
    // 非 Linux 构建使用同目录硬链接保持“目标已存在即失败”的原子语义。
    std::error_code error;
    std::filesystem::create_hard_link(temporary_path, final_path, error);
    if (error) {
        error_message = "无法无覆盖发布 " + final_path.string() + ": " +
                        error.message();
        return false;
    }
    error.clear();
    const bool removed = std::filesystem::remove(temporary_path, error);
    if (!removed || error) {
        std::error_code rollback_error;
        (void)std::filesystem::remove(final_path, rollback_error);
        error_message = "发布后无法移除临时文件 " +
                        temporary_path.string();
        return false;
    }
    return true;
#endif
}

}  // namespace volleyball::tools::hardware
