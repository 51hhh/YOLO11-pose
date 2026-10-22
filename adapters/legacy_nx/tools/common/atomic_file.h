#ifndef VOLLEYBALL_LEGACY_NX_TOOLS_ATOMIC_FILE_H_
#define VOLLEYBALL_LEGACY_NX_TOOLS_ATOMIC_FILE_H_

#include <filesystem>
#include <string>

namespace legacy_nx_tools {

/**
 * 在同一文件系统内把临时文件发布为最终文件，且绝不覆盖已有目标。
 * 成功后临时路径消失；失败时临时文件仍由调用方负责清理。
 */
[[nodiscard]] bool publishFileNoReplace(
    const std::filesystem::path& temporary_path,
    const std::filesystem::path& final_path,
    std::string& error_message);

}  // namespace legacy_nx_tools

#endif  // VOLLEYBALL_LEGACY_NX_TOOLS_ATOMIC_FILE_H_
