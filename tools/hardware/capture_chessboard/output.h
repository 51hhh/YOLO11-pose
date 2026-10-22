#ifndef VOLLEYBALL_CAPTURE_CHESSBOARD_OUTPUT_H_
#define VOLLEYBALL_CAPTURE_CHESSBOARD_OUTPUT_H_

#include "volleyball/adapters/hik_mvs/hikvision_camera.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>

namespace capture_chessboard {

struct SyncInfo {
    std::int64_t frame_number_delta = 0;
    std::int64_t frame_counter_delta = 0;
    std::int64_t trigger_delta = 0;
    std::int64_t timestamp_delta_ns = 0;
};

[[nodiscard]] SyncInfo makeSyncInfo(
    const volleyball::adapters::hik_mvs::GrabResult& left,
    const volleyball::adapters::hik_mvs::GrabResult& right);

[[nodiscard]] bool writeMetadataHeader(std::ostream& metadata);

[[nodiscard]] bool writeMetadataRow(
    std::ostream& metadata,
    int pair_index,
    const std::string& file_name,
    const volleyball::adapters::hik_mvs::GrabResult& left,
    const volleyball::adapters::hik_mvs::GrabResult& right,
    const SyncInfo& sync);

using PairFileWriter =
    std::function<bool(const std::filesystem::path& temporary_path)>;
using MetadataCommit = std::function<bool()>;

/**
 * 先写入同目录临时文件，再发布左右图片并提交 metadata。
 *
 * 任一步失败时清理本次调用产生的临时文件和已发布图片；已有目标文件
 * 永不覆盖。metadata 流自身无法回滚，因此调用方必须把 false 视为整次
 * 标定采集失败并停止继续写入。
 */
[[nodiscard]] bool publishPair(
    const std::filesystem::path& final_left,
    const std::filesystem::path& final_right,
    const PairFileWriter& write_left,
    const PairFileWriter& write_right,
    const MetadataCommit& commit_metadata,
    std::string& error_message);

}  // namespace capture_chessboard

#endif  // VOLLEYBALL_CAPTURE_CHESSBOARD_OUTPUT_H_
