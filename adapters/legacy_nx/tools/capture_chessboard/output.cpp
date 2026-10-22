#include "output.h"
#include "../common/atomic_file.h"

#include <exception>
#include <ostream>
#include <system_error>

namespace capture_chessboard {
namespace {

std::filesystem::path partialPath(const std::filesystem::path& final_path) {
    return final_path.parent_path() /
           (final_path.stem().string() + ".partial" +
            final_path.extension().string());
}

bool pathExists(
    const std::filesystem::path& path,
    bool& exists,
    std::string& error_message) {
    std::error_code error;
    exists = std::filesystem::exists(path, error);
    if (!error) return true;
    error_message = "无法检查输出路径 " + path.string() + ": " +
                    error.message();
    return false;
}

void removeIfPresent(
    const std::filesystem::path& path,
    std::string* cleanup_error = nullptr) {
    std::error_code error;
    (void)std::filesystem::remove(path, error);
    if (error && cleanup_error != nullptr && cleanup_error->empty()) {
        *cleanup_error = "无法清理 " + path.string() + ": " +
                         error.message();
    }
}

void appendCleanupError(
    std::string& error_message,
    const std::string& cleanup_error) {
    if (cleanup_error.empty()) return;
    if (!error_message.empty()) error_message += "; ";
    error_message += cleanup_error;
}

}  // namespace

SyncInfo makeSyncInfo(
    const volleyball::adapters::hik_mvs::GrabResult& left,
    const volleyball::adapters::hik_mvs::GrabResult& right) {
    SyncInfo sync;
    sync.frame_number_delta =
        static_cast<std::int64_t>(left.frame_number) -
        static_cast<std::int64_t>(right.frame_number);
    sync.frame_counter_delta =
        static_cast<std::int64_t>(left.frame_counter) -
        static_cast<std::int64_t>(right.frame_counter);
    sync.trigger_delta =
        static_cast<std::int64_t>(left.trigger_index) -
        static_cast<std::int64_t>(right.trigger_index);
    sync.timestamp_delta_ns =
        static_cast<std::int64_t>(left.timestamp_us) -
        static_cast<std::int64_t>(right.timestamp_us);
    return sync;
}

bool writeMetadataHeader(std::ostream& metadata) {
    metadata << "pair,file,left_frame_number,right_frame_number,"
             << "left_frame_counter,right_frame_counter,"
             << "left_trigger_index,right_trigger_index,"
             << "left_timestamp_ns,right_timestamp_ns,"
             << "frame_number_delta,frame_counter_delta,"
             << "trigger_delta,timestamp_delta_ns\n";
    metadata.flush();
    return metadata.good();
}

bool writeMetadataRow(
    std::ostream& metadata,
    int pair_index,
    const std::string& file_name,
    const volleyball::adapters::hik_mvs::GrabResult& left,
    const volleyball::adapters::hik_mvs::GrabResult& right,
    const SyncInfo& sync) {
    metadata << pair_index << ',' << file_name << ','
             << left.frame_number << ',' << right.frame_number << ','
             << left.frame_counter << ',' << right.frame_counter << ','
             << left.trigger_index << ',' << right.trigger_index << ','
             << left.timestamp_us << ',' << right.timestamp_us << ','
             << sync.frame_number_delta << ',' << sync.frame_counter_delta << ','
             << sync.trigger_delta << ',' << sync.timestamp_delta_ns << '\n';
    metadata.flush();
    return metadata.good();
}

bool publishPair(
    const std::filesystem::path& final_left,
    const std::filesystem::path& final_right,
    const PairFileWriter& write_left,
    const PairFileWriter& write_right,
    const MetadataCommit& commit_metadata,
    std::string& error_message) {
    error_message.clear();
    if (!write_left || !write_right || !commit_metadata) {
        error_message = "成对输出回调不能为空";
        return false;
    }
    if (final_left.empty() || final_right.empty() ||
        final_left == final_right) {
        error_message = "左右输出路径必须非空且不同";
        return false;
    }

    const auto partial_left = partialPath(final_left);
    const auto partial_right = partialPath(final_right);
    for (const auto& path : {
             final_left, final_right, partial_left, partial_right}) {
        bool exists = false;
        if (!pathExists(path, exists, error_message)) return false;
        if (exists) {
            error_message = "拒绝覆盖已有输出: " + path.string();
            return false;
        }
    }

    try {
        if (!write_left(partial_left) || !write_right(partial_right)) {
            error_message = "图片编码器未完成左右成对写入";
            removeIfPresent(partial_left);
            removeIfPresent(partial_right);
            return false;
        }
    } catch (const std::exception& error) {
        error_message = "图片编码异常: " + std::string(error.what());
        removeIfPresent(partial_left);
        removeIfPresent(partial_right);
        return false;
    } catch (...) {
        error_message = "图片编码发生未知异常";
        removeIfPresent(partial_left);
        removeIfPresent(partial_right);
        return false;
    }

    if (!legacy_nx_tools::publishFileNoReplace(
            partial_left, final_left, error_message)) {
        removeIfPresent(partial_left);
        removeIfPresent(partial_right);
        return false;
    }
    if (!legacy_nx_tools::publishFileNoReplace(
            partial_right, final_right, error_message)) {
        std::string cleanup_error;
        removeIfPresent(final_left, &cleanup_error);
        removeIfPresent(partial_right, &cleanup_error);
        appendCleanupError(error_message, cleanup_error);
        return false;
    }

    bool metadata_ok = false;
    try {
        metadata_ok = commit_metadata();
    } catch (const std::exception& error) {
        error_message = "metadata 提交异常: " + std::string(error.what());
    } catch (...) {
        error_message = "metadata 提交发生未知异常";
    }
    if (!metadata_ok) {
        if (error_message.empty()) error_message = "metadata 提交失败";
        std::string cleanup_error;
        removeIfPresent(final_left, &cleanup_error);
        removeIfPresent(final_right, &cleanup_error);
        appendCleanupError(error_message, cleanup_error);
        return false;
    }
    return true;
}

}  // namespace capture_chessboard
