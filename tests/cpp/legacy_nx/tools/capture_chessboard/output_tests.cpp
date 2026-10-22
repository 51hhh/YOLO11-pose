#include "tools/capture_chessboard/output.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

std::filesystem::path uniqueTemporaryPath() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    return std::filesystem::temp_directory_path() /
           ("capture_chessboard_output_tests_" + std::to_string(suffix));
}

bool writeTextFile(
    const std::filesystem::path& path,
    const std::string& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    output.flush();
    return output.good();
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void testSyncAndMetadataContract() {
    volleyball::adapters::hik_mvs::GrabResult left;
    left.frame_number = 20;
    left.frame_counter = 30;
    left.trigger_index = 40;
    left.device_timestamp_ns = 5000;
    volleyball::adapters::hik_mvs::GrabResult right;
    right.frame_number = 18;
    right.frame_counter = 29;
    right.trigger_index = 37;
    right.device_timestamp_ns = 4500;

    const auto sync = capture_chessboard::makeSyncInfo(left, right);
    check(sync.frame_number_delta == 2, "frame-number delta must be signed");
    check(sync.frame_counter_delta == 1, "frame-counter delta must be signed");
    check(sync.trigger_delta == 3, "trigger delta must be signed");
    check(sync.timestamp_delta_ns == 500, "timestamp delta must remain nanoseconds");

    std::ostringstream output;
    check(capture_chessboard::writeMetadataHeader(output),
          "metadata header must write to a healthy stream");
    check(capture_chessboard::writeMetadataRow(
              output, 7, "0007.png", left, right, sync),
          "metadata row must write to a healthy stream");
    const std::string value = output.str();
    check(value.find(
              "left_device_timestamp_ns,right_device_timestamp_ns") !=
              std::string::npos,
          "metadata header must state timestamp units");
    check(value.find("7,0007.png,20,18,30,29,40,37,5000,4500,2,1,3,500\n") !=
              std::string::npos,
          "metadata row must preserve frame identity and timing evidence");

    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    check(!capture_chessboard::writeMetadataHeader(failed),
          "metadata write failure must be observable");
}

void testPairPublishIsAllOrNothing() {
    const auto root = uniqueTemporaryPath();
    const auto left_dir = root / "left";
    const auto right_dir = root / "right";
    std::error_code error;
    std::filesystem::create_directories(left_dir, error);
    check(!error, "left test directory must be created");
    error.clear();
    std::filesystem::create_directories(right_dir, error);
    check(!error, "right test directory must be created");

    const auto left = left_dir / "0001.png";
    const auto right = right_dir / "0001.png";
    std::string message;
    bool metadata_committed = false;
    check(capture_chessboard::publishPair(
              left,
              right,
              [](const auto& path) { return writeTextFile(path, "left"); },
              [](const auto& path) { return writeTextFile(path, "right"); },
              [&]() {
                  metadata_committed = true;
                  return true;
              },
              message),
          "healthy pair and metadata must publish");
    check(metadata_committed, "metadata must commit after both images publish");
    check(readTextFile(left) == "left" && readTextFile(right) == "right",
          "published images must preserve writer output");
    check(!std::filesystem::exists(left_dir / "0001.partial.png") &&
              !std::filesystem::exists(right_dir / "0001.partial.png"),
          "successful publish must not leave temporary images");

    bool writer_called = false;
    check(!capture_chessboard::publishPair(
              left,
              right,
              [&](const auto&) {
                  writer_called = true;
                  return true;
              },
              [&](const auto&) {
                  writer_called = true;
                  return true;
              },
              []() { return true; },
              message),
          "existing pair must be refused");
    check(!writer_called, "overwrite refusal must happen before encoding");
    check(readTextFile(left) == "left" && readTextFile(right) == "right",
          "overwrite refusal must preserve existing images");

    const auto failed_left = left_dir / "0002.png";
    const auto failed_right = right_dir / "0002.png";
    check(!capture_chessboard::publishPair(
              failed_left,
              failed_right,
              [](const auto& path) { return writeTextFile(path, "left"); },
              [](const auto&) { return false; },
              []() { return true; },
              message),
          "one-sided encoder failure must reject the pair");
    check(!std::filesystem::exists(failed_left) &&
              !std::filesystem::exists(failed_right) &&
              !std::filesystem::exists(left_dir / "0002.partial.png"),
          "one-sided failure must leave no visible or temporary pair");

    const auto metadata_left = left_dir / "0003.png";
    const auto metadata_right = right_dir / "0003.png";
    check(!capture_chessboard::publishPair(
              metadata_left,
              metadata_right,
              [](const auto& path) { return writeTextFile(path, "left"); },
              [](const auto& path) { return writeTextFile(path, "right"); },
              []() { return false; },
              message),
          "metadata failure must reject the pair");
    check(!std::filesystem::exists(metadata_left) &&
              !std::filesystem::exists(metadata_right),
          "metadata failure must roll back both published images");

    const auto exception_left = left_dir / "0004.png";
    const auto exception_right = right_dir / "0004.png";
    check(!capture_chessboard::publishPair(
              exception_left,
              exception_right,
              [](const auto&) -> bool {
                  throw std::runtime_error("encoder failure");
              },
              [](const auto&) { return true; },
              []() { return true; },
              message),
          "encoder exception must become a normal failure result");
    check(!std::filesystem::exists(exception_left) &&
              !std::filesystem::exists(exception_right),
          "encoder exception must leave no published pair");

    const auto race_left = left_dir / "0005.png";
    const auto race_right = right_dir / "0005.png";
    check(!capture_chessboard::publishPair(
              race_left,
              race_right,
              [&](const auto& path) {
                  return writeTextFile(path, "new-left") &&
                         writeTextFile(race_left, "external-writer");
              },
              [](const auto& path) { return writeTextFile(path, "new-right"); },
              []() { return true; },
              message),
          "target created during encoding must not be overwritten");
    check(readTextFile(race_left) == "external-writer",
          "no-replace publish must preserve a racing writer's file");
    check(!std::filesystem::exists(race_right) &&
              !std::filesystem::exists(left_dir / "0005.partial.png") &&
              !std::filesystem::exists(right_dir / "0005.partial.png"),
          "racing target failure must clean only this transaction's files");

    std::filesystem::remove_all(root, error);
    check(!error, "test output directory must be removed");
}

}  // namespace

int main() {
    testSyncAndMetadataContract();
    testPairPublishIsAllOrNothing();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "capture_chessboard output tests passed\n";
    return EXIT_SUCCESS;
}
