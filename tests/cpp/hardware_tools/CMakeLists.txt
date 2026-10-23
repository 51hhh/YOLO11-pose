set(HARDWARE_TOOL_DIR "${PROJECT_SOURCE_DIR}/tools/hardware")
set(MVS_FAKE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/tests/cpp/fakes/mvs")

find_package(Threads REQUIRED)

# 生产探针入口在主机构建中没有真实 MVS 动态库可链接，但仍应使用 SDK 桩
# 编译，以免只有到 NX 验收时才发现 main.cpp 的接口或告警错误。
add_library(hik_frame_metadata_probe_main_compile_contract OBJECT
  ${HARDWARE_TOOL_DIR}/hik_frame_metadata_probe/main.cpp
)
target_include_directories(
  hik_frame_metadata_probe_main_compile_contract BEFORE PRIVATE
  "${MVS_FAKE_INCLUDE_DIR}"
  "${HARDWARE_TOOL_DIR}"
)
target_link_libraries(
  hik_frame_metadata_probe_main_compile_contract PRIVATE volleyball::gpio
)
target_compile_features(
  hik_frame_metadata_probe_main_compile_contract PRIVATE cxx_std_17
)
target_compile_options(
  hik_frame_metadata_probe_main_compile_contract PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)

function(add_hardware_tool_test target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE
    "${HARDWARE_TOOL_DIR}"
    "${PROJECT_SOURCE_DIR}/adapters/hik_mvs/include"
  )
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_hardware_tool_test(
  hik_frame_metadata_probe_args_tests
  hik_frame_metadata_probe/args_tests.cpp
  ${HARDWARE_TOOL_DIR}/hik_frame_metadata_probe/args.cpp
)

add_hardware_tool_test(
  hik_frame_metadata_probe_camera_session_tests
  hik_frame_metadata_probe/camera_session_tests.cpp
  ${HARDWARE_TOOL_DIR}/hik_frame_metadata_probe/camera_session.cpp
)
target_include_directories(
  hik_frame_metadata_probe_camera_session_tests BEFORE PRIVATE
  "${MVS_FAKE_INCLUDE_DIR}"
)

add_hardware_tool_test(
  hik_frame_metadata_probe_pair_audit_tests
  hik_frame_metadata_probe/pair_audit_tests.cpp
  ${HARDWARE_TOOL_DIR}/hik_frame_metadata_probe/pair_audit.cpp
)
target_include_directories(
  hik_frame_metadata_probe_pair_audit_tests BEFORE PRIVATE
  "${MVS_FAKE_INCLUDE_DIR}"
)

add_hardware_tool_test(
  capture_chessboard_args_tests
  capture_chessboard/args_tests.cpp
  ${HARDWARE_TOOL_DIR}/capture_chessboard/args.cpp
)

add_hardware_tool_test(
  capture_chessboard_output_tests
  capture_chessboard/output_tests.cpp
  ${HARDWARE_TOOL_DIR}/capture_chessboard/output.cpp
  ${HARDWARE_TOOL_DIR}/common/atomic_file.cpp
)

add_hardware_tool_test(
  common_parallel_tasks_tests
  common/parallel_tasks_tests.cpp
  ${HARDWARE_TOOL_DIR}/common/parallel_tasks.cpp
)
target_link_libraries(common_parallel_tasks_tests PRIVATE Threads::Threads)

add_hardware_tool_test(
  stereo_calibrate_args_tests
  stereo_calibrate/args_tests.cpp
  ${HARDWARE_TOOL_DIR}/stereo_calibrate/args.cpp
)
