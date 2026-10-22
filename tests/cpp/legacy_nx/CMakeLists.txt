set(LEGACY_NX_SOURCE_DIR "${PROJECT_SOURCE_DIR}/adapters/legacy_nx/src")
set(VOLLEYBALL_LOGGING_INCLUDE_DIR
  "${PROJECT_SOURCE_DIR}/libs/logging/include")
set(VOLLEYBALL_HIK_MVS_INCLUDE_DIR
  "${PROJECT_SOURCE_DIR}/adapters/hik_mvs/include")
set(VOLLEYBALL_MVS_FAKE_INCLUDE_DIR
  "${PROJECT_SOURCE_DIR}/tests/cpp/fakes/mvs")
set(LEGACY_NX_TOOL_DIR
  "${PROJECT_SOURCE_DIR}/adapters/legacy_nx/tools")

function(add_legacy_nx_test target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE
    "${LEGACY_NX_SOURCE_DIR}"
    "${VOLLEYBALL_HIK_MVS_INCLUDE_DIR}"
    "${VOLLEYBALL_LOGGING_INCLUDE_DIR}"
  )
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

find_package(Threads REQUIRED)

# 抓帧与曝光时间
add_legacy_nx_test(
  async_grab_state_tests
  capture/async_grab_state_tests.cpp
)
target_link_libraries(async_grab_state_tests PRIVATE Threads::Threads)

add_legacy_nx_test(
  exposure_timestamp_tests
  capture/exposure_timestamp_tests.cpp
)

# 实时流水线状态与入口
add_legacy_nx_test(
  async_roi_buffer_state_tests
  pipeline/async_roi_buffer_state_tests.cpp
)

add_legacy_nx_test(
  pipeline_run_state_tests
  pipeline/pipeline_run_state_tests.cpp
)
target_link_libraries(pipeline_run_state_tests PRIVATE Threads::Threads)

add_legacy_nx_test(
  pipeline_streams_tests
  pipeline/pipeline_streams_tests.cpp
)
target_include_directories(pipeline_streams_tests BEFORE PRIVATE fakes/sync)

add_legacy_nx_test(
  main_cli_options_tests
  pipeline/main_cli_options_tests.cpp
  ${LEGACY_NX_SOURCE_DIR}/main_cli_options.cpp
)

# 双目与 VPI/CUDA 桩
add_legacy_nx_test(
  depth_match_contract_tests
  stereo/depth_match_contract_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_match_contract.cpp"
)

add_legacy_nx_test(
  dual_yolo_depth_gpu_tests
  stereo/dual_yolo_depth_gpu_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/dual_yolo_depth_gpu.cpp"
)
target_include_directories(dual_yolo_depth_gpu_tests BEFORE PRIVATE fakes/gpu)

foreach(VPI_FAKE_MAJOR IN ITEMS 3 4)
  set(VPI_PLANE_TEST_TARGET vpi${VPI_FAKE_MAJOR}_plane_access_tests)
  add_legacy_nx_test(
    ${VPI_PLANE_TEST_TARGET}
    stereo/vpi_plane_access_tests.cpp
  )
  target_include_directories(
    ${VPI_PLANE_TEST_TARGET} BEFORE PRIVATE fakes/vpi${VPI_FAKE_MAJOR}
  )
endforeach()

add_legacy_nx_test(
  depth_candidate_selection_tests
  stereo/depth_candidate_selection_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_builder.cpp"
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_builder_feature.cpp"
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_contract.cpp"
)

# 深度融合与时间语义
add_legacy_nx_test(
  hybrid_depth_tests
  fusion/hybrid_depth_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_track.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_candidate_copy.cpp"
)

add_legacy_nx_test(
  fusion_timing_tests
  fusion/fusion_timing_tests.cpp
)

# ROS2 发布边界中的纯逻辑
add_legacy_nx_test(
  nx_observation_quality_tests
  ros/nx_observation_quality_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/ros/nx_observation_quality.cpp"
)

add_legacy_nx_test(
  source_epoch_tests
  ros/source_epoch_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/ros/source_epoch.cpp"
)

# 标定与硬件验收辅助工具
add_legacy_nx_test(
  hik_frame_metadata_probe_args_tests
  tools/hik_frame_metadata_probe/args_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/hik_frame_metadata_probe/args.cpp
)
target_include_directories(
  hik_frame_metadata_probe_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  hik_frame_metadata_probe_camera_session_tests
  tools/hik_frame_metadata_probe/camera_session_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/hik_frame_metadata_probe/camera_session.cpp
)
target_include_directories(
  hik_frame_metadata_probe_camera_session_tests BEFORE PRIVATE
  "${VOLLEYBALL_MVS_FAKE_INCLUDE_DIR}"
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  hik_frame_metadata_probe_pair_audit_tests
  tools/hik_frame_metadata_probe/pair_audit_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/hik_frame_metadata_probe/pair_audit.cpp
)
target_include_directories(
  hik_frame_metadata_probe_pair_audit_tests BEFORE PRIVATE
  "${VOLLEYBALL_MVS_FAKE_INCLUDE_DIR}"
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  capture_chessboard_args_tests
  tools/capture_chessboard/args_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/capture_chessboard/args.cpp
)
target_include_directories(
  capture_chessboard_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  capture_chessboard_output_tests
  tools/capture_chessboard/output_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/capture_chessboard/output.cpp
  ${LEGACY_NX_TOOL_DIR}/common/atomic_file.cpp
)
target_include_directories(
  capture_chessboard_output_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  common_parallel_tasks_tests
  tools/common/parallel_tasks_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/common/parallel_tasks.cpp
)
target_include_directories(
  common_parallel_tasks_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)
target_link_libraries(common_parallel_tasks_tests PRIVATE Threads::Threads)

add_legacy_nx_test(
  stereo_calibrate_args_tests
  tools/stereo_calibrate/args_tests.cpp
  ${LEGACY_NX_TOOL_DIR}/stereo_calibrate/args.cpp
)
target_include_directories(
  stereo_calibrate_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)
