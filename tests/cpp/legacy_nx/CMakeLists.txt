set(LEGACY_NX_SOURCE_DIR "${PROJECT_SOURCE_DIR}/adapters/legacy_nx/src")
set(VOLLEYBALL_LOGGING_INCLUDE_DIR
  "${PROJECT_SOURCE_DIR}/libs/logging/include")

function(add_legacy_nx_test target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE
    "${LEGACY_NX_SOURCE_DIR}"
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
target_link_libraries(depth_match_contract_tests PRIVATE
  volleyball::tensorrt_types
)

add_legacy_nx_test(
  dual_yolo_depth_gpu_tests
  stereo/dual_yolo_depth_gpu_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/dual_yolo_depth_gpu.cpp"
)
target_include_directories(dual_yolo_depth_gpu_tests BEFORE PRIVATE fakes/gpu)
target_link_libraries(dual_yolo_depth_gpu_tests PRIVATE
  volleyball::tensorrt_types
)

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
target_link_libraries(depth_candidate_selection_tests PRIVATE
  volleyball::tensorrt_types
)

# 深度融合与时间语义
add_legacy_nx_test(
  hybrid_depth_tests
  fusion/hybrid_depth_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_track.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_candidate_copy.cpp"
)
target_link_libraries(hybrid_depth_tests PRIVATE
  volleyball::tensorrt_types
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
