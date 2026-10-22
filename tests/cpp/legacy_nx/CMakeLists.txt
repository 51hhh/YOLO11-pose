set(LEGACY_NX_SOURCE_DIR "${PROJECT_SOURCE_DIR}/adapters/legacy_nx/src")

function(add_legacy_nx_test target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE "${LEGACY_NX_SOURCE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

find_package(Threads REQUIRED)

add_legacy_nx_test(
  depth_match_contract_tests
  depth_match_contract_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_match_contract.cpp"
)

add_legacy_nx_test(trigger_edge_history_tests trigger_edge_history_tests.cpp)
target_compile_definitions(trigger_edge_history_tests PRIVATE NO_GPIOD=1)

add_legacy_nx_test(
  hik_camera_sync_tests
  hik_camera_sync_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/capture/hikvision_camera.cpp"
)
target_include_directories(hik_camera_sync_tests BEFORE PRIVATE fakes/mvs)
target_link_libraries(hik_camera_sync_tests PRIVATE Threads::Threads)

add_legacy_nx_test(async_grab_state_tests async_grab_state_tests.cpp)
target_link_libraries(async_grab_state_tests PRIVATE Threads::Threads)

add_legacy_nx_test(
  async_roi_buffer_state_tests
  async_roi_buffer_state_tests.cpp
)

add_legacy_nx_test(pipeline_run_state_tests pipeline_run_state_tests.cpp)
target_link_libraries(pipeline_run_state_tests PRIVATE Threads::Threads)

add_legacy_nx_test(pipeline_streams_tests pipeline_streams_tests.cpp)
target_include_directories(pipeline_streams_tests BEFORE PRIVATE fakes/sync)

add_legacy_nx_test(
  dual_yolo_depth_gpu_tests
  dual_yolo_depth_gpu_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/dual_yolo_depth_gpu.cpp"
)
target_include_directories(dual_yolo_depth_gpu_tests BEFORE PRIVATE fakes/gpu)

foreach(VPI_FAKE_MAJOR IN ITEMS 3 4)
  set(VPI_PLANE_TEST_TARGET vpi${VPI_FAKE_MAJOR}_plane_access_tests)
  add_legacy_nx_test(
    ${VPI_PLANE_TEST_TARGET}
    vpi_plane_access_tests.cpp
  )
  target_include_directories(
    ${VPI_PLANE_TEST_TARGET} BEFORE PRIVATE fakes/vpi${VPI_FAKE_MAJOR}
  )
endforeach()

add_legacy_nx_test(
  trt_engine_contract_tests
  trt_engine_contract_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/detect/trt_engine_contract.cpp"
)

add_legacy_nx_test(
  hybrid_depth_tests
  hybrid_depth_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_track.cpp"
  "${LEGACY_NX_SOURCE_DIR}/fusion/hybrid_depth_candidate_copy.cpp"
)

add_legacy_nx_test(
  depth_candidate_selection_tests
  depth_candidate_selection_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_builder.cpp"
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_builder_feature.cpp"
  "${LEGACY_NX_SOURCE_DIR}/stereo/depth_candidate_contract.cpp"
)

add_legacy_nx_test(fusion_timing_tests fusion_timing_tests.cpp)

add_legacy_nx_test(exposure_timestamp_tests exposure_timestamp_tests.cpp)

add_legacy_nx_test(
  main_cli_options_tests
  main_cli_options_tests.cpp
  ${LEGACY_NX_SOURCE_DIR}/main_cli_options.cpp
)

add_legacy_nx_test(
  hik_metadata_probe_args_tests
  hik_metadata_probe_args_tests.cpp
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx/tools/hik_frame_metadata_probe/args.cpp
)
target_include_directories(
  hik_metadata_probe_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  capture_chessboard_args_tests
  capture_chessboard_args_tests.cpp
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx/tools/capture_chessboard/args.cpp
)
target_include_directories(
  capture_chessboard_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  stereo_calibrate_args_tests
  stereo_calibrate_args_tests.cpp
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx/tools/stereo_calibrate/args.cpp
)
target_include_directories(
  stereo_calibrate_args_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/adapters/legacy_nx
)

add_legacy_nx_test(
  nx_observation_quality_tests
  nx_observation_quality_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/ros/nx_observation_quality.cpp"
)

add_legacy_nx_test(
  source_epoch_tests
  source_epoch_tests.cpp
  "${LEGACY_NX_SOURCE_DIR}/ros/source_epoch.cpp"
)
