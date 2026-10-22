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

function(add_legacy_nx_gnu_test target)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PRIVATE "${LEGACY_NX_SOURCE_DIR}")
  target_compile_features(${target} PRIVATE cxx_std_17)
  # 旧 logger 宏使用 GNU 空 __VA_ARGS__ 扩展，暂时不能启用 -Wpedantic。
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror>
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

add_legacy_nx_gnu_test(
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

add_legacy_nx_gnu_test(
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
