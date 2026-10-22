add_executable(core_contract_tests
  core_contract_tests.cpp
)
target_link_libraries(core_contract_tests PRIVATE
  volleyball::core
  volleyball::ports
  volleyball::fake_adapters
)
target_compile_features(core_contract_tests PRIVATE cxx_std_17)
target_compile_options(core_contract_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME core_contract_tests COMMAND core_contract_tests)

add_executable(legacy_pair_audit_tests
  legacy_pair_audit_tests.cpp
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src/stereo/depth_match_contract.cpp
)
target_include_directories(legacy_pair_audit_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_compile_features(legacy_pair_audit_tests PRIVATE cxx_std_17)
target_compile_options(legacy_pair_audit_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME legacy_pair_audit_tests COMMAND legacy_pair_audit_tests)

add_executable(trigger_edge_history_tests
  trigger_edge_history_tests.cpp
)
target_include_directories(trigger_edge_history_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_compile_definitions(trigger_edge_history_tests PRIVATE NO_GPIOD=1)
target_compile_features(trigger_edge_history_tests PRIVATE cxx_std_17)
target_compile_options(trigger_edge_history_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME trigger_edge_history_tests COMMAND trigger_edge_history_tests)

add_executable(stereo_observation_service_tests
  stereo_observation_service_tests.cpp
)
target_link_libraries(stereo_observation_service_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)
target_compile_features(stereo_observation_service_tests PRIVATE cxx_std_17)
target_compile_options(stereo_observation_service_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(
  NAME stereo_observation_service_tests
  COMMAND stereo_observation_service_tests
)

find_package(Threads REQUIRED)
add_executable(hik_camera_sync_tests
  hik_camera_sync_tests.cpp
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src/capture/hikvision_camera.cpp
)
target_include_directories(hik_camera_sync_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/mvs_stub
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_link_libraries(hik_camera_sync_tests PRIVATE Threads::Threads)
target_compile_features(hik_camera_sync_tests PRIVATE cxx_std_17)
target_compile_options(hik_camera_sync_tests PRIVATE
  # 旧 logger 宏使用 GNU 的空 __VA_ARGS__ 扩展，不能与 -Wpedantic 同开。
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Werror>
)
add_test(NAME hik_camera_sync_tests COMMAND hik_camera_sync_tests)

add_executable(async_grab_state_tests
  async_grab_state_tests.cpp
)
target_include_directories(async_grab_state_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_link_libraries(async_grab_state_tests PRIVATE Threads::Threads)
target_compile_features(async_grab_state_tests PRIVATE cxx_std_17)
target_compile_options(async_grab_state_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME async_grab_state_tests COMMAND async_grab_state_tests)

add_executable(trt_engine_contract_tests
  trt_engine_contract_tests.cpp
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src/detect/trt_engine_contract.cpp
)
target_include_directories(trt_engine_contract_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_compile_features(trt_engine_contract_tests PRIVATE cxx_std_17)
target_compile_options(trt_engine_contract_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME trt_engine_contract_tests COMMAND trt_engine_contract_tests)

