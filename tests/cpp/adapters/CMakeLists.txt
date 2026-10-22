find_package(Threads REQUIRED)

function(add_adapter_test target)
  add_executable(${target} ${ARGN})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_adapter_test(
  capture_v21_stage_schema_tests
  capture_v21_stage_schema_tests.cpp
)
target_link_libraries(capture_v21_stage_schema_tests PRIVATE
  volleyball::capture_v21
)

add_adapter_test(
  mainline_stage_schema_tests
  mainline_stage_schema_tests.cpp
)
target_link_libraries(mainline_stage_schema_tests PRIVATE
  volleyball::capture_v21
)

add_adapter_test(
  mainline_stage_trial_writer_tests
  mainline_stage_trial_writer_tests.cpp
)
target_link_libraries(mainline_stage_trial_writer_tests PRIVATE
  volleyball::capture_v21
)

add_adapter_test(gpio_trigger_tests gpio_trigger_tests.cpp)
target_link_libraries(gpio_trigger_tests PRIVATE volleyball::gpio)

add_adapter_test(
  tensorrt_engine_contract_tests
  tensorrt_engine_contract_tests.cpp
)
target_link_libraries(
  tensorrt_engine_contract_tests PRIVATE volleyball::tensorrt_contract
)

add_adapter_test(
  hik_mvs_camera_tests
  hik_mvs_camera_tests.cpp
)
if(TARGET volleyball::hik_mvs)
  target_link_libraries(hik_mvs_camera_tests PRIVATE volleyball::hik_mvs)
else()
  target_sources(hik_mvs_camera_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/src/hik_stereo_camera_source.cpp
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/src/hikvision_camera.cpp
  )
  target_include_directories(hik_mvs_camera_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/include
  )
endif()
target_include_directories(hik_mvs_camera_tests BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/tests/cpp/fakes/mvs
)
target_link_libraries(hik_mvs_camera_tests PRIVATE
  volleyball::logging
  volleyball::ports
  Threads::Threads
)
