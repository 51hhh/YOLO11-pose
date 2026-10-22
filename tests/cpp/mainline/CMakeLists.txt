function(add_mainline_test target)
  add_executable(${target} ${ARGN})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_mainline_test(core_contract_tests core_contract_tests.cpp)
target_link_libraries(core_contract_tests PRIVATE
  volleyball::core
  volleyball::ports
  volleyball::fake_adapters
)

add_mainline_test(stereo_algorithms_tests stereo_algorithms_tests.cpp)
target_link_libraries(stereo_algorithms_tests PRIVATE
  volleyball::stereo_algorithms
)

add_mainline_test(logging_tests logging_tests.cpp)
target_link_libraries(logging_tests PRIVATE volleyball::logging)

add_mainline_test(
  stereo_observation_service_tests
  stereo_observation_service_tests.cpp
)
target_link_libraries(stereo_observation_service_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)

add_mainline_test(
  stereo_pipeline_runner_tests
  stereo_pipeline_runner_tests.cpp
)
target_link_libraries(stereo_pipeline_runner_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)

add_mainline_test(
  managed_stereo_trial_tests
  managed_stereo_trial_tests.cpp
)
target_link_libraries(managed_stereo_trial_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)

# 主机不链接硬件 SDK，但必须持续编译原生 NX 应用的装配边界，防止只有到
# Jetson 上才发现接口漂移。ROS2 使用最小桩，其余均编译真实公开头。
find_package(OpenCV QUIET COMPONENTS core calib3d)
find_package(OpenSSL QUIET COMPONENTS Crypto)
find_package(yaml-cpp QUIET)
if(OpenCV_FOUND AND OpenSSL_FOUND AND yaml-cpp_FOUND)
  if(TARGET yaml-cpp::yaml-cpp)
    set(VOLLEYBALL_TEST_YAML_TARGET yaml-cpp::yaml-cpp)
  else()
    set(VOLLEYBALL_TEST_YAML_TARGET yaml-cpp)
  endif()
  add_mainline_test(
    nx_mainline_runtime_config_tests
    nx_mainline_runtime_config_tests.cpp
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline/runtime_config.cpp
  )
  target_include_directories(nx_mainline_runtime_config_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline
  )
  target_link_libraries(nx_mainline_runtime_config_tests PRIVATE
    ${VOLLEYBALL_TEST_YAML_TARGET}
  )
  add_library(nx_mainline_compile_contract OBJECT
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline/assembly.cpp
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline/main.cpp
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline/runtime_config.cpp
  )
  target_compile_features(nx_mainline_compile_contract PRIVATE cxx_std_17)
  target_compile_options(nx_mainline_compile_contract PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror -Wconversion>
  )
  target_include_directories(nx_mainline_compile_contract PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/nx_mainline
    ${PROJECT_SOURCE_DIR}/tests/cpp/fakes/ros2
    ${PROJECT_SOURCE_DIR}/adapters/capture_v21/include
    ${PROJECT_SOURCE_DIR}/adapters/cuda_depth/include
    ${PROJECT_SOURCE_DIR}/adapters/cuda_preprocess/include
    ${PROJECT_SOURCE_DIR}/adapters/gpio/include
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/include
    ${PROJECT_SOURCE_DIR}/adapters/ros2/include
    ${PROJECT_SOURCE_DIR}/adapters/tensorrt/include
  )
  target_include_directories(nx_mainline_compile_contract SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(nx_mainline_compile_contract PRIVATE
    volleyball::services
    ${VOLLEYBALL_TEST_YAML_TARGET}
    OpenSSL::Crypto
  )
endif()
