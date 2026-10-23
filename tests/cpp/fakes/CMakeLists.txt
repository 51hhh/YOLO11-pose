add_library(volleyball_test_fakes STATIC
  mainline/src/in_memory_trial_writer.cpp
  mainline/src/scripted_camera_source.cpp
)
add_library(volleyball::test_fakes ALIAS volleyball_test_fakes)

target_include_directories(volleyball_test_fakes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mainline/include>
)
target_link_libraries(volleyball_test_fakes PUBLIC volleyball::ports)
target_compile_features(volleyball_test_fakes PUBLIC cxx_std_17)
target_compile_options(volleyball_test_fakes PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

add_library(volleyball_tensorrt_runtime_fake STATIC
  tensorrt_runtime/src/tensorrt_runtime.cpp
)
add_library(
  volleyball::tensorrt_runtime_fake ALIAS volleyball_tensorrt_runtime_fake
)

target_include_directories(volleyball_tensorrt_runtime_fake
  BEFORE
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_runtime/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt
    ${PROJECT_SOURCE_DIR}/adapters/tensorrt/include
)
target_link_libraries(volleyball_tensorrt_runtime_fake PUBLIC
  volleyball::tensorrt_contract
  volleyball::tensorrt_postprocess
)
target_compile_features(volleyball_tensorrt_runtime_fake PUBLIC cxx_std_17)
target_compile_options(volleyball_tensorrt_runtime_fake PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
