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
