add_library(volleyball_capture_stage STATIC
  src/mainline_stage_schema.cpp
  src/mainline_stage_trial_writer.cpp
)
add_library(volleyball::capture_stage ALIAS volleyball_capture_stage)

target_include_directories(volleyball_capture_stage PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_capture_stage PUBLIC volleyball::ports)
target_compile_features(volleyball_capture_stage PUBLIC cxx_std_17)
target_compile_options(volleyball_capture_stage PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
