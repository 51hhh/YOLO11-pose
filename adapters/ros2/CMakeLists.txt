add_library(volleyball_ros2_mapping STATIC
  src/observation_message.cpp
)
add_library(volleyball::ros2_mapping ALIAS volleyball_ros2_mapping)

target_include_directories(volleyball_ros2_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_ros2_mapping PUBLIC volleyball::ports)
target_compile_features(volleyball_ros2_mapping PUBLIC cxx_std_17)
target_compile_options(volleyball_ros2_mapping PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
