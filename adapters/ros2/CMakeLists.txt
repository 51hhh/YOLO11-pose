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

option(
  VOLLEYBALL_BUILD_ROS2_ADAPTER
  "构建依赖 rclcpp 和 volleyball_interfaces 的观测发布器"
  OFF
)

if(NOT VOLLEYBALL_BUILD_ROS2_ADAPTER)
  return()
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(volleyball_interfaces REQUIRED)

add_library(volleyball_ros2_publisher STATIC
  src/observation_publisher.cpp
)
add_library(volleyball::ros2_publisher ALIAS volleyball_ros2_publisher)
target_include_directories(volleyball_ros2_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_ros2_publisher PUBLIC
  volleyball::ros2_mapping
  volleyball::ports
)
target_compile_features(volleyball_ros2_publisher PUBLIC cxx_std_17)
target_compile_options(volleyball_ros2_publisher PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
ament_target_dependencies(
  volleyball_ros2_publisher
  rclcpp
  volleyball_interfaces
)
