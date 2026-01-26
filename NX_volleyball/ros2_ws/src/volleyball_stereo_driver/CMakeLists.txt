cmake_minimum_required(VERSION 3.8)
project(volleyball_stereo_driver)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 查找 libgpiod
pkg_check_modules(GPIOD REQUIRED libgpiod)

# CUDA (必需，用于GPU加速预处理)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION}")
  include_directories(${CUDA_INCLUDE_DIRS})
  # Jetson Orin NX: sm_87
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_87,code=sm_87 -std=c++17")
  message(STATUS "CUDA flags: ${CUDA_NVCC_FLAGS}")
else()
  message(FATAL_ERROR "CUDA is required for GPU-accelerated preprocessing")
endif()

# TensorRT (必需)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu
  PATHS /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/aarch64-linux-gnu
  PATHS /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT is required for YOLO inference")
else()
  message(STATUS "TensorRT found: ${TENSORRT_LIBRARY}")
  set(BUILD_TRACKER ON)
endif()

# 海康 MVS SDK
set(HIK_SDK_DIR "/opt/MVS")
if(EXISTS ${HIK_SDK_DIR})
  set(BUILD_HIK ON)
  message(STATUS "Hikvision SDK found: ${HIK_SDK_DIR}")
else()
  message(WARNING "Hikvision SDK not found at ${HIK_SDK_DIR}")
  set(BUILD_HIK OFF)
endif()

# 包含目录
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${GPIOD_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${HIK_SDK_DIR}/include
)

link_directories(
  ${HIK_SDK_DIR}/lib/aarch64
)

# ==================== 排球追踪节点 (All-in-One: PWM + 相机 + YOLO + 立体 + 追踪) ====================
if(BUILD_HIK AND BUILD_TRACKER)
  message(STATUS "Building volleyball_tracker_node with CUDA GPU acceleration")
  
  # 使用cuda_add_executable支持CUDA kernel
  cuda_add_executable(volleyball_tracker_node
    src/volleyball_tracker_node.cpp
    src/high_precision_pwm.cpp
    src/hik_camera_wrapper.cpp
    src/yolo_detector.cpp
    src/yolo_preprocessor.cu  # CUDA kernel文件
    src/stereo_matcher.cpp
    src/kalman_filter_3d.cpp
    src/roi_manager.cpp
  )

  ament_target_dependencies(volleyball_tracker_node
    rclcpp
    std_msgs
    geometry_msgs
    sensor_msgs
    ament_index_cpp
  )

  target_link_libraries(volleyball_tracker_node
    ${OpenCV_LIBS}
    ${GPIOD_LIBRARIES}
    ${TENSORRT_LIBRARY}
    MvCameraControl
    ${CUDA_LIBRARIES}  # CUDA库
    cudart
    pthread
  )

  install(TARGETS volleyball_tracker_node
    DESTINATION lib/${PROJECT_NAME}
  )
else()
  message(WARNING "volleyball_tracker_node will not be built (missing HIK SDK or TensorRT)")
endif()

# ==================== 可视化节点 (调试用，不依赖 TensorRT 和相机) ====================
add_executable(volleyball_visualizer_node
  src/volleyball_visualizer_node.cpp
)

ament_target_dependencies(volleyball_visualizer_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
)

target_link_libraries(volleyball_visualizer_node
  ${OpenCV_LIBS}
  pthread
)

install(TARGETS volleyball_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# ==================== 安装 ====================
install(DIRECTORY
  config
  scripts
  model
  calibration
  DESTINATION share/${PROJECT_NAME}
)

# 安装脚本并设置可执行权限
install(PROGRAMS
  scripts/start_pwm_node.sh
  scripts/start_system.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)

# ==================== 测试 ====================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
