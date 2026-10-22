cmake_minimum_required(VERSION 3.18)
project(stereo_3d_pipeline LANGUAGES CXX CUDA)

set(VOLLEYBALL_CAPTURE_V21_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../capture_v21")
if(NOT EXISTS
   "${VOLLEYBALL_CAPTURE_V21_ADAPTER_DIR}/include/volleyball/adapters/capture_v21/stage_schema.hpp")
  message(FATAL_ERROR
    "Mainline capture_v21 adapter not found next to legacy_nx: "
    "${VOLLEYBALL_CAPTURE_V21_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_GPIO_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../gpio")
if(NOT EXISTS
   "${VOLLEYBALL_GPIO_ADAPTER_DIR}/include/volleyball/adapters/gpio/pwm_trigger.hpp")
  message(FATAL_ERROR
    "Mainline GPIO adapter not found next to legacy_nx: "
    "${VOLLEYBALL_GPIO_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_HIK_MVS_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../hik_mvs")
if(NOT EXISTS
   "${VOLLEYBALL_HIK_MVS_ADAPTER_DIR}/include/volleyball/adapters/hik_mvs/hikvision_camera.hpp")
  message(FATAL_ERROR
    "Mainline hik_mvs adapter not found next to legacy_nx: "
    "${VOLLEYBALL_HIK_MVS_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_TENSORRT_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt")
if(NOT EXISTS
   "${VOLLEYBALL_TENSORRT_ADAPTER_DIR}/include/volleyball/adapters/tensorrt/engine_contract.hpp")
  message(FATAL_ERROR
    "Mainline tensorrt adapter not found next to legacy_nx: "
    "${VOLLEYBALL_TENSORRT_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_CUDA_PREPROCESS_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../cuda_preprocess")
if(NOT EXISTS
   "${VOLLEYBALL_CUDA_PREPROCESS_ADAPTER_DIR}/include/volleyball/adapters/cuda_preprocess/stereo_calibration.hpp")
  message(FATAL_ERROR
    "Mainline CUDA preprocess adapter not found next to legacy_nx: "
    "${VOLLEYBALL_CUDA_PREPROCESS_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_ROS2_ADAPTER_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../ros2")
if(NOT EXISTS
   "${VOLLEYBALL_ROS2_ADAPTER_DIR}/include/volleyball/adapters/ros2/observation_publisher.hpp")
  message(FATAL_ERROR
    "Mainline ROS2 adapter not found next to legacy_nx: "
    "${VOLLEYBALL_ROS2_ADAPTER_DIR}")
endif()
set(VOLLEYBALL_LOGGING_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/logging")
if(NOT EXISTS
   "${VOLLEYBALL_LOGGING_DIR}/include/volleyball/logging/log.hpp")
  message(FATAL_ERROR
    "Mainline logging component not found: ${VOLLEYBALL_LOGGING_DIR}")
endif()
set(VOLLEYBALL_CORE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/core")
set(VOLLEYBALL_PORTS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../ports")
if(NOT EXISTS
   "${VOLLEYBALL_CORE_DIR}/include/volleyball/core/observation.hpp" OR
   NOT EXISTS
   "${VOLLEYBALL_PORTS_DIR}/include/volleyball/ports/camera_source.hpp")
  message(FATAL_ERROR "Mainline core/ports components are incomplete")
endif()
add_subdirectory(
  "${VOLLEYBALL_LOGGING_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/mainline_logging"
  EXCLUDE_FROM_ALL
)
add_subdirectory(
  "${VOLLEYBALL_CORE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/mainline_core"
  EXCLUDE_FROM_ALL
)
add_subdirectory(
  "${VOLLEYBALL_PORTS_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/mainline_ports"
  EXCLUDE_FROM_ALL
)
# ==================== ROS2 ament_cmake ====================
option(REQUIRE_ROS2 "Fail configuration when ROS2 dependencies are unavailable" OFF)
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  set(USE_ROS2 ON)
  find_package(rclcpp REQUIRED)
  find_package(volleyball_interfaces REQUIRED)
  set(VOLLEYBALL_BUILD_ROS2_ADAPTER ON CACHE BOOL "" FORCE)
  add_subdirectory(
    "${VOLLEYBALL_ROS2_ADAPTER_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/mainline_ros2"
    EXCLUDE_FROM_ALL
  )
  message(STATUS "ROS2: ament_cmake found, ROS2 support enabled")
else()
  if(REQUIRE_ROS2)
    message(FATAL_ERROR
      "REQUIRE_ROS2=ON but ament_cmake was not found. Source ROS2 and the "
      "volleyball_interfaces workspace before configuring.")
  endif()
  set(USE_ROS2 OFF)
  message(STATUS "ROS2: not found, standalone build")
endif()

# ==================== 编译标准 ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 编译优化
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Jetson Xavier NX: sm_72  /  Jetson Orin NX: sm_87
# 默认支持两者, 用户可通过 -DCUDA_ARCH 覆盖
set(CUDA_ARCH "72;87" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ==================== 依赖 ====================

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

# VPI (NVIDIA Vision Programming Interface)
find_package(vpi REQUIRED)
message(STATUS "VPI found")

# OpenCV（标定、图像处理与 CUDA 立体匹配）
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d highgui imgcodecs cudastereo cudaimgproc cudafilters)
set(STEREO_OPENCV_LIBS ${OpenCV_LIBS})
set(STEREO_OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV: ${OpenCV_VERSION}")

add_subdirectory(
  "${VOLLEYBALL_CUDA_PREPROCESS_ADAPTER_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/mainline_cuda_preprocess"
  EXCLUDE_FROM_ALL
)

# OpenCV CUDA ORB 是可选后端。缺少 cudafeatures2d 时禁用 ORB 候选，
# 但不阻塞默认 90 Hz 实时管线构建。
find_package(OpenCV QUIET COMPONENTS cudafeatures2d)
if(OpenCV_cudafeatures2d_FOUND)
  set(HAS_OPENCV_CUDAFEATURES2D ON)
  list(APPEND STEREO_OPENCV_LIBS ${OpenCV_LIBS})
  list(REMOVE_DUPLICATES STEREO_OPENCV_LIBS)
  message(STATUS "OpenCV cudafeatures2d: found (CUDA ORB enabled)")
else()
  set(HAS_OPENCV_CUDAFEATURES2D OFF)
  message(STATUS "OpenCV cudafeatures2d not found, CUDA ORB disabled")
endif()

find_package(OpenCV QUIET COMPONENTS cudaoptflow)
if(OpenCV_cudaoptflow_FOUND)
  set(HAS_OPENCV_CUDAOPTFLOW ON)
  list(APPEND STEREO_OPENCV_LIBS ${OpenCV_LIBS})
  list(REMOVE_DUPLICATES STEREO_OPENCV_LIBS)
  message(STATUS "OpenCV cudaoptflow: found (CUDA LK enabled)")
else()
  set(HAS_OPENCV_CUDAOPTFLOW OFF)
  message(STATUS "OpenCV cudaoptflow not found, CUDA LK disabled")
endif()

# Fixstars libSGM 是可选 P2 diagnostic 后端。未安装时保持
# roi_libsgm=unsupported，不阻塞默认实时管线构建。
set(LIBSGM_ROOT "" CACHE PATH "Fixstars libSGM install or source root")
find_path(LIBSGM_INCLUDE_DIR libsgm.h
  HINTS
    ${LIBSGM_ROOT}/include
    /usr/local/include
    /usr/include)
find_library(LIBSGM_LIBRARY
  NAMES sgm libsgm
  HINTS
    ${LIBSGM_ROOT}/lib
    ${LIBSGM_ROOT}/lib64
    ${LIBSGM_ROOT}/build
    ${LIBSGM_ROOT}/build/src
    /usr/local/lib
    /usr/lib
    /usr/lib/aarch64-linux-gnu)
if(LIBSGM_INCLUDE_DIR AND LIBSGM_LIBRARY)
  set(HAS_FIXSTARS_LIBSGM ON)
  message(STATUS "Fixstars libSGM: ${LIBSGM_LIBRARY}")
else()
  set(HAS_FIXSTARS_LIBSGM OFF)
  message(STATUS "Fixstars libSGM not found, roi_libsgm will report unsupported")
endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include)
find_library(NVINFER nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(NVINFER_PLUGIN nvinfer_plugin
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER)
  message(FATAL_ERROR "TensorRT not found")
endif()
message(STATUS "TensorRT: ${NVINFER}")

# TensorRT adapter 的 runtime 目标需要父工程先解析 CUDA/TensorRT 依赖；
# 纯 CXX 根工程仍会在自己的构建中只生成无 SDK 契约目标。
add_subdirectory(
  "${VOLLEYBALL_TENSORRT_ADAPTER_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/mainline_tensorrt"
  EXCLUDE_FROM_ALL
)

# 海康 MVS SDK
option(REQUIRE_HIK "Fail configuration when Hikvision MVS SDK is unavailable" OFF)
set(MVS_SDK_ROOT "/opt/MVS" CACHE PATH "Hikvision MVS SDK root")
find_path(MVS_INCLUDE_DIR MvCameraControl.h
  HINTS
    "${MVS_SDK_ROOT}/include"
    "${MVS_SDK_ROOT}/include/MvCameraControl"
  NO_DEFAULT_PATH)
find_library(MVS_LIBRARY MvCameraControl
  HINTS
    "${MVS_SDK_ROOT}/lib/aarch64"
    "${MVS_SDK_ROOT}/lib/arm64"
    "${MVS_SDK_ROOT}/lib"
  NO_DEFAULT_PATH)
if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
  set(HIK_FOUND ON)
  message(STATUS
    "Hikvision SDK: include=${MVS_INCLUDE_DIR}, library=${MVS_LIBRARY}")
else()
  set(HIK_FOUND OFF)
  if(REQUIRE_HIK)
    message(FATAL_ERROR
      "REQUIRE_HIK=ON but Hikvision MVS SDK header/library was not found "
      "under MVS_SDK_ROOT=${MVS_SDK_ROOT}")
  endif()
  message(WARNING
    "Hikvision MVS SDK header/library not found under "
    "MVS_SDK_ROOT=${MVS_SDK_ROOT}; camera module disabled")
endif()

if(HIK_FOUND)
  set(VOLLEYBALL_BUILD_HIK_MVS ON CACHE BOOL "" FORCE)
  set(
    VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR "${MVS_INCLUDE_DIR}"
    CACHE PATH "" FORCE
  )
  set(
    VOLLEYBALL_HIK_MVS_SDK_LIBRARY "${MVS_LIBRARY}"
    CACHE FILEPATH "" FORCE
  )
  add_subdirectory(
    "${VOLLEYBALL_HIK_MVS_ADAPTER_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/mainline_hik_mvs"
    EXCLUDE_FROM_ALL
  )
endif()

# yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_library(YAML_CPP_LIB yaml-cpp)
  find_path(YAML_CPP_INCLUDE yaml-cpp/yaml.h)
  if(YAML_CPP_LIB AND YAML_CPP_INCLUDE)
    message(STATUS "yaml-cpp: ${YAML_CPP_LIB}")
  else()
    message(FATAL_ERROR "yaml-cpp not found")
  endif()
endif()

# libgpiod (PWM trigger for calibration capture)
find_library(GPIOD_LIB gpiod)
find_path(GPIOD_INCLUDE gpiod.h)
if(GPIOD_LIB AND GPIOD_INCLUDE)
  set(GPIOD_FOUND ON)
  message(STATUS "libgpiod: ${GPIOD_LIB}")
else()
  set(GPIOD_FOUND OFF)
  message(STATUS "libgpiod not found, capture_chessboard PWM disabled")
endif()

# ==================== 包含目录 ====================
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${VOLLEYBALL_CAPTURE_V21_ADAPTER_DIR}/include
  ${VOLLEYBALL_GPIO_ADAPTER_DIR}/include
  ${VOLLEYBALL_HIK_MVS_ADAPTER_DIR}/include
  ${VOLLEYBALL_LOGGING_DIR}/include
  ${STEREO_OPENCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

if(HIK_FOUND)
  include_directories(${MVS_INCLUDE_DIR})
endif()

# ==================== 源文件 ====================
set(PIPELINE_SOURCES
  src/main.cpp
  ${VOLLEYBALL_GPIO_ADAPTER_DIR}/src/pwm_trigger.cpp
  src/main_cli_options.cpp
  src/main_config_loaders.cpp
  src/main_dual_yolo_config.cpp
  src/diagnostics/realtime_debug_dump.cpp
  src/diagnostics/realtime_debug_dump_writer.cpp
  src/main_optional_config_loaders.cpp
  src/pipeline/pipeline.cpp
  src/pipeline/pipeline_async_roi.cpp
  src/pipeline/pipeline_detection.cpp
  src/pipeline/pipeline_depth_modes.cpp
  src/pipeline/pipeline_dual_yolo_match.cpp
  src/pipeline/pipeline_dual_yolo_host.cpp
  src/pipeline/pipeline_feature_jobs.cpp
  src/pipeline/pipeline_image_access.cpp
  src/pipeline/pipeline_lifecycle.cpp
  src/pipeline/pipeline_loops.cpp
  src/pipeline/pipeline_roi_match_helpers.cpp
  src/pipeline/pipeline_roi_stage2.cpp
  src/pipeline/pipeline_stages.cpp
  src/rectify/vpi_rectifier.cpp
  src/stereo/vpi_stereo.cpp
  src/stereo/roi_stereo_matcher.cpp
  src/stereo/depth_candidate_contract.cpp
  src/stereo/depth_candidate_builder.cpp
  src/stereo/depth_candidate_builder_feature.cpp
  src/stereo/depth_match_contract.cpp
  src/stereo/p0p1_soft_gate.cpp
  src/stereo/dual_yolo_depth_gpu.cpp
  src/stereo/roi_feature_validation.cpp
  src/stereo/roi_feature_match_common.cpp
  src/stereo/roi_patch_match_cpu.cpp
  src/stereo/roi_geometry_cpu.cpp
  src/stereo/roi_geometry_circle_cpu.cpp
  src/stereo/roi_feature_match_cpu.cpp
  src/stereo/roi_feature_match_cpu_sparse.cpp
  src/stereo/roi_feature_match_cpu_opencv.cpp
  src/stereo/roi_feature_match_cpu_debug.cpp
  src/stereo/roi_feature_match_gpu.cpp
  src/stereo/roi_feature_match_experimental.cpp
  src/stereo/neural_feature_matcher.cpp
  src/stereo/neural_feature_matcher_xfeat.cpp
  src/stereo/neural_feature_matcher_direct.cpp
  src/stereo/neural_feature_matcher_fused.cpp
  src/fusion/coordinate_3d.cpp
  src/fusion/hybrid_depth_candidate_copy.cpp
  src/fusion/hybrid_depth.cpp
  src/fusion/hybrid_depth_track.cpp
  ${VOLLEYBALL_CAPTURE_V21_ADAPTER_DIR}/src/stage_schema.cpp
  src/capture/capture_v21_stage_recorder.cpp
  src/utils/vpi_image_copy.cpp
)

# CUDA 源文件
set(CUDA_SOURCES
  src/fusion/depth_extract.cu
  src/stereo/roi_stereo_match.cu
  src/stereo/roi_circle_match.cu
  src/stereo/dual_yolo_depth_gpu.cu
  src/stereo/roi_feature_match_gpu_reduce.cu
  src/stereo/roi_feature_match_gpu_profile.cu
  src/stereo/neural_feature_xfeat_gpu_postprocess.cu
  src/stereo/neural_feature_direct_gpu_postprocess.cu
  src/stereo/neural_feature_crop_resize.cu
)

# 相机源文件 (可选)
if(HIK_FOUND)
  add_definitions(-DHIK_CAMERA_ENABLED)
endif()

# ROS2 源文件 (条件编译)
if(USE_ROS2)
  list(APPEND PIPELINE_SOURCES
    src/ros/nx_ball_observation_publisher.cpp
    src/ros/nx_observation_quality.cpp
  )
endif()

# ==================== 目标 ====================
add_executable(stereo_pipeline
  ${PIPELINE_SOURCES}
  ${CUDA_SOURCES}
)

set_target_properties(stereo_pipeline PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ==================== 链接 ====================
target_link_libraries(stereo_pipeline
  volleyball::stereo_calibration
  volleyball::cuda_preprocess_kernels
  volleyball::cuda_stereo_preprocessor
  volleyball::tensorrt_runtime
  # VPI
  vpi
  # TensorRT
  ${NVINFER}
  ${NVINFER_PLUGIN}
  # CUDA
  CUDA::cudart
  CUDA::cuda_driver
  # OpenCV
  ${STEREO_OPENCV_LIBS}
  # yaml-cpp
  yaml-cpp
  # 系统
  ${CMAKE_DL_LIBS}
  pthread
  stdc++fs
)

if(HAS_OPENCV_CUDAFEATURES2D)
  target_compile_definitions(stereo_pipeline PRIVATE HAS_OPENCV_CUDAFEATURES2D)
endif()
if(HAS_OPENCV_CUDAOPTFLOW)
  target_compile_definitions(stereo_pipeline PRIVATE HAS_OPENCV_CUDAOPTFLOW)
endif()
if(HAS_FIXSTARS_LIBSGM)
  target_include_directories(stereo_pipeline PRIVATE ${LIBSGM_INCLUDE_DIR})
  target_link_libraries(stereo_pipeline ${LIBSGM_LIBRARY})
  target_compile_definitions(stereo_pipeline PRIVATE HAS_FIXSTARS_LIBSGM)
endif()

if(HIK_FOUND)
  target_link_libraries(stereo_pipeline volleyball::hik_mvs)
endif()

# PWM trigger: libgpiod 或 sysfs fallback
if(GPIOD_FOUND)
  target_link_libraries(stereo_pipeline ${GPIOD_LIB})
else()
  target_compile_definitions(stereo_pipeline PRIVATE NO_GPIOD)
endif()

# ROS2 链接
if(USE_ROS2)
  target_link_libraries(stereo_pipeline
    volleyball::ros2_publisher
    volleyball::ros2_epoch
  )
  ament_target_dependencies(stereo_pipeline rclcpp volleyball_interfaces)
  target_compile_definitions(stereo_pipeline PRIVATE HAS_ROS2)
endif()

# ==================== 标定与硬件验收工具 ====================
set(VOLLEYBALL_HARDWARE_TOOLS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/hardware")
if(NOT EXISTS "${VOLLEYBALL_HARDWARE_TOOLS_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "Hardware tools are missing: ${VOLLEYBALL_HARDWARE_TOOLS_DIR}")
endif()
add_subdirectory(
  "${VOLLEYBALL_HARDWARE_TOOLS_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/hardware_tools"
)

# ==================== 安装 ====================
install(TARGETS stereo_pipeline DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/stereo_3d_pipeline)

# ==================== 信息汇总 ====================
message(STATUS "")
message(STATUS "=== stereo_3d_pipeline Build Summary ===")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:      ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  VPI:            YES")
message(STATUS "  TensorRT:       YES")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "  Hikvision SDK:  ${HIK_FOUND}")
message(STATUS "  ROS2:           ${USE_ROS2}")
message(STATUS "  yaml-cpp:       YES")
message(STATUS "========================================")

# ==================== ament 收尾 ====================
if(USE_ROS2)
  ament_package()
endif()
