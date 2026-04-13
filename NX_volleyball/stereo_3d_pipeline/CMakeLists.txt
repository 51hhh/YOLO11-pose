cmake_minimum_required(VERSION 3.18)
project(stereo_3d_pipeline LANGUAGES CXX CUDA)

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

# OpenCV (标定加载 + 可视化 + 标定工具 + CUDA 立体匹配)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui imgcodecs cudastereo cudaimgproc cudafilters)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

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

# 海康 MVS SDK
set(MVS_SDK_ROOT "/opt/MVS" CACHE PATH "Hikvision MVS SDK root")
if(EXISTS "${MVS_SDK_ROOT}/include/MvCameraControl.h")
  set(HIK_FOUND ON)
  message(STATUS "Hikvision SDK: ${MVS_SDK_ROOT}")
else()
  # 也尝试 third_party 目录
  set(MVS_SDK_ROOT "${CMAKE_SOURCE_DIR}/third_party/MVS_SDK")
  if(EXISTS "${MVS_SDK_ROOT}/include/MvCameraControl.h")
    set(HIK_FOUND ON)
    message(STATUS "Hikvision SDK (third_party): ${MVS_SDK_ROOT}")
  else()
    set(HIK_FOUND OFF)
    message(WARNING "Hikvision SDK not found, camera module disabled")
  endif()
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

# ONNX Runtime (可选, DL 立体匹配模型推理)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS /usr/include/onnxruntime /usr/local/include/onnxruntime
        /usr/include /usr/local/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
  set(ONNXRUNTIME_FOUND ON)
  message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
  set(ONNXRUNTIME_FOUND OFF)
  message(STATUS "ONNX Runtime not found, DL stereo disabled")
endif()

# OpenCV ximgproc (可选, WLS 视差后处理滤波)
# JetPack OpenCV 通常包含 contrib 模块
find_package(OpenCV QUIET COMPONENTS ximgproc)
if(OpenCV_ximgproc_FOUND)
  set(XIMGPROC_FOUND ON)
  message(STATUS "OpenCV ximgproc: found (WLS filter enabled)")
else()
  set(XIMGPROC_FOUND OFF)
  message(STATUS "OpenCV ximgproc not found, WLS filter disabled")
endif()

# ==================== 包含目录 ====================
include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

if(HIK_FOUND)
  include_directories(${MVS_SDK_ROOT}/include)
  link_directories(${MVS_SDK_ROOT}/lib/aarch64)
endif()

# ==================== 源文件 ====================
set(PIPELINE_SOURCES
  src/main.cpp
  src/pipeline/pipeline.cpp
  src/calibration/stereo_calibration.cpp
  src/rectify/vpi_rectifier.cpp
  src/stereo/vpi_stereo.cpp
  src/stereo/roi_stereo_matcher.cpp
  src/detect/trt_detector.cpp
  src/fusion/coordinate_3d.cpp
  src/fusion/hybrid_depth.cpp
  src/fusion/trajectory_predictor.cpp
  src/utils/trajectory_recorder.cpp
  src/track/nanotrack_trt.cpp
  src/track/mixformer_trt.cpp
)

# CUDA 源文件
set(CUDA_SOURCES
  src/detect/detect_preprocess.cu
  src/detect/dfl_postprocess.cu
  src/fusion/depth_extract.cu
  src/stereo/roi_stereo_match.cu
  src/stereo/roi_circle_match.cu
  src/track/crop_resize.cu
)

# 相机源文件 (可选)
if(HIK_FOUND)
  list(APPEND PIPELINE_SOURCES src/capture/hikvision_camera.cpp)
  add_definitions(-DHIK_CAMERA_ENABLED)
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
  # VPI
  vpi
  # TensorRT
  ${NVINFER}
  ${NVINFER_PLUGIN}
  # CUDA
  CUDA::cudart
  CUDA::cuda_driver
  # OpenCV
  ${OpenCV_LIBS}
  # yaml-cpp
  yaml-cpp
  # 系统
  pthread
)

if(HIK_FOUND)
  target_link_libraries(stereo_pipeline MvCameraControl)
endif()

# PWM trigger: libgpiod 或 sysfs fallback
if(GPIOD_FOUND)
  target_link_libraries(stereo_pipeline ${GPIOD_LIB})
else()
  target_compile_definitions(stereo_pipeline PRIVATE NO_GPIOD)
endif()

# ==================== 标定工具 ====================

# stereo_calibrate (纯 OpenCV, 不依赖相机 SDK)
add_executable(stereo_calibrate
  src/calibration/stereo_calibrate.cpp
)
set_target_properties(stereo_calibrate PROPERTIES CXX_STANDARD 17)
target_link_libraries(stereo_calibrate ${OpenCV_LIBS} pthread stdc++fs)

# capture_chessboard (需要海康 SDK; libgpiod 可选)
if(HIK_FOUND)
  add_executable(capture_chessboard
    src/calibration/capture_chessboard.cpp
    src/capture/hikvision_camera.cpp
  )
  set_target_properties(capture_chessboard PROPERTIES CXX_STANDARD 17)
  target_include_directories(capture_chessboard PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(capture_chessboard
    ${OpenCV_LIBS}
    MvCameraControl
    pthread
    stdc++fs
  )
  if(GPIOD_FOUND)
    target_include_directories(capture_chessboard PRIVATE ${GPIOD_INCLUDE})
    target_link_libraries(capture_chessboard ${GPIOD_LIB})
  else()
    target_compile_definitions(capture_chessboard PRIVATE NO_GPIOD)
  endif()
  message(STATUS "Calibration tools: capture_chessboard + stereo_calibrate")
else()
  message(STATUS "Calibration tools: stereo_calibrate only (no HIK SDK)")
endif()

# ==================== 安装 ====================
install(TARGETS stereo_pipeline DESTINATION bin)

# ==================== 深度查看器 ====================
if(HIK_FOUND)
  set(VIEWER_SOURCES
    src/stereo_depth_viewer.cpp
    src/calibration/stereo_calibration.cpp
    src/capture/hikvision_camera.cpp
  )

  # ONNX Runtime DL 立体匹配 (可选)
  if(ONNXRUNTIME_FOUND)
    list(APPEND VIEWER_SOURCES src/stereo/onnx_stereo.cpp)
  endif()

  add_executable(stereo_depth_viewer ${VIEWER_SOURCES})
  set_target_properties(stereo_depth_viewer PROPERTIES CXX_STANDARD 17)
  target_include_directories(stereo_depth_viewer PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(stereo_depth_viewer
    vpi
    CUDA::cudart
    ${OpenCV_LIBS}
    MvCameraControl
    pthread
  )

  # ONNX Runtime
  if(ONNXRUNTIME_FOUND)
    target_include_directories(stereo_depth_viewer PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(stereo_depth_viewer ${ONNXRUNTIME_LIB})
    target_compile_definitions(stereo_depth_viewer PRIVATE HAS_ONNXRUNTIME)
  endif()

  # ximgproc WLS filter
  if(XIMGPROC_FOUND)
    target_compile_definitions(stereo_depth_viewer PRIVATE HAS_XIMGPROC)
  endif()

  if(GPIOD_FOUND)
    target_link_libraries(stereo_depth_viewer ${GPIOD_LIB})
  else()
    target_compile_definitions(stereo_depth_viewer PRIVATE NO_GPIOD)
  endif()
  message(STATUS "stereo_depth_viewer: enabled")
endif()
install(TARGETS stereo_calibrate DESTINATION bin)
if(HIK_FOUND)
  install(TARGETS capture_chessboard DESTINATION bin)
endif()
install(DIRECTORY config/ DESTINATION etc/stereo_3d_pipeline)
install(DIRECTORY scripts/ DESTINATION bin
  FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# ==================== 信息汇总 ====================
message(STATUS "")
message(STATUS "=== stereo_3d_pipeline Build Summary ===")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA arch:      ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  VPI:            YES")
message(STATUS "  TensorRT:       YES")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime:   ${ONNXRUNTIME_FOUND}")
message(STATUS "  ximgproc (WLS): ${XIMGPROC_FOUND}")
message(STATUS "  Hikvision SDK:  ${HIK_FOUND}")
message(STATUS "  yaml-cpp:       YES")
message(STATUS "========================================")
