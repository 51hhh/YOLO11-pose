cmake_minimum_required(VERSION 3.24)

project(volleyball_vision VERSION 0.1.0 LANGUAGES CXX)

option(VOLLEYBALL_BUILD_TESTS "构建无硬件单元测试" ON)
option(VOLLEYBALL_BUILD_CUDA_ADAPTERS "构建可独立验证的 CUDA 适配器" OFF)
option(VOLLEYBALL_BUILD_NX_MAINLINE "构建原生 Jetson Orin NX 主线应用" OFF)
option(VOLLEYBALL_BUILD_HARDWARE_TOOLS "构建相机验收与双目标定工具" OFF)
set(VOLLEYBALL_CUDA_ARCH "87" CACHE STRING "NX 管线 CUDA 架构")
set(VOLLEYBALL_MVS_SDK_ROOT "/opt/MVS" CACHE PATH "海康 MVS SDK 根目录")

if(VOLLEYBALL_BUILD_NX_MAINLINE)
  set(VOLLEYBALL_BUILD_CUDA_ADAPTERS ON CACHE BOOL "" FORCE)
  set(VOLLEYBALL_BUILD_HIK_MVS ON CACHE BOOL "" FORCE)
  set(VOLLEYBALL_GPIO_WITH_GPIOD ON CACHE BOOL "" FORCE)
  set(VOLLEYBALL_BUILD_ROS2_ADAPTER ON CACHE BOOL "" FORCE)
  set(VOLLEYBALL_BUILD_HARDWARE_TOOLS ON CACHE BOOL "" FORCE)
endif()

if(VOLLEYBALL_BUILD_NX_MAINLINE OR VOLLEYBALL_BUILD_HARDWARE_TOOLS)
  set(VOLLEYBALL_BUILD_HIK_MVS ON CACHE BOOL "" FORCE)

  find_path(VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR MvCameraControl.h
    HINTS
      "${VOLLEYBALL_MVS_SDK_ROOT}/include"
      "${VOLLEYBALL_MVS_SDK_ROOT}/include/MvCameraControl"
    REQUIRED
  )
  find_library(VOLLEYBALL_HIK_MVS_SDK_LIBRARY MvCameraControl
    HINTS
      "${VOLLEYBALL_MVS_SDK_ROOT}/lib/aarch64"
      "${VOLLEYBALL_MVS_SDK_ROOT}/lib/arm64"
      "${VOLLEYBALL_MVS_SDK_ROOT}/lib"
    REQUIRED
  )
endif()

if(VOLLEYBALL_BUILD_NX_MAINLINE)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include
    REQUIRED
  )
  find_library(NVINFER nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib
    REQUIRED
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(VOLLEYBALL_BUILD_CUDA_ADAPTERS)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "VOLLEYBALL_BUILD_CUDA_ADAPTERS=ON, but CMake could not compile a "
      "CUDA compiler probe. nvcc may be missing, or CUDA may be incompatible "
      "with the host compiler/C library. Inspect "
      "CMakeFiles/CMakeConfigureLog.yaml (CMake >= 3.26) or "
      "CMakeFiles/CMakeError.log.")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "${VOLLEYBALL_CUDA_ARCH}")
endif()

add_subdirectory(libs/core)
add_subdirectory(libs/logging)
add_subdirectory(ports)
add_subdirectory(algorithms/stereo)
add_subdirectory(services)
add_subdirectory(adapters/capture_stage)
add_subdirectory(adapters/gpio)
add_subdirectory(adapters/hik_mvs)
add_subdirectory(adapters/cuda_preprocess)
add_subdirectory(adapters/cuda_depth)
add_subdirectory(adapters/tensorrt)
add_subdirectory(adapters/ros2)
add_subdirectory(adapters/fake)
add_subdirectory(tools/hardware)

if(VOLLEYBALL_BUILD_NX_MAINLINE)
  add_subdirectory(apps/nx_mainline)
endif()

if(VOLLEYBALL_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

