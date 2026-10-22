cmake_minimum_required(VERSION 3.24)

project(volleyball_vision VERSION 0.1.0 LANGUAGES CXX)

option(VOLLEYBALL_BUILD_TESTS "构建无硬件单元测试" ON)
option(VOLLEYBALL_BUILD_LEGACY "构建重构前的 NX CUDA 管线" OFF)
option(VOLLEYBALL_BUILD_CUDA_ADAPTERS "构建可独立验证的 CUDA 适配器" OFF)
set(VOLLEYBALL_CUDA_ARCH "87" CACHE STRING "NX 管线 CUDA 架构")
option(VOLLEYBALL_REQUIRE_ROS2 "NX 管线配置时必须找到 ROS2" ON)
option(VOLLEYBALL_REQUIRE_HIK "NX 管线配置时必须找到海康 MVS SDK" ON)
set(VOLLEYBALL_MVS_SDK_ROOT "/opt/MVS" CACHE PATH "海康 MVS SDK 根目录")
set(VOLLEYBALL_LIBSGM_ROOT "" CACHE PATH "可选 Fixstars libSGM 安装目录")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(VOLLEYBALL_BUILD_CUDA_ADAPTERS)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "VOLLEYBALL_BUILD_CUDA_ADAPTERS=ON but no CUDA compiler was found")
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
add_subdirectory(adapters/capture_v21)
add_subdirectory(adapters/gpio)
add_subdirectory(adapters/hik_mvs)
add_subdirectory(adapters/cuda_preprocess)
add_subdirectory(adapters/cuda_depth)
add_subdirectory(adapters/tensorrt)
add_subdirectory(adapters/ros2)
add_subdirectory(adapters/fake)

if(VOLLEYBALL_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(VOLLEYBALL_BUILD_LEGACY)
  include(ExternalProject)
  set(VOLLEYBALL_LEGACY_NX_BINARY_DIR "${CMAKE_BINARY_DIR}/legacy_nx")
  # NX 适配器自己声明 CUDA 语言和硬件 SDK。作为独立外部工程构建，
  # 避免它的全局 include/flags 泄漏进新的纯 CPU 核心层。
  ExternalProject_Add(volleyball_legacy_nx
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/adapters/legacy_nx"
    BINARY_DIR "${VOLLEYBALL_LEGACY_NX_BINARY_DIR}"
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCUDA_ARCH=${VOLLEYBALL_CUDA_ARCH}"
      "-DREQUIRE_ROS2=${VOLLEYBALL_REQUIRE_ROS2}"
      "-DREQUIRE_HIK=${VOLLEYBALL_REQUIRE_HIK}"
      "-DMVS_SDK_ROOT=${VOLLEYBALL_MVS_SDK_ROOT}"
      "-DLIBSGM_ROOT=${VOLLEYBALL_LIBSGM_ROOT}"
    BUILD_COMMAND
      "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      "${VOLLEYBALL_LEGACY_NX_BINARY_DIR}/stereo_pipeline"
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
  )
endif()

