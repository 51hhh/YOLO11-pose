cmake_minimum_required(VERSION 3.24)

project(volleyball_vision VERSION 0.1.0 LANGUAGES CXX)

option(VOLLEYBALL_BUILD_TESTS "构建无硬件单元测试" ON)
option(VOLLEYBALL_BUILD_LEGACY "构建重构前的 NX CUDA 管线" OFF)
set(VOLLEYBALL_CUDA_ARCH "87" CACHE STRING "NX 管线 CUDA 架构")
option(VOLLEYBALL_REQUIRE_ROS2 "NX 管线配置时必须找到 ROS2" ON)
set(VOLLEYBALL_MVS_SDK_ROOT "/opt/MVS" CACHE PATH "海康 MVS SDK 根目录")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(libs/core)
add_subdirectory(ports)
add_subdirectory(algorithms/stereo)
add_subdirectory(services)
add_subdirectory(adapters/fake)

if(VOLLEYBALL_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(VOLLEYBALL_BUILD_LEGACY)
  include(ExternalProject)
  # 旧 NX 工程自己声明 CUDA 语言和硬件 SDK。作为独立外部工程构建，
  # 避免它的全局 include/flags 泄漏进新的纯 CPU 核心层。
  ExternalProject_Add(volleyball_legacy_nx
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline"
    BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/build"
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCUDA_ARCH=${VOLLEYBALL_CUDA_ARCH}"
      "-DREQUIRE_ROS2=${VOLLEYBALL_REQUIRE_ROS2}"
      "-DMVS_SDK_ROOT=${VOLLEYBALL_MVS_SDK_ROOT}"
    BUILD_COMMAND
      "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      "${CMAKE_CURRENT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/build/stereo_pipeline"
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD TRUE
  )
endif()

