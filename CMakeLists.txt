cmake_minimum_required(VERSION 3.24)

project(volleyball_vision VERSION 0.1.0 LANGUAGES CXX)

option(VOLLEYBALL_BUILD_TESTS "构建无硬件单元测试" ON)
option(VOLLEYBALL_BUILD_LEGACY "构建重构前的 NX CUDA 管线" OFF)

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
  message(FATAL_ERROR
    "VOLLEYBALL_BUILD_LEGACY 尚未接入根工程；请使用 nx-legacy-release preset "
    "在 NX_volleyball/stereo_3d_pipeline 中构建。")
endif()

