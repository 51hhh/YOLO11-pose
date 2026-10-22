option(
  VOLLEYBALL_BUILD_HIK_MVS
  "构建需要海康 MVS SDK 的双目相机适配器"
  OFF
)

if(NOT VOLLEYBALL_BUILD_HIK_MVS)
  return()
endif()

find_package(Threads REQUIRED)

set(
  VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR ""
  CACHE PATH "海康 MVS SDK 头文件目录"
)
set(
  VOLLEYBALL_HIK_MVS_SDK_LIBRARY ""
  CACHE FILEPATH "海康 MVS SDK 动态库"
)

if(NOT EXISTS "${VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR}/MvCameraControl.h")
  message(FATAL_ERROR
    "VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR 缺少 MvCameraControl.h: "
    "${VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${VOLLEYBALL_HIK_MVS_SDK_LIBRARY}")
  message(FATAL_ERROR
    "VOLLEYBALL_HIK_MVS_SDK_LIBRARY 不存在: "
    "${VOLLEYBALL_HIK_MVS_SDK_LIBRARY}")
endif()
if(NOT TARGET volleyball::logging)
  message(FATAL_ERROR "hik_mvs 需要 volleyball::logging 目标")
endif()
if(NOT TARGET volleyball::ports)
  message(FATAL_ERROR "hik_mvs 需要 volleyball::ports 目标")
endif()

add_library(volleyball_hik_mvs STATIC
  src/hik_stereo_camera_source.cpp
  src/hikvision_camera.cpp
)
add_library(volleyball::hik_mvs ALIAS volleyball_hik_mvs)

target_include_directories(volleyball_hik_mvs
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR}
)
target_compile_features(volleyball_hik_mvs PUBLIC cxx_std_17)
target_compile_options(volleyball_hik_mvs PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
target_link_libraries(volleyball_hik_mvs
  PUBLIC
    volleyball::ports
  PRIVATE
    Threads::Threads
    volleyball::logging
    ${VOLLEYBALL_HIK_MVS_SDK_LIBRARY}
)
