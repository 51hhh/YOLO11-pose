if(NOT VOLLEYBALL_BUILD_HARDWARE_TOOLS AND
   NOT VOLLEYBALL_BUILD_STEREO_CALIBRATE)
  return()
endif()

# 三个保留工具只负责实机验收和标定，不参与实时主程序。纯 OpenCV 标定可在
# 普通主机单独构建；相机工具只消费根工程的正式适配器目标。
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
  core
  calib3d
  highgui
  imgcodecs
  imgproc
)

# stereo_calibrate：纯 OpenCV 离线标定，不依赖相机 SDK。
if(VOLLEYBALL_BUILD_STEREO_CALIBRATE OR VOLLEYBALL_BUILD_HARDWARE_TOOLS)
  add_executable(stereo_calibrate
    stereo_calibrate/main.cpp
    stereo_calibrate/args.cpp
    stereo_calibrate/images.cpp
    common/atomic_file.cpp
    common/parallel_tasks.cpp
  )
  set_target_properties(stereo_calibrate PROPERTIES CXX_STANDARD 17)
  target_link_libraries(stereo_calibrate
    ${OpenCV_LIBS}
    Threads::Threads
  )
  install(TARGETS stereo_calibrate DESTINATION bin)
endif()

if(NOT VOLLEYBALL_BUILD_HARDWARE_TOOLS)
  return()
endif()

if(NOT TARGET volleyball::hik_mvs OR NOT TARGET volleyball::gpio)
  message(FATAL_ERROR
    "Hardware tools require volleyball::hik_mvs and volleyball::gpio")
endif()
if(NOT EXISTS "${VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR}/MvCameraControl.h" OR
   NOT EXISTS "${VOLLEYBALL_HIK_MVS_SDK_LIBRARY}")
  message(FATAL_ERROR "Hardware tools require the configured Hikvision MVS SDK")
endif()

# capture_chessboard：使用与生产链一致的海康相机与 GPIO 硬触发。
add_executable(capture_chessboard
  capture_chessboard/main.cpp
  capture_chessboard/args.cpp
  capture_chessboard/output.cpp
  common/atomic_file.cpp
)
set_target_properties(capture_chessboard PROPERTIES CXX_STANDARD 17)
target_link_libraries(capture_chessboard
  ${OpenCV_LIBS}
  volleyball::hik_mvs
  volleyball::gpio
  Threads::Threads
)

# hik_frame_metadata_probe：独立验证帧水印、触发计数与到达顺序。
add_executable(hik_frame_metadata_probe
  hik_frame_metadata_probe/main.cpp
  hik_frame_metadata_probe/args.cpp
  hik_frame_metadata_probe/camera_session.cpp
  hik_frame_metadata_probe/pair_audit.cpp
  common/parallel_tasks.cpp
)
set_target_properties(hik_frame_metadata_probe PROPERTIES CXX_STANDARD 17)
target_include_directories(hik_frame_metadata_probe PRIVATE
  ${VOLLEYBALL_HIK_MVS_SDK_INCLUDE_DIR}
)
target_link_libraries(hik_frame_metadata_probe
  volleyball::gpio
  ${VOLLEYBALL_HIK_MVS_SDK_LIBRARY}
  Threads::Threads
)

install(TARGETS
  capture_chessboard
  hik_frame_metadata_probe
  DESTINATION bin
)
