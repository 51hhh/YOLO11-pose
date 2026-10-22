# 只定义标定与硬件验收辅助程序。主实时管线目标留在父目录，避免两类入口
# 的源文件、依赖和安装规则继续混在同一个构建文件中。

# stereo_calibrate：纯 OpenCV 离线标定，不依赖相机 SDK。
add_executable(stereo_calibrate
  stereo_calibrate/main.cpp
  stereo_calibrate/args.cpp
  stereo_calibrate/images.cpp
  common/atomic_file.cpp
  common/parallel_tasks.cpp
)
set_target_properties(stereo_calibrate PROPERTIES CXX_STANDARD 17)
target_link_libraries(stereo_calibrate
  ${STEREO_OPENCV_LIBS}
  pthread
  stdc++fs
)

if(HIK_FOUND)
  # capture_chessboard：使用与生产链一致的海康相机与 GPIO 硬触发。
  add_executable(capture_chessboard
    capture_chessboard/main.cpp
    capture_chessboard/args.cpp
    capture_chessboard/output.cpp
    common/atomic_file.cpp
    ${VOLLEYBALL_GPIO_ADAPTER_DIR}/src/pwm_trigger.cpp
  )
  set_target_properties(capture_chessboard PROPERTIES CXX_STANDARD 17)
  target_include_directories(capture_chessboard PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(capture_chessboard
    ${STEREO_OPENCV_LIBS}
    volleyball::hik_mvs
    pthread
    stdc++fs
  )
  if(GPIOD_FOUND)
    target_include_directories(capture_chessboard PRIVATE ${GPIOD_INCLUDE})
    target_link_libraries(capture_chessboard ${GPIOD_LIB})
  else()
    target_compile_definitions(capture_chessboard PRIVATE NO_GPIOD)
  endif()

  # hik_frame_metadata_probe：独立验证帧水印、触发计数与到达顺序。
  add_executable(hik_frame_metadata_probe
    hik_frame_metadata_probe/main.cpp
    hik_frame_metadata_probe/args.cpp
    hik_frame_metadata_probe/camera_session.cpp
    hik_frame_metadata_probe/pair_audit.cpp
    common/parallel_tasks.cpp
    ${VOLLEYBALL_GPIO_ADAPTER_DIR}/src/pwm_trigger.cpp
  )
  set_target_properties(hik_frame_metadata_probe PROPERTIES CXX_STANDARD 17)
  target_include_directories(
    hik_frame_metadata_probe PRIVATE ${CMAKE_SOURCE_DIR}/src
  )
  target_link_libraries(hik_frame_metadata_probe
    ${MVS_LIBRARY}
    pthread
  )
  if(GPIOD_FOUND)
    target_include_directories(
      hik_frame_metadata_probe PRIVATE ${GPIOD_INCLUDE}
    )
    target_link_libraries(hik_frame_metadata_probe ${GPIOD_LIB})
  else()
    target_compile_definitions(hik_frame_metadata_probe PRIVATE NO_GPIOD)
  endif()

  install(TARGETS capture_chessboard hik_frame_metadata_probe DESTINATION bin)
  message(STATUS "Calibration tools: capture_chessboard + stereo_calibrate")
  message(STATUS "Hikvision tools: hik_frame_metadata_probe")
else()
  message(STATUS "Calibration tools: stereo_calibrate only (no HIK SDK)")
endif()

install(TARGETS stereo_calibrate DESTINATION bin)
