find_package(OpenCV QUIET COMPONENTS core calib3d)
if(NOT OpenCV_FOUND)
  return()
endif()

add_library(volleyball_stereo_calibration STATIC
  src/stereo_calibration.cpp
)
add_library(
  volleyball::stereo_calibration ALIAS volleyball_stereo_calibration
)
target_include_directories(volleyball_stereo_calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(volleyball_stereo_calibration PUBLIC
  volleyball::logging
  ${OpenCV_LIBS}
)
target_compile_features(volleyball_stereo_calibration PUBLIC cxx_std_17)
target_compile_options(volleyball_stereo_calibration PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
