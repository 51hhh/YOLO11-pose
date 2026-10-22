add_library(volleyball_stereo_algorithms STATIC
  src/association.cpp
  src/rectified_center_estimator.cpp
  src/triangulation.cpp
)
add_library(volleyball::stereo_algorithms ALIAS volleyball_stereo_algorithms)

target_include_directories(volleyball_stereo_algorithms PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_stereo_algorithms PUBLIC volleyball::ports)
target_compile_features(volleyball_stereo_algorithms PUBLIC cxx_std_17)
target_compile_options(volleyball_stereo_algorithms PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
