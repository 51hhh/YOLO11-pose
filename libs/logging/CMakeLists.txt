add_library(volleyball_logging INTERFACE)
add_library(volleyball::logging ALIAS volleyball_logging)

target_include_directories(volleyball_logging INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(volleyball_logging INTERFACE cxx_std_17)
