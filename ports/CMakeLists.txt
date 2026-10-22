add_library(volleyball_ports INTERFACE)
add_library(volleyball::ports ALIAS volleyball_ports)

target_include_directories(volleyball_ports INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(volleyball_ports INTERFACE volleyball::core)
target_compile_features(volleyball_ports INTERFACE cxx_std_17)

