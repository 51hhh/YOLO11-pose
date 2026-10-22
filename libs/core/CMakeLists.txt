add_library(volleyball_core STATIC
  src/clock_mapping.cpp
  src/domain_validation.cpp
)

add_library(volleyball::core ALIAS volleyball_core)

target_include_directories(volleyball_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(volleyball_core PUBLIC cxx_std_17)
target_compile_options(volleyball_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

