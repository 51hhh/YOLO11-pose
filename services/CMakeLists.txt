find_package(Threads REQUIRED)

add_library(volleyball_services STATIC
  src/managed_stereo_trial.cpp
  src/stereo_detector_executor.cpp
  src/stereo_observation_service.cpp
  src/stereo_pipeline_runner.cpp
)
add_library(volleyball::services ALIAS volleyball_services)

target_include_directories(volleyball_services PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_services PUBLIC
  volleyball::ports
  volleyball::stereo_algorithms
  Threads::Threads
)
target_compile_features(volleyball_services PUBLIC cxx_std_17)
target_compile_options(volleyball_services PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
