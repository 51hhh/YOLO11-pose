find_package(yaml-cpp REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

foreach(required_target IN ITEMS
    volleyball::capture_stage
    volleyball::cuda_direct_observation_estimator
    volleyball::cuda_stereo_preprocessor
    volleyball::stereo_calibration
    volleyball::gpio
    volleyball::hik_mvs
    volleyball::tensorrt_detector
    volleyball::ros2_publisher
    volleyball::services)
  if(NOT TARGET ${required_target})
    message(FATAL_ERROR
      "NX mainline requires missing target ${required_target}")
  endif()
endforeach()

if(TARGET yaml-cpp::yaml-cpp)
  set(VOLLEYBALL_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(VOLLEYBALL_YAML_CPP_TARGET yaml-cpp)
else()
  message(FATAL_ERROR "yaml-cpp package did not export a supported target")
endif()

add_executable(volleyball_nx_mainline
  assembly.cpp
  main.cpp
  runtime_config.cpp
)
target_include_directories(volleyball_nx_mainline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(volleyball_nx_mainline PRIVATE
  volleyball::capture_stage
  volleyball::cuda_direct_observation_estimator
  volleyball::cuda_stereo_preprocessor
  volleyball::stereo_calibration
  volleyball::gpio
  volleyball::hik_mvs
  volleyball::tensorrt_detector
  volleyball::ros2_publisher
  volleyball::services
  ${VOLLEYBALL_YAML_CPP_TARGET}
  OpenSSL::Crypto
)
target_compile_features(volleyball_nx_mainline PRIVATE cxx_std_17)
target_compile_options(volleyball_nx_mainline PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)
