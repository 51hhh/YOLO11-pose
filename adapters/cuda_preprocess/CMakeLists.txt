find_package(OpenCV QUIET COMPONENTS core calib3d)
if(OpenCV_FOUND)
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
endif()

if(CMAKE_CUDA_COMPILER)
  add_library(volleyball_cuda_preprocess_kernels STATIC
    src/cuda/preprocess_kernels.cu
  )
  add_library(
    volleyball::cuda_preprocess_kernels
    ALIAS volleyball_cuda_preprocess_kernels
  )
  target_include_directories(volleyball_cuda_preprocess_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(volleyball_cuda_preprocess_kernels PUBLIC
    CUDA::cudart
  )
  set_target_properties(volleyball_cuda_preprocess_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  add_library(volleyball_cuda_stereo_preprocessor STATIC
    src/cuda_stereo_preprocessor.cpp
  )
  add_library(
    volleyball::cuda_stereo_preprocessor
    ALIAS volleyball_cuda_stereo_preprocessor
  )
  target_include_directories(volleyball_cuda_stereo_preprocessor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(volleyball_cuda_stereo_preprocessor PUBLIC
    volleyball::cuda_preprocess_kernels
    volleyball::ports
    CUDA::cudart
  )
  target_compile_features(
    volleyball_cuda_stereo_preprocessor PUBLIC cxx_std_17
  )
  target_compile_options(volleyball_cuda_stereo_preprocessor PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
  )
endif()
