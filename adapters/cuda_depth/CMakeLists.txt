add_library(volleyball_cuda_depth_selection STATIC
  src/depth_candidate.cpp
)
add_library(
  volleyball::cuda_depth_selection ALIAS volleyball_cuda_depth_selection
)
target_include_directories(volleyball_cuda_depth_selection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_cuda_depth_selection PUBLIC volleyball::core)
target_compile_features(volleyball_cuda_depth_selection PUBLIC cxx_std_17)
target_compile_options(volleyball_cuda_depth_selection PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
)

if(CMAKE_CUDA_COMPILER)
  add_library(volleyball_cuda_roi_matcher STATIC
    src/roi_matcher.cpp
    src/cuda/roi_matcher.cu
  )
  add_library(volleyball::cuda_roi_matcher ALIAS volleyball_cuda_roi_matcher)
  target_include_directories(volleyball_cuda_roi_matcher PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(volleyball_cuda_roi_matcher PUBLIC
    volleyball::logging
    CUDA::cudart
  )
  target_compile_features(volleyball_cuda_roi_matcher PUBLIC cxx_std_17)
  target_compile_options(volleyball_cuda_roi_matcher PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wconversion>
  )
  set_target_properties(volleyball_cuda_roi_matcher PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()
