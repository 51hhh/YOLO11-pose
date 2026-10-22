add_library(volleyball_tensorrt_types INTERFACE)
add_library(volleyball::tensorrt_types ALIAS volleyball_tensorrt_types)
target_include_directories(volleyball_tensorrt_types INTERFACE include)
target_compile_features(volleyball_tensorrt_types INTERFACE cxx_std_17)

add_library(volleyball_tensorrt_contract
  src/engine_contract.cpp
)
add_library(volleyball::tensorrt_contract ALIAS volleyball_tensorrt_contract)

target_include_directories(volleyball_tensorrt_contract PUBLIC include)
target_compile_features(volleyball_tensorrt_contract PUBLIC cxx_std_17)

add_library(volleyball_tensorrt_postprocess
  src/postprocess.cpp
)
add_library(volleyball::tensorrt_postprocess ALIAS volleyball_tensorrt_postprocess)

target_include_directories(volleyball_tensorrt_postprocess PUBLIC include)
target_link_libraries(volleyball_tensorrt_postprocess PUBLIC
  volleyball::tensorrt_types
)
target_compile_features(volleyball_tensorrt_postprocess PUBLIC cxx_std_17)

add_library(volleyball_tensorrt_mapping
  src/detector_mapping.cpp
)
add_library(volleyball::tensorrt_mapping ALIAS volleyball_tensorrt_mapping)

target_include_directories(volleyball_tensorrt_mapping PUBLIC include)
target_link_libraries(volleyball_tensorrt_mapping PUBLIC
  volleyball::tensorrt_postprocess
  volleyball::ports
)
target_compile_features(volleyball_tensorrt_mapping PUBLIC cxx_std_17)

# 根工程只启用 CXX，因此主机测试不会看到这个目标。NX 外部工程已经启用
# CUDA，届时才构建硬件 kernel，避免把 CUDA SDK 泄漏进平台无关层。
if(CMAKE_CUDA_COMPILER)
  add_library(volleyball_tensorrt_cuda
    src/cuda/preprocess.cu
    src/cuda/dfl_decode.cu
  )
  add_library(volleyball::tensorrt_cuda ALIAS volleyball_tensorrt_cuda)

  target_include_directories(volleyball_tensorrt_cuda PUBLIC include)
  set_target_properties(volleyball_tensorrt_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )

  if(TENSORRT_INCLUDE_DIR AND NVINFER AND TARGET volleyball::logging)
    add_library(volleyball_tensorrt_runtime
      src/runtime.cpp
    )
    add_library(volleyball::tensorrt_runtime ALIAS volleyball_tensorrt_runtime)

    target_include_directories(volleyball_tensorrt_runtime PUBLIC
      include
      "${TENSORRT_INCLUDE_DIR}"
    )
    target_link_libraries(volleyball_tensorrt_runtime PUBLIC
      volleyball::tensorrt_contract
      volleyball::tensorrt_postprocess
      volleyball::tensorrt_cuda
      volleyball::logging
      "${NVINFER}"
      CUDA::cudart
    )
    target_compile_features(volleyball_tensorrt_runtime PUBLIC cxx_std_17)

    add_library(volleyball_tensorrt_detector
      src/detector_adapter.cpp
    )
    add_library(volleyball::tensorrt_detector ALIAS volleyball_tensorrt_detector)

    target_include_directories(volleyball_tensorrt_detector PUBLIC include)
    target_link_libraries(volleyball_tensorrt_detector PUBLIC
      volleyball::tensorrt_runtime
      volleyball::tensorrt_mapping
      volleyball::ports
    )
    target_compile_features(volleyball_tensorrt_detector PUBLIC cxx_std_17)
  endif()
endif()
