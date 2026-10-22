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
target_compile_features(volleyball_tensorrt_postprocess PUBLIC cxx_std_17)

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
endif()
