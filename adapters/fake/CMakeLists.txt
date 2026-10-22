add_library(volleyball_fake_adapters STATIC
  src/in_memory_trial_writer.cpp
)
add_library(volleyball::fake_adapters ALIAS volleyball_fake_adapters)

target_include_directories(volleyball_fake_adapters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(volleyball_fake_adapters PUBLIC volleyball::ports)
target_compile_features(volleyball_fake_adapters PUBLIC cxx_std_17)
target_compile_options(volleyball_fake_adapters PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

