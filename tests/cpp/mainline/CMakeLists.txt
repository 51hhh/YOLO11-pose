function(add_mainline_test target)
  add_executable(${target} ${ARGN})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_mainline_test(core_contract_tests core_contract_tests.cpp)
target_link_libraries(core_contract_tests PRIVATE
  volleyball::core
  volleyball::ports
  volleyball::fake_adapters
)

add_mainline_test(stereo_algorithms_tests stereo_algorithms_tests.cpp)
target_link_libraries(stereo_algorithms_tests PRIVATE
  volleyball::stereo_algorithms
)

add_mainline_test(logging_tests logging_tests.cpp)
target_link_libraries(logging_tests PRIVATE volleyball::logging)

add_mainline_test(
  stereo_observation_service_tests
  stereo_observation_service_tests.cpp
)
target_link_libraries(stereo_observation_service_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)

add_mainline_test(
  stereo_pipeline_runner_tests
  stereo_pipeline_runner_tests.cpp
)
target_link_libraries(stereo_pipeline_runner_tests PRIVATE
  volleyball::services
  volleyball::fake_adapters
)
