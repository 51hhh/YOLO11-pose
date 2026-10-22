add_executable(core_contract_tests
  core_contract_tests.cpp
)
target_link_libraries(core_contract_tests PRIVATE
  volleyball::core
  volleyball::ports
  volleyball::fake_adapters
)
target_compile_features(core_contract_tests PRIVATE cxx_std_17)
target_compile_options(core_contract_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME core_contract_tests COMMAND core_contract_tests)

