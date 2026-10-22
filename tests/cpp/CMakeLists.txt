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

add_executable(legacy_pair_audit_tests
  legacy_pair_audit_tests.cpp
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src/stereo/depth_match_contract.cpp
)
target_include_directories(legacy_pair_audit_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_compile_features(legacy_pair_audit_tests PRIVATE cxx_std_17)
target_compile_options(legacy_pair_audit_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME legacy_pair_audit_tests COMMAND legacy_pair_audit_tests)

add_executable(trigger_edge_history_tests
  trigger_edge_history_tests.cpp
)
target_include_directories(trigger_edge_history_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/NX_volleyball/stereo_3d_pipeline/src
)
target_compile_definitions(trigger_edge_history_tests PRIVATE NO_GPIOD=1)
target_compile_features(trigger_edge_history_tests PRIVATE cxx_std_17)
target_compile_options(trigger_edge_history_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
add_test(NAME trigger_edge_history_tests COMMAND trigger_edge_history_tests)

