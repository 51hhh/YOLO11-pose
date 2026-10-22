find_package(Threads REQUIRED)

add_executable(
  hik_mvs_camera_tests
  hik_mvs_camera_tests.cpp
)
if(TARGET volleyball::hik_mvs)
  target_link_libraries(hik_mvs_camera_tests PRIVATE volleyball::hik_mvs)
else()
  target_sources(hik_mvs_camera_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/src/hik_stereo_camera_source.cpp
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/src/hikvision_camera.cpp
  )
  target_include_directories(hik_mvs_camera_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/adapters/hik_mvs/include
  )
endif()
target_include_directories(hik_mvs_camera_tests BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/tests/cpp/fakes/mvs
)
target_compile_features(hik_mvs_camera_tests PRIVATE cxx_std_17)
target_compile_options(hik_mvs_camera_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
)
target_link_libraries(hik_mvs_camera_tests PRIVATE
  volleyball::logging
  volleyball::ports
  Threads::Threads
)
add_test(NAME hik_mvs_camera_tests COMMAND hik_mvs_camera_tests)
