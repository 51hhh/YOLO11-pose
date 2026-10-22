find_package(Threads REQUIRED)

option(
  VOLLEYBALL_GPIO_WITH_GPIOD
  "使用 libgpiod 实现 GPIO 触发；关闭时仅保留 sysfs 兼容路径"
  OFF
)

add_library(volleyball_gpio STATIC
  src/pwm_trigger.cpp
)
add_library(volleyball::gpio ALIAS volleyball_gpio)

target_include_directories(volleyball_gpio PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(volleyball_gpio PUBLIC cxx_std_17)
target_compile_options(volleyball_gpio PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
target_link_libraries(volleyball_gpio PUBLIC
  Threads::Threads
  volleyball::ports
)

if(VOLLEYBALL_GPIO_WITH_GPIOD)
  find_path(VOLLEYBALL_GPIOD_INCLUDE_DIR gpiod.h REQUIRED)
  find_library(VOLLEYBALL_GPIOD_LIBRARY gpiod REQUIRED)
  target_include_directories(
    volleyball_gpio PRIVATE ${VOLLEYBALL_GPIOD_INCLUDE_DIR}
  )
  target_link_libraries(volleyball_gpio PRIVATE ${VOLLEYBALL_GPIOD_LIBRARY})
else()
  # NO_GPIOD 会改变公开头中 PWMTrigger 的成员布局，必须传播给所有调用方，
  # 否则库与消费者会按不同 ABI 编译同一个类。
  target_compile_definitions(volleyball_gpio PUBLIC NO_GPIOD=1)
endif()
