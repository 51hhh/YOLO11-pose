/**
 * @file pwm_trigger.hpp
 * @brief 双目相机同步所用 GPIO PWM 触发器接口。
 */

#pragma once

#include "volleyball/adapters/gpio/trigger_events.hpp"
#include "volleyball/ports/trigger_controller.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <string>
#include <thread>

#ifndef NO_GPIOD
struct gpiod_chip;
struct gpiod_line;
#endif

namespace volleyball::adapters::gpio {

/**
 * 通过 libgpiod 或 sysfs 输出周期触发信号。
 *
 * 工作线程使用 CLOCK_MONOTONIC 绝对时间休眠，并记录每个上升沿在
 * MONOTONIC/REALTIME 两个时钟域中的观测时间。默认对应 Orin NX 的
 * gpiochip2 line 7。
 */
class PWMTrigger final : public ports::TriggerController {
public:
    PWMTrigger(
        const std::string& chip_name = "gpiochip2",
        unsigned int line_offset = 7,
        double frequency = 10.0,
        double duty_cycle = 50.0);
    ~PWMTrigger();

    PWMTrigger(const PWMTrigger&) = delete;
    PWMTrigger& operator=(const PWMTrigger&) = delete;

    bool start();
    [[nodiscard]] bool start(std::string& error) override;
    void stop() noexcept override;

    /** 返回频率和占空比能否表示为非零的整数纳秒高、低电平。 */
    [[nodiscard]] static bool isTimingValid(
        double frequency,
        double duty_cycle) noexcept;
    [[nodiscard]] bool isRunning() const noexcept override;
    [[nodiscard]] bool eventForSequence(
        std::uint64_t sequence,
        PWMTriggerEdgeEvent& event) const;
    [[nodiscard]] std::uint64_t latestEventSequence() const;

private:
    static bool calculateTimingNs(
        double frequency,
        double duty_cycle,
        long& high_ns,
        long& low_ns) noexcept;
    static std::uint64_t timespecNs(const struct timespec& value);
    static std::uint64_t midpoint(std::uint64_t first, std::uint64_t second);
    void recordRisingEdge(
        const struct timespec& scheduled,
        const struct timespec& monotonic_before,
        const struct timespec& monotonic_after,
        const struct timespec& realtime_before,
        const struct timespec& realtime_after);
    static bool writeExact(int fd, const char* data, std::size_t len);
    void unexportGpio();
    static void setupRealtimeThread();
    static void timespecAdd(struct timespec& ts, long ns);
    static void sleepUntil(const struct timespec& target);
    int computeSysfsGpioNum() const;
    bool startSysfs();
    void loopSysfs() noexcept;
#ifndef NO_GPIOD
    bool startGpiod();
    void loopGpiod() noexcept;
#endif

    std::string chip_name_;
    unsigned int line_offset_;
    double frequency_;
    double duty_cycle_;
    long high_ns_ = 0;
    long low_ns_ = 0;
#ifndef NO_GPIOD
    struct gpiod_chip* chip_;
    struct gpiod_line* line_;
#endif
    int sysfs_fd_;
    int sysfs_gpio_num_ = -1;
    std::atomic<bool> running_;
    std::atomic<std::uint64_t> next_sequence_{0};
    PWMTriggerEventHistory event_history_;
    std::thread thread_;
};

}  // namespace volleyball::adapters::gpio
