/**
 * @file pwm_trigger.h
 * @brief 双目相机同步所用 GPIO PWM 触发器接口。
 */

#ifndef STEREO_3D_PIPELINE_PWM_TRIGGER_H_
#define STEREO_3D_PIPELINE_PWM_TRIGGER_H_

#include "pwm_trigger_events.h"

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

namespace stereo3d {

/**
 * 通过 libgpiod 或 sysfs 输出周期触发信号。
 *
 * 工作线程使用 CLOCK_MONOTONIC 绝对时间休眠，并记录每个上升沿在
 * MONOTONIC/REALTIME 两个时钟域中的观测时间。默认对应 Orin NX 的
 * gpiochip2 line 7。
 */
class PWMTrigger {
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
    void stop();

    [[nodiscard]] bool isRunning() const;
    [[nodiscard]] bool eventForSequence(
        std::uint64_t sequence,
        PWMTriggerEdgeEvent& event) const;
    [[nodiscard]] std::uint64_t latestEventSequence() const;

private:
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
    void loopSysfs();
#ifndef NO_GPIOD
    bool startGpiod();
    void loopGpiod();
#endif

    std::string chip_name_;
    unsigned int line_offset_;
    double frequency_;
    double duty_cycle_;
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

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_PWM_TRIGGER_H_
