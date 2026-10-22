#include "pwm_trigger.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

#include <fcntl.h>
#ifndef NO_GPIOD
#include <gpiod.h>
#endif
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace stereo3d {

PWMTrigger::PWMTrigger(
    const std::string& chip_name,
    unsigned int line_offset,
    double frequency,
    double duty_cycle)
    : chip_name_(chip_name),
      line_offset_(line_offset),
      frequency_(frequency),
      duty_cycle_(duty_cycle),
#ifndef NO_GPIOD
      chip_(nullptr),
      line_(nullptr),
#endif
      sysfs_fd_(-1),
      running_(false) {}

PWMTrigger::~PWMTrigger() {
    stop();
}

bool PWMTrigger::start() {
    if (running_.load()) return true;
    // 上一次工作线程可能因 GPIO 写入失败而自行退出；先 join 并释放
    // 旧 line/chip，避免给 joinable std::thread 重新赋值导致终止进程。
    stop();
    if (!calculateTimingNs(
            frequency_, duty_cycle_, high_ns_, low_ns_)) {
        std::fprintf(
            stderr,
            "[PWM] Invalid or unrepresentable timing: "
            "freq=%.17g duty=%.17g\n",
            frequency_,
            duty_cycle_);
        return false;
    }
    event_history_.clear();
    next_sequence_.store(0);
#ifndef NO_GPIOD
    if (startGpiod()) return true;
    std::fprintf(stderr, "[PWM] libgpiod failed, trying sysfs fallback...\n");
#endif
    if (startSysfs()) return true;

    std::fprintf(stderr, "[PWM] All GPIO methods failed, PWM disabled\n");
    return false;
}

void PWMTrigger::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();

#ifndef NO_GPIOD
    if (line_) {
        gpiod_line_set_value(line_, 0);
        gpiod_line_release(line_);
        line_ = nullptr;
    }
    if (chip_) {
        gpiod_chip_close(chip_);
        chip_ = nullptr;
    }
#endif
    if (sysfs_fd_ >= 0) {
        (void)writeExact(sysfs_fd_, "0", 1);
        ::close(sysfs_fd_);
        sysfs_fd_ = -1;
        unexportGpio();
    }
}

bool PWMTrigger::isRunning() const {
    return running_.load();
}

bool PWMTrigger::isTimingValid(
    double frequency,
    double duty_cycle) noexcept {
    long high_ns = 0;
    long low_ns = 0;
    return calculateTimingNs(
        frequency, duty_cycle, high_ns, low_ns);
}

bool PWMTrigger::calculateTimingNs(
    double frequency,
    double duty_cycle,
    long& high_ns,
    long& low_ns) noexcept {
    high_ns = 0;
    low_ns = 0;
    if (!std::isfinite(frequency) || frequency <= 0.0 ||
        !std::isfinite(duty_cycle) || duty_cycle <= 0.0 ||
        duty_cycle >= 100.0) {
        return false;
    }

    const double period_ns_value = 1e9 / frequency;
    const double max_long =
        static_cast<double>(std::numeric_limits<long>::max());
    if (!std::isfinite(period_ns_value) ||
        period_ns_value < 2.0 || period_ns_value >= max_long) {
        return false;
    }

    const long period_ns = static_cast<long>(period_ns_value);
    const double high_ns_value =
        period_ns_value * (duty_cycle / 100.0);
    if (!std::isfinite(high_ns_value) || high_ns_value < 1.0 ||
        high_ns_value >= static_cast<double>(period_ns)) {
        return false;
    }
    high_ns = static_cast<long>(high_ns_value);
    low_ns = period_ns - high_ns;
    return high_ns > 0 && low_ns > 0;
}

bool PWMTrigger::eventForSequence(
    std::uint64_t sequence,
    PWMTriggerEdgeEvent& event) const {
    return event_history_.find(sequence, event);
}

std::uint64_t PWMTrigger::latestEventSequence() const {
    return event_history_.latestSequence();
}

std::uint64_t PWMTrigger::timespecNs(const struct timespec& value) {
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}

std::uint64_t PWMTrigger::midpoint(
    std::uint64_t first,
    std::uint64_t second) {
    return first / 2ULL + second / 2ULL +
           (first % 2ULL + second % 2ULL) / 2ULL;
}

void PWMTrigger::recordRisingEdge(
    const struct timespec& scheduled,
    const struct timespec& monotonic_before,
    const struct timespec& monotonic_after,
    const struct timespec& realtime_before,
    const struct timespec& realtime_after) {
    const std::uint64_t before_mono = timespecNs(monotonic_before);
    const std::uint64_t after_mono = timespecNs(monotonic_after);
    const std::uint64_t before_real = timespecNs(realtime_before);
    const std::uint64_t after_real = timespecNs(realtime_after);
    PWMTriggerEdgeEvent event;
    event.sequence = next_sequence_.fetch_add(1) + 1;
    event.scheduled_monotonic_ns = timespecNs(scheduled);
    event.edge_monotonic_ns = midpoint(before_mono, after_mono);
    event.edge_realtime_ns = midpoint(before_real, after_real);
    event.write_uncertainty_ns =
        after_mono >= before_mono ? after_mono - before_mono : 0;
    event_history_.push(event);
}

bool PWMTrigger::writeExact(int fd, const char* data, std::size_t len) {
    std::size_t total = 0;
    while (total < len) {
        const ssize_t written = ::write(fd, data + total, len - total);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (written == 0) return false;
        total += static_cast<std::size_t>(written);
    }
    return true;
}

void PWMTrigger::unexportGpio() {
    if (sysfs_gpio_num_ < 0) return;
    int fd = ::open("/sys/class/gpio/unexport", O_WRONLY);
    if (fd >= 0) {
        char buf[16];
        const int len = std::snprintf(
            buf,
            sizeof(buf),
            "%d",
            sysfs_gpio_num_);
        if (len > 0) {
            (void)writeExact(fd, buf, static_cast<std::size_t>(len));
        }
        ::close(fd);
    }
}

void PWMTrigger::setupRealtimeThread() {
    struct sched_param sp {};
    sp.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) < 0) {
        std::fprintf(
            stderr,
            "[PWM] SCHED_FIFO failed (need root): %s\n",
            std::strerror(errno));
    }

    if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0) {
        std::fprintf(stderr, "[PWM] mlockall failed: %s\n", std::strerror(errno));
    }

    const int ncpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if (ncpus > 1) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(ncpus - 1, &cpuset);
        const int affinity_status = pthread_setaffinity_np(
            pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (affinity_status != 0) {
            std::fprintf(
                stderr,
                "[PWM] CPU affinity failed: %s\n",
                std::strerror(affinity_status));
        }
    }
}

void PWMTrigger::timespecAdd(struct timespec& ts, long ns) {
    ts.tv_nsec += ns;
    while (ts.tv_nsec >= 1000000000L) {
        ts.tv_nsec -= 1000000000L;
        ++ts.tv_sec;
    }
}

void PWMTrigger::sleepUntil(const struct timespec& target) {
    // TIMER_ABSTIME 在信号中断后仍使用同一个截止时刻，必须重试。
    while (clock_nanosleep(
               CLOCK_MONOTONIC,
               TIMER_ABSTIME,
               &target,
               nullptr) == EINTR) {
    }
}

int PWMTrigger::computeSysfsGpioNum() const {
    char path[256];
    std::snprintf(
        path,
        sizeof(path),
        "/sys/class/gpio/%s/base",
        chip_name_.c_str());
    FILE* file = std::fopen(path, "r");
    int base = -1;
    if (file) {
        if (std::fscanf(file, "%d", &base) != 1) base = -1;
        std::fclose(file);
    }
    if (base < 0) {
        if (chip_name_ == "gpiochip0") {
            base = 316;
        } else if (chip_name_ == "gpiochip1") {
            base = 348;
        } else if (chip_name_ == "gpiochip2") {
            base = 386;
        } else {
            std::fprintf(
                stderr,
                "[PWM] Unknown chip base for %s\n",
                chip_name_.c_str());
            return -1;
        }
    }
    return base + static_cast<int>(line_offset_);
}

bool PWMTrigger::startSysfs() {
    sysfs_gpio_num_ = computeSysfsGpioNum();
    if (sysfs_gpio_num_ < 0) return false;

    std::fprintf(
        stderr,
        "[PWM] WARNING: sysfs GPIO is deprecated on JetPack 6+, "
        "install libgpiod for reliable operation\n");

    int fd = ::open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) {
        std::fprintf(
            stderr,
            "[PWM] Cannot open /sys/class/gpio/export: %s\n",
            std::strerror(errno));
        return false;
    }
    char buf[16];
    const int len = std::snprintf(
        buf,
        sizeof(buf),
        "%d",
        sysfs_gpio_num_);
    if (len <= 0 || !writeExact(fd, buf, static_cast<std::size_t>(len))) {
        std::fprintf(
            stderr,
            "[PWM] Cannot export GPIO %d: %s\n",
            sysfs_gpio_num_,
            std::strerror(errno));
        ::close(fd);
        return false;
    }
    ::close(fd);

    usleep(50000);

    char direction_path[256];
    std::snprintf(
        direction_path,
        sizeof(direction_path),
        "/sys/class/gpio/gpio%d/direction",
        sysfs_gpio_num_);
    fd = ::open(direction_path, O_WRONLY);
    if (fd < 0) {
        std::fprintf(
            stderr,
            "[PWM] Cannot set GPIO %d direction: %s\n",
            sysfs_gpio_num_,
            std::strerror(errno));
        unexportGpio();
        return false;
    }
    if (!writeExact(fd, "out", 3)) {
        std::fprintf(
            stderr,
            "[PWM] Cannot write GPIO %d direction: %s\n",
            sysfs_gpio_num_,
            std::strerror(errno));
        ::close(fd);
        unexportGpio();
        return false;
    }
    ::close(fd);

    char value_path[256];
    std::snprintf(
        value_path,
        sizeof(value_path),
        "/sys/class/gpio/gpio%d/value",
        sysfs_gpio_num_);
    sysfs_fd_ = ::open(value_path, O_WRONLY);
    if (sysfs_fd_ < 0) {
        std::fprintf(
            stderr,
            "[PWM] Cannot open GPIO %d value: %s\n",
            sysfs_gpio_num_,
            std::strerror(errno));
        unexportGpio();
        return false;
    }

    running_.store(true);
    try {
        thread_ = std::thread(&PWMTrigger::loopSysfs, this);
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[PWM] Cannot start sysfs worker: %s\n",
            error.what());
        running_.store(false);
        ::close(sysfs_fd_);
        sysfs_fd_ = -1;
        unexportGpio();
        return false;
    } catch (...) {
        std::fprintf(stderr, "[PWM] Cannot start sysfs worker: unknown error\n");
        running_.store(false);
        ::close(sysfs_fd_);
        sysfs_fd_ = -1;
        unexportGpio();
        return false;
    }

    std::fprintf(
        stderr,
        "[PWM] Started %.1f Hz via sysfs GPIO %d\n",
        frequency_,
        sysfs_gpio_num_);
    return true;
}

void PWMTrigger::loopSysfs() noexcept {
    try {
        setupRealtimeThread();
        struct timespec next {};
        clock_gettime(CLOCK_MONOTONIC, &next);

        while (running_.load(std::memory_order_relaxed)) {
            struct timespec monotonic_before {};
            struct timespec monotonic_after {};
            struct timespec realtime_before {};
            struct timespec realtime_after {};
            clock_gettime(CLOCK_MONOTONIC, &monotonic_before);
            clock_gettime(CLOCK_REALTIME, &realtime_before);
            const bool edge_ok = writeExact(sysfs_fd_, "1", 1);
            clock_gettime(CLOCK_REALTIME, &realtime_after);
            clock_gettime(CLOCK_MONOTONIC, &monotonic_after);
            if (!edge_ok) {
                std::fprintf(
                    stderr,
                    "[PWM] GPIO rising-edge write failed: %s\n",
                    std::strerror(errno));
                running_.store(false);
                break;
            }
            recordRisingEdge(
                next,
                monotonic_before,
                monotonic_after,
                realtime_before,
                realtime_after);
            lseek(sysfs_fd_, 0, SEEK_SET);
            timespecAdd(next, high_ns_);
            sleepUntil(next);

            (void)writeExact(sysfs_fd_, "0", 1);
            lseek(sysfs_fd_, 0, SEEK_SET);
            timespecAdd(next, low_ns_);
            sleepUntil(next);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[PWM] sysfs worker failed: %s\n", error.what());
        running_.store(false);
    } catch (...) {
        std::fprintf(stderr, "[PWM] sysfs worker failed: unknown error\n");
        running_.store(false);
    }
    munlockall();
}

#ifndef NO_GPIOD
bool PWMTrigger::startGpiod() {
    if (running_.load()) return false;

    chip_ = gpiod_chip_open_by_name(chip_name_.c_str());
    if (!chip_) {
        std::fprintf(
            stderr,
            "[PWM] Cannot open GPIO chip: %s\n",
            chip_name_.c_str());
        return false;
    }

    line_ = gpiod_chip_get_line(chip_, line_offset_);
    if (!line_) {
        std::fprintf(stderr, "[PWM] Cannot get GPIO line: %u\n", line_offset_);
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        return false;
    }

    if (gpiod_line_request_output(line_, "calib_pwm", 0) < 0) {
        std::fprintf(
            stderr,
            "[PWM] Cannot request output: %s\n",
            std::strerror(errno));
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        line_ = nullptr;
        return false;
    }

    running_.store(true);
    try {
        thread_ = std::thread(&PWMTrigger::loopGpiod, this);
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[PWM] Cannot start libgpiod worker: %s\n",
            error.what());
        running_.store(false);
        gpiod_line_set_value(line_, 0);
        gpiod_line_release(line_);
        line_ = nullptr;
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        return false;
    } catch (...) {
        std::fprintf(
            stderr,
            "[PWM] Cannot start libgpiod worker: unknown error\n");
        running_.store(false);
        gpiod_line_set_value(line_, 0);
        gpiod_line_release(line_);
        line_ = nullptr;
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        return false;
    }

    std::fprintf(
        stderr,
        "[PWM] Started %.1f Hz on %s line %u (libgpiod)\n",
        frequency_,
        chip_name_.c_str(),
        line_offset_);
    return true;
}

void PWMTrigger::loopGpiod() noexcept {
    try {
        setupRealtimeThread();
        struct timespec next {};
        clock_gettime(CLOCK_MONOTONIC, &next);

        while (running_.load(std::memory_order_relaxed)) {
            struct timespec monotonic_before {};
            struct timespec monotonic_after {};
            struct timespec realtime_before {};
            struct timespec realtime_after {};
            clock_gettime(CLOCK_MONOTONIC, &monotonic_before);
            clock_gettime(CLOCK_REALTIME, &realtime_before);
            const int edge_status = gpiod_line_set_value(line_, 1);
            clock_gettime(CLOCK_REALTIME, &realtime_after);
            clock_gettime(CLOCK_MONOTONIC, &monotonic_after);
            if (edge_status < 0) {
                std::fprintf(
                    stderr,
                    "[PWM] GPIO rising-edge write failed: %s\n",
                    std::strerror(errno));
                running_.store(false);
                break;
            }
            recordRisingEdge(
                next,
                monotonic_before,
                monotonic_after,
                realtime_before,
                realtime_after);
            timespecAdd(next, high_ns_);
            sleepUntil(next);

            gpiod_line_set_value(line_, 0);
            timespecAdd(next, low_ns_);
            sleepUntil(next);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[PWM] libgpiod worker failed: %s\n", error.what());
        running_.store(false);
    } catch (...) {
        std::fprintf(stderr, "[PWM] libgpiod worker failed: unknown error\n");
        running_.store(false);
    }
    munlockall();
}
#endif

}  // namespace stereo3d
