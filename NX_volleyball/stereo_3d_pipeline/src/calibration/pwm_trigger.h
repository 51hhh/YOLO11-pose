/**
 * @file pwm_trigger.h
 * @brief GPIO PWM trigger for stereo camera sync
 *
 * Timing strategy:
 *   clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME) for jitter-free
 *   absolute-time wakeup. Combined with SCHED_FIFO, mlockall, and
 *   CPU affinity to minimize scheduling interference.
 *
 * Priority:
 *   1. libgpiod (if available at compile time)
 *   2. sysfs GPIO fallback (/sys/class/gpio)
 *   3. NO_GPIOD + NO_SYSFS_GPIO → disabled
 *
 * Default: gpiochip2 line 7 on Orin NX
 */

#pragma once

#include <atomic>
#include <thread>
#include <string>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <pthread.h>

#ifndef NO_GPIOD
#include <gpiod.h>
#endif

namespace stereo3d {

class PWMTrigger {
public:
    PWMTrigger(const std::string& chip_name = "gpiochip2",
               unsigned int line_offset = 7,
               double frequency = 10.0,
               double duty_cycle = 50.0)
        : chip_name_(chip_name), line_offset_(line_offset),
          frequency_(frequency), duty_cycle_(duty_cycle),
#ifndef NO_GPIOD
          chip_(nullptr), line_(nullptr),
#endif
          sysfs_fd_(-1),
          running_(false) {}

    ~PWMTrigger() { stop(); }

    PWMTrigger(const PWMTrigger&) = delete;
    PWMTrigger& operator=(const PWMTrigger&) = delete;

    bool start() {
        if (frequency_ <= 0 || duty_cycle_ < 0 || duty_cycle_ > 100) {
            fprintf(stderr, "[PWM] Invalid params: freq=%.1f duty=%.1f\n", frequency_, duty_cycle_);
            return false;
        }
#ifndef NO_GPIOD
        // Try libgpiod first
        if (startGpiod()) return true;
        fprintf(stderr, "[PWM] libgpiod failed, trying sysfs fallback...\n");
#endif
        // Fallback to sysfs
        if (startSysfs()) return true;

        fprintf(stderr, "[PWM] All GPIO methods failed, PWM disabled\n");
        return false;
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (thread_.joinable()) thread_.join();

#ifndef NO_GPIOD
        if (line_) { gpiod_line_set_value(line_, 0); gpiod_line_release(line_); line_ = nullptr; }
        if (chip_) { gpiod_chip_close(chip_); chip_ = nullptr; }
#endif
        if (sysfs_fd_ >= 0) {
            (void)write(sysfs_fd_, "0", 1);
            close(sysfs_fd_);
            sysfs_fd_ = -1;
            unexportGpio();
        }
    }

    bool isRunning() const { return running_.load(); }

private:
    /// Unexport sysfs GPIO (safe to call multiple times)
    void unexportGpio() {
        if (sysfs_gpio_num_ < 0) return;
        int fd = ::open("/sys/class/gpio/unexport", O_WRONLY);
        if (fd >= 0) {
            char buf[16];
            int len = snprintf(buf, sizeof(buf), "%d", sysfs_gpio_num_);
            (void)::write(fd, buf, len);
            ::close(fd);
        }
    }
    // ==================== RT thread setup ====================

    /// Configure real-time scheduling, memory lock, and CPU affinity
    static void setupRealtimeThread() {
        // SCHED_FIFO priority 80 — preempts normal threads
        struct sched_param sp{};
        sp.sched_priority = 80;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) < 0)
            fprintf(stderr, "[PWM] SCHED_FIFO failed (need root): %s\n", strerror(errno));

        // Lock all pages to prevent page-fault jitter
        if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0)
            fprintf(stderr, "[PWM] mlockall failed: %s\n", strerror(errno));

        // Pin to last CPU core to avoid migration jitter
        int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpus > 1) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(ncpus - 1, &cpuset);
            if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
                fprintf(stderr, "[PWM] CPU affinity failed: %s\n", strerror(errno));
        }
    }

    /// Add nanoseconds to timespec (handles overflow)
    static void timespecAdd(struct timespec& ts, long ns) {
        ts.tv_nsec += ns;
        while (ts.tv_nsec >= 1000000000L) {
            ts.tv_nsec -= 1000000000L;
            ts.tv_sec++;
        }
    }

    /// Absolute-time sleep using kernel hrtimer (jitter < 50μs with SCHED_FIFO)
    static void sleepUntil(const struct timespec& target) {
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
    }

    // ==================== sysfs GPIO backend ====================

    int computeSysfsGpioNum() const {
        char path[256];
        snprintf(path, sizeof(path), "/sys/class/gpio/%s/base", chip_name_.c_str());
        FILE* f = fopen(path, "r");
        int base = -1;
        if (f) {
            if (fscanf(f, "%d", &base) != 1) base = -1;
            fclose(f);
        }
        if (base < 0) {
            // Orin NX known defaults
            if      (chip_name_ == "gpiochip0") base = 316;
            else if (chip_name_ == "gpiochip1") base = 348;
            else if (chip_name_ == "gpiochip2") base = 386;
            else {
                fprintf(stderr, "[PWM] Unknown chip base for %s\n", chip_name_.c_str());
                return -1;
            }
        }
        return base + static_cast<int>(line_offset_);
    }

    bool startSysfs() {
        sysfs_gpio_num_ = computeSysfsGpioNum();
        if (sysfs_gpio_num_ < 0) return false;

        fprintf(stderr, "[PWM] WARNING: sysfs GPIO is deprecated on JetPack 6+, "
                        "install libgpiod for reliable operation\n");

        int fd = ::open("/sys/class/gpio/export", O_WRONLY);
        if (fd < 0) {
            fprintf(stderr, "[PWM] Cannot open /sys/class/gpio/export: %s\n", strerror(errno));
            return false;
        }
        char buf[16];
        int len = snprintf(buf, sizeof(buf), "%d", sysfs_gpio_num_);
        (void)::write(fd, buf, len);
        ::close(fd);

        usleep(50000);

        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "/sys/class/gpio/gpio%d/direction", sysfs_gpio_num_);
        fd = ::open(dir_path, O_WRONLY);
        if (fd < 0) {
            fprintf(stderr, "[PWM] Cannot set GPIO %d direction: %s\n", sysfs_gpio_num_, strerror(errno));
            unexportGpio();
            return false;
        }
        (void)::write(fd, "out", 3);
        ::close(fd);

        char val_path[256];
        snprintf(val_path, sizeof(val_path), "/sys/class/gpio/gpio%d/value", sysfs_gpio_num_);
        sysfs_fd_ = ::open(val_path, O_WRONLY);
        if (sysfs_fd_ < 0) {
            fprintf(stderr, "[PWM] Cannot open GPIO %d value: %s\n", sysfs_gpio_num_, strerror(errno));
            unexportGpio();
            return false;
        }

        running_.store(true);
        thread_ = std::thread(&PWMTrigger::loopSysfs, this);

        fprintf(stderr, "[PWM] Started %.1f Hz via sysfs GPIO %d\n",
                frequency_, sysfs_gpio_num_);
        return true;
    }

    void loopSysfs() {
        setupRealtimeThread();

        long high_ns = static_cast<long>(1e9 / frequency_ * (duty_cycle_ / 100.0));
        long low_ns  = static_cast<long>(1e9 / frequency_) - high_ns;

        struct timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);

        while (running_.load(std::memory_order_relaxed)) {
            (void)::write(sysfs_fd_, "1", 1);
            lseek(sysfs_fd_, 0, SEEK_SET);
            timespecAdd(next, high_ns);
            sleepUntil(next);

            (void)::write(sysfs_fd_, "0", 1);
            lseek(sysfs_fd_, 0, SEEK_SET);
            timespecAdd(next, low_ns);
            sleepUntil(next);
        }

        munlockall();
    }

    // ==================== libgpiod backend ====================
#ifndef NO_GPIOD
    bool startGpiod() {
        if (running_.load()) return false;

        chip_ = gpiod_chip_open_by_name(chip_name_.c_str());
        if (!chip_) {
            fprintf(stderr, "[PWM] Cannot open GPIO chip: %s\n", chip_name_.c_str());
            return false;
        }

        line_ = gpiod_chip_get_line(chip_, line_offset_);
        if (!line_) {
            fprintf(stderr, "[PWM] Cannot get GPIO line: %u\n", line_offset_);
            gpiod_chip_close(chip_); chip_ = nullptr;
            return false;
        }

        if (gpiod_line_request_output(line_, "calib_pwm", 0) < 0) {
            fprintf(stderr, "[PWM] Cannot request output: %s\n", strerror(errno));
            gpiod_chip_close(chip_); chip_ = nullptr; line_ = nullptr;
            return false;
        }

        running_.store(true);
        thread_ = std::thread(&PWMTrigger::loopGpiod, this);

        fprintf(stderr, "[PWM] Started %.1f Hz on %s line %u (libgpiod)\n",
                frequency_, chip_name_.c_str(), line_offset_);
        return true;
    }

    void loopGpiod() {
        setupRealtimeThread();

        long high_ns = static_cast<long>(1e9 / frequency_ * (duty_cycle_ / 100.0));
        long low_ns  = static_cast<long>(1e9 / frequency_) - high_ns;

        struct timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);

        while (running_.load(std::memory_order_relaxed)) {
            gpiod_line_set_value(line_, 1);
            timespecAdd(next, high_ns);
            sleepUntil(next);

            gpiod_line_set_value(line_, 0);
            timespecAdd(next, low_ns);
            sleepUntil(next);
        }

        munlockall();
    }
#endif

    // ==================== members ====================
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
    std::thread thread_;
};

}  // namespace stereo3d
