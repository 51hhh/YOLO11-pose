/**
 * @file pwm_trigger.h
 * @brief GPIO PWM trigger for stereo camera sync
 *
 * Priority:
 *   1. libgpiod (if available at compile time)
 *   2. sysfs GPIO fallback (/sys/class/gpio) — works without any library
 *   3. NO_GPIOD + NO_SYSFS_GPIO → disabled
 *
 * Default: gpiochip2 line 7 → sysfs GPIO 393 on Orin NX
 * (gpiochip2 base=386, line 7 → 386+7=393)
 */

#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>

#ifndef NO_GPIOD
#include <gpiod.h>
#include <sched.h>
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
            // Unexport
            int fd = ::open("/sys/class/gpio/unexport", O_WRONLY);
            if (fd >= 0) {
                char buf[16];
                int len = snprintf(buf, sizeof(buf), "%d", sysfs_gpio_num_);
                (void)::write(fd, buf, len);
                ::close(fd);
            }
        }
    }

    bool isRunning() const { return running_.load(); }

private:
    // ==================== sysfs GPIO backend ====================
    // Compute sysfs GPIO number from chip + line offset
    int computeSysfsGpioNum() const {
        // Try reading chip base from /sys/bus/gpio/devices/<chip_name>/base
        char path[256];
        snprintf(path, sizeof(path), "/sys/class/gpio/%s/base", chip_name_.c_str());
        FILE* f = fopen(path, "r");
        int base = -1;
        if (f) {
            if (fscanf(f, "%d", &base) != 1) base = -1;
            fclose(f);
        }
        if (base < 0) {
            // Orin NX known defaults: gpiochip0=316, gpiochip1=348, gpiochip2=386
            if (chip_name_ == "gpiochip0") base = 316;
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

        // Export
        int fd = ::open("/sys/class/gpio/export", O_WRONLY);
        if (fd < 0) {
            fprintf(stderr, "[PWM] Cannot open /sys/class/gpio/export: %s\n", strerror(errno));
            return false;
        }
        char buf[16];
        int len = snprintf(buf, sizeof(buf), "%d", sysfs_gpio_num_);
        (void)::write(fd, buf, len);  // May fail if already exported — ok
        ::close(fd);

        // Wait a moment for sysfs node to appear
        usleep(50000);

        // Set direction
        char dir_path[256];
        snprintf(dir_path, sizeof(dir_path), "/sys/class/gpio/gpio%d/direction", sysfs_gpio_num_);
        fd = ::open(dir_path, O_WRONLY);
        if (fd < 0) {
            fprintf(stderr, "[PWM] Cannot set GPIO %d direction: %s\n", sysfs_gpio_num_, strerror(errno));
            return false;
        }
        (void)::write(fd, "out", 3);
        ::close(fd);

        // Open value file for fast toggling
        char val_path[256];
        snprintf(val_path, sizeof(val_path), "/sys/class/gpio/gpio%d/value", sysfs_gpio_num_);
        sysfs_fd_ = ::open(val_path, O_WRONLY);
        if (sysfs_fd_ < 0) {
            fprintf(stderr, "[PWM] Cannot open GPIO %d value: %s\n", sysfs_gpio_num_, strerror(errno));
            return false;
        }

        running_.store(true);
        thread_ = std::thread(&PWMTrigger::loopSysfs, this);

        fprintf(stderr, "[PWM] Started %.1f Hz via sysfs GPIO %d\n",
                frequency_, sysfs_gpio_num_);
        return true;
    }

    void loopSysfs() {
        using clock = std::chrono::high_resolution_clock;
        using dsec  = std::chrono::duration<double>;

        double period  = 1.0 / frequency_;
        double high_t  = period * (duty_cycle_ / 100.0);
        double low_t   = period - high_t;

        auto next = clock::now();
        while (running_.load()) {
            (void)::write(sysfs_fd_, "1", 1);
            lseek(sysfs_fd_, 0, SEEK_SET);
            next += std::chrono::duration_cast<clock::duration>(dsec(high_t));
            accurateSleep(dsec(next - clock::now()).count());

            (void)::write(sysfs_fd_, "0", 1);
            lseek(sysfs_fd_, 0, SEEK_SET);
            next += std::chrono::duration_cast<clock::duration>(dsec(low_t));
            accurateSleep(dsec(next - clock::now()).count());
        }
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
        struct sched_param sp;
        sp.sched_priority = 50;
        sched_setscheduler(0, SCHED_FIFO, &sp);

        using clock = std::chrono::high_resolution_clock;
        using dsec  = std::chrono::duration<double>;

        double period  = 1.0 / frequency_;
        double high_t  = period * (duty_cycle_ / 100.0);
        double low_t   = period - high_t;

        auto next = clock::now();
        while (running_.load()) {
            gpiod_line_set_value(line_, 1);
            next += std::chrono::duration_cast<clock::duration>(dsec(high_t));
            accurateSleep(dsec(next - clock::now()).count());

            gpiod_line_set_value(line_, 0);
            next += std::chrono::duration_cast<clock::duration>(dsec(low_t));
            accurateSleep(dsec(next - clock::now()).count());
        }
    }
#endif

    // ==================== common ====================
    static void accurateSleep(double dur) {
        if (dur <= 0) return;
        constexpr double BUSY_THRESHOLD = 0.0005;
        using clock = std::chrono::high_resolution_clock;
        using dsec  = std::chrono::duration<double>;
        if (dur > BUSY_THRESHOLD)
            std::this_thread::sleep_for(dsec(dur - BUSY_THRESHOLD));
        auto target = clock::now() + std::chrono::duration_cast<clock::duration>(dsec(std::min(dur, BUSY_THRESHOLD)));
        while (clock::now() < target) {}
    }

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
