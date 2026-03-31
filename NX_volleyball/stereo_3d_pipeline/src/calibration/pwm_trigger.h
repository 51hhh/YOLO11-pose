/**
 * @file pwm_trigger.h
 * @brief libgpiod PWM trigger for stereo camera sync
 *
 * High-precision software PWM using libgpiod + busy-wait.
 * Same approach as volleyball_stereo_driver/high_precision_pwm.
 * Define NO_GPIOD to compile without libgpiod (PWM disabled).
 */

#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <cstdio>

#ifndef NO_GPIOD
#include <gpiod.h>
#include <cstring>
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
          running_(false) {}

    ~PWMTrigger() { stop(); }

    PWMTrigger(const PWMTrigger&) = delete;
    PWMTrigger& operator=(const PWMTrigger&) = delete;

    bool start() {
#ifdef NO_GPIOD
        fprintf(stderr, "[PWM] libgpiod not available, PWM disabled\n");
        return false;
#else
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
        thread_ = std::thread(&PWMTrigger::loop, this);

        fprintf(stderr, "[PWM] Started %.1f Hz on %s line %u\n",
                frequency_, chip_name_.c_str(), line_offset_);
        return true;
#endif
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (thread_.joinable()) thread_.join();
#ifndef NO_GPIOD
        if (line_) { gpiod_line_set_value(line_, 0); gpiod_line_release(line_); line_ = nullptr; }
        if (chip_) { gpiod_chip_close(chip_); chip_ = nullptr; }
#endif
    }

    bool isRunning() const { return running_.load(); }

private:
#ifndef NO_GPIOD
    void loop() {
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
#endif

    std::string chip_name_;
    unsigned int line_offset_;
    double frequency_;
    double duty_cycle_;
#ifndef NO_GPIOD
    struct gpiod_chip* chip_;
    struct gpiod_line* line_;
#endif
    std::atomic<bool> running_;
    std::thread thread_;
};

}  // namespace stereo3d
