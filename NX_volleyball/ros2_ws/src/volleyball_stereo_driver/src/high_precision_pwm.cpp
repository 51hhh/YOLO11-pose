/**
 * @file high_precision_pwm.cpp
 * @brief 高精度软件 PWM 实现
 */

#include "volleyball_stereo_driver/high_precision_pwm.hpp"
#include <iostream>
#include <sched.h>
#include <cstring>
#include <stdexcept>

namespace volleyball {

HighPrecisionPWM::HighPrecisionPWM(
    const std::string& chip_name,
    unsigned int line_offset,
    double frequency,
    double duty_cycle
) : chip_name_(chip_name),
    line_offset_(line_offset),
    chip_(nullptr),
    line_(nullptr),
    frequency_(frequency),
    duty_cycle_(duty_cycle),
    running_(false),
    actual_frequency_(0.0)
{
    // 计算时间参数
    period_ = 1.0 / frequency_;
    high_time_ = period_ * (duty_cycle_ / 100.0);
    low_time_ = period_ - high_time_;
}

HighPrecisionPWM::~HighPrecisionPWM() {
    stop();
    
    // 释放 GPIO 资源
    if (line_) {
        gpiod_line_release(line_);
    }
    if (chip_) {
        gpiod_chip_close(chip_);
    }
}

bool HighPrecisionPWM::start() {
    if (running_.load()) {
        std::cerr << "PWM 已在运行" << std::endl;
        return false;
    }

    // 打开 GPIO 芯片
    chip_ = gpiod_chip_open_by_name(chip_name_.c_str());
    if (!chip_) {
        std::cerr << "无法打开 GPIO 芯片: " << chip_name_ << std::endl;
        return false;
    }

    // 获取 GPIO 线
    line_ = gpiod_chip_get_line(chip_, line_offset_);
    if (!line_) {
        std::cerr << "无法获取 GPIO line: " << line_offset_ << std::endl;
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        return false;
    }

    // 请求输出模式
    int ret = gpiod_line_request_output(line_, "pwm_trigger", 0);
    if (ret < 0) {
        std::cerr << "无法请求 GPIO 输出: " << std::strerror(errno) << std::endl;
        gpiod_chip_close(chip_);
        chip_ = nullptr;
        line_ = nullptr;
        return false;
    }

    // 启动 PWM 线程
    running_.store(true);
    pwm_thread_ = std::thread(&HighPrecisionPWM::pwmLoop, this);

    std::cout << "✅ 高精度 PWM 已启动: " 
              << frequency_ << " Hz, " 
              << duty_cycle_ << "%" << std::endl;

    return true;
}

void HighPrecisionPWM::stop() {
    if (!running_.load()) {
        return;
    }

    // 停止线程
    running_.store(false);
    if (pwm_thread_.joinable()) {
        pwm_thread_.join();
    }

    // 确保输出低电平
    if (line_) {
        gpiod_line_set_value(line_, 0);
    }

    std::cout << "✅ PWM 已停止" << std::endl;
    std::cout << "   最终频率: " << actual_frequency_.load() << " Hz" << std::endl;
}

void HighPrecisionPWM::setFrequency(double frequency) {
    frequency_ = frequency;
    period_ = 1.0 / frequency_;
    high_time_ = period_ * (duty_cycle_ / 100.0);
    low_time_ = period_ - high_time_;
}

void HighPrecisionPWM::setDutyCycle(double duty_cycle) {
    duty_cycle_ = duty_cycle;
    high_time_ = period_ * (duty_cycle_ / 100.0);
    low_time_ = period_ - high_time_;
}

double HighPrecisionPWM::getActualFrequency() const {
    return actual_frequency_.load();
}

void HighPrecisionPWM::pwmLoop() {
    // 设置实时优先级
    setRealtimePriority();

    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    auto next_edge_time = clock::now();
    int cycle_count = 0;
    auto last_stat_time = clock::now();
    double total_period = 0.0;

    while (running_.load()) {
        auto cycle_start = clock::now();

        // === 高电平 ===
        gpiod_line_set_value(line_, 1);
        next_edge_time += std::chrono::duration_cast<clock::duration>(
            duration(high_time_)
        );

        // 精确等待
        auto wait_time = duration(next_edge_time - clock::now()).count();
        if (wait_time > 0) {
            accurateSleep(wait_time);
        }

        // === 低电平 ===
        gpiod_line_set_value(line_, 0);
        next_edge_time += std::chrono::duration_cast<clock::duration>(
            duration(low_time_)
        );

        // 精确等待
        wait_time = duration(next_edge_time - clock::now()).count();
        if (wait_time > 0) {
            accurateSleep(wait_time);
        }

        // 统计实际周期
        auto cycle_end = clock::now();
        double actual_period = duration(cycle_end - cycle_start).count();
        total_period += actual_period;
        cycle_count++;

        // 每 5 秒打印一次统计
        if (duration(cycle_end - last_stat_time).count() >= 5.0) {
            double avg_period = total_period / cycle_count;
            double freq = 1.0 / avg_period;
            actual_frequency_.store(freq);

            std::cout << "  周期: " << cycle_count 
                      << " | 实际频率: " << freq << " Hz"
                      << " | 误差: " << (freq - frequency_) << " Hz"
                      << std::endl;

            cycle_count = 0;
            total_period = 0.0;
            last_stat_time = cycle_end;
        }
    }
}

void HighPrecisionPWM::accurateSleep(double duration) {
    using clock = std::chrono::high_resolution_clock;
    using dur = std::chrono::duration<double>;

    if (duration > BUSY_WAIT_THRESHOLD) {
        // 长时间: 使用 sleep 节省 CPU
        auto sleep_duration = duration - BUSY_WAIT_THRESHOLD;
        std::this_thread::sleep_for(dur(sleep_duration));
        duration = BUSY_WAIT_THRESHOLD;
    }

    // 短时间: 使用忙等待
    auto target = clock::now() + std::chrono::duration_cast<clock::duration>(dur(duration));
    while (clock::now() < target) {
        // 忙等待
    }
}

void HighPrecisionPWM::setRealtimePriority() {
    // 设置 SCHED_FIFO 实时调度
    struct sched_param param;
    param.sched_priority = 50;  // 中等优先级

    if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
        std::cout << "  ✅ 线程优先级已提升 (SCHED_FIFO)" << std::endl;
    }
    // 失败时静默（不需要 sudo 也能正常工作）
}

}  // namespace volleyball
