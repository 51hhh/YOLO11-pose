/**
 * @file high_precision_pwm.hpp
 * @brief 高精度软件 PWM 类 (C++ 实现)
 * 
 * 使用 libgpiod 和误差补偿算法实现精确的 PWM 输出
 * 适用于 gpiochip2 line 7
 */

#ifndef HIGH_PRECISION_PWM_HPP_
#define HIGH_PRECISION_PWM_HPP_

#include <gpiod.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>

namespace volleyball {

/**
 * @class HighPrecisionPWM
 * @brief 高精度软件 PWM 实现
 */
class HighPrecisionPWM {
public:
    /**
     * @brief 构造函数
     * @param chip_name GPIO 芯片名称 (如 "gpiochip2")
     * @param line_offset GPIO 线偏移量 (如 7)
     * @param frequency 目标频率 (Hz)
     * @param duty_cycle 占空比 (0-100)
     */
    HighPrecisionPWM(
        const std::string& chip_name,
        unsigned int line_offset,
        double frequency,
        double duty_cycle
    );

    /**
     * @brief 析构函数
     */
    ~HighPrecisionPWM();

    /**
     * @brief 启动 PWM
     * @return true 成功, false 失败
     */
    bool start();

    /**
     * @brief 停止 PWM
     */
    void stop();

    /**
     * @brief 改变频率
     * @param frequency 新频率 (Hz)
     */
    void setFrequency(double frequency);

    /**
     * @brief 改变占空比
     * @param duty_cycle 新占空比 (0-100)
     */
    void setDutyCycle(double duty_cycle);

    /**
     * @brief 获取实际频率
     * @return 实际频率 (Hz)
     */
    double getActualFrequency() const;

    /**
     * @brief 是否正在运行
     * @return true 运行中, false 已停止
     */
    bool isRunning() const { return running_.load(); }

private:
    /**
     * @brief PWM 循环线程
     */
    void pwmLoop();

    /**
     * @brief 高精度睡眠
     * @param duration 睡眠时间 (秒)
     */
    void accurateSleep(double duration);

    /**
     * @brief 设置实时优先级
     */
    void setRealtimePriority();

    // GPIO 相关
    std::string chip_name_;
    unsigned int line_offset_;
    struct gpiod_chip* chip_;
    struct gpiod_line* line_;

    // PWM 参数
    double frequency_;
    double duty_cycle_;
    double period_;
    double high_time_;
    double low_time_;

    // 线程控制
    std::atomic<bool> running_;
    std::thread pwm_thread_;

    // 统计信息
    mutable std::atomic<double> actual_frequency_;
    static constexpr double BUSY_WAIT_THRESHOLD = 0.0005;  // 0.5ms
};

}  // namespace volleyball

#endif  // HIGH_PRECISION_PWM_HPP_
