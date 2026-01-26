# 阶段2实施计划 - volleyball_tracker_node.cpp 重构

## 📋 需要修改的方法

### 1. **initialize()** - 修改线程启动逻辑
```cpp
// 原代码：启动两个线程
capture_thread_ = std::thread(&VolleyballTrackerNode::captureLoop, this);
inference_thread_ = std::thread(&VolleyballTrackerNode::inferenceLoop, this);

// 新代码：只启动推理线程
inference_thread_ = std::thread(&VolleyballTrackerNode::inferenceLoop, this);
```

### 2. **移除 captureLoop()**  
完全删除此方法，相机回调已经是异步的

### 3. **新增 updateLeftFrame()** - 左相机更新逻辑
```cpp
void VolleyballTrackerNode::updateLeftFrame() {
    // 轮询相机回调是否有新帧（非阻塞，5ms超时）
    if (!cam_left_->waitForNewFrame(5)) {
        return;  // 无新帧，直接返回
    }
    
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 如果上一帧还未被处理，丢弃（推理太慢）
    if (left_frame_.ready.load()) {
        left_dropped_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "⚠️  Left frame dropped (inference too slow)");
    }
    
    // 获取新帧 + 元数据
    left_frame_.image = cam_left_->getLatestImage().clone();
    left_frame_.metadata = cam_left_->getFrameMetadata();
    left_frame_.ready.store(true);
}
```

### 4. **新增 updateRightFrame()** - 右相机更新逻辑
同 updateLeftFrame()，处理右相机

### 5. **新增 waitForSyncedPair()** - 核心同步逻辑
```cpp
bool VolleyballTrackerNode::waitForSyncedPair(
    cv::Mat& left, cv::Mat& right,
    FrameMetadata& left_meta, FrameMetadata& right_meta)
{
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 检查两帧是否都ready
    if (!left_frame_.ready.load() || !right_frame_.ready.load()) {
        return false;
    }
    
    // ========== PWM时间戳同步：帧号匹配 ==========
    uint32_t left_num = left_frame_.metadata.frame_number;
    uint32_t right_num = right_frame_.metadata.frame_number;
    
    // 严格匹配帧号
    if (left_num != right_num) {
        sync_mismatch_count_++;
        
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "⚠️  帧号不匹配: Left#%u vs Right#%u (USB延迟差异)",
            left_num, right_num);
        
        // 丢弃旧帧，保留新帧
        if (left_num < right_num) {
            left_frame_.reset();  // 丢弃左帧
        } else {
            right_frame_.reset();  // 丢弃右帧
        }
        return false;
    }
    
    // 同步成功
    sync_success_count_++;
    
    // 拷贝数据
    left = left_frame_.image.clone();
    right = right_frame_.image.clone();
    left_meta = left_frame_.metadata;
    right_meta = right_frame_.metadata;
    
    // 重置标志位
    left_frame_.reset();
    right_frame_.reset();
    
    return true;
}
```

### 6. **重构 inferenceLoop()** - 主推理循环
```cpp
void VolleyballTrackerNode::inferenceLoop() {
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已启动 (PWM时间戳同步模式)");
    
    auto stats_start = this->now();
    int frame_count = 0;
    
    while (rclcpp::ok() && running_) {
        // ========== 更新相机帧 (轮询回调) ==========
        updateLeftFrame();
        updateRightFrame();
        
        // ========== 等待同步帧对 ==========
        cv::Mat left, right;
        FrameMetadata left_meta, right_meta;
        
        if (!waitForSyncedPair(left, right, left_meta, right_meta)) {
            // 无同步帧，短暂休眠后重试
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        
        // ========== Batch=2 推理 ==========
        auto t_start = std::chrono::steady_clock::now();
        
        if (!detectVolleyball(left, right)) {
            continue;
        }
        
        auto t_detect = std::chrono::steady_clock::now();
        
        // 立体匹配 + 卡尔曼滤波
        if (computeStereoMatch()) {
            updateTracker();
            publishResults();
        }
        
        auto t_end = std::chrono::steady_clock::now();
        
        // ========== 统计 ==========
        frame_count++;
        if (frame_count % 100 == 0) {
            auto dt_detect = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_detect - t_start).count();
            auto dt_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_end - t_start).count();
            
            RCLCPP_INFO(this->get_logger(),
                "🚀 [推理 100帧] FPS: %.1f | 检测: %ldms | "
                "同步成功: %lu | 失配: %lu | 丢帧: L=%lu R=%lu",
                100000.0 / (this->now() - stats_start).seconds(),
                dt_detect,
                sync_success_count_.load(),
                sync_mismatch_count_.load(),
                left_dropped_.load(),
                right_dropped_.load()
            );
            
            stats_start = this->now();
            frame_count = 0;
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已退出");
}
```

### 7. **修改 detectVolleyball()** 签名
```cpp
// 原签名
bool VolleyballTrackerNode::detectVolleyball() {
    // 从成员变量读取 img_left_, img_right_
}

// 新签名
bool VolleyballTrackerNode::detectVolleyball(const cv::Mat& left, const cv::Mat& right) {
    // 直接使用参数
    img_left_ = left;   // 保存到成员变量供后续使用
    img_right_ = right;
    
    // 原有检测逻辑不变
    ...
}
```

### 8. **修改析构函数**
```cpp
VolleyballTrackerNode::~VolleyballTrackerNode() {
    running_.store(false);
    
    // 原代码：等待两个线程
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    
    // 新代码：只等待推理线程
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
    
    // ... 其他清理代码
}
```

---

## 🔍 关键修改点总结

| 修改项 | 原设计 | 新设计 | 理由 |
|--------|--------|--------|------|
| 线程模型 | 采集线程 + 推理线程 | 只有推理线程 | 相机回调已异步 |
| 数据传递 | FrameQueue<3> 队列 | 全局帧缓冲 + 标志位 | 参考RC项目 |
| 同步策略 | 无同步保证 | 帧号严格匹配 | PWM时间戳同步 |
| 帧获取 | waitForNewFrame阻塞 | 轮询标志位非阻塞 | 避免串行等待 |
| 错帧处理 | 无检测 | 自动丢弃旧帧 | 容错机制 |

---

## 📝 完整实施步骤

1. 在 cpp 文件中添加 `updateLeftFrame()` 和 `updateRightFrame()`
2. 添加 `waitForSyncedPair()` 核心同步逻辑
3. 重写 `inferenceLoop()` 为轮询模式
4. 修改 `detectVolleyball()` 接受参数
5. 删除 `captureLoop()` 方法
6. 修改 `initialize()` 和 `~VolleyballTrackerNode()`
7. 编译测试

---

## ✅ 下一步

准备好后，我将逐个实施这些修改。
