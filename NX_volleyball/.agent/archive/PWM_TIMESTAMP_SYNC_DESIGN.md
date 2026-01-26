# PWM时间戳同步设计方案

## 📋 问题分析

### 当前状态
- ✅ **PWM同步触发**: 100Hz，两个相机同时曝光
- ✅ **曝光时间一致**: 9867us (约10ms)
- ❌ **到达时间不一致**: USB传输延迟差异（2-8ms不等）

### 风险场景

```
PWM周期: 10ms (100Hz)

理想同步:
T0      T10     T20     T30 (ms)
├───────┼───────┼───────┤
│ #100  │ #101  │ #102  │  Left Camera
│ #100  │ #101  │ #102  │  Right Camera
└───────┴───────┴───────┘

风险场景 (USB延迟导致错配):
T0      T10     T20     T30 (ms)
├───────┼───────┼───────┤
│ #100  │ #101  │ #102  │  Left (到达: 12ms, 22ms, 32ms)
  │ #100  │ #101  │ #102│  Right (到达: 18ms, 28ms, 38ms)
  └───────┴───────┴───────┘
      ↑ 错误配对: Left#100 + Right#101 ❌
```

---

## 🎯 解决方案：帧号匹配法

### 核心原理

**海康SDK保证**：
- PWM触发时，两个相机**同步曝光**
- 相机内部帧号 `nFrameNum` **同步递增**
- 虽然到达时间不同，但 `frame_number` **相同即同源**

### 实现策略

```cpp
// 扩展帧数据结构
struct SyncedFrame {
    cv::Mat image;
    uint32_t frame_number;      // 海康SDK: pFrame->stFrameInfo.nFrameNum
    uint64_t device_timestamp;  // 可选: 设备时间戳(微秒)
    rclcpp::Time host_timestamp; // 主机接收时间
};

// 同步逻辑
struct FramePair {
    SyncedFrame left;
    SyncedFrame right;
    bool is_synced = false;
};

// 匹配判断
bool isSynced(const SyncedFrame& left, const SyncedFrame& right) {
    // 方案1: 帧号完全匹配 (推荐)
    return left.frame_number == right.frame_number;
    
    // 方案2: 容忍1帧误差 (宽松)
    // return abs((int)left.frame_number - (int)right.frame_number) <= 1;
    
    // 方案3: 设备时间戳匹配 (高精度)
    // return abs(left.device_timestamp - right.device_timestamp) < 500; // 500us
}
```

---

## 💻 代码实现

### 1. 修改相机回调获取帧号

#### 文件: `hik_camera_wrapper.cpp`

```cpp
// 回调函数修改
void HikCamera::onImageCallback(MV_FRAME_OUT* pFrame) {
    if (!pFrame || !pFrame->pBufAddr) return;
    
    // ========== 新增: 提取帧号和时间戳 ==========
    uint32_t frame_num = pFrame->stFrameInfo.nFrameNum;
    uint64_t dev_timestamp = ((uint64_t)pFrame->stFrameInfo.nDevTimeStampHigh << 32) 
                           | pFrame->stFrameInfo.nDevTimeStampLow;
    
    // 获取写入缓冲区索引
    int idx = write_index_.load();
    cv::Mat& dst = frame_buffer_[idx];
    
    // 图像数据转换 (原有代码)
    MvGvspPixelType pixelType = pFrame->stFrameInfo.enPixelType;
    // ... (转换逻辑不变)
    
    // ========== 新增: 保存元数据 ==========
    frame_metadata_[idx].frame_number = frame_num;
    frame_metadata_[idx].device_timestamp = dev_timestamp;
    frame_metadata_[idx].host_timestamp = std::chrono::steady_clock::now();
    
    // 切换写入索引
    write_index_.store(1 - idx);
    
    // 通知有新帧
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        read_index_.store(idx);
        new_frame_ready_.store(true);
    }
    frame_cv_.notify_one();
}

// 新增: 获取帧元数据
FrameMetadata HikCamera::getFrameMetadata() const {
    int idx = read_index_.load();
    return frame_metadata_[idx];
}
```

#### 文件: `hik_camera_wrapper.hpp`

```cpp
// 新增结构体
struct FrameMetadata {
    uint32_t frame_number = 0;
    uint64_t device_timestamp = 0;  // 微秒
    std::chrono::steady_clock::time_point host_timestamp;
};

class HikCamera {
private:
    std::array<FrameMetadata, 2> frame_metadata_;  // 双缓冲元数据
    
public:
    FrameMetadata getFrameMetadata() const;
};
```

---

### 2. 修改主节点为标志位同步模式

#### 文件: `volleyball_tracker_node.hpp`

```cpp
class VolleyballTrackerNode : public rclcpp::Node {
private:
    // ========== 移除队列，改为全局帧 + 标志位 ==========
    // FrameQueue<3> frame_queue_;  // 删除
    
    // 全局帧缓冲 + 元数据
    struct CameraFrame {
        cv::Mat image;
        FrameMetadata metadata;
        std::atomic<bool> ready{false};
    };
    
    CameraFrame left_frame_;
    CameraFrame right_frame_;
    std::mutex frame_sync_mutex_;
    
    // 统计
    std::atomic<uint64_t> sync_success_count_{0};
    std::atomic<uint64_t> sync_mismatch_count_{0};
    std::atomic<uint64_t> left_dropped_{0};
    std::atomic<uint64_t> right_dropped_{0};
    
    // 线程
    std::thread inference_thread_;  // 只需1个推理线程
    
    // 方法
    void leftCameraCallback();      // 左相机回调
    void rightCameraCallback();     // 右相机回调
    void inferenceLoop();           // 推理线程
    bool waitForSyncedPair(cv::Mat& left, cv::Mat& right, 
                          FrameMetadata& left_meta, 
                          FrameMetadata& right_meta);
};
```

#### 文件: `volleyball_tracker_node.cpp`

```cpp
// ========== 相机回调改造 ==========
void VolleyballTrackerNode::leftCameraCallback() {
    // 等待新帧（非阻塞，回调模式已经异步）
    if (!cam_left_->waitForNewFrame(5)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 如果上一帧还未被处理，丢弃
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

void VolleyballTrackerNode::rightCameraCallback() {
    // 同理
    if (!cam_right_->waitForNewFrame(5)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    if (right_frame_.ready.load()) {
        right_dropped_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "⚠️  Right frame dropped");
    }
    
    right_frame_.image = cam_right_->getLatestImage().clone();
    right_frame_.metadata = cam_right_->getFrameMetadata();
    right_frame_.ready.store(true);
}

// ========== 推理线程 (轮询标志位) ==========
void VolleyballTrackerNode::inferenceLoop() {
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已启动 (PWM时间戳同步模式)");
    
    auto stats_start = this->now();
    int frame_count = 0;
    
    while (rclcpp::ok() && running_) {
        cv::Mat left, right;
        FrameMetadata left_meta, right_meta;
        
        // 等待同步帧对
        if (!waitForSyncedPair(left, right, left_meta, right_meta)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        
        // ========== Batch=2 推理 ==========
        auto t_start = std::chrono::steady_clock::now();
        
        std::vector<Detection> left_dets, right_dets;
        detector_->detectBatch2(left, right, left_dets, right_dets);
        
        auto t_detect = std::chrono::steady_clock::now();
        
        // 立体匹配 + 卡尔曼滤波
        auto ball_3d = stereo_matcher_->match(left_dets, right_dets);
        kalman_filter_->update(ball_3d);
        
        auto t_end = std::chrono::steady_clock::now();
        
        // 发布结果
        publishResults(ball_3d, left, right, left_meta.frame_number);
        
        // 统计
        frame_count++;
        if (frame_count % 100 == 0) {
            auto dt_detect = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_detect - t_start).count();
            auto dt_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_end - t_start).count();
            
            RCLCPP_INFO(this->get_logger(), 
                "🚀 [推理 100帧] FPS: %.1f | 检测: %ldms | 总计: %ldms | "
                "同步成功: %lu | 失配: %lu",
                100000.0 / (this->now() - stats_start).seconds(),
                dt_detect, dt_total,
                sync_success_count_.load(),
                sync_mismatch_count_.load()
            );
            
            stats_start = this->now();
            frame_count = 0;
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "🧠 推理线程已退出");
}

// ========== 等待同步帧对 ==========
bool VolleyballTrackerNode::waitForSyncedPair(
    cv::Mat& left, cv::Mat& right,
    FrameMetadata& left_meta, FrameMetadata& right_meta) 
{
    std::lock_guard<std::mutex> lock(frame_sync_mutex_);
    
    // 检查两帧是否都ready
    if (!left_frame_.ready.load() || !right_frame_.ready.load()) {
        return false;
    }
    
    // ========== 帧号同步检查 ==========
    uint32_t left_num = left_frame_.metadata.frame_number;
    uint32_t right_num = right_frame_.metadata.frame_number;
    
    // 方案1: 严格匹配 (推荐)
    if (left_num != right_num) {
        sync_mismatch_count_++;
        
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "⚠️  帧号不匹配: Left#%u vs Right#%u (USB延迟差异)",
            left_num, right_num);
        
        // 丢弃旧帧，保留新帧
        if (left_num < right_num) {
            left_frame_.ready.store(false);  // 丢弃左帧
        } else {
            right_frame_.ready.store(false);  // 丢弃右帧
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
    left_frame_.ready.store(false);
    right_frame_.ready.store(false);
    
    return true;
}

// ========== 初始化修改 ==========
bool VolleyballTrackerNode::initialize() {
    // ... (PWM、相机初始化不变)
    
    // 启动推理线程 (只需1个)
    inference_thread_ = std::thread(&VolleyballTrackerNode::inferenceLoop, this);
    
    RCLCPP_INFO(this->get_logger(), "✅ 排球追踪节点已启动 (PWM时间戳同步)");
    RCLCPP_INFO(this->get_logger(), "   同步策略: 帧号严格匹配");
    
    return true;
}
```

---

## 📊 优势对比

### 当前队列方案 vs 时间戳同步方案

| 维度 | 队列方案 | 时间戳同步方案 |
|------|---------|---------------|
| **时间同步精度** | 无保证 (顺序到达≠同源) | **严格保证** (帧号匹配) |
| **延迟容忍** | 差 (队列阻塞) | 优秀 (异步丢旧帧) |
| **内存开销** | 高 (队列3帧) | 低 (单帧缓冲) |
| **锁竞争** | 中 (push/pop) | 低 (轮询标志位) |
| **错帧检测** | ❌ 无法检测 | ✅ **自动检测并丢弃** |
| **代码复杂度** | 高 (队列管理) | 低 (标志位) |

---

## 🎯 预期效果

### 1. **时间同步保证**
```
✅ Left#100 + Right#100  → Batch推理
✅ Left#101 + Right#101  → Batch推理
❌ Left#100 + Right#101  → 检测失配，丢弃Left#100，等待Right#100
```

### 2. **性能提升**
- **采集FPS**: 100 Hz (PWM频率上限)
- **推理FPS**: 55-60 Hz (16ms batch2推理 + 采集并行)
- **丢帧策略**: 智能丢旧帧，保持最新

### 3. **错误恢复**
```
场景: USB拔插导致Right相机掉帧

T0: Left#100, Right#100 → ✅ 同步推理
T1: Left#101, Right#??? → ⏸ Right未到达，等待
T2: Left#102, Right#101 → ❌ 检测失配 (102≠101)
                        → 丢弃Right#101
T3: Left#102, Right#102 → ✅ 恢复同步
```

---

## 🛠️ 实施步骤

### 阶段1: 扩展相机元数据 ✅
1. 修改 `hik_camera_wrapper.hpp`: 添加 `FrameMetadata` 结构
2. 修改 `hik_camera_wrapper.cpp`: 回调提取 `frame_number` + `device_timestamp`
3. 添加 `getFrameMetadata()` 接口

### 阶段2: 重构主节点 🔄
1. 移除 `FrameQueue<3>`
2. 添加 `left_frame_` + `right_frame_` 全局缓冲
3. 实现 `leftCameraCallback()` + `rightCameraCallback()`
4. 实现 `waitForSyncedPair()` 匹配逻辑
5. 修改 `inferenceLoop()` 为单线程轮询

### 阶段3: 测试验证 🧪
1. 日志输出帧号差异统计
2. 人为制造USB延迟（拔插测试）
3. 验证错帧自动恢复
4. 性能测试（预期55-60 FPS）

---

## 📝 完全参考RC_Volleyball_vision

### RC项目的精髓

```cpp
// RC项目: grab_hikcam 线程 (生产者)
void grab_img_hikcam() {
    while (state.load()) {
        if (hik_img_flag == SUCCESS) {
            usleep(10);  // 上一帧未处理，小延迟
            continue;
        }
        
        hik_cam.get_one_frame(frame_hik, 0);  // 采集到全局
        hik_img_flag = SUCCESS;               // 设置标志
    }
}

// RC项目: detect_process 线程 (消费者)
void detect_process() {
    while (state.load()) {
        if (usb_img_flag && hik_img_flag) {  // 两帧都ready
            hik_img_flag = NO_IMAGE;  // 立即重置
            usb_img_flag = NO_IMAGE;
            
            detector.push_img(frame_hik, 1);  // clone
            detector.push_img(frame_usb, 2);
            
            detector.preprocess();  // cuda_2batch_preprocess
            detector.infer();       // enqueueV2
            detector.postprocess();
        }
    }
}
```

**我们的改进**：
1. ✅ **保留标志位模式** (RC的核心设计)
2. ✅ **保留全局帧缓冲** (zero-copy)
3. ✅ **保留batch2推理** (单次enqueueV3)
4. ➕ **增加帧号匹配** (解决时间同步问题) ← 新增
5. ➕ **增加错帧检测** (生产级容错) ← 新增

---

## ✅ 结论

**完全可行且强烈推荐！**

这个方案：
1. ✅ **技术成熟**: 基于RC_Volleyball_vision验证的标志位架构
2. ✅ **硬件保证**: PWM触发→帧号同步，无需软件时钟同步
3. ✅ **容错性强**: 自动检测并恢复错帧
4. ✅ **性能优秀**: 预期55-60 FPS（接近硬件上限）
5. ✅ **代码简洁**: 比队列方案更简单

**立即开始实施？**
