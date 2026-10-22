/**
 * @file zero_copy_alloc.h
 * @brief Xavier NX Zero-Copy 统一内存分配器
 *
 * Xavier NX 是 SoC 统一内存架构 (CPU/GPU 共享 LPDDR4x)，
 * cudaHostAllocMapped 映射到同一物理页面，真正零开销。
 */

#ifndef STEREO_3D_PIPELINE_ZERO_COPY_ALLOC_H_
#define STEREO_3D_PIPELINE_ZERO_COPY_ALLOC_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace stereo3d {

/**
 * @brief 零拷贝缓冲区：CPU/GPU 共享同一物理内存
 *
 * 在 Xavier NX 统一内存架构上，cudaHostAllocMapped 分配的内存
 * CPU 和 GPU 可同时访问同一物理地址，无需 cudaMemcpy。
 */
struct ZeroCopyBuffer {
    void* host_ptr   = nullptr;   ///< CPU 可访问指针
    void* device_ptr = nullptr;   ///< GPU 可访问指针 (与 host_ptr 指向同一物理内存)
    size_t size      = 0;         ///< 缓冲区大小 (bytes)

    bool valid() const { return host_ptr != nullptr && device_ptr != nullptr; }
};

/**
 * @brief 分配 Zero-Copy 共享内存
 * @param size 需要的字节数
 * @return ZeroCopyBuffer 包含 CPU/GPU 双指针
 */
inline ZeroCopyBuffer allocZeroCopy(size_t size) {
    ZeroCopyBuffer buf;
    buf.size = size;

    cudaError_t err = cudaHostAlloc(&buf.host_ptr, size, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ZeroCopy] cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
        buf.host_ptr = nullptr;
        return buf;
    }

    err = cudaHostGetDevicePointer(&buf.device_ptr, buf.host_ptr, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ZeroCopy] cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
        cudaFreeHost(buf.host_ptr);
        buf.host_ptr = nullptr;
        buf.device_ptr = nullptr;
        return buf;
    }

    return buf;
}

/**
 * @brief 释放 Zero-Copy 缓冲区
 */
inline void freeZeroCopy(ZeroCopyBuffer& buf) {
    if (buf.host_ptr) {
        cudaFreeHost(buf.host_ptr);
        buf.host_ptr = nullptr;
        buf.device_ptr = nullptr;
        buf.size = 0;
    }
}

/**
 * @brief Zero-Copy 缓冲池 (用于三缓冲 Ring Buffer)
 *
 * 预分配 N 组缓冲区，Pipeline 各 Stage 通过索引轮流使用。
 */
class ZeroCopyPool {
public:
    ZeroCopyPool() = default;
    ~ZeroCopyPool() { release(); }

    ZeroCopyPool(const ZeroCopyPool&) = delete;
    ZeroCopyPool& operator=(const ZeroCopyPool&) = delete;

    /**
     * @brief 初始化缓冲池
     * @param count 缓冲区数量 (通常 3 = 三缓冲)
     * @param size_per_buffer 每个缓冲区大小 (bytes)
     * @return true 全部分配成功
     */
    bool init(int count, size_t size_per_buffer) {
        release();
        buffers_.resize(count);
        for (int i = 0; i < count; ++i) {
            buffers_[i] = allocZeroCopy(size_per_buffer);
            if (!buffers_[i].valid()) {
                fprintf(stderr, "[ZeroCopyPool] Failed to allocate buffer %d/%d\n", i, count);
                release();
                return false;
            }
        }
        return true;
    }

    void release() {
        for (auto& buf : buffers_) {
            freeZeroCopy(buf);
        }
        buffers_.clear();
    }

    int count() const { return static_cast<int>(buffers_.size()); }
    ZeroCopyBuffer& operator[](int idx) { return buffers_[idx % count()]; }
    const ZeroCopyBuffer& operator[](int idx) const { return buffers_[idx % count()]; }

    bool empty() const { return buffers_.empty(); }

private:
    std::vector<ZeroCopyBuffer> buffers_;
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_ZERO_COPY_ALLOC_H_
