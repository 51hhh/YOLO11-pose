/**
 * @file sync.h
 * @brief CUDA Event/Stream 同步工具
 *
 * 管理 Pipeline 使用的所有 CUDA Streams:
 *   - streamPVA : Stage 0 校正 (PVA backend)
 *   - streamDLA : Stage 1 检测 (DLA 推理)
 *   - streamGPU : Stage 2 视差 (GPU CUDA)
 *   - streamFuse: Stage 3 融合
 */

#ifndef STEREO_3D_PIPELINE_SYNC_H_
#define STEREO_3D_PIPELINE_SYNC_H_

#include <cuda_runtime.h>
#include <vpi/Stream.h>
#include <cstdio>

namespace stereo3d {

/**
 * @brief Pipeline 同步上下文
 *
 * 封装 Pipeline 所需的全部 CUDA Stream 和 VPI Stream。
 * DLA Stream 和 GPU Stream 独立工作，实现 Stage 1 / Stage 2 并行。
 */
struct PipelineStreams {
    // ===== VPI Streams =====
    VPIStream vpiStreamPVA  = nullptr;   ///< PVA backend (Stage 0 校正)
    VPIStream vpiStreamGPU  = nullptr;   ///< CUDA backend (Stage 2 视差)

    // ===== CUDA Streams =====
    cudaStream_t cudaStreamDLA  = nullptr;   ///< DLA 推理 CUDA Stream
    cudaStream_t cudaStreamGPU  = nullptr;   ///< GPU 视差 CUDA Stream
    cudaStream_t cudaStreamFuse = nullptr;   ///< 融合 Stage CUDA Stream

    /**
     * @brief 初始化所有 Streams
     * @return true 全部成功
     */
    bool init() {
        cudaError_t err;

        // VPI Streams
        VPIStatus vpiErr;
        vpiErr = vpiStreamCreate(VPI_BACKEND_PVA | VPI_BACKEND_CUDA, &vpiStreamPVA);
        if (vpiErr != VPI_SUCCESS) {
            fprintf(stderr, "[Sync] Failed to create VPI PVA stream\n");
            return false;
        }

        vpiErr = vpiStreamCreate(VPI_BACKEND_CUDA, &vpiStreamGPU);
        if (vpiErr != VPI_SUCCESS) {
            fprintf(stderr, "[Sync] Failed to create VPI GPU stream\n");
            return false;
        }

        // CUDA Streams (non-blocking)
        err = cudaStreamCreateWithFlags(&cudaStreamDLA, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create DLA stream: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamGPU, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create GPU stream: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamFuse, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create Fuse stream: %s\n", cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    /**
     * @brief 等待所有 Streams 完成当前工作
     */
    void syncAll() {
        if (vpiStreamPVA) vpiStreamSync(vpiStreamPVA);
        if (vpiStreamGPU) vpiStreamSync(vpiStreamGPU);
        if (cudaStreamDLA) cudaStreamSynchronize(cudaStreamDLA);
        if (cudaStreamGPU) cudaStreamSynchronize(cudaStreamGPU);
        if (cudaStreamFuse) cudaStreamSynchronize(cudaStreamFuse);
    }

    /**
     * @brief 销毁所有 Streams
     */
    void destroy() {
        if (vpiStreamPVA)  { vpiStreamDestroy(vpiStreamPVA);  vpiStreamPVA = nullptr;  }
        if (vpiStreamGPU)  { vpiStreamDestroy(vpiStreamGPU);  vpiStreamGPU = nullptr;  }
        if (cudaStreamDLA) { cudaStreamDestroy(cudaStreamDLA); cudaStreamDLA = nullptr; }
        if (cudaStreamGPU) { cudaStreamDestroy(cudaStreamGPU); cudaStreamGPU = nullptr; }
        if (cudaStreamFuse){ cudaStreamDestroy(cudaStreamFuse); cudaStreamFuse = nullptr; }
    }
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_SYNC_H_
