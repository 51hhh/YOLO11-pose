/**
 * @file sync.h
 * @brief CUDA Event/Stream 同步工具
 *
 * 管理 Pipeline 使用的所有 CUDA Streams:
 *   - vpiStreamPVA : Stage 0 校正（历史名称，backend 由配置决定）
 *   - cudaStreamDLA : Stage 1 左目检测（历史名称，不代表必然使用 DLA）
 *   - cudaStreamDLA_R : Stage 1 右目检测（历史名称，不代表必然使用 DLA）
 *   - streamGPU : Stage 2 视差 (GPU CUDA)
 *   - streamFuse: Stage 3 融合
 */

#ifndef STEREO_3D_PIPELINE_SYNC_H_
#define STEREO_3D_PIPELINE_SYNC_H_

#include <cuda_runtime.h>
#include <vpi/Stream.h>
#include <cstdio>

namespace stereo3d {

struct PipelineStreamRequirements {
    bool full_frame_stereo = false;
    bool p2_ncc = false;
    bool p2_xfeat = false;
    bool p2_superpoint = false;
    bool p2_aliked = false;
};

/**
 * @brief Pipeline 同步上下文
 *
 * 封装 Pipeline 主执行图固定创建的 CUDA/VPI stream。异步 ROI 与 P2 诊断的
 * 条件 stream 由 Pipeline 单独持有，不在本结构中。左右检测 stream 和 GPU
 * stream 可独立工作，但实际 backend 与并发仍须由最终配置和运行轨迹确认。
 */
struct PipelineStreams {
    // ===== VPI Streams =====
    VPIStream vpiStreamPVA  = nullptr;   ///< 历史名称：Stage 0 校正，可选 PVA/CUDA/VIC
    VPIStream vpiStreamGPU  = nullptr;   ///< CUDA backend (Stage 2 视差)

    // ===== CUDA Streams =====
    cudaStream_t cudaStreamDLA  = nullptr;   ///< 历史名称：左目检测 CUDA Stream
    cudaStream_t cudaStreamDLA_R = nullptr;  ///< 历史名称：右目检测 CUDA Stream
    cudaStream_t cudaStreamGPU  = nullptr;   ///< GPU 视差 CUDA Stream
    cudaStream_t cudaStreamFuse = nullptr;   ///< 融合 Stage CUDA Stream
    cudaStream_t cudaStreamBGR  = nullptr;   ///< Bayer→BGR CUDA kernel Stream
    cudaStream_t cudaStreamP2Ncc = nullptr;  ///< P2 inline NCC/Template Stream
    cudaStream_t cudaStreamP2XFeat = nullptr; ///< P2 inline XFeat Stream
    cudaStream_t cudaStreamP2SuperPoint = nullptr; ///< P2 inline SuperPoint Stream
    cudaStream_t cudaStreamP2Aliked = nullptr; ///< P2 inline ALIKED Stream

    /**
     * @brief 初始化所有 Streams
     * @return true 全部成功
     */
    bool init(const PipelineStreamRequirements& requirements) {
        cudaError_t err;

        // VPI Streams (VIC backend added for hardware remap)
        VPIStatus vpiErr;
        vpiErr = vpiStreamCreate(VPI_BACKEND_PVA | VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &vpiStreamPVA);
        if (vpiErr != VPI_SUCCESS) {
            fprintf(stderr, "[Sync] Failed to create VPI PVA stream\n");
            return false;
        }

        if (requirements.full_frame_stereo) {
            vpiErr = vpiStreamCreate(VPI_BACKEND_CUDA, &vpiStreamGPU);
            if (vpiErr != VPI_SUCCESS) {
                fprintf(stderr, "[Sync] Failed to create VPI GPU stream\n");
                destroy();
                return false;
            }
        }

        // CUDA Streams (non-blocking)
        err = cudaStreamCreateWithFlags(&cudaStreamDLA, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create DLA stream: %s\n", cudaGetErrorString(err));
            destroy();
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamDLA_R, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create right DLA stream: %s\n", cudaGetErrorString(err));
            destroy();
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamGPU, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create GPU stream: %s\n", cudaGetErrorString(err));
            destroy();
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamFuse, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create Fuse stream: %s\n", cudaGetErrorString(err));
            destroy();
            return false;
        }

        err = cudaStreamCreateWithFlags(&cudaStreamBGR, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Sync] Failed to create BGR stream: %s\n", cudaGetErrorString(err));
            destroy();
            return false;
        }

        if (requirements.p2_ncc) {
            err = cudaStreamCreateWithFlags(
                &cudaStreamP2Ncc, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                fprintf(stderr, "[Sync] Failed to create P2 NCC stream: %s\n",
                        cudaGetErrorString(err));
                destroy();
                return false;
            }
        }

        if (requirements.p2_xfeat) {
            err = cudaStreamCreateWithFlags(
                &cudaStreamP2XFeat, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                fprintf(stderr, "[Sync] Failed to create P2 XFeat stream: %s\n",
                        cudaGetErrorString(err));
                destroy();
                return false;
            }
        }

        if (requirements.p2_superpoint) {
            err = cudaStreamCreateWithFlags(
                &cudaStreamP2SuperPoint, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[Sync] Failed to create P2 SuperPoint stream: %s\n",
                        cudaGetErrorString(err));
                destroy();
                return false;
            }
        }

        if (requirements.p2_aliked) {
            err = cudaStreamCreateWithFlags(
                &cudaStreamP2Aliked, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "[Sync] Failed to create P2 ALIKED stream: %s\n",
                        cudaGetErrorString(err));
                destroy();
                return false;
            }
        }

        return true;
    }

    /**
     * @brief 等待所有 Streams 完成当前工作
     */
    bool syncAll() {
        bool ok = true;
        auto sync_vpi = [&ok](VPIStream stream, const char* name) {
            if (!stream) return;
            const VPIStatus status = vpiStreamSync(stream);
            if (status != VPI_SUCCESS) {
                fprintf(stderr, "[Sync] Failed to synchronize %s: status=%d\n",
                        name, static_cast<int>(status));
                ok = false;
            }
        };
        auto sync_cuda = [&ok](cudaStream_t stream, const char* name) {
            if (!stream) return;
            const cudaError_t status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                fprintf(stderr, "[Sync] Failed to synchronize %s: %s\n",
                        name, cudaGetErrorString(status));
                ok = false;
            }
        };

        sync_vpi(vpiStreamPVA, "VPI rectification stream");
        sync_vpi(vpiStreamGPU, "VPI stereo stream");
        sync_cuda(cudaStreamDLA, "left TensorRT stream");
        sync_cuda(cudaStreamDLA_R, "right TensorRT stream");
        sync_cuda(cudaStreamGPU, "pipeline GPU stream");
        sync_cuda(cudaStreamFuse, "fusion stream");
        sync_cuda(cudaStreamBGR, "Bayer-to-BGR stream");
        sync_cuda(cudaStreamP2Ncc, "P2 NCC stream");
        sync_cuda(cudaStreamP2XFeat, "P2 XFeat stream");
        sync_cuda(cudaStreamP2SuperPoint, "P2 SuperPoint stream");
        sync_cuda(cudaStreamP2Aliked, "P2 ALIKED stream");
        return ok;
    }

    /**
     * @brief 销毁所有 Streams
     */
    void destroy() {
        if (vpiStreamPVA)  { vpiStreamDestroy(vpiStreamPVA);  vpiStreamPVA = nullptr;  }
        if (vpiStreamGPU)  { vpiStreamDestroy(vpiStreamGPU);  vpiStreamGPU = nullptr;  }
        if (cudaStreamDLA) { cudaStreamDestroy(cudaStreamDLA); cudaStreamDLA = nullptr; }
        if (cudaStreamDLA_R) { cudaStreamDestroy(cudaStreamDLA_R); cudaStreamDLA_R = nullptr; }
        if (cudaStreamGPU) { cudaStreamDestroy(cudaStreamGPU); cudaStreamGPU = nullptr; }
        if (cudaStreamFuse){ cudaStreamDestroy(cudaStreamFuse); cudaStreamFuse = nullptr; }
        if (cudaStreamBGR) { cudaStreamDestroy(cudaStreamBGR);  cudaStreamBGR = nullptr; }
        if (cudaStreamP2Ncc) {
            cudaStreamDestroy(cudaStreamP2Ncc);
            cudaStreamP2Ncc = nullptr;
        }
        if (cudaStreamP2XFeat) {
            cudaStreamDestroy(cudaStreamP2XFeat);
            cudaStreamP2XFeat = nullptr;
        }
        if (cudaStreamP2SuperPoint) {
            cudaStreamDestroy(cudaStreamP2SuperPoint);
            cudaStreamP2SuperPoint = nullptr;
        }
        if (cudaStreamP2Aliked) {
            cudaStreamDestroy(cudaStreamP2Aliked);
            cudaStreamP2Aliked = nullptr;
        }
    }
};

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_SYNC_H_
