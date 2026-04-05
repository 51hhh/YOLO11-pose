/**
 * @file dfl_postprocess.cu
 * @brief DFL 6-tensor 后处理 CUDA Kernel
 *
 * 将 6-tensor DFL 解码 (softmax + dist2bbox + NMS) 从 CPU 移到 GPU:
 *   - sigmoid(cls) → 置信度筛选
 *   - DFL softmax (4 edges × 16 bins) → 4 偏移量
 *   - dist2bbox: anchor ± offset × stride → (cx, cy, w, h)
 *   - 输出: 候选框列表 (GPU→CPU 后做 NMS)
 *
 * 预期加速: CPU ~0.3ms → GPU <0.05ms
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 共享内存中的 DFL softmax 计算
__device__ float dfl_decode_edge(const float* dfl, int reg_max) {
    // Numerically stable softmax across reg_max bins
    float max_val = dfl[0];
    for (int k = 1; k < reg_max; ++k)
        max_val = fmaxf(max_val, dfl[k]);

    float sum_exp = 0.0f;
    float weighted = 0.0f;
    for (int k = 0; k < reg_max; ++k) {
        float e = __expf(dfl[k] - max_val);
        sum_exp += e;
        weighted += e * (float)k;
    }
    return weighted / (sum_exp + 1e-9f);
}

/**
 * @brief DFL 解码 Kernel - 处理单个尺度
 *
 * 每个线程处理一个 grid cell:
 *   1. 读取 cls logit → sigmoid → 阈值过滤
 *   2. DFL softmax decode → 4 offsets
 *   3. dist2bbox → (cx, cy, w, h)
 *   4. 原子写入输出缓冲
 *
 * @param cls_data     分类数据 [H, W, nc] (NHWC without batch)
 * @param bbox_data    边界框数据 [H, W, 4*reg_max]
 * @param out_boxes    输出: [max_det, 6] (cx, cy, w, h, score, cls_id)
 * @param out_count    输出: 检测到的框数 (atomic)
 * @param H, W         空间维度
 * @param nc           类别数
 * @param reg_max      DFL bin 数 (16)
 * @param stride       当前尺度步长 (8/16/32)
 * @param conf_thresh  置信度阈值
 * @param max_det      最大检测数
 */
__global__ void dflDecodeKernel(
    const float* __restrict__ cls_data,
    const float* __restrict__ bbox_data,
    float* __restrict__ out_boxes,
    int* __restrict__ out_count,
    int H, int W, int nc, int reg_max,
    int stride, float conf_thresh, int max_det)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;

    int y = idx / W;
    int x = idx % W;

    // Find max class score
    float max_score = 0.0f;
    int max_cls = 0;
    for (int c = 0; c < nc; ++c) {
        float logit = cls_data[idx * nc + c];
        // Sigmoid: 1 / (1 + exp(-x))
        float score = 1.0f / (1.0f + __expf(-logit));
        if (score > max_score) {
            max_score = score;
            max_cls = c;
        }
    }

    if (max_score < conf_thresh) return;

    // DFL decode: 4 edges
    const float* bbox_row = bbox_data + idx * (4 * reg_max);
    float l = dfl_decode_edge(bbox_row + 0 * reg_max, reg_max);
    float t = dfl_decode_edge(bbox_row + 1 * reg_max, reg_max);
    float r = dfl_decode_edge(bbox_row + 2 * reg_max, reg_max);
    float b = dfl_decode_edge(bbox_row + 3 * reg_max, reg_max);

    // dist2bbox
    float ax = (x + 0.5f) * stride;
    float ay = (y + 0.5f) * stride;
    float x1 = ax - l * stride;
    float y1 = ay - t * stride;
    float x2 = ax + r * stride;
    float y2 = ay + b * stride;

    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = x2 - x1;
    float h = y2 - y1;

    // Atomic output
    int pos = atomicAdd(out_count, 1);
    if (pos < max_det) {
        float* row = out_boxes + pos * 6;
        row[0] = cx;
        row[1] = cy;
        row[2] = w;
        row[3] = h;
        row[4] = max_score;
        row[5] = (float)max_cls;
    }
}

// ==================== C 接口 ====================

/**
 * @brief 启动 DFL 解码 kernel
 *
 * 调用方负责:
 *   1. cls_data 和 bbox_data 已在 GPU 上（enqueue 后直接使用 device buffer）
 *   2. out_boxes 预分配 [max_det * 6] floats 在 GPU 上
 *   3. out_count 初始化为 0 (在 GPU 上)
 *   4. 调用后 cudaMemcpy out_count 和 out_boxes 到 host
 *
 * @return 需要的输出 buffer 字节数 = max_det * 6 * sizeof(float) + sizeof(int)
 */
extern "C" void launchDFLDecodeKernel(
    const float* cls_data,
    const float* bbox_data,
    float* out_boxes,
    int* out_count,
    int H, int W, int nc, int reg_max,
    int stride, float conf_thresh, int max_det,
    cudaStream_t stream)
{
    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dflDecodeKernel<<<blocks, threads, 0, stream>>>(
        cls_data, bbox_data, out_boxes, out_count,
        H, W, nc, reg_max, stride, conf_thresh, max_det);
}
