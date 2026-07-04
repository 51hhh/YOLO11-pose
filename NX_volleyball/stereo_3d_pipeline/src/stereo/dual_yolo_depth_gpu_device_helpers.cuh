// Device math, patch scoring, and robust aggregation helpers for dual_yolo_depth_gpu.cu.
// Included inside the .cu anonymous namespace.

__device__ __forceinline__ int clampInt(int v, int lo, int hi) {
    return max(lo, min(hi, v));
}
__device__ __forceinline__ float clampFloat(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}

__device__ __forceinline__ float readGray(
    const uint8_t* img, int pitch, int x, int y) {
    return static_cast<float>(img[y * pitch + x]);
}

__device__ __forceinline__ void readBgr(
    const uint8_t* img, int pitch, int x, int y,
    float* b, float* g, float* r) {
    const uint8_t* px = img + y * pitch + x * 3;
    *b = static_cast<float>(px[0]);
    *g = static_cast<float>(px[1]);
    *r = static_cast<float>(px[2]);
}

__device__ __forceinline__ int colorLabel(float b, float g, float r) {
    const float hi = fmaxf(r, fmaxf(g, b));
    const float lo = fminf(r, fminf(g, b));
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    const float sat = hi > 1.0f ? (hi - lo) / hi : 0.0f;
    if (luma > 150.0f && sat < 0.25f) return 1;     // white panel
    if (r > 95.0f && g > 85.0f && b < 0.78f * fminf(r, g)) return 2; // yellow
    if (b > 80.0f && b > 1.18f * r && b > 1.08f * g) return 3;       // blue
    if (hi < 45.0f) return 4;                       // dark edge/shadow
    return 0;
}

__device__ __forceinline__ float colorBallLikelihood(float b, float g, float r) {
    const int label = colorLabel(b, g, r);
    if (label == 1 || label == 2 || label == 3) return 1.0f;
    const float hi = fmaxf(r, fmaxf(g, b));
    const float lo = fminf(r, fminf(g, b));
    const float sat = hi > 1.0f ? (hi - lo) / hi : 0.0f;
    const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
    return clampFloat(0.35f * sat + 0.35f * (luma / 255.0f), 0.0f, 1.0f);
}

__device__ __forceinline__ float sobelMag(
    const uint8_t* img, int pitch, int x, int y) {
    const float gx =
        readGray(img, pitch, x + 1, y) - readGray(img, pitch, x - 1, y);
    const float gy =
        readGray(img, pitch, x, y + 1) - readGray(img, pitch, x, y - 1);
    return sqrtf(gx * gx + gy * gy);
}

__device__ __forceinline__ bool patchInside(
    int w, int h, int x, int y, int r) {
    return x >= r && y >= r && x < w - r && y < h - r;
}

__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ double blockReduceSum(double val) {
    __shared__ double warp_sums[4];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = lane < 4 ? warp_sums[lane] : 0.0;
        val = warpReduceSum(val);
    }
    __syncthreads();
    return val;
}

__device__ bool solve3x3(
    double A00, double A01, double A02,
    double A10, double A11, double A12,
    double A20, double A21, double A22,
    double b0, double b1, double b2,
    double& x0, double& x1, double& x2) {
    const double det = A00 * (A11 * A22 - A12 * A21) -
                       A01 * (A10 * A22 - A12 * A20) +
                       A02 * (A10 * A21 - A11 * A20);
    if (fabs(det) < 1e-12) return false;
    const double inv = 1.0 / det;
    x0 = (b0 * (A11 * A22 - A12 * A21) -
          A01 * (b1 * A22 - A12 * b2) +
          A02 * (b1 * A21 - A11 * b2)) * inv;
    x1 = (A00 * (b1 * A22 - A12 * b2) -
          b0 * (A10 * A22 - A12 * A20) +
          A02 * (A10 * b2 - b1 * A20)) * inv;
    x2 = (A00 * (A11 * b2 - b1 * A21) -
          A01 * (A10 * b2 - b1 * A20) +
          b0 * (A10 * A21 - A11 * A20)) * inv;
    return true;
}

__device__ __forceinline__ void clearCircle(stereo3d::DualYoloGpuCircle* c) {
    c->cx = 0.0f;
    c->cy = 0.0f;
    c->radius = 0.0f;
    c->confidence = 0.0f;
    c->source = 0;
    c->valid = 0;
}

__device__ __forceinline__ void clearPoint(stereo3d::DualYoloGpuPointMeasure* p) {
    p->cx = 0.0f;
    p->cy = 0.0f;
    p->confidence = 0.0f;
    p->valid = 0;
}

__device__ __forceinline__ void clearDisparity(stereo3d::DualYoloGpuDisparity* d) {
    d->disparity = -1.0f;
    d->confidence = 0.0f;
    d->stddev = -1.0f;
    d->delta_gate_px = 0.0f;
    d->anchor_cx = 0.0f;
    d->anchor_cy = 0.0f;
    d->support = 0;
    d->attempted = 0;
    d->low_confidence = 0;
    d->valid = 0;
    d->debug_match_count = 0;
}

__device__ float znccScore(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    double sum_l = 0.0;
    double sum_r = 0.0;
    double sum_l2 = 0.0;
    double sum_r2 = 0.0;
    double sum_lr = 0.0;
    int n = 0;
    for (int yy = -patch_radius; yy <= patch_radius; ++yy) {
        const uint8_t* lrow = left_img + (y_left + yy) * left_pitch;
        const uint8_t* rrow = right_img + (y_right + yy) * right_pitch;
        for (int xx = -patch_radius; xx <= patch_radius; ++xx) {
            const double lv = static_cast<double>(lrow[x_left + xx]);
            const double rv = static_cast<double>(rrow[x_right + xx]);
            sum_l += lv;
            sum_r += rv;
            sum_l2 += lv * lv;
            sum_r2 += rv * rv;
            sum_lr += lv * rv;
            ++n;
        }
    }
    if (n <= 1) return -2.0f;
    const double inv_n = 1.0 / static_cast<double>(n);
    const double mean_l = sum_l * inv_n;
    const double mean_r = sum_r * inv_n;
    const double var_l = sum_l2 - static_cast<double>(n) * mean_l * mean_l;
    const double var_r = sum_r2 - static_cast<double>(n) * mean_r * mean_r;
    if (var_l <= 1e-6 || var_r <= 1e-6) return -2.0f;
    const double cov = sum_lr - static_cast<double>(n) * mean_l * mean_r;
    return static_cast<float>(cov / sqrt(var_l * var_r));
}

__device__ uint32_t census5x5(
    const uint8_t* img, int pitch, int x, int y) {
    const uint8_t center = img[y * pitch + x];
    uint32_t bits = 0u;
    int bit = 0;
    for (int yy = -2; yy <= 2; ++yy) {
        for (int xx = -2; xx <= 2; ++xx) {
            if (xx == 0 && yy == 0) continue;
            bits |= (img[(y + yy) * pitch + x + xx] > center ? 1u : 0u) << bit;
            ++bit;
        }
    }
    return bits;
}

__device__ float binaryPatchScore(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    if (patch_radius < 2) return -2.0f;
    const uint32_t lc = census5x5(left_img, left_pitch, x_left, y_left);
    const uint32_t rc = census5x5(right_img, right_pitch, x_right, y_right);
    const float census = 1.0f - static_cast<float>(__popc(lc ^ rc)) / 24.0f;
    const float zncc = znccScore(left_img, left_pitch, right_img, right_pitch,
                                 x_left, y_left, x_right, y_right, patch_radius);
    return zncc > -1.5f ? 0.55f * census + 0.45f * (0.5f + 0.5f * zncc) : census;
}

__device__ float colorPatchScore(
    const uint8_t* left_gray, int left_pitch,
    const uint8_t* right_gray, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int x_left, int y_left, int x_right, int y_right,
    int patch_radius) {
    if (!left_bgr || !right_bgr || left_bgr_pitch <= 0 || right_bgr_pitch <= 0) {
        const float zncc = znccScore(left_gray, left_pitch, right_gray, right_pitch,
                                     x_left, y_left, x_right, y_right, patch_radius);
        return zncc > -1.5f ? 0.5f + 0.5f * zncc : -2.0f;
    }

    float color_diff = 0.0f;
    float label_same = 0.0f;
    float gray_sim = 0.0f;
    float color_support = 0.0f;
    int n = 0;
    for (int yy = -patch_radius; yy <= patch_radius; ++yy) {
        for (int xx = -patch_radius; xx <= patch_radius; ++xx) {
            float lb, lg, lr, rb, rg, rr;
            readBgr(left_bgr, left_bgr_pitch, x_left + xx, y_left + yy,
                    &lb, &lg, &lr);
            readBgr(right_bgr, right_bgr_pitch, x_right + xx, y_right + yy,
                    &rb, &rg, &rr);
            const float diff =
                (fabsf(lb - rb) + fabsf(lg - rg) + fabsf(lr - rr)) / (3.0f * 255.0f);
            color_diff += diff;
            const int ll = colorLabel(lb, lg, lr);
            const int rl = colorLabel(rb, rg, rr);
            label_same += (ll == rl && ll != 0) ? 1.0f : 0.0f;
            color_support += 0.5f * (colorBallLikelihood(lb, lg, lr) +
                                     colorBallLikelihood(rb, rg, rr));
            const float lv = readGray(left_gray, left_pitch, x_left + xx, y_left + yy);
            const float rv = readGray(right_gray, right_pitch, x_right + xx, y_right + yy);
            gray_sim += 1.0f - fminf(1.0f, fabsf(lv - rv) / 255.0f);
            ++n;
        }
    }
    if (n <= 0) return -2.0f;

    const float inv_n = 1.0f / static_cast<float>(n);
    const float color_similarity = clampFloat(1.0f - color_diff * inv_n * 2.4f,
                                              0.0f, 1.0f);
    const float label_iou = clampFloat(label_same * inv_n, 0.0f, 1.0f);
    const float support = clampFloat(color_support * inv_n, 0.0f, 1.0f);
    const float gray_consistency = clampFloat(gray_sim * inv_n, 0.0f, 1.0f);
    const float zncc = znccScore(left_gray, left_pitch, right_gray, right_pitch,
                                 x_left, y_left, x_right, y_right, patch_radius);
    const float zncc01 = zncc > -1.5f ? clampFloat(0.5f + 0.5f * zncc, 0.0f, 1.0f) : 0.0f;

    return clampFloat(0.30f * zncc01 +
                      0.25f * color_similarity +
                      0.25f * label_iou +
                      0.12f * gray_consistency +
                      0.08f * support,
                      0.0f, 1.0f);
}

__device__ float localVariance(
    const uint8_t* img, int pitch, int x, int y) {
    float sum = 0.0f;
    float sum2 = 0.0f;
    int n = 0;
    for (int yy = -1; yy <= 1; ++yy) {
        for (int xx = -1; xx <= 1; ++xx) {
            const float v = readGray(img, pitch, x + xx, y + yy);
            sum += v;
            sum2 += v * v;
            ++n;
        }
    }
    const float mean = sum / static_cast<float>(n);
    return fmaxf(0.0f, sum2 / static_cast<float>(n) - mean * mean);
}

__device__ float sparseResponse(
    const uint8_t* img, int pitch, int x, int y, int mode) {
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (int yy = -1; yy <= 1; ++yy) {
        for (int xx = -1; xx <= 1; ++xx) {
            const float gx = readGray(img, pitch, x + xx + 1, y + yy) -
                             readGray(img, pitch, x + xx - 1, y + yy);
            const float gy = readGray(img, pitch, x + xx, y + yy + 1) -
                             readGray(img, pitch, x + xx, y + yy - 1);
            sxx += static_cast<double>(gx) * gx;
            syy += static_cast<double>(gy) * gy;
            sxy += static_cast<double>(gx) * gy;
        }
    }
    const double tr = sxx + syy;
    const double det = sxx * syy - sxy * sxy;
    float corner = 0.0f;
    if (tr > 1e-6 && det > 0.0) {
        const double disc = fmax(0.0, tr * tr - 4.0 * det);
        corner = static_cast<float>(0.5 * (tr - sqrt(disc)));
    }
    const float gx = readGray(img, pitch, x + 1, y) - readGray(img, pitch, x - 1, y);
    const float gy = readGray(img, pitch, x, y + 1) - readGray(img, pitch, x, y - 1);
    const float texture = sqrtf(gx * gx + gy * gy) *
                          sqrtf(fmaxf(0.0f, localVariance(img, pitch, x, y)));
    if (mode == 0) return corner;
    if (mode == 1) return texture;
    return 0.65f * corner + 0.35f * texture;
}

__device__ float colorSparseResponse(
    const uint8_t* gray, int gray_pitch,
    const uint8_t* bgr, int bgr_pitch,
    int x, int y,
    int mode) {
    if (!bgr || bgr_pitch <= 0) return 0.0f;
    float b, g, r;
    readBgr(bgr, bgr_pitch, x, y, &b, &g, &r);
    const float support = colorBallLikelihood(b, g, r);
    const float edge = sobelMag(gray, gray_pitch, x, y);
    const float variance = sqrtf(fmaxf(0.0f, localVariance(gray, gray_pitch, x, y)));
    if (mode == 3) {
        return 64.0f * support + 0.18f * variance + 0.08f * edge;
    }
    return edge * (0.35f + 0.65f * support) + 0.10f * variance;
}

__device__ float disparityDeltaGate(
    float initial_disp,
    float focal,
    float baseline,
    float max_disp_delta_px,
    float max_disp_delta_ratio,
    float max_depth_delta_m) {
    float gate = fmaxf(max_disp_delta_px, fabsf(initial_disp) * max_disp_delta_ratio);
    const float fb = focal * baseline;
    if (fb > 0.0f && max_depth_delta_m > 0.0f && initial_disp > 0.5f) {
        const float z = fb / initial_disp;
        const float near_z = fmaxf(0.01f, z - max_depth_delta_m);
        const float far_z = z + max_depth_delta_m;
        gate = fmaxf(gate, fabsf(fb / near_z - initial_disp));
        gate = fmaxf(gate, fabsf(initial_disp - fb / far_z));
    }
    return fmaxf(gate, 0.5f);
}

__device__ __forceinline__ float sampleWeight(float score) {
    return fmaxf(0.05f, score);
}

__device__ float medianSortedValues(const float* values, int n) {
    if (n <= 0) return 0.0f;
    return (n & 1) ? values[n / 2]
                   : 0.5f * (values[n / 2 - 1] + values[n / 2]);
}

__device__ void sortSamplesByDisparity(
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* sample_right_y,
    int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (sample_disp[j] <= sample_disp[j + 1]) continue;
            const float td = sample_disp[j];
            const float ts = sample_score[j];
            const float tx = sample_x[j];
            const float ty = sample_y[j];
            const float try_ = sample_right_y[j];
            sample_disp[j] = sample_disp[j + 1];
            sample_score[j] = sample_score[j + 1];
            sample_x[j] = sample_x[j + 1];
            sample_y[j] = sample_y[j + 1];
            sample_right_y[j] = sample_right_y[j + 1];
            sample_disp[j + 1] = td;
            sample_score[j + 1] = ts;
            sample_x[j + 1] = tx;
            sample_y[j + 1] = ty;
            sample_right_y[j + 1] = try_;
        }
    }
}

__device__ void sortValues(float* values, int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (values[j] <= values[j + 1]) continue;
            const float tmp = values[j];
            values[j] = values[j + 1];
            values[j + 1] = tmp;
        }
    }
}

__device__ bool robustAggregateSamples(
    int n,
    int min_points,
    float initial_disp,
    float max_delta,
    float max_stddev,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* sample_right_y,
    float* scratch,
    float* out_disp,
    float* out_anchor_x,
    float* out_anchor_y,
    float* out_stddev,
    float* out_avg_score,
    int* out_support) {
    if (n < min_points) return false;

    sortSamplesByDisparity(sample_disp, sample_score, sample_x, sample_y,
                           sample_right_y, n);
    const float median = medianSortedValues(sample_disp, n);

    for (int i = 0; i < n; ++i) {
        scratch[i] = fabsf(sample_disp[i] - median);
    }
    sortValues(scratch, n);
    const float mad = medianSortedValues(scratch, n);
    const float robust_sigma = 1.4826f * mad;
    const float min_gate = clampFloat(feature_ransac_gate_px, 0.25f, 3.0f);
    const float mad_gate =
        fmaxf(min_gate, robust_sigma * fmaxf(1.0f, feature_mad_scale));
    const float gate = fminf(fmaxf(0.35f, max_delta), mad_gate);

    float best_center = median;
    float best_weight = -1.0f;
    int best_support = 0;
    for (int i = 0; i < n; ++i) {
        float support_weight = 0.0f;
        int support = 0;
        for (int j = 0; j < n; ++j) {
            if (fabsf(sample_disp[j] - sample_disp[i]) > gate) continue;
            support_weight += sampleWeight(sample_score[j]);
            ++support;
        }
        if (support > best_support ||
            (support == best_support && support_weight > best_weight)) {
            best_support = support;
            best_weight = support_weight;
            best_center = sample_disp[i];
        }
    }

    const float median_gate = fmaxf(gate, min_gate * 1.5f);
    int inliers = 0;
    float total_w = 0.0f;
    for (int i = 0; i < n; ++i) {
        const bool keep =
            fabsf(sample_disp[i] - best_center) <= gate &&
            fabsf(sample_disp[i] - median) <= median_gate;
        scratch[i] = keep ? 1.0f : 0.0f;
        if (!keep) continue;
        total_w += sampleWeight(sample_score[i]);
        ++inliers;
    }
    if (inliers < min_points || total_w <= 0.0f) return false;

    const float half_w = total_w * 0.5f;
    float accum_w = 0.0f;
    float weighted_median = best_center;
    for (int i = 0; i < n; ++i) {
        if (scratch[i] <= 0.0f) continue;
        accum_w += sampleWeight(sample_score[i]);
        if (accum_w >= half_w) {
            weighted_median = sample_disp[i];
            break;
        }
    }
    if (weighted_median <= 0.5f ||
        fabsf(weighted_median - initial_disp) > max_delta) {
        return false;
    }

    float sum_w = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_score = 0.0f;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (scratch[i] <= 0.0f) continue;
        const float w = sampleWeight(sample_score[i]);
        const float diff = sample_disp[i] - weighted_median;
        sum_w += w;
        sum_x += w * sample_x[i];
        sum_y += w * sample_y[i];
        sum_score += w;
        var += w * diff * diff;
    }
    if (sum_w <= 0.0f) return false;

    const float stddev = sqrtf(var / sum_w);
    if (stddev > max_stddev) return false;

    *out_disp = weighted_median;
    *out_anchor_x = sum_x / sum_w;
    *out_anchor_y = sum_y / sum_w;
    *out_stddev = stddev;
    *out_avg_score = sum_score / static_cast<float>(max(1, inliers));
    *out_support = inliers;
    return true;
}

__device__ void copyDualYoloDebugMatches(
    int n,
    const float* sample_disp,
    const float* sample_score,
    const float* sample_x,
    const float* sample_y,
    const float* sample_right_y,
    const float* inlier_flags,
    stereo3d::DualYoloGpuDisparity* out) {
    int count = 0;
    for (int i = 0;
         i < n && count < stereo3d::kMaxDualYoloGpuDebugMatches;
         ++i) {
        if (inlier_flags[i] <= 0.0f || sample_disp[i] <= 0.5f) {
            continue;
        }
        out->debug_left_x[count] = sample_x[i];
        out->debug_left_y[count] = sample_y[i];
        out->debug_right_x[count] = sample_x[i] - sample_disp[i];
        out->debug_right_y[count] = sample_right_y[i];
        out->debug_disparity[count] = sample_disp[i];
        out->debug_score[count] = sample_score[i];
        ++count;
    }
    out->debug_match_count = count;
}

__device__ bool pointInsideDetectionEllipse(
    const stereo3d::DualYoloGpuDetection& det,
    float x,
    float y,
    float scale) {
    if (det.width <= 1.0f || det.height <= 1.0f) return false;
    const float rx = fmaxf(1.0f, det.width * scale);
    const float ry = fmaxf(1.0f, det.height * scale);
    const float nx = (x - det.cx) / rx;
    const float ny = (y - det.cy) / ry;
    return nx * nx + ny * ny <= 1.0f;
}

__device__ __forceinline__ float expectedFeatureYDelta(
    float left_x,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_slope,
    float feature_y_offset_px) {
    return feature_y_offset_px + feature_y_slope * (left_x - left_det.cx);
}

__device__ __forceinline__ bool passesFeatureYGate(
    float left_x,
    float left_y,
    float right_y,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px) {
    const float expected = expectedFeatureYDelta(
        left_x, left_det, feature_y_slope, feature_y_offset_px);
    const float residual = (left_y - right_y) - expected;
    return fabsf(residual) <= clampFloat(feature_y_tolerance_px, 0.5f, 8.0f);
}

__device__ float reverseSparseMatchError(
    const uint8_t* left_img,
    int left_pitch,
    const uint8_t* right_img,
    int right_pitch,
    const uint8_t* left_bgr,
    int left_bgr_pitch,
    const uint8_t* right_bgr,
    int right_bgr_pitch,
    int img_w,
    int img_h,
    float left_x,
    float left_y,
    float right_x,
    float right_y,
    int patch_radius,
    int d_start,
    int d_end,
    int y_radius,
    int mode,
    const stereo3d::DualYoloGpuDetection& left_det,
    float feature_y_slope,
    float feature_y_offset_px) {
    const int rx = static_cast<int>(rintf(right_x));
    const int ry = static_cast<int>(rintf(right_y));
    if (!patchInside(img_w, img_h, rx, ry, patch_radius)) return FLT_MAX;

    float best_score = -2.0f;
    int best_lx = -1;
    int best_ly = -1;
    for (int d = d_start; d <= d_end; ++d) {
        const int lx = rx + d;
        const float expected_y = expectedFeatureYDelta(
            static_cast<float>(lx), left_det,
            feature_y_slope, feature_y_offset_px);
        const int dy_center = static_cast<int>(rintf(expected_y));
        for (int dy = dy_center - y_radius; dy <= dy_center + y_radius; ++dy) {
            const int ly = ry + dy;
            if (!patchInside(img_w, img_h, lx, ly, patch_radius)) continue;
            const float score = mode == 2
                ? binaryPatchScore(right_img, right_pitch, left_img, left_pitch,
                                   rx, ry, lx, ly, patch_radius)
                : (mode >= 3
                    ? colorPatchScore(right_img, right_pitch,
                                      left_img, left_pitch,
                                      right_bgr, right_bgr_pitch,
                                      left_bgr, left_bgr_pitch,
                                      rx, ry, lx, ly, patch_radius)
                    : znccScore(right_img, right_pitch, left_img, left_pitch,
                                rx, ry, lx, ly, patch_radius));
            if (score > best_score) {
                best_score = score;
                best_lx = lx;
                best_ly = ly;
            }
        }
    }
    if (best_lx < 0) return FLT_MAX;
    const float dx = static_cast<float>(best_lx) - left_x;
    const float dy = static_cast<float>(best_ly) - left_y;
    return sqrtf(dx * dx + dy * dy);
}

__device__ bool passesFeatureOverlapGate(
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float left_x,
    float left_y,
    float right_x,
    float right_y,
    float initial_disp,
    float feature_overlap_scale) {
    const float scale = clampFloat(feature_overlap_scale, 0.35f, 0.90f);
    const float projection_scale = fminf(0.98f, scale + 0.12f);
    if (!pointInsideDetectionEllipse(left_det, left_x, left_y, scale) ||
        !pointInsideDetectionEllipse(right_det, right_x, right_y, scale)) {
        return false;
    }
    if (!pointInsideDetectionEllipse(right_det,
                                     left_x - initial_disp,
                                     left_y,
                                     projection_scale)) {
        return false;
    }
    if (!pointInsideDetectionEllipse(left_det,
                                     right_x + initial_disp,
                                     right_y,
                                     projection_scale)) {
        return false;
    }
    return true;
}

__device__ bool passesSphereRadiusGate(
    float left_x,
    float left_y,
    float ball_cx,
    float ball_cy,
    float disparity,
    float initial_disp,
    float focal,
    float baseline,
    float radius_m,
    float radius_scale,
    float margin_m) {
    if (radius_m <= 0.0f || focal <= 1e-3f || baseline <= 1e-6f ||
        initial_disp <= 0.5f || disparity <= 0.5f) {
        return true;
    }
    const float fb = focal * baseline;
    const float center_z = fb / initial_disp;
    const float z = fb / disparity;
    if (!isfinite(center_z) || !isfinite(z)) return false;
    const float dx = (left_x - ball_cx) * z / focal;
    const float dy = (left_y - ball_cy) * z / focal;
    const float dz = z - center_z;
    const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    const float max_distance =
        radius_m * fmaxf(1.0f, radius_scale) + fmaxf(0.0f, margin_m);
    return distance <= max_distance;
}
