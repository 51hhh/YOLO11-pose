// ROI geometry and sparse patch matching helpers for dual_yolo_depth_gpu.cu.
// Included inside the .cu anonymous namespace after device helpers.

__device__ void fitGeometryInBBox(
    const uint8_t* img, int pitch, int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& det,
    int max_roi_pixels,
    float* edge_x,
    float* edge_y,
    float* edge_w,
    int* edge_n,
    float* max_grad,
    stereo3d::DualYoloGpuCircle* circle,
    stereo3d::DualYoloGpuPointMeasure* edge_centroid,
    stereo3d::DualYoloGpuPointMeasure* radial_center,
    stereo3d::DualYoloGpuPointMeasure* edge_pair_center) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *edge_n = 0;
        *max_grad = 0.0f;
        clearCircle(circle);
        clearPoint(edge_centroid);
        clearPoint(radial_center);
        clearPoint(edge_pair_center);
    }
    __syncthreads();

    int x1 = static_cast<int>(floorf(det.cx - det.width * 0.5f));
    int y1 = static_cast<int>(floorf(det.cy - det.height * 0.5f));
    int x2 = static_cast<int>(ceilf(det.cx + det.width * 0.5f));
    int y2 = static_cast<int>(ceilf(det.cy + det.height * 0.5f));
    x1 = clampInt(x1, 1, img_w - 2);
    y1 = clampInt(y1, 1, img_h - 2);
    x2 = clampInt(x2, 1, img_w - 2);
    y2 = clampInt(y2, 1, img_h - 2);
    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    if (roi_w < 12 || roi_h < 12) {
        __syncthreads();
        return;
    }

    int stride = 1;
    const int area = roi_w * roi_h;
    while (max_roi_pixels > 0 && area / (stride * stride) > max_roi_pixels) {
        ++stride;
    }
    const int inner_w = max(1, (roi_w - 2 + stride - 1) / stride);
    const int inner_h = max(1, (roi_h - 2 + stride - 1) / stride);
    const int total = inner_w * inner_h;

    float local_max = 0.0f;
    for (int i = tid; i < total; i += kThreads) {
        const int x = x1 + 1 + (i % inner_w) * stride;
        const int y = y1 + 1 + (i / inner_w) * stride;
        if (x <= 0 || x >= img_w - 1 || y <= 0 || y >= img_h - 1) continue;
        local_max = fmaxf(local_max, sobelMag(img, pitch, x, y));
    }
    atomicMax(reinterpret_cast<int*>(max_grad), __float_as_int(local_max));
    __syncthreads();

    const float threshold = fmaxf(10.0f, *max_grad * 0.25f);
    for (int i = tid; i < total; i += kThreads) {
        const int x = x1 + 1 + (i % inner_w) * stride;
        const int y = y1 + 1 + (i / inner_w) * stride;
        if (x <= 0 || x >= img_w - 1 || y <= 0 || y >= img_h - 1) continue;
        const float mag = sobelMag(img, pitch, x, y);
        if (mag <= threshold) continue;
        const int idx = atomicAdd(edge_n, 1);
        if (idx < kMaxEdges) {
            edge_x[idx] = static_cast<float>(x);
            edge_y[idx] = static_cast<float>(y);
            edge_w[idx] = mag;
        }
    }
    __syncthreads();

    const int n = min(*edge_n, kMaxEdges);
    if (n < 12) {
        __syncthreads();
        return;
    }

    double sw = 0.0, swx = 0.0, swy = 0.0;
    double swxx = 0.0, swyy = 0.0, swxy = 0.0;
    double swxz = 0.0, swyz = 0.0, swz = 0.0;
    int local_min_x = 0x3fffffff;
    int local_max_x = -0x3fffffff;
    int local_min_y = 0x3fffffff;
    int local_max_y = -0x3fffffff;

    for (int i = tid; i < n; i += kThreads) {
        const double w = static_cast<double>(edge_w[i]);
        const double x = static_cast<double>(edge_x[i]);
        const double y = static_cast<double>(edge_y[i]);
        const double z = x * x + y * y;
        sw += w;
        swx += w * x;
        swy += w * y;
        swxx += w * x * x;
        swyy += w * y * y;
        swxy += w * x * y;
        swxz += w * x * z;
        swyz += w * y * z;
        swz += w * z;
        local_min_x = min(local_min_x, static_cast<int>(edge_x[i]));
        local_max_x = max(local_max_x, static_cast<int>(edge_x[i]));
        local_min_y = min(local_min_y, static_cast<int>(edge_y[i]));
        local_max_y = max(local_max_y, static_cast<int>(edge_y[i]));
    }

    sw = blockReduceSum(sw);
    swx = blockReduceSum(swx);
    swy = blockReduceSum(swy);
    swxx = blockReduceSum(swxx);
    swyy = blockReduceSum(swyy);
    swxy = blockReduceSum(swxy);
    swxz = blockReduceSum(swxz);
    swyz = blockReduceSum(swyz);
    swz = blockReduceSum(swz);

    __shared__ int min_x, max_x, min_y, max_y;
    if (tid == 0) {
        min_x = 0x3fffffff;
        max_x = -0x3fffffff;
        min_y = 0x3fffffff;
        max_y = -0x3fffffff;
    }
    __syncthreads();
    atomicMin(&min_x, local_min_x);
    atomicMax(&max_x, local_max_x);
    atomicMin(&min_y, local_min_y);
    atomicMax(&max_y, local_max_y);
    __syncthreads();

    if (tid == 0) {
        const float edge_score = clampFloat(static_cast<float>(n) / 80.0f, 0.0f, 1.0f);
        if (sw > 1e-6) {
            edge_centroid->cx = static_cast<float>(swx / sw);
            edge_centroid->cy = static_cast<float>(swy / sw);
            edge_centroid->confidence = edge_score;
            edge_centroid->valid = 1;
        }
        if (min_x < max_x && min_y < max_y) {
            edge_pair_center->cx = 0.5f * static_cast<float>(min_x + max_x);
            edge_pair_center->cy = 0.5f * static_cast<float>(min_y + max_y);
            edge_pair_center->confidence = edge_score * 0.85f;
            edge_pair_center->valid = 1;
        }

        double a = 0.0, b = 0.0, c = 0.0;
        if (solve3x3(swxx, swxy, swx,
                     swxy, swyy, swy,
                     swx, swy, sw,
                     swxz, swyz, swz,
                     a, b, c)) {
            const double cx = a * 0.5;
            const double cy = b * 0.5;
            const double r2 = c + cx * cx + cy * cy;
            const float r = r2 > 0.0 ? static_cast<float>(sqrt(r2)) : 0.0f;
            const float bbox_cx = det.cx;
            const float bbox_cy = det.cy;
            const float max_r = fmaxf(det.width, det.height) * 0.75f;
            const float min_r = fmaxf(4.0f, fminf(det.width, det.height) * 0.20f);
            const float center_gate = fmaxf(det.width, det.height) * 0.55f;
            if (r >= min_r && r <= max_r &&
                fabsf(static_cast<float>(cx) - bbox_cx) <= center_gate &&
                fabsf(static_cast<float>(cy) - bbox_cy) <= center_gate) {
                circle->cx = static_cast<float>(cx);
                circle->cy = static_cast<float>(cy);
                circle->radius = r;
                circle->confidence = fmaxf(0.15f, edge_score);
                circle->source = 2;
                circle->valid = 1;
                radial_center->cx = circle->cx;
                radial_center->cy = circle->cy;
                radial_center->confidence = circle->confidence * 0.9f;
                radial_center->valid = 1;
            }
        }
    }
    __syncthreads();
}
__device__ void matchPatchAtPoint(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    float x_left_f, float y_left_f,
    float initial_disp,
    int patch_radius,
    int search_radius,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    stereo3d::DualYoloGpuDisparity* out) {
    if (threadIdx.x == 0) {
        clearDisparity(out);
        out->attempted = 0;
    }
    __syncthreads();

    const int x_left = static_cast<int>(rintf(x_left_f));
    const int y_left = static_cast<int>(rintf(y_left_f));
    const int d0 = static_cast<int>(rintf(initial_disp));
    const int d_start = max(1, d0 - search_radius);
    const int d_end = min(max_disparity, d0 + search_radius);
    if (!patchInside(img_w, img_h, x_left, y_left, patch_radius) ||
        d_start >= d_end) {
        __syncthreads();
        return;
    }

    __shared__ float best_score_parts[kThreadsPerPoint];
    __shared__ int best_disp_parts[kThreadsPerPoint];
    __shared__ float second_score_parts[kThreadsPerPoint];
    if (threadIdx.x < kThreadsPerPoint) {
        best_score_parts[threadIdx.x] = -2.0f;
        best_disp_parts[threadIdx.x] = -1;
        second_score_parts[threadIdx.x] = -2.0f;
    }
    __syncthreads();

    if (threadIdx.x < kThreadsPerPoint) {
        const int range = d_end - d_start + 1;
        const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
        const int begin = d_start + threadIdx.x * per;
        const int end = min(d_end, begin + per - 1);
        float best_score = -2.0f;
        float second_score = -2.0f;
        int best_disp = -1;
        for (int d = begin; d <= end; ++d) {
            const int xr = x_left - d;
            if (!patchInside(img_w, img_h, xr, y_left, patch_radius)) continue;
            const float score = znccScore(left_img, left_pitch, right_img, right_pitch,
                                          x_left, y_left, xr, y_left, patch_radius);
            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_disp = d;
            } else if (score > second_score) {
                second_score = score;
            }
        }
        best_score_parts[threadIdx.x] = best_score;
        best_disp_parts[threadIdx.x] = best_disp;
        second_score_parts[threadIdx.x] = second_score;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_score = best_score_parts[0];
        float second_score = second_score_parts[0];
        int best_disp = best_disp_parts[0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[i] > best_score) {
                second_score = fmaxf(second_score, fmaxf(best_score, second_score_parts[i]));
                best_score = best_score_parts[i];
                best_disp = best_disp_parts[i];
            } else {
                second_score = fmaxf(second_score, best_score_parts[i]);
            }
        }
        out->attempted = d_end - d_start + 1;
        bool accept = true;
        if (best_disp <= 0 || best_score < fmaxf(0.10f, min_confidence * 0.60f)) {
            out->low_confidence = 1;
            accept = false;
        }

        float sub_disp = static_cast<float>(best_disp);
        if (accept && best_disp > d_start && best_disp < d_end) {
            const int xr_m = x_left - (best_disp - 1);
            const int xr_p = x_left - (best_disp + 1);
            if (patchInside(img_w, img_h, xr_m, y_left, patch_radius) &&
                patchInside(img_w, img_h, xr_p, y_left, patch_radius)) {
                const float s_m = znccScore(left_img, left_pitch, right_img, right_pitch,
                                            x_left, y_left, xr_m, y_left, patch_radius);
                const float s_p = znccScore(left_img, left_pitch, right_img, right_pitch,
                                            x_left, y_left, xr_p, y_left, patch_radius);
                const float denom = s_m - 2.0f * best_score + s_p;
                if (s_m > -1.5f && s_p > -1.5f && denom < -1e-5f) {
                    sub_disp += clampFloat(0.5f * (s_m - s_p) / denom, -1.0f, 1.0f);
                }
            }
        }

        const float uniqueness = second_score > -1.5f ? best_score - second_score : 1.0f;
        const float z = focal * baseline / fmaxf(0.5f, sub_disp);
        if (accept &&
            ((uniqueness < 0.01f && best_score < 0.75f) ||
             fabsf(sub_disp - initial_disp) > max_delta ||
             sub_disp <= 0.5f || sub_disp > static_cast<float>(max_disparity) ||
             z < min_depth || z > max_depth)) {
            out->low_confidence = 1;
            accept = false;
        }
        if (accept) {
            out->disparity = sub_disp;
            out->confidence = clampFloat((best_score - 0.10f) / 0.80f, 0.0f, 1.0f);
            out->stddev = 0.0f;
            out->delta_gate_px = max_delta;
            out->anchor_cx = x_left_f;
            out->anchor_cy = y_left_f;
            out->support = 1;
            out->valid = out->confidence >= min_confidence ? 1 : 0;
            out->low_confidence = out->valid ? 0 : 1;
        }
    }
    __syncthreads();
}

__device__ void matchSparsePoints(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    const uint8_t* left_bgr, int left_bgr_pitch,
    const uint8_t* right_bgr, int right_bgr_pitch,
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float center_x, float center_y, float radius,
    float initial_disp,
    int mode,
    int patch_radius,
    int search_radius,
    int max_points,
    int min_points,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float max_stddev,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* sample_right_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint],
    int* valid_count,
    stereo3d::DualYoloGpuDisparity* out) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *valid_count = 0;
        clearDisparity(out);
    }
    for (int i = tid; i < kMaxFeaturePoints; i += kThreads) {
        sample_disp[i] = 0.0f;
        sample_score[i] = 0.0f;
        sample_x[i] = 0.0f;
        sample_y[i] = 0.0f;
        sample_right_y[i] = 0.0f;
        point_x[i] = 0.0f;
        point_y[i] = 0.0f;
        for (int j = 0; j < kThreadsPerPoint; ++j) {
            best_score_parts[i][j] = -2.0f;
            best_disp_parts[i][j] = -1.0f;
            best_dy_parts[i][j] = 0.0f;
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1,
                          min(kMaxFeaturePoints, kMaxParallelFeaturePoints));
    min_points = clampInt(min_points, 1, max_points);
    patch_radius = clampInt(patch_radius, 2, 8);
    const int point_idx = tid / kThreadsPerPoint;
    const int sub_idx = tid % kThreadsPerPoint;
    if (point_idx < max_points) {
        const int grid = max(2, static_cast<int>(ceilf(sqrtf(static_cast<float>(max_points)))));
        const int gx = point_idx % grid;
        const int gy = point_idx / grid;
        float sample_cx = center_x;
        float sample_cy = center_y;
        float rx = fmaxf(4.0f, fminf(radius * 0.82f, left_det.width * 0.42f));
        float ry = fmaxf(4.0f, fminf(radius * 0.82f, left_det.height * 0.42f));
        if (mode >= 3) {
            const float projected_right_cx = right_det.cx + initial_disp;
            sample_cx = 0.5f * (left_det.cx + projected_right_cx);
            sample_cy = 0.5f * (left_det.cy + right_det.cy);
            rx = fmaxf(4.0f, fminf(radius * 0.90f,
                                   fminf(left_det.width, right_det.width) * 0.42f));
            ry = fmaxf(4.0f, fminf(radius * 0.90f,
                                   fminf(left_det.height, right_det.height) * 0.42f));
        }
        const float u = (static_cast<float>(gx) + 0.5f) / static_cast<float>(grid);
        const float v = (static_cast<float>(gy) + 0.5f) / static_cast<float>(grid);
        const float x_f = sample_cx + (u - 0.5f) * 2.0f * rx;
        const float y_f = sample_cy + (v - 0.5f) * 2.0f * ry;
        const float nx = (x_f - sample_cx) / fmaxf(rx, 1.0f);
        const float ny = (y_f - sample_cy) / fmaxf(ry, 1.0f);
        const int x = static_cast<int>(rintf(x_f));
        const int y = static_cast<int>(rintf(y_f));
        if (nx * nx + ny * ny <= 0.92f * 0.92f &&
            patchInside(img_w, img_h, x, y, patch_radius)) {
            const float response = mode >= 3
                ? colorSparseResponse(left_img, left_pitch,
                                      left_bgr, left_bgr_pitch, x, y, mode)
                : sparseResponse(left_img, left_pitch, x, y, mode);
            const float response_floor =
                mode == 3 ? 24.0f : (mode == 4 ? 12.0f : (mode == 1 ? 20.0f : 8.0f));
            if (mode >= 3 || response > response_floor) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                if (d_start < d_end) {
                    const int range = d_end - d_start + 1;
                    const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
                    const int begin = d_start + sub_idx * per;
                    const int end = min(d_end, begin + per - 1);
                    const float expected_y = expectedFeatureYDelta(
                        static_cast<float>(x), left_det,
                        feature_y_slope, feature_y_offset_px);
                    const int y_radius = mode >= 3
                        ? 1
                        : clampInt(static_cast<int>(ceilf(
                                       clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                                   1, 3);
                    const int dy_center = static_cast<int>(rintf(-expected_y));
                    float best_score = -2.0f;
                    float best_disp = -1.0f;
                    float best_dy = 0.0f;
                    for (int d = begin; d <= end; ++d) {
                        const int xr = x - d;
                        for (int dy = dy_center - y_radius;
                             dy <= dy_center + y_radius; ++dy) {
                            const int yr = y + dy;
                            if (!patchInside(img_w, img_h, xr, yr, patch_radius)) continue;
                            const float score = mode == 2
                                ? binaryPatchScore(left_img, left_pitch, right_img, right_pitch,
                                                   x, y, xr, yr, patch_radius)
                                : (mode >= 3
                                    ? colorPatchScore(left_img, left_pitch,
                                                      right_img, right_pitch,
                                                      left_bgr, left_bgr_pitch,
                                                      right_bgr, right_bgr_pitch,
                                                      x, y, xr, yr, patch_radius)
                                    : znccScore(left_img, left_pitch, right_img, right_pitch,
                                                x, y, xr, yr, patch_radius));
                            if (score > best_score) {
                                best_score = score;
                                best_disp = static_cast<float>(d);
                                best_dy = static_cast<float>(dy);
                            }
                        }
                    }
                    best_score_parts[point_idx][sub_idx] = best_score;
                    best_disp_parts[point_idx][sub_idx] = best_disp;
                    best_dy_parts[point_idx][sub_idx] = best_dy;
                    if (sub_idx == 0) {
                        point_x[point_idx] = static_cast<float>(x);
                        point_y[point_idx] = static_cast<float>(y);
                    }
                }
            }
        }
    }
    __syncthreads();

    if (point_idx < max_points && sub_idx == 0) {
        float best_score = best_score_parts[point_idx][0];
        float best_disp = best_disp_parts[point_idx][0];
        float best_dy = best_dy_parts[point_idx][0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
                best_dy = best_dy_parts[point_idx][i];
            }
        }
        const float min_score = mode == 2
            ? fmaxf(0.58f, 0.50f + min_confidence * 0.35f)
            : (mode >= 3
                ? fmaxf(0.40f, 0.36f + min_confidence * 0.30f)
                : fmaxf(0.12f, min_confidence * 0.60f));
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            const float left_x = point_x[point_idx];
            const float left_y = point_y[point_idx];
            const float right_x = left_x - best_disp;
            const float right_y = left_y + best_dy;
            const bool cheap_ok =
                z >= min_depth && z <= max_depth &&
                passesFeatureYGate(left_x, left_y, right_y, left_det,
                                   feature_y_tolerance_px,
                                   feature_y_slope,
                                   feature_y_offset_px) &&
                passesFeatureOverlapGate(left_det, right_det,
                                         left_x, left_y, right_x, right_y,
                                         initial_disp, feature_overlap_scale) &&
                passesSphereRadiusGate(left_x, left_y,
                                       center_x, center_y,
                                       best_disp, initial_disp,
                                       focal, baseline,
                                       feature_sphere_radius_m,
                                       feature_sphere_radius_scale,
                                       feature_sphere_margin_m);
            if (cheap_ok) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                const int reverse_y_radius = mode >= 3
                    ? 1
                    : clampInt(static_cast<int>(ceilf(
                                   clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                               1, 3);
                const float reverse_err = feature_reverse_check_px >= 0.0f
                    ? reverseSparseMatchError(left_img, left_pitch,
                                              right_img, right_pitch,
                                              left_bgr, left_bgr_pitch,
                                              right_bgr, right_bgr_pitch,
                                              img_w, img_h,
                                              left_x, left_y,
                                              right_x, right_y,
                                              patch_radius,
                                              d_start, d_end,
                                              reverse_y_radius,
                                              mode,
                                              left_det,
                                              feature_y_slope,
                                              feature_y_offset_px)
                    : 0.0f;
                if (feature_reverse_check_px < 0.0f ||
                    reverse_err <= fmaxf(0.25f, feature_reverse_check_px)) {
                    const int idx = atomicAdd(valid_count, 1);
                    if (idx < kMaxFeaturePoints) {
                        sample_disp[idx] = best_disp;
                        sample_score[idx] = best_score;
                        sample_x[idx] = left_x;
                        sample_y[idx] = left_y;
                        sample_right_y[idx] = right_y;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        out->support = n;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            float disparity = 0.0f;
            float anchor_x = 0.0f;
            float anchor_y = 0.0f;
            float stddev = 0.0f;
            float avg_score = 0.0f;
            int support = 0;
            if (!robustAggregateSamples(n, min_points, initial_disp, max_delta,
                                        max_stddev, feature_mad_scale,
                                        feature_ransac_gate_px,
                                        sample_disp, sample_score,
                                        sample_x, sample_y, sample_right_y,
                                        point_x,
                                        &disparity, &anchor_x, &anchor_y,
                                        &stddev, &avg_score, &support)) {
                out->low_confidence = 1;
            } else {
                const float z = focal * baseline / fmaxf(disparity, 0.5f);
                if (disparity > static_cast<float>(max_disparity) ||
                    z < min_depth || z > max_depth) {
                    out->low_confidence = 1;
                } else {
                    const float min_score = mode == 2
                        ? fmaxf(0.58f, 0.50f + min_confidence * 0.35f)
                        : (mode >= 3
                            ? fmaxf(0.40f, 0.36f + min_confidence * 0.30f)
                            : fmaxf(0.12f, min_confidence * 0.60f));
                    const float score_conf =
                        clampFloat((avg_score - min_score) /
                                   fmaxf(0.01f, 1.0f - min_score),
                                   0.0f, 1.0f);
                    const float support_ratio =
                        static_cast<float>(support) /
                        static_cast<float>(max(1, max_points));
                    const float consistency =
                        clampFloat(1.0f / (1.0f + stddev), 0.0f, 1.0f);
                    const float delta_conf =
                        1.0f - fminf(1.0f, fabsf(disparity - initial_disp) / max_delta);
                    out->disparity = disparity;
                    out->confidence = clampFloat(0.30f * support_ratio +
                                                 0.35f * score_conf +
                                                 0.25f * consistency +
                                                 0.10f * delta_conf,
                                                 0.0f, 1.0f);
                    out->stddev = stddev;
                    out->delta_gate_px = max_delta;
                    out->anchor_cx = anchor_x;
                    out->anchor_cy = anchor_y;
                    out->support = support;
                    out->valid = out->confidence >= min_confidence ? 1 : 0;
                    out->low_confidence = out->valid ? 0 : 1;
                    copyDualYoloDebugMatches(n, sample_disp, sample_score,
                                             sample_x, sample_y,
                                             sample_right_y, point_x, out);
                }
            }
        }
    }
    __syncthreads();
}

__device__ void matchMultiPointPatch(
    const uint8_t* left_img, int left_pitch,
    const uint8_t* right_img, int right_pitch,
    int img_w, int img_h,
    const stereo3d::DualYoloGpuDetection& left_det,
    const stereo3d::DualYoloGpuDetection& right_det,
    float center_x, float center_y, float radius,
    float initial_disp,
    int patch_radius,
    int search_radius,
    int max_points,
    int min_points,
    int max_disparity,
    float min_confidence,
    float max_delta,
    float max_stddev,
    float focal,
    float baseline,
    float min_depth,
    float max_depth,
    float feature_y_tolerance_px,
    float feature_y_slope,
    float feature_y_offset_px,
    float feature_reverse_check_px,
    float feature_overlap_scale,
    float feature_mad_scale,
    float feature_ransac_gate_px,
    float feature_sphere_radius_m,
    float feature_sphere_radius_scale,
    float feature_sphere_margin_m,
    float* sample_disp,
    float* sample_score,
    float* sample_x,
    float* sample_y,
    float* sample_right_y,
    float* point_x,
    float* point_y,
    float best_score_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_disp_parts[kMaxFeaturePoints][kThreadsPerPoint],
    float best_dy_parts[kMaxFeaturePoints][kThreadsPerPoint],
    int* valid_count,
    stereo3d::DualYoloGpuDisparity* out) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        *valid_count = 0;
        clearDisparity(out);
    }
    for (int i = tid; i < kMaxFeaturePoints; i += kThreads) {
        sample_disp[i] = 0.0f;
        sample_score[i] = 0.0f;
        sample_x[i] = 0.0f;
        sample_y[i] = 0.0f;
        sample_right_y[i] = 0.0f;
        point_x[i] = 0.0f;
        point_y[i] = 0.0f;
        for (int j = 0; j < kThreadsPerPoint; ++j) {
            best_score_parts[i][j] = -2.0f;
            best_disp_parts[i][j] = -1.0f;
            best_dy_parts[i][j] = 0.0f;
        }
    }
    __syncthreads();

    max_points = clampInt(max_points, 1,
                          min(kMaxFeaturePoints, kMaxParallelFeaturePoints));
    min_points = clampInt(min_points, 1, max_points);
    patch_radius = clampInt(patch_radius, 2, 8);
    const int point_idx = tid / kThreadsPerPoint;
    const int sub_idx = tid % kThreadsPerPoint;
    if (point_idx < max_points) {
        float x_f = center_x;
        float y_f = center_y;
        if (point_idx > 0) {
            const int ring_idx = point_idx - 1;
            const int angle_idx = ring_idx % 8;
            const int ring = ring_idx / 8;
            const float ring_frac = ring == 0 ? 0.28f : (ring == 1 ? 0.48f : 0.66f);
            const float angle = 0.78539816339f * static_cast<float>(angle_idx);
            const float rr = fmaxf(static_cast<float>(patch_radius + 2), radius * ring_frac);
            x_f += rr * cosf(angle);
            y_f += rr * sinf(angle);
        }
        const int x = static_cast<int>(rintf(x_f));
        const int y = static_cast<int>(rintf(y_f));
        if (patchInside(img_w, img_h, x, y, patch_radius)) {
            const int d0 = static_cast<int>(rintf(initial_disp));
            const int d_start = max(1, d0 - search_radius);
            const int d_end = min(max_disparity, d0 + search_radius);
            if (d_start < d_end) {
                const int range = d_end - d_start + 1;
                const int per = (range + kThreadsPerPoint - 1) / kThreadsPerPoint;
                const int begin = d_start + sub_idx * per;
                const int end = min(d_end, begin + per - 1);
                const float expected_y = expectedFeatureYDelta(
                    static_cast<float>(x), left_det,
                    feature_y_slope, feature_y_offset_px);
                const int y_radius = clampInt(
                    static_cast<int>(ceilf(
                        clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                    1, 3);
                const int dy_center = static_cast<int>(rintf(-expected_y));
                float best_score = -2.0f;
                float best_disp = -1.0f;
                float best_dy = 0.0f;
                for (int d = begin; d <= end; ++d) {
                    const int xr = x - d;
                    for (int dy = dy_center - y_radius;
                         dy <= dy_center + y_radius; ++dy) {
                        const int yr = y + dy;
                        if (!patchInside(img_w, img_h, xr, yr, patch_radius)) continue;
                        const float score = znccScore(left_img, left_pitch,
                                                      right_img, right_pitch,
                                                      x, y, xr, yr, patch_radius);
                        if (score > best_score) {
                            best_score = score;
                            best_disp = static_cast<float>(d);
                            best_dy = static_cast<float>(dy);
                        }
                    }
                }
                best_score_parts[point_idx][sub_idx] = best_score;
                best_disp_parts[point_idx][sub_idx] = best_disp;
                best_dy_parts[point_idx][sub_idx] = best_dy;
                if (sub_idx == 0) {
                    point_x[point_idx] = static_cast<float>(x);
                    point_y[point_idx] = static_cast<float>(y);
                }
            }
        }
    }
    __syncthreads();

    if (point_idx < max_points && sub_idx == 0) {
        float best_score = best_score_parts[point_idx][0];
        float best_disp = best_disp_parts[point_idx][0];
        float best_dy = best_dy_parts[point_idx][0];
        for (int i = 1; i < kThreadsPerPoint; ++i) {
            if (best_score_parts[point_idx][i] > best_score) {
                best_score = best_score_parts[point_idx][i];
                best_disp = best_disp_parts[point_idx][i];
                best_dy = best_dy_parts[point_idx][i];
            }
        }
        const float min_score = fmaxf(0.10f, min_confidence * 0.60f);
        if (best_disp > 0.5f && best_score >= min_score &&
            fabsf(best_disp - initial_disp) <= max_delta) {
            const float z = focal * baseline / best_disp;
            const float left_x = point_x[point_idx];
            const float left_y = point_y[point_idx];
            const float right_x = left_x - best_disp;
            const float right_y = left_y + best_dy;
            const bool cheap_ok =
                z >= min_depth && z <= max_depth &&
                passesFeatureYGate(left_x, left_y, right_y, left_det,
                                   feature_y_tolerance_px,
                                   feature_y_slope,
                                   feature_y_offset_px) &&
                passesFeatureOverlapGate(left_det, right_det,
                                         left_x, left_y, right_x, right_y,
                                         initial_disp, feature_overlap_scale) &&
                passesSphereRadiusGate(left_x, left_y,
                                       center_x, center_y,
                                       best_disp, initial_disp,
                                       focal, baseline,
                                       feature_sphere_radius_m,
                                       feature_sphere_radius_scale,
                                       feature_sphere_margin_m);
            if (cheap_ok) {
                const int d0 = static_cast<int>(rintf(initial_disp));
                const int d_start = max(1, d0 - search_radius);
                const int d_end = min(max_disparity, d0 + search_radius);
                const int reverse_y_radius = clampInt(
                    static_cast<int>(ceilf(
                        clampFloat(feature_y_tolerance_px, 0.5f, 8.0f))),
                    1, 3);
                const float reverse_err = feature_reverse_check_px >= 0.0f
                    ? reverseSparseMatchError(left_img, left_pitch,
                                              right_img, right_pitch,
                                              nullptr, 0,
                                              nullptr, 0,
                                              img_w, img_h,
                                              left_x, left_y,
                                              right_x, right_y,
                                              patch_radius,
                                              d_start, d_end,
                                              reverse_y_radius,
                                              1,
                                              left_det,
                                              feature_y_slope,
                                              feature_y_offset_px)
                    : 0.0f;
                if (feature_reverse_check_px < 0.0f ||
                    reverse_err <= fmaxf(0.25f, feature_reverse_check_px)) {
                    const int idx = atomicAdd(valid_count, 1);
                    if (idx < kMaxFeaturePoints) {
                        sample_disp[idx] = best_disp;
                        sample_score[idx] = best_score;
                        sample_x[idx] = left_x;
                        sample_y[idx] = left_y;
                        sample_right_y[idx] = right_y;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int n = min(*valid_count, kMaxFeaturePoints);
        out->attempted = max_points;
        out->support = n;
        if (n < min_points) {
            out->low_confidence = 1;
        } else {
            float disparity = 0.0f;
            float anchor_x = 0.0f;
            float anchor_y = 0.0f;
            float stddev = 0.0f;
            float avg_score = 0.0f;
            int support = 0;
            if (!robustAggregateSamples(n, min_points, initial_disp, max_delta,
                                        max_stddev, feature_mad_scale,
                                        feature_ransac_gate_px,
                                        sample_disp, sample_score,
                                        sample_x, sample_y, sample_right_y,
                                        point_x,
                                        &disparity, &anchor_x, &anchor_y,
                                        &stddev, &avg_score, &support)) {
                out->low_confidence = 1;
            } else {
                const float z = focal * baseline / fmaxf(disparity, 0.5f);
                if (disparity > static_cast<float>(max_disparity) ||
                    z < min_depth || z > max_depth) {
                    out->low_confidence = 1;
                } else {
                    const float min_score = fmaxf(0.10f, min_confidence * 0.60f);
                    const float score_conf =
                        clampFloat((avg_score - min_score) /
                                   fmaxf(0.01f, 1.0f - min_score),
                                   0.0f, 1.0f);
                    const float support_ratio =
                        static_cast<float>(support) /
                        static_cast<float>(max(1, max_points));
                    const float consistency =
                        clampFloat(1.0f / (1.0f + stddev), 0.0f, 1.0f);
                    const float delta_conf =
                        1.0f - fminf(1.0f, fabsf(disparity - initial_disp) / max_delta);
                    out->disparity = disparity;
                    out->confidence = clampFloat(0.30f * support_ratio +
                                                 0.35f * score_conf +
                                                 0.25f * consistency +
                                                 0.10f * delta_conf,
                                                 0.0f, 1.0f);
                    out->stddev = stddev;
                    out->delta_gate_px = max_delta;
                    out->anchor_cx = anchor_x;
                    out->anchor_cy = anchor_y;
                    out->support = support;
                    out->valid = out->confidence >= min_confidence ? 1 : 0;
                    out->low_confidence = out->valid ? 0 : 1;
                    copyDualYoloDebugMatches(n, sample_disp, sample_score,
                                             sample_x, sample_y,
                                             sample_right_y, point_x, out);
                }
            }
        }
    }
    __syncthreads();
}
