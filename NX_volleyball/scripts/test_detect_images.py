#!/usr/bin/env python3
"""
test_detect_images.py - TensorRT GPU detect test on images
Test detection quality by running TRT engine on calib_500 images.
Usage:
  python3 test_detect_images.py --engine <path> --images <dir> [--save <dir>] [--conf 0.25]
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

try:
    import tensorrt as trt
    import ctypes
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

# ---- TRT inference via plain ctypes/cudart when pycuda unavailable ----
if not HAS_PYCUDA:
    import tensorrt as trt
    import ctypes

    _cudart = ctypes.CDLL("libcudart.so")

    def _cuda_malloc(nbytes):
        ptr = ctypes.c_void_p()
        _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        return ptr.value

    def _cuda_free(ptr):
        _cudart.cudaFree(ctypes.c_void_p(ptr))

    def _cuda_memcpy_h2d(dst, src_np, nbytes: int, stream=None):
        _cudart.cudaMemcpy(ctypes.c_void_p(dst),
                           src_np.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_size_t(nbytes), 1)  # cudaMemcpyHostToDevice

    def _cuda_memcpy_d2h(dst_np, src, nbytes: int, stream=None):
        _cudart.cudaMemcpy(dst_np.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_void_p(src),
                           ctypes.c_size_t(nbytes), 2)  # cudaMemcpyDeviceToHost


class TRTInference:
    """Minimal TensorRT inference wrapper (no pycuda dependency)."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover I/O tensors
        self.input_name = None
        self.output_names = []
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
                print(f"  Input: {name} {self.input_shape}")
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)
                print(f"  Output: {name} {tuple(shape)}")

        self.input_size = self.input_shape[2]  # H == W

        # Allocate device buffers
        self._d_input = _cuda_malloc(int(np.prod(self.input_shape) * 4))
        self._d_outputs = {}
        self._h_outputs = {}
        for name in self.output_names:
            shape = self.output_shapes[name]
            nbytes = int(np.prod(shape) * 4)
            self._d_outputs[name] = _cuda_malloc(nbytes)
            self._h_outputs[name] = np.empty(shape, dtype=np.float32)

    def infer(self, input_chw: np.ndarray):
        """Run inference. input_chw: float32 [1,3,H,W]."""
        nbytes_in = int(np.prod(self.input_shape) * 4)
        _cuda_memcpy_h2d(self._d_input, np.ascontiguousarray(input_chw), nbytes_in)

        self.context.set_tensor_address(self.input_name, self._d_input)
        for name in self.output_names:
            self.context.set_tensor_address(name, self._d_outputs[name])
        self.context.execute_async_v3(0)
        _cudart.cudaDeviceSynchronize()

        results = {}
        for name in self.output_names:
            shape = self.output_shapes[name]
            nbytes = int(np.prod(shape) * 4)
            _cuda_memcpy_d2h(self._h_outputs[name], self._d_outputs[name], nbytes)
            results[name] = self._h_outputs[name].copy()
        return results

    def __del__(self):
        if hasattr(self, "_d_input") and self._d_input:
            _cuda_free(self._d_input)
        if hasattr(self, "_d_outputs"):
            for ptr in self._d_outputs.values():
                _cuda_free(ptr)


# ---- Preprocessing ----

def letterbox(img, new_shape=640):
    """Letterbox resize keeping aspect ratio, pad with gray (114)."""
    h, w = img.shape[:2]
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    pad_x = (new_shape - nw) // 2
    pad_y = (new_shape - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img_bgr, input_size=640):
    """BGR image -> float32 NCHW tensor [1,3,H,W] in RGB order."""
    canvas, scale, pad_x, pad_y = letterbox(img_bgr, input_size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, 0)  # NCHW
    return blob, scale, pad_x, pad_y


# ---- Postprocessing ----

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def nms(boxes, scores, iou_thresh=0.45):
    """Simple NMS. boxes: Nx4 (x1,y1,x2,y2), scores: N."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (a1 + a2 - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def softmax_1d(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def dfl_decode(bbox_raw, reg_max=16):
    """Decode DFL bbox from raw [4*reg_max] to [4] (dist l,t,r,b)."""
    dists = np.zeros(4, dtype=np.float32)
    for i in range(4):
        chunk = bbox_raw[i * reg_max:(i + 1) * reg_max]
        w = softmax_1d(chunk)
        dists[i] = np.sum(w * np.arange(reg_max, dtype=np.float32))
    return dists


def postprocess_single_output(output, conf_thresh, inputsz):
    """Handle single-tensor output: [1, 4+nc, N] or [1, N, 4+nc] or [1, N, 6]."""
    out = output.squeeze()  # remove batch dim
    if out.ndim != 2:
        return np.empty((0, 6))

    d0, d1 = out.shape
    # Post-NMS format: [N, 6] => x1,y1,x2,y2,conf,cls
    if d1 == 6 and d0 <= 1000:
        mask = out[:, 4] >= conf_thresh
        dets = out[mask]
        # Convert to cx,cy,w,h format for consistency
        result = np.zeros((len(dets), 6))
        result[:, 0] = (dets[:, 0] + dets[:, 2]) / 2  # cx
        result[:, 1] = (dets[:, 1] + dets[:, 3]) / 2  # cy
        result[:, 2] = dets[:, 2] - dets[:, 0]         # w
        result[:, 3] = dets[:, 3] - dets[:, 1]         # h
        result[:, 4] = dets[:, 4]                       # conf
        result[:, 5] = dets[:, 5]                       # cls
        return result

    # Transposed: [4+nc, N] where d0 < d1
    if d0 < d1:
        out = out.T  # -> [N, 4+nc]

    N, channels = out.shape
    nc = channels - 4
    if nc <= 0:
        return np.empty((0, 6))

    cx = out[:, 0]
    cy = out[:, 1]
    w = out[:, 2]
    h = out[:, 3]

    scores = out[:, 4:].max(axis=1)
    cls_ids = out[:, 4:].argmax(axis=1)

    mask = scores >= conf_thresh
    if not mask.any():
        return np.empty((0, 6))

    dets = np.column_stack([cx[mask], cy[mask], w[mask], h[mask],
                            scores[mask], cls_ids[mask].astype(np.float32)])

    # NMS
    x1 = dets[:, 0] - dets[:, 2] / 2
    y1 = dets[:, 1] - dets[:, 3] / 2
    x2 = dets[:, 0] + dets[:, 2] / 2
    y2 = dets[:, 1] + dets[:, 3] / 2
    keep = nms(np.column_stack([x1, y1, x2, y2]), dets[:, 4])
    return dets[keep]


def postprocess_multiscale(outputs, conf_thresh, inputsz, reg_max=16):
    """Handle 6-tensor DFL output: cls[H,W,nc] + bbox[H,W,64] per scale."""
    # Group by spatial dims: sort tensors, pair cls (small C) with bbox (C=64)
    tensors = []
    for name, data in outputs.items():
        d = data.squeeze()
        tensors.append((name, d))

    # Pair: cls has channels <= 4, bbox has channels == 64 (or 4*reg_max)
    scales = []
    used = set()
    for i, (n1, d1) in enumerate(tensors):
        if i in used:
            continue
        if d1.ndim != 3:
            continue
        h1, w1, c1 = d1.shape
        if c1 > 4:
            continue
        # This is cls tensor, find matching bbox
        for j, (n2, d2) in enumerate(tensors):
            if j in used or j == i:
                continue
            if d2.ndim != 3:
                continue
            h2, w2, c2 = d2.shape
            if h2 == h1 and w2 == w1 and c2 == 4 * reg_max:
                stride = inputsz // h1
                scales.append((d1, d2, stride))
                used.add(i)
                used.add(j)
                break

    all_dets = []
    for cls_map, bbox_map, stride in scales:
        H, W, nc = cls_map.shape
        cls_scores = sigmoid(cls_map)
        for y_idx in range(H):
            for x_idx in range(W):
                max_score = cls_scores[y_idx, x_idx].max()
                if max_score < conf_thresh:
                    continue
                cls_id = cls_scores[y_idx, x_idx].argmax()
                dist = dfl_decode(bbox_map[y_idx, x_idx], reg_max)
                # dist = [left, top, right, bottom]
                cx = (x_idx + 0.5) * stride
                cy = (y_idx + 0.5) * stride
                x1 = cx - dist[0] * stride
                y1 = cy - dist[1] * stride
                x2 = cx + dist[2] * stride
                y2 = cy + dist[3] * stride
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                all_dets.append([cx, cy, w, h, float(max_score), float(cls_id)])

    if not all_dets:
        return np.empty((0, 6))

    dets = np.array(all_dets)
    x1 = dets[:, 0] - dets[:, 2] / 2
    y1 = dets[:, 1] - dets[:, 3] / 2
    x2 = dets[:, 0] + dets[:, 2] / 2
    y2 = dets[:, 1] + dets[:, 3] / 2
    keep = nms(np.column_stack([x1, y1, x2, y2]), dets[:, 4])
    return dets[keep]


def postprocess(outputs, conf_thresh, input_size):
    """Auto-detect format and postprocess."""
    if len(outputs) == 1:
        key = list(outputs.keys())[0]
        return postprocess_single_output(outputs[key], conf_thresh, input_size)
    elif len(outputs) == 6:
        return postprocess_multiscale(outputs, conf_thresh, input_size)
    else:
        # Try single-output on first tensor
        key = list(outputs.keys())[0]
        return postprocess_single_output(outputs[key], conf_thresh, input_size)


def rescale_dets(dets, scale, pad_x, pad_y):
    """Convert detections from letterbox input coords back to original image coords."""
    if len(dets) == 0:
        return dets
    dets = dets.copy()
    dets[:, 0] = (dets[:, 0] - pad_x) / scale  # cx
    dets[:, 1] = (dets[:, 1] - pad_y) / scale  # cy
    dets[:, 2] = dets[:, 2] / scale              # w
    dets[:, 3] = dets[:, 3] / scale              # h
    return dets


def draw_dets(img, dets):
    """Draw detection boxes on image. dets: Nx6 [cx,cy,w,h,conf,cls]."""
    vis = img.copy()
    for det in dets:
        cx, cy, w, h, conf, cls = det
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"volleyball {conf:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="TRT detection test on images")
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--save", default="", help="Save annotated images to dir")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--show-summary", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Engine: {args.engine}")
    print(f"Images: {args.images}")
    print(f"Conf threshold: {args.conf}")

    # Load engine
    model = TRTInference(args.engine)
    input_size = model.input_size
    print(f"Input size: {input_size}x{input_size}")

    # Find images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(args.images, ext)))
    image_files.sort()
    if args.max_images > 0:
        image_files = image_files[:args.max_images]
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("ERROR: No images found!")
        return

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    # Process
    total_dets = 0
    images_with_det = 0
    total_time = 0.0
    det_confs = []

    for idx, fpath in enumerate(image_files):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  SKIP (unreadable): {os.path.basename(fpath)}")
            continue

        blob, scale, pad_x, pad_y = preprocess(img, input_size)
        t0 = time.time()
        outputs = model.infer(blob)
        t1 = time.time()
        total_time += (t1 - t0)

        dets = postprocess(outputs, args.conf, input_size)
        dets = rescale_dets(dets, scale, pad_x, pad_y)

        n = len(dets)
        total_dets += n
        if n > 0:
            images_with_det += 1
            for d in dets:
                det_confs.append(d[4])

        if idx < 10 or (idx % 50 == 0):
            print(f"  [{idx+1}/{len(image_files)}] {os.path.basename(fpath)}: "
                  f"{n} det(s), {(t1-t0)*1000:.1f}ms"
                  + (f", best conf={dets[:,4].max():.3f}" if n > 0 else ""))

        if args.save and n > 0:
            vis = draw_dets(img, dets)
            save_path = os.path.join(args.save, os.path.basename(fpath))
            cv2.imwrite(save_path, vis)

    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY:")
    print(f"  Total images:         {len(image_files)}")
    print(f"  Images with detect:   {images_with_det} ({100*images_with_det/max(len(image_files),1):.1f}%)")
    print(f"  Total detections:     {total_dets}")
    print(f"  Avg inference time:   {total_time/max(len(image_files),1)*1000:.1f} ms")
    if det_confs:
        det_confs = np.array(det_confs)
        print(f"  Confidence: mean={det_confs.mean():.3f}, "
              f"min={det_confs.min():.3f}, max={det_confs.max():.3f}")
    print("=" * 50)

    if images_with_det == 0:
        print("\nWARNING: Zero detections! Check model/engine/confidence threshold.")
    elif images_with_det < len(image_files) * 0.3:
        print(f"\nWARNING: Low detection rate ({100*images_with_det/len(image_files):.0f}%). "
              "Try lowering --conf threshold.")


if __name__ == "__main__":
    main()
