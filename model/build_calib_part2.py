#!/usr/bin/env python3
"""
Generate calibration data for part2 from raw images.

Step 1: run part1 on the calibration images → save outputs 199 + 202.
Step 2: apply unshuffle_concat(202, 199) → part2 input [N,1280,13,13].

Usage:
  python3 build_calib_part2.py /path/to/images --part1-onnx model_neural_part1.onnx --out calib_part2.npy

The image preprocessing is identical to build_calib.py (letterbox 416,
BGR→RGB, /255) — the same images used for part1 calibration are reused here.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

from rewrite_s2d import unshuffle_concat

INPUT_SIZE = 416


def letterbox_resize(img, size):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--part1-onnx", default="model_neural_part1.onnx")
    parser.add_argument("--out", default="calib_part2.npy")
    parser.add_argument("--max", type=int, default=300)
    args = parser.parse_args()

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(args.image_dir, e)))
        files.extend(glob.glob(os.path.join(args.image_dir, "**", e), recursive=True))
    files = sorted(set(files))[:args.max]
    if not files:
        raise SystemExit(f"No images found under {args.image_dir}")

    # Load part1 as ONNX session
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.part1_onnx, so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    batch199 = []
    batch202 = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        img = letterbox_resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        t199, t202 = sess.run(None, {inp_name: img})
        batch199.append(t199)
        batch202.append(t202)

    # S2D transform on host
    part2_data = np.concatenate(
        [unshuffle_concat(t202, t199) for t199, t202 in zip(batch199, batch202)],
        axis=0,
    )
    np.save(args.out, part2_data)
    print(f"Wrote {part2_data.shape} calibration tensor to {args.out}")


if __name__ == "__main__":
    main()
