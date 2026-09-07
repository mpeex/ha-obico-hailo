#!/usr/bin/env python3
"""
Build the Hailo calibration set from arbitrary images (no labels needed).

Calibration is unsupervised: Hailo only needs the *distribution* of the
floating-point activations to derive per-layer quantization ranges, so fresh
images from the target domain are fine. Uses the same preprocessing as Obico's
runtime (see ml_api/lib/onnx.py): letterbox resize to 416, BGR->RGB, /255.

Usage:  python3 build_calib.py DIR_WITH_IMAGES --out calib.npy
"""
import argparse
import glob
import os

import cv2
import numpy as np

INPUT_SIZE = 416
NORMALIZE = True  # divide by 255 like the runtime


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
    parser.add_argument("--out", default="calib.npy")
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

    batch = []
    for f in files:
        img = cv2.imread(f)  # BGR
        if img is None:
            continue
        img = letterbox_resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        if NORMALIZE:
            img /= 255.0
        batch.append(img)

    arr = np.stack(batch).astype(np.float32)  # [N,416,416,3]
    arr = arr.transpose(0, 3, 1, 2)           # -> [N,3,416,416]
    np.save(args.out, arr)
    print(f"Wrote {arr.shape} calibration tensor to {args.out} ({len(batch)} images)")


if __name__ == "__main__":
    main()
