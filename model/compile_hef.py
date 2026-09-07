#!/usr/bin/env python3
"""
Compile the (split) Obico YOLOv5 ONNX models into HEF files for Hailo.

Two-part deployment (avoids the Hailo ONNX parser S2D bug):
  Part 1 (backbone):  input [1,3,416,416] → outputs 199 [1,1024,13,13] + 202 [1,64,26,26]
  Part 2 (detection): input [1,1280,13,13] → output  218 [1,30,13,13]

Host-side bridge between the two HEFs:
  unshuffle_concat(202) + concat(·, 199) → 214 [1,1280,13,13]   (see ml_api/lib/hailo.py)

Usage:
  python3 compile_hef.py --hw-arch hailo8 --part1 --calib calib.npy
  python3 compile_hef.py --hw-arch hailo8 --part2 --calib calib_part2.npy
"""
import argparse

import numpy as np
from hailo_sdk_client import ClientRunner

from rewrite_s2d import unshuffle_concat


def load_calib(path, batch_size=1, max_samples=None):
    data = np.load(path)
    if max_samples:
        data = data[:max_samples]
    n = data.shape[0]
    for i in range(0, n, batch_size):
        yield data[i:i + batch_size]


def to_runtime_layout(calib):
    """Convert an [N,C,H,W] calibration array to the runtime (NHWC) layout.

    Newer Hailo DFC releases feed calibration to the statistics collector in
    the runtime layout of the parsed graph, which is NHWC [N,H,W,C] for all
    4D inputs (the .npy from build_calib*.py is [N,C,H,W]). Only 4D inputs
    need the transpose; anything else (e.g. 2D) is passed through.
    """
    if calib.ndim == 4:
        return calib.transpose(0, 2, 3, 1)
    return calib


def build_part2_calib(part1_onnx, part1_hef_path, images_npy, hw_arch,
                      batch_size=1, max_samples=None):
    """Run part1 on calibration images, apply S2D → part2 calibration data.

    This is done here because part2's calibration tensors depend on part1's
    actual floating-point outputs; we cannot derive them from the raw images.
    """
    # NOTE: This path is only needed when calibrating part2 from raw images.
    # In practice, compile_hef.py is run twice: once for part1 (with --calib
    # pointing to raw images .npy) and once for part2 (with --calib pointing
    # to part1's output .npy, produced by this script or manually).

    # For simplicity, just load part1 outputs that were saved externally.
    raise NotImplementedError(
        "Automatic part2 calibration from raw images not yet implemented.\n"
        "Run compile_hef.py --part1 first, then generate part2 calibration\n"
        "data by running part1 inference on the calibration images and\n"
        "applying unshuffle_concat().  See ml_api/lib/hailo.py for the\n"
        "inference path, or generate with a small offline script.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1", action="store_true",
                        help="Compile part1 (backbone): input [1,3,416,416] → 199+202")
    parser.add_argument("--part2", action="store_true",
                        help="Compile part2 (detection): input [1,1280,13,13] → 218")
    parser.add_argument("--onnx", default=None,
                        help="ONNX path (default: model_neural_part{1,2}.onnx)")
    parser.add_argument("--har", default=None, help="Output HAR path")
    parser.add_argument("--hef", default=None, help="Output HEF path")
    parser.add_argument("--hw-arch", default="hailo8",
                        help="hailo8 / hailo8l / hailo10 ...")
    parser.add_argument("--calib", default=None, help=".npy calibration data")
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--calib-max", type=int, default=None)
    args = parser.parse_args()

    if not args.part1 and not args.part2:
        raise SystemExit("Specify --part1 or --part2")

    # ---- Resolve defaults per part ----------------------------------------
    if args.part1:
        default_onnx = "model_neural_part1.onnx"
        default_har  = "obico_part1.har"
        default_hef  = "obico_part1.hef"
        net_name     = "obico_part1"
        input_shape  = [1, 3, 416, 416]
        end_nodes    = ["199", "202"]
    else:
        default_onnx = "model_neural_part2.onnx"
        default_har  = "obico_part2.har"
        default_hef  = "obico_part2.hef"
        net_name     = "obico_part2"
        input_shape  = [1, 1280, 13, 13]
        end_nodes    = ["218"]

    onnx_path = args.onnx or default_onnx
    har_path  = args.har  or default_har
    hef_path  = args.hef  or default_hef

    runner = ClientRunner(hw_arch=args.hw_arch)

    # ---- 1. PARSE ---------------------------------------------------------
    if args.part1:
        runner.translate_onnx_model(
            onnx_path, net_name,
            start_node_names=["input"],
            end_node_names=end_nodes,
            net_input_shapes={"input": input_shape},
        )
    else:
        # Part 2 has a single input: 214 [1,1280,13,13] (the host already
        # folds 199 into 214 via the S2D+concat step).
        runner.translate_onnx_model(
            onnx_path, net_name,
            start_node_names=["214"],
            end_node_names=end_nodes,
            net_input_shapes={"214": input_shape},
        )
    runner.save_har(har_path)

    # ---- 2. OPTIMIZE / QUANTIZE -------------------------------------------
    alls = "model_optimization_config(calibration, batch_size=1)"
    if args.calib:
        calib_batches = load_calib(args.calib, batch_size=args.calib_batch_size,
                                   max_samples=args.calib_max)
        first = next(calib_batches)
        if first.shape[0] == 0:
            raise SystemExit("Calibration tensor is empty")
        runner.optimize_full_precision(calib_data=to_runtime_layout(first[:1]))
        # Pass all calibration data as a single numpy array (not a generator),
        # in the runtime (NHWC) layout this DFC expects.
        all_calib = np.concatenate([first] + list(calib_batches), axis=0)
        runner.optimize(calib_data=to_runtime_layout(all_calib))
    else:
        runner.optimize_full_precision(calib_data=np.zeros(input_shape, dtype=np.float32))

    # ---- 3. COMPILE --------------------------------------------------------
    hef = runner.compile()
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"Wrote {hef_path}")


if __name__ == "__main__":
    main()
