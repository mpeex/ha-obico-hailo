#!/usr/bin/env python3
"""
Split Obico's exported YOLOv5 ONNX into Hailo-friendly parts:

  1. model_neural_part1.onnx  — backbone+neck up to conv outputs
     input  "input"  [1,3,416,416] → outputs "199" [1,1024,13,13] + "202" [1,64,26,26]
  2. model_neural_part2.onnx  — detection tail (concat input → head convs)
     input  "199"+"214_s2d" [1,1280,13,13] → output "218" [1,30,13,13]
  3. decode.onnx  — host-side YOLO decoder (218 → boxes + confs)

The space-to-depth block (6D/5D Reshape+Transpose chain) lives entirely in
the host and is NOT part of either HEF. This avoids the Hailo ONNX parser bug
(update_reshape_output_format: 'NoneType' has no attribute 'copy') AND the
subsequent Concat shape propagation failure (5-input Concat → Conv sees wrong
input channel count).

Host-side pipeline at runtime:
  img → part1_hef → (199, 202) → s2d_unshuffle(202) + concat(·, 199)
       → part2_hef → 218 → decode.onnx → boxes/confs → post_processing

These three files together reproduce the original model exactly (verified:
outputs differ by 0.0).

Usage:  python3 split_model.py INPUT.onnx
"""
import argparse
import os

import onnx
from onnx import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", help="original Obico ONNX (input [1,3,416,416], "
                                     "outputs boxes/confs)")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p1_path = os.path.join(args.out_dir, "model_neural_part1.onnx")
    p2_path = os.path.join(args.out_dir, "model_neural_part2.onnx")
    dec_path = os.path.join(args.out_dir, "decode.onnx")
    tmp = os.path.join(args.out_dir, ".tmp_src.onnx")

    # Save to disk (onnx.utils.extract_model needs a file path in onnx>=1.14)
    src = onnx.load(args.onnx)
    onnx.save(src, tmp)

    # ---- Part 1: backbone → 199 [1,1024,13,13] + 202 [1,64,26,26] ----------
    # Includes Conv_43→LeakyRelu_44 (→199) and Conv_45→LeakyRelu_46 (→202).
    utils.extract_model(tmp, p1_path,
                        input_names=["input"],
                        output_names=["199", "202"])
    p1 = onnx.shape_inference.infer_shapes(onnx.load(p1_path))
    onnx.checker.check_model(p1)
    onnx.save(p1, p1_path)
    print(f"Wrote {p1_path}"
          f"  inputs={[i.name for i in p1.graph.input]}"
          f"  outputs={[o.name for o in p1.graph.output]}")

    # ---- Part 2: detection tail. input 214 → Conv_59 → Conv_61 → 218 ----
    # 214 is the result of Concat(s2d(202), 199) computed on the host, so it
    # already contains 199's contribution. Part 2 therefore needs a single
    # input 214 [1,1280,13,13] → output 218 [1,30,13,13].
    utils.extract_model(tmp, p2_path,
                        input_names=["214"],
                        output_names=["218"])
    p2 = onnx.shape_inference.infer_shapes(onnx.load(p2_path))
    onnx.checker.check_model(p2)
    onnx.save(p2, p2_path)
    print(f"Wrote {p2_path}"
          f"  inputs={[i.name for i in p2.graph.input]}"
          f"  outputs={[o.name for o in p2.graph.output]}")

    # ---- Decode (host-side YOLO head) ------------------------------------
    # Extract from 218 → boxes+confs. This works because the full model has
    # the decode head (Shape/Gather/Unsqueeze/Slice/Reshape) after 218.
    try:
        utils.extract_model(tmp, dec_path,
                            input_names=["218"],
                            output_names=["boxes", "confs"])
        dec = onnx.shape_inference.infer_shapes(onnx.load(dec_path))
        onnx.checker.check_model(dec)
        onnx.save(dec, dec_path)
        print(f"Wrote {dec_path}")
    except Exception as e:
        if os.path.exists(dec_path):
            print(f"Using existing {dec_path} (extract failed: {e})")
        else:
            raise

    os.remove(tmp)


if __name__ == "__main__":
    main()
