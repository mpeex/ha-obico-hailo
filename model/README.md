# From the official Obico ONNX to Hailo HEFs

This folder contains everything needed to **rebuild the compiled Hailo models**
(`app/model/*.hef`) that the addon ships, starting from Obico's official
YOLOv5 weights export.

The Model is a **shared-weight YOLOv5** re-exported from darknet format:
input `[1,3,416,416]`, 1 class (`failure`), 5 anchors on a single 13×13 scale
(`boxes`/`confs`). Hailo executes **only the neural part**; the decoding head is
not supported on the accelerator and is handled on the host via `decode.onnx`
+ `post_processing`.

## Two-part architecture

The graph is split into two HEF models, with the neck's **space-to-depth**
fusion (6D/5D Reshape + Transpose block) executed **on the host** (numpy). This
avoids two known Hailo ONNX parser bugs:

1. `update_reshape_output_format` crashes on Reshape 6D/5D (`'NoneType' has no
   attribute 'copy'`) because it misinterprets them as attention-window
   reshapes.
2. Even rewriting S2D as Slice+Concat, the 5-input Concat did not propagate
   shapes correctly → the following Conv sees the wrong channel count.

```
input [1,3,416,416]
  ↓
┌──────────────────────────────────────┐
│  Part 1 HEF  (backbone+neck)        │
│  Conv / LeakyRelu / MaxPool × 47    │
│  → 199  [1,1024,13,13]             │
│  → 202  [1, 64, 26,26]             │
└──────────────────────────────────────┘
  ↓                          ↓
  199                    202
  │                        │
  │                   ┌────────────────┐
  │                   │ Host: unshuffle │
  │                   │   202 → [1,256,13,13]
  │                   └────────────────┘
  │                        │
  │              concat(·, 199) → 214 [1,1280,13,13]
  │                        │
  ↓                        ↓
┌──────────────────────────────────────┐
│  Part 2 HEF  (detection tail)       │
│  input → Conv_59 → LeakyRelu →      │
│          Conv_61 → 218 [1,30,13,13] │
└──────────────────────────────────────┘
  ↓
decode.onnx (onnxruntime) → boxes + confs
  ↓
post_processing → detections
```

## Files

| File | Purpose |
|------|---------|
| `split_model.py` | Extract the original Obico ONNX → part1 + part2 + decode |
| `rewrite_s2d.py` | `unshuffle_concat()` used by the runtime and the calib scripts (plus an optional Slice+Concat S2D rewrite for the parser) |
| `build_calib.py` | Generates `calib.npy` `[N,3,416,416]` from domain images (no labels) — for part1 |
| `build_calib_part2.py` | Generates `calib_part2.npy` `[N,1280,13,13]` for part2 (runs part1 internally) |
| `compile_hef.py` | Compiles part1 or part2 → `.har` → `.hef` (`--part1` or `--part2`; `--hw-arch hailo8l` for the AI HAT+ 8L) |

The compiled models shipped in `app/model/` (`obico_part1/2.hef`,
`obico_part1/2_8l.hef`, `decode.onnx`) are the output of this toolchain.

## Full workflow

```bash
# 0) Environment: venv with onnx, onnxruntime and the Hailo toolchain
#    (Hailo Dataflow Compiler → hailo_sdk_client)

# 1) SPLIT: create part1 + part2 + decode from the original Obico ONNX
python3 split_model.py model-weights-5a6b1be1fa.onnx

#    produces:
#      model_neural_part1.onnx  → input [1,3,416,416], outputs 199+202
#      model_neural_part2.onnx  → input [1,1280,13,13], output 218 [1,30,13,13]
#      decode.onnx             → input 218, output boxes + confs

# 2) Calibrate part1 (domain images, no annotations)
python3 build_calib.py /path/to/images --out calib.npy

# 3) Compile HEF part1 (backbone+neck)
python3 compile_hef.py --part1 --hw-arch hailo8 --calib calib.npy

# 4) Calibrate part2 (same images, forwarded through part1)
python3 build_calib_part2.py /path/to/images \
    --part1-onnx model_neural_part1.onnx \
    --out calib_part2.npy

# 5) Compile HEF part2 (detection head)
python3 compile_hef.py --part2 --hw-arch hailo8 --calib calib_part2.npy
```

> On the **Raspberry Pi AI HAT+ 8L** (Hailo-8L) use `--hw-arch hailo8l`.

## Expected outputs

- `obico_part1.har` / `obico_part1.hef` — backbone+neck (43 MB for Hailo-8)
- `obico_part2.har` / `obico_part2.hef` — detection head (13 MB for Hailo-8)

Compile the `HAILO8L` set on a machine with a Hailo-8L (so calibration matches
the architecture) and rename the outputs to `obico_part1_8l.hef` /
`obico_part2_8l.hef`, matching the `.hef` LFS pattern in `app/model/`.
`decode.onnx` is architecture-agnostic (runs on the CPU host) so it is shared,
not duplicated per arch.

## Runtime on HailoRT

The addon runtime (`app/lib/hailo.py`) loads `part1` + `part2` on the same
`VDevice`, performs the **space-to-depth + concat on the host** (numpy) and
then runs `decode.onnx` (ONNX Runtime CPU). Conventions used by the runtime:

- The part2 HEF path is derived from part1's: `.../obico_part1.hef` →
  `.../obico_part2.hef`, and `.../obico_part1_8l.hef` → `.../obico_part2_8l.hef`.
- `decode.onnx` by default sits next to the HEFs.
- HEF vstream names (`obico_part1/conv…`) are **not** the ONNX tensor names
  (`199`/`202`): part1 outputs are resolved **by shape** (26×26 →
  space-to-depth, 13×13 → skip), robust to any naming.
- HailoRT returns NHWC outputs → converted to NCHW before decoding.
- Architecture is detected at load via a control-only `Device.control.identify()`
  (`HAILO8` vs `HAILO8L`) and selects the matching HEF set.

## Validation (100-image dataset)

- ~72 ms/frame on Hailo-8 (RPi5).
- HEF vs ONNX reference: 799 detections on both; **hit rate 92.5%** (IoU≥0.5),
  mean IoU **0.936**, mean confidence drift **0.016**.
- mAP50 identical ONNX/HEF (0.001–0.003): a model limitation, not a conversion
  one; Obico's default threshold (0.08) is below the maximum the model produces
  (~0.13–0.25) → no false FAILs.