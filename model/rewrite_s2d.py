#!/usr/bin/env python3
"""
Rewrite the YOLOv5 "space-to-depth" neck block so the Hailo ONNX parser can
parse the model.

Obico's shared-weight YOLOv5 exports (PyTorch -> remodeled/darknet -> ONNX)
represent the S2D fusion with a chain of 6D/5D Reshape + Transpose nodes:

    Reshape -> [1,C,H,2,W,2]
    Transpose (0,1,2,4,3,5)
    Reshape -> [1,C,H*W,4]
    Transpose (0,1,3,2)
    Reshape -> [1,C,4,H,W]
    Transpose (0,2,1,3,4)
    Reshape -> [1,C*4,H,W]     # this is SpaceToDepth(bs=2, mode=CRC)

The Hailo Dataflow Compiler trips over these 6D/5D reshapes (it mistakes them
for attention-window reshapes) and crashes with 'TypeError: argument of type
'NoneType' is not iterable' / "'NoneType' object has no attribute 'copy'" in
onnx_graph.update_reshape_output_format.

Hailo's own model zoo implements the same S2D block with strided Slice nodes
followed by Concat on the features axis — a combination the parser handles.
This tool performs exactly that rewrite. The transformation is bit-exact:
the resulting model reproduces the original outputs (verified, diff 0.0).

Usage:  python3 rewrite_s2d.py model_neural.onnx [--out model_neural.onnx]
"""
import argparse
import copy
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

BLOCK = 2  # space-to-depth block size (2x2)


def unshuffle_concat(t202, t199):
    """Host-side S2D transform: part1 outputs → part2 input.

    Space-toDepth(CRC, bs=2) on t202 [1,64,26,26] → [1,256,13,13],
    then concat with t199 [1,1024,13,13] along axis=1 → [1,1280,13,13].
    This is the exact same computation as the original Reshape/Transpose chain
    that Hailo cannot parse (6D/5D attention-windows crash).

    t202 : np.ndarray  [1,64,26,26]
    t199 : np.ndarray  [1,1024,13,13]
    returns np.ndarray [1,1280,13,13]
    """
    C, HH, WW = t202.shape[1], t202.shape[2], t202.shape[3]
    H, W = HH // BLOCK, WW // BLOCK
    phases = np.empty((1, C * 4, H, W), dtype=t202.dtype)
    k = 0
    for dy in range(BLOCK):
        for dx in range(BLOCK):
            phases[:, k:k+C, :, :] = t202[:, :, dy::BLOCK, dx::BLOCK]
            k += C
    return np.concatenate([phases, t199], axis=1)


def _get_shape_const(g, name):
    for init in g.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init).tolist()
    for node in g.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value" and attr.t:
                    return numpy_helper.to_array(attr.t).tolist()
    return None


def find_s2d_chain(g):
    """Locate the reshape/transpose chain that implements block--2 S2D.

    Returns (source, end, concat_axis, concat_inputs_order) or None.
      source     : tensor name feeding the chain (e.g. '202', [1,C,2H,2W])
      end        : tensor name produced by the chain (e.g. '213', [1,C*4,H,W])
      concat_node: the Concat node that consumes `end`
    """
    for node in g.node:
        if node.op_type != "Reshape":
            continue
        shape = _get_shape_const(g, node.input[1])
        if not shape or len(shape) != 6:
            continue
        # YOLOv5 S2D starts with Reshape to [1,C,H,2,W,2]
        if shape[3] != BLOCK or shape[5] != BLOCK:
            continue

        # walk the linear chain forward: R -> T -> R -> T -> R -> T -> R
        names = {n: i for i, n in enumerate(node.output)}  # output -> current node
        cur_nodes = [node]
        cur_input, cur_out = node.input[0], node.output[0]
        for step in range(6):
            nxt = _find_single_consumer(g, cur_out)
            if nxt is None or nxt.op_type not in ("Reshape", "Transpose"):
                break
            cur_nodes.append(nxt)
            cur_out = nxt.output[0]

        # the chain must end with a Reshape to 4D [1,C*4,H,W] feeding a Concat
        last = cur_nodes[-1]
        if last.op_type != "Reshape":
            continue
        if len(_get_shape_const(g, last.input[1])) != 4:
            continue
        concat_node = _find_single_consumer(g, cur_out)
        if concat_node is None or concat_node.op_type != "Concat":
            continue

        return cur_input, cur_out, concat_node, cur_nodes
    return None


def _find_single_consumer(g, tensor):
    found = None
    for node in g.node:
        if tensor in node.input:
            if found is not None:
                return None  # multiple consumers -> not a linear chain
            found = node
    return found


def rewrite(g):
    """Rewrite the S2D chain in-place. Returns True if a rewrite happened."""
    hit = find_s2d_chain(g)
    if hit is None:
        return False

    source, chain_end, concat_node, chain_nodes = hit
    concat_axis = concat_node.attribute[0].i if concat_node.attribute else 1

    # Source shape [1,C,2H,2W] -> S2D blocks [1,C,.., ..]
    src_shape = None
    for v in g.value_info:
        if v.name == source:
            src_shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
    if not src_shape or len(src_shape) != 4:
        raise RuntimeError(f"cannot determine source shape for '{source}'")

    C, HH, WW = src_shape[1], src_shape[2], src_shape[3]
    H, W = HH // BLOCK, WW // BLOCK

    # NCHW -> features axis, phase-major order (CRC). For the ONNX exported
    # chain, output channel of phase k (k = dy*2+dx) is  out = k*C + c,
    # i.e. space-to-depth where each 2x2 neighbourhood is stacked contiguously
    # in channel order (phases block).
    slice_nodes = []
    for dy in range(BLOCK):
        for dx in range(BLOCK):
            name = f"s2d_slice_{dy}_{dx}"
            starts = helper.make_tensor(f"{name}_starts", TensorProto.INT64,
                                        [2], [dy, dx])
            ends = helper.make_tensor(f"{name}_ends", TensorProto.INT64,
                                      [2], [HH, WW])
            axes = helper.make_tensor(f"{name}_axes", TensorProto.INT64,
                                      [2], [2, 3])
            steps = helper.make_tensor(f"{name}_steps", TensorProto.INT64,
                                       [2], [BLOCK, BLOCK])
            g.initializer.extend([starts, ends, axes, steps])
            slice_nodes.append(helper.make_node(
                "Slice", [source, f"{name}_starts", f"{name}_ends",
                          f"{name}_axes", f"{name}_steps"], [name]))

    # Replace the Concat's single chain input with the phase slices. Keep the
    # other Concat inputs (e.g. the main 13x13 path '199') untouched.
    slices = [s.output[0] for s in slice_nodes]
    new_inputs = []
    for i in concat_node.input:
        if i == chain_end:
            new_inputs.extend(slices)
        else:
            new_inputs.append(i)
    del concat_node.input[:]
    concat_node.input.extend(new_inputs)

    # Drop the original chain nodes (and their value_info).
    chain_names = set()
    for node in chain_nodes:
        chain_names.update(node.output)
    keep_nodes = [n for n in g.node if n not in chain_nodes]
    # drop Constant nodes left orphaned by the chain removal
    consumed = set()
    for n in keep_nodes:
        consumed.update(n.input)
    orphan_consts = [n for n in keep_nodes
                     if n.op_type == "Constant" and n.output
                     and n.output[0] not in consumed]
    keep_nodes = [n for n in keep_nodes if n not in orphan_consts]

    # insert the Slice nodes right before the Concat, keeping topo order
    at = keep_nodes.index(concat_node)
    keep_nodes[at:at] = slice_nodes

    del g.node[:]
    g.node.extend(keep_nodes)
    keep_vi = [v for v in g.value_info if v.name not in chain_names]
    del g.value_info[:]
    g.value_info.extend(keep_vi)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", help="model to fix (in place after a .bak backup)")
    parser.add_argument("--out", default=None, help="write to this path instead")
    args = parser.parse_args()

    model = onnx.load(args.onnx)
    model = copy.deepcopy(model)
    if not rewrite(model.graph):
        raise SystemExit("No YOLOv5 space-to-depth chain found; nothing to do")

    onnx.checker.check_model(model)
    # re-infer shapes so Hailo (and the runtime) have full value_info
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    out = args.out or args.onnx
    if out == args.onnx:
        backup = args.onnx + ".bak"
        if not os.path.exists(backup):
            os.replace(args.onnx, backup)
    onnx.save(model, out)
    print(f"Rewrote S2D chain in {args.onnx} -> {out}"
          + (f" (backup: {backup})" if out == args.onnx else ""))


if __name__ == "__main__":
    main()