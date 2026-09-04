from typing import List, Tuple
import os
import numpy as np
import cv2
import onnxruntime

from lib.meta import Meta

from hailo_platform import (HEF, VDevice, Device, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

try:
    from hailo_platform import InferVStreams
    _HAS_INFER_VSTREAMS = True
except ImportError:
    _HAS_INFER_VSTREAMS = False

try:
    from hailo_platform.pyhailort.pyhailort import DeviceArchitectureTypes
except ImportError:  # pragma: no cover - defensive
    DeviceArchitectureTypes = None

try:
    from hailo_platform.pyhailort._pyhailort import SchedulingAlgorithm
except ImportError:  # pragma: no cover - defensive
    try:
        from hailo_platform.pyhailort.pyhailort import SchedulingAlgorithm
    except ImportError:  # pragma: no cover - defensive
        SchedulingAlgorithm = None

_INPUT_SIZE = 416

# Architecture → set of HEF files to load. An HEF compiled for one architecture
# cannot run on the other, so each architecture has its own compiled files.
# The default (no suffix) is the HAILO8 set; HAILO8L appends `_8l` to the base
# filenames (e.g. obico_part1_8l.hef / obico_part2_8l.hef).
_ARCH_SUFFIX = {
    "HAILO8": "",
    "HAILO8L": "_8l",
}


def _to_vstream(arr):
    """[N,C,H,W] NCHW → [N,H,W,C] NHWC for HailoRT input vstreams."""
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr


def _from_vstream(arr):
    """[N,H,W,C] NHWC output from HailoRT → [N,C,H,W] NCHW."""
    if arr.ndim == 4:
        return arr.transpose(0, 3, 1, 2)
    return arr


def _unshuffle_concat(t202, t199):
    """SpaceToDepth(CRC, bs=2) + concat, performed on host.

    t202 [1,64,26,26]  → [1,256,13,13]
    t199 [1,1024,13,13]
    concat → [1,1280,13,13]
    """
    C, H, W = t202.shape[1], t202.shape[2] // 2, t202.shape[3] // 2
    phases = np.empty((1, C * 4, H, W), dtype=t202.dtype)
    k = 0
    for dy in range(2):
        for dx in range(2):
            phases[:, k:k+C, :, :] = t202[:, :, dy::2, dx::2]
            k += C
    return np.concatenate([phases, t199], axis=1)


class HailoNet:
    vdevice: VDevice
    meta: Meta
    input_h: int
    input_w: int

    def __init__(self, hef_path: str, meta_path: str, decode_onnx_path: str = None,
                 use_gpu: bool = False):
        self.meta = Meta(meta_path)
        self.input_h = _INPUT_SIZE
        self.input_w = _INPUT_SIZE

        # Use explicit device_ids from Device.scan() so that VDevice() works
        # even when hailort_service is running (HailoRT quirk on single-device
        # platforms: bare VDevice() reports HAILO_OUT_OF_PHYSICAL_DEVICES).
        # multi_process_service=True + a HailoRT scheduling algorithm lets other
        # processes (e.g. face detection) share the same physical Hailo device
        # via the HailoRT service; the two options are required together.
        from hailo_platform import Device as _Device
        _ids = _Device.scan()
        _vdevice_params = VDevice.create_params()
        _vdevice_params.multi_process_service = True
        if SchedulingAlgorithm is not None:
            _vdevice_params.scheduling_algorithm = SchedulingAlgorithm.ROUND_ROBIN
        if _ids:
            _vdevice_params.device_ids = _ids
        self.vdevice = VDevice(_vdevice_params)

        # ---- Detect device architecture and pick the matching HEF set ------
        # Hailo-8 and Hailo-8L are different architectures: an HEF compiled for
        # one cannot run on the other. They are distinguished at runtime via a
        # control-only Device (coexists with the inference VDevice) and used to
        # select the right compiled HEF files.
        arch_name = self._detect_arch(_ids)
        self.arch_suffix = _ARCH_SUFFIX.get(arch_name, "")
        print(f'[hailo] Detected device architecture: {arch_name} '
              f'(HEF suffix: "{self.arch_suffix}")', flush=True)

        # ---- Load part1 + part2 HEFs --------------------------------------
        # hef_path should point to the part1 .hef; part2 .hef is placed
        # alongside it with a _part2 suffix, e.g.:
        #   obico_part1.hef   (hef_path)
        #   obico_part2.hef   (hef_path with 'part1' → 'part2')
        # For HAILO8L the base files are suffixed _8l (obico_part1_8l.hef) and
        # the part2 file is derived from that suffixed base.
        base1, ext = os.path.splitext(hef_path)
        if self.arch_suffix:
            # e.g. .../obico_part1.hef -> .../obico_part1_8l.hef
            hef1_path = base1 + self.arch_suffix + ext
        else:
            hef1_path = hef_path
        if "part1" in os.path.basename(hef1_path):
            # e.g. obico_part1_8l.hef -> obico_part2_8l.hef
            hef2_path = hef1_path.replace("part1", "part2")
        else:
            # base name without 'part1' (e.g. model-weights) -> _part2 suffix
            hef2_path = os.path.splitext(hef1_path)[0] + "_part2" + ext
        if not os.path.exists(hef2_path):
            raise FileNotFoundError(
                f"Part2 HEF for architecture {arch_name} not found. "
                f"Expected at: {hef2_path}\n"
                f"Please compile both parts: compile_hef.py --hw-arch "
                f"{arch_name.lower()} --part1 ... && compile_hef.py --hw-arch "
                f"{arch_name.lower()} --part2 ...")

        self.hef1 = HEF(hef1_path)
        self.hef2 = HEF(hef2_path)

        # API compat: HailoRT < 4.19 uses VDevice.create_target(), >= 4.19
        # configures the network groups directly on the VDevice.
        target = getattr(self.vdevice, "create_target", None)
        if target is not None:
            target = target()
            self.ng1 = target.configure(self.hef1)[0]
            self.ng2 = target.configure(self.hef2)[0]
        else:
            self.ng1 = self.vdevice.configure(self.hef1)[0]
            self.ng2 = self.vdevice.configure(self.hef2)[0]
        self.ng1_params = self.ng1.create_params()
        self.ng2_params = self.ng2.create_params()

        # Part1 outputs: 199 [1,1024,13,13], 202 [1,64,26,26]
        # NOTE: HEF vstream names (e.g. 'obico_part1/conv21') are NOT the
        # ONNX tensor names ('199'/'202'), so we resolve them by shape.
        self.p1_output_names = [o.name for o in self.hef1.get_output_vstream_infos()]
        # Part2 output: 218 [1,30,13,13]
        self.p2_output_names = [o.name for o in self.hef2.get_output_vstream_infos()]

        # Host-side decode: 218 → boxes + confs
        if decode_onnx_path is None:
            decode_onnx_path = os.path.join(os.path.dirname(hef_path), "decode.onnx")
        self.decode_session = onnxruntime.InferenceSession(
            decode_onnx_path, providers=["CPUExecutionProvider"])

    @staticmethod
    def _detect_arch(device_ids):
        """Return the device architecture name ('HAILO8' / 'HAILO8L').

        Reads the architecture via a control-only Device, which can coexist
        with the inference VDevice (the firmware allows parallel control
        access). DeviceArchitectureTypes is not exposed in the public
        hailo_platform namespace, so we fall back to the integer value
        (HAILO8=1, HAILO8L=2) for robustness.
        """
        if not device_ids:
            return "HAILO8"
        dev = Device(device_id=device_ids[0])
        try:
            arch = dev.control.identify().device_architecture
            if DeviceArchitectureTypes is not None and arch == DeviceArchitectureTypes.HAILO8L:
                return "HAILO8L"
            if int(arch) == 2:
                return "HAILO8L"
            return "HAILO8"
        finally:
            dev.release()

    def detect(self, meta, image, alt_names, thresh=.5, hier_thresh=.5, nms=.45,
               debug=False) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        width = image.shape[1]
        height = image.shape[0]

        # Preprocess (identical to onnx.py: resize 416, RGB, /255)
        resized = cv2.resize(image, (self.input_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        # ---- Part 1: backbone → 199 + 202 ---------------------------------
        p1_inp = self.hef1.get_input_vstream_infos()[0].name
        p1_outs = self._infer(self.ng1, self.ng1_params, self.hef1,
                              {p1_inp: _to_vstream(img_in)},
                              self.p1_output_names, nhwc_out=True)

        # Resolve by shape (robust to vstream naming differences):
        #   26x26 → t202 (source of space-to-depth)
        #   13x13 → t199 (skip/backbone output)
        t202 = next(o for o in p1_outs if o.shape[2] == 26 and o.shape[3] == 26)
        t199 = next(o for o in p1_outs if o.shape[2] == 13 and o.shape[3] == 13)

        # ---- Host bridge: S2D + Concat → 214 [1,1280,13,13] ----------------
        # The original Concat(s2d(202), 199) → 214 is done here on host;
        # part2 receives the already-concatenated 214 as its single input.
        t214 = _unshuffle_concat(t202, t199)

        # ---- Part 2: detection → 218 [1,30,13,13] --------------------------
        p2_input_names = [i.name for i in self.hef2.get_input_vstream_infos()]
        p2_feed = {name: _to_vstream(t214) for name in p2_input_names}
        p2_outs = self._infer(self.ng2, self.ng2_params, self.hef2,
                              p2_feed, self.p2_output_names, nhwc_out=True)

        t218 = p2_outs[0]  # [1,30,13,13]

        # ---- Host decode: 218 → boxes + confs ------------------------------
        feed = {self.decode_session.get_inputs()[0].name: t218}
        boxes, confs = self.decode_session.run(None, feed)

        detections = post_processing([boxes, confs], width, height, thresh, nms, meta.names)
        return detections[0]

    def _infer(self, network_group, ng_params, hef, inputs, output_names, nhwc_out=False):
        input_vstreams_params = InputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(
            network_group, quantized=False, format_type=FormatType.FLOAT32)

        if _HAS_INFER_VSTREAMS:
            # HailoRT >= 4.19: InferVStreams context manager; the network
            # group must be activated explicitly during inference.
            with InferVStreams(network_group, input_vstreams_params,
                               output_vstreams_params) as ivs:
                with network_group.activate(ng_params):
                    results = ivs.infer(inputs)
            results = [results[n] for n in output_names]
        else:
            # HailoRT < 4.19: InputVStreams/OutputVStreams managers
            from hailo_platform import InputVStreams, OutputVStreams
            with InputVStreams(network_group, input_vstreams_params) as in_streams, \
                 OutputVStreams(network_group, output_vstreams_params) as out_streams:
                for k, v in inputs.items():
                    in_streams.get_by_name(k).send(v)
                results = [out_streams.get_by_name(n).recv() for n in output_names]
        # HailoRT returns NHWC [N,H,W,C]; the rest of the pipeline (S2D,
        # decode.onnx, post_processing) expects NCHW [N,C,H,W].
        if nhwc_out:
            results = [_from_vstream(r) for r in results]
        return results

    def __del__(self):
        try:
            self.vdevice.release()
        except Exception:
            pass


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(output, width, height, conf_thresh, nms_thresh, names):
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    box_x1x1x2y2_to_xcycwh_scaled = lambda b: \
        (
            float(0.5 * width * (b[0] + b[2])),
            float(0.5 * height * (b[1] + b[3])),
            float(width * (b[2] - b[0])),
            float(width * (b[3] - b[1]))
         )
    dets_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        detections = [(names[b[6]], float(b[4]), box_x1x1x2y2_to_xcycwh_scaled((b[0], b[1], b[2], b[3]))) for b in bboxes]
        dets_batch.append(detections)


    return dets_batch
