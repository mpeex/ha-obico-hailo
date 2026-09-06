#!/usr/bin/env python

import time
import flask
from flask_compress import Compress
from flask import request, jsonify
from os import path, environ
import cv2
import numpy as np
import logging
import threading
import base64

from lib.detection_model import load_net, detect
from lib import ha as ha_lib
from lib.config_store import load_config, save_config


app = flask.Flask(__name__)
Compress(app)

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection worker.
#
# The addon pulls the configured camera's snapshot from HA (see lib/ha.py),
# runs detection on the Hailo and exposes the annotated frame + result through
# its REST API. The companion integration drives it and mirrors the outcome as
# registered HA entities.
# ---------------------------------------------------------------------------
cam_state = {
    "running": False,
    "last_error": None,
    "last_detections": [],
    "last_avg_confidence": 0.0,
    "last_update": None,
    "last_image": None,
}
_cam_lock = threading.Lock()
_stop_event = threading.Event()


def _publish_result(detections):
    avg = 0.0
    if detections:
        avg = round((sum(d[1] for d in detections) / len(detections)) * 100, 2)
    with _cam_lock:
        cam_state["last_detections"] = detections
        cam_state["last_avg_confidence"] = avg
        cam_state["last_update"] = time.time()


def _camera_worker():
    while not _stop_event.is_set():
        cfg = load_config()
        camera = cfg.get("camera_entity", "")
        interval = int(cfg.get("detection_interval", 1))
        threshold = float(cfg.get("threshold", 0.2))
        if camera:
            with _cam_lock:
                cam_state["running"] = True
            try:
                image_bytes = ha_lib.fetch_camera_image(camera)
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
                detections = detect(net_main, img, thresh=threshold)
                img_with_boxes = draw_bounding_boxes(img, detections)
                _, buffer = cv2.imencode('.jpg', img_with_boxes)
                with _cam_lock:
                    cam_state["last_image"] = buffer.tobytes()
                _publish_result(detections)
                with _cam_lock:
                    cam_state["last_error"] = None
            except Exception as err:
                logger.error("Camera worker error: %s", err)
                with _cam_lock:
                    cam_state["last_error"] = str(err)
        else:
            with _cam_lock:
                cam_state["running"] = False
        # Sleep in small steps so /stop can interrupt promptly.
        for _ in range(interval):
            if _stop_event.wait(1.0):
                break


def _ensure_camera_worker_started():
    if not getattr(_ensure_camera_worker_started, "started", False):
        _ensure_camera_worker_started.started = True
        threading.Thread(target=_camera_worker, daemon=True, name="camera-worker").start()

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))


def _auto_start_worker():
    """Start the camera worker on boot when auto_start is enabled and a
    camera is configured, so detection resumes without the integration having
    to toggle the switch after every restart."""
    cfg = load_config()
    if cfg.get("auto_start") and cfg.get("camera_entity"):
        _stop_event.clear()
        _ensure_camera_worker_started()


def draw_bounding_boxes(image, detections):
    """Overlay the detection state on `image`.

    Always draws a status banner (top-left): a green 'NO FAILURE' when no
    detections, or a red 'FAILURE xx.x%' with the average confidence when the
    model fires. Any detected object is additionally boxed in red with its own
    label/confidence. This keeps the exposed camera stream self-explanatory.
    """
    avg = 0.0
    if detections:
        avg = round((sum(d[1] for d in detections) / len(detections)) * 100, 1)
    if detections:
        banner = f"FAILURE ({avg:.1f}%)"
        color = (0, 0, 255)  # red
    else:
        banner = "NO FAILURE"
        color = (0, 200, 0)  # green

    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    bx, by = 10, 10
    cv2.rectangle(
        image, (bx, by), (bx + bw + 20, by + bh + 20), (0, 0, 0), -1
    )
    cv2.putText(
        image, banner, (bx + 10, by + bh + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2,
    )

    for detection in detections:
        label, confidence, bbox = detection
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    return image


@app.route('/detect/', methods=['POST'])
def failure_detect():
    data = request.get_json()

    img_base64 = data.get("img", None)
    if img_base64 is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_bytes = base64.b64decode(img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        threshold = float(data.get("threshold", 0.2))

        detections = detect(net_main, img, thresh=threshold)

        img_with_boxes = draw_bounding_boxes(img, detections)

        _, buffer = cv2.imencode('.jpg', img_with_boxes)
        img_with_boxes_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "detections": detections,
            "image_with_detections": img_with_boxes_base64
        }), 200

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image - {str(e)}"}), 500


@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        cfg = save_config(data)
        return jsonify({"ok": True, "config": cfg})
    return jsonify({"ok": True, "config": load_config()})


@app.route('/status', methods=['GET'])
def api_status():
    with _cam_lock:
        snap = {k: v for k, v in cam_state.items() if k != "last_image"}
    return jsonify({
        "ok": True,
        "config": load_config(),
        "state": snap,
    })


@app.route('/last_image', methods=['GET'])
def api_last_image():
    with _cam_lock:
        img = cam_state.get("last_image")
    if not img:
        return jsonify({"error": "No image available yet"}), 404
    resp = flask.Response(img, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.get_json(silent=True) or {}
    cfg = save_config(data)
    _stop_event.clear()
    _ensure_camera_worker_started()
    return jsonify({"ok": True, "config": cfg})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    _stop_event.set()
    with _cam_lock:
        cam_state["running"] = False
    return jsonify({"ok": True})


# Auto-start the camera worker at server boot (gunicorn imports this module).
_auto_start_worker()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
