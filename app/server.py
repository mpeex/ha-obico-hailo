#!/usr/bin/env python

import time
import flask
from flask_compress import Compress
from flask import request, jsonify, render_template_string
from os import path, environ
import cv2
import numpy as np
import logging
import threading
import base64

from lib.detection_model import load_net, detect
from lib import ha as ha_lib
from lib.config_store import load_config, save_config


# Optional last-frame-wins frame-rate limiter. When MAX_FPS is set (e.g. "1"),
# detection is throttled to at most MAX_FPS inferences per second; requests that
# arrive within an already-consumed time slot immediately reuse the previous
# detection list instead of running the Hailo device (which is thereby left free
# for other consumers). Unset by default => no limiting. Set via the add-on
# configuration option `max_fps`.
MAX_FPS = environ.get('MAX_FPS')


class FrameRateLimiter:
    """Last-frame-wins throttle: runs at most MAX_FPS inferences/sec, reuses the
    last detection result for requests that exceed the budget."""

    def __init__(self, max_fps=None):
        self.min_interval = 1.0 / float(max_fps) if max_fps else 0.0
        self._last_ts = None
        self._last_detections = []

    def run(self, fn):
        """Execute fn only if the time budget allows; otherwise return the last
        cached frame's detections without consuming Hailo."""
        now = time.monotonic()
        if self.min_interval > 0.0:
            if self._last_ts is not None:
                elapsed = now - self._last_ts
                if elapsed < self.min_interval:
                    # Frame is throttled: reuse the previous result (drop, no infer).
                    return self._last_detections
            # First frame, or budget elapsed: always run and set the clock.
            self._last_ts = now
        result = fn()
        self._last_detections = result
        return result

app = flask.Flask(__name__)
Compress(app)

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-contained camera selection.
#
# The addon can discover the user's camera.* entities via the HA core REST
# API (see lib/ha.py), pull the selected camera's snapshot, run detection and
# publish the outcome back to HA entities. This makes the addon independent of
# a companion integration. The worker runs in a daemon thread; selecting a
# camera from the web UI (or a config option) starts it.
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
    error = bool(detections)
    try:
        ha_lib.set_state(
            "binary_sensor.obico_failure",
            "on" if error else "off",
            {
                "friendly_name": "Obico ML failure",
                "detections": detections,
                "avg_confidence": avg,
                "camera": load_config().get("camera_entity", ""),
                "icon": "mdi:alert" if error else "mdi:check",
            },
        )
        ha_lib.set_state(
            "sensor.obico_confidence",
            str(avg),
            {
                "friendly_name": "Obico ML confidence",
                "unit_of_measurement": "%",
                "camera": load_config().get("camera_entity", ""),
            },
        )
    except Exception as err:
        logger.error("Failed to publish state to HA: %s", err)


def _camera_worker():
    while not _stop_event.is_set():
        cfg = load_config()
        camera = cfg.get("camera_entity", "")
        interval = int(cfg.get("interval", 10))
        threshold = float(cfg.get("threshold", 0.2))
        if camera:
            with _cam_lock:
                cam_state["running"] = True
            try:
                image_bytes = ha_lib.fetch_camera_image(camera)
                img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
                detections = _rate_limiter.run(
                    lambda: detect(net_main, img, thresh=threshold)
                )
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

# One shared limiter per worker. With gunicorn --workers 1 this is a single
# process-wide throttle; the budget is global to the Hailo device.
_rate_limiter = FrameRateLimiter(MAX_FPS)


def _auto_start_worker():
    """Start the camera worker on boot when auto_start is enabled and a
    camera is configured, so detection resumes without clicking Start in the
    web UI."""
    cfg = load_config()
    if cfg.get("auto_start") and cfg.get("camera_entity"):
        _stop_event.clear()
        _ensure_camera_worker_started()


def draw_bounding_boxes(image, detections):
    for detection in detections:
        label, confidence, bbox = detection
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 0, 255)  # Red color for bounding box
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

@app.route('/', methods=['GET'])
def index():
    cfg = load_config()
    cameras = []
    ha_ok = ha_lib.ha_available()
    if ha_ok:
        try:
            cameras = ha_lib.list_cameras()
        except Exception as err:
            logger.error("Failed to list cameras: %s", err)
    return render_template_string(_UI_PAGE, config=cfg, cameras=cameras, ha_ok=ha_ok)


@app.route('/api/cameras', methods=['GET'])
def api_cameras():
    if not ha_lib.ha_available():
        return jsonify({"ok": False, "error": "Not running under HAOS (no Supervisor token)"}), 503
    try:
        return jsonify({"ok": True, "cameras": ha_lib.list_cameras()})
    except Exception as err:
        logger.error("Failed to list cameras: %s", err)
        return jsonify({"ok": False, "error": str(err)}), 500


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
        "ha_connected": ha_lib.ha_available(),
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


_UI_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Obico ML (Hailo) — Camera</title>
<style>
  body{font-family:system-ui,sans-serif;background:#f4f6f8;margin:0;padding:24px;color:#1c2733}
  .card{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:24px;max-width:560px;margin:0 auto}
  h1{font-size:20px;margin-top:0}
  label{display:block;font-weight:600;margin:14px 0 6px;font-size:14px}
  select,input{width:100%;box-sizing:border-box;padding:9px;border:1px solid #cbd5e0;border-radius:6px;font-size:14px}
  .btn{display:inline-block;border:0;border-radius:6px;padding:11px 18px;font-size:14px;cursor:pointer;color:#fff;margin-top:16px}
  .btn-start{background:#2f8132}.btn-stop{background:#c53030}.btn:disabled{opacity:.5;cursor:not-allowed}
  .msg{margin-top:16px;padding:12px;border-radius:6px;font-size:14px}
  .ok{background:#e6ffed;color:#22543d}.err{background:#ffebeb;color:#742a2a}
  .muted{color:#718096;font-size:12px;margin-top:8px}
</style>
</head>
<body>
<div class="card">
  <h1>Obico ML (Hailo) — Camera selection</h1>
  {% if not ha_ok %}
    <div class="msg err">Home Assistant (Supervisor) is not reachable from this addon.
      Install it on HAOS so it can list and snapshot cameras automatically.</div>
  {% endif %}
  <label for="camera">Camera entity</label>
  <select id="camera">
    <option value="">— select a camera —</option>
    {% for c in cameras %}
      <option value="{{ c }}" {% if config.camera_entity == c %}selected{% endif %}>{{ c }}</option>
    {% endfor %}
  </select>
  <div class="muted">Cameras are enumerated from Home Assistant.{% if not cameras %}{% if ha_ok %} No camera.* entities found.{% endif %}{% endif %}</div>

  <label for="interval">Interval (seconds)</label>
  <input id="interval" type="number" min="1" value="{{ config.interval }}" />

  <label for="threshold">Threshold (0–1)</label>
  <input id="threshold" type="number" step="0.01" min="0" max="1" value="{{ config.threshold }}" />

  <label><input id="auto_start" type="checkbox" {% if config.auto_start %}checked{% endif %} /> Auto-start detection on boot</label>

  <button class="btn btn-start" id="start">Start detection</button>
  <button class="btn btn-stop" id="stop">Stop</button>

  <div id="msg"></div>
</div>
<script>
const msg = el => { const d = document.getElementById('msg'); d.className='msg '+(el.ok?'ok':'err');
  d.innerHTML = el.ok ? (el.detail||'Saved.') : ('Error: '+(el.error||'unknown')); };
function cfg(){return {camera_entity:document.getElementById('camera').value,
  interval:parseInt(document.getElementById('interval').value)||10,
  threshold:parseFloat(document.getElementById('threshold').value)||0.2,
  auto_start:document.getElementById('auto_start').checked};}
document.getElementById('start').onclick = async () => {
  const r = await fetch('api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg())});
  msg(await r.json());
};
document.getElementById('stop').onclick = async () => {
  const r = await fetch('api/stop',{method:'POST'});
  msg(await r.json());
};
</script>
</body>
</html>
"""

# Auto-start the camera worker at server boot (gunicorn imports this module).
_auto_start_worker()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
