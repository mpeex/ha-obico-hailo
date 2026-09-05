# spaghetti-rpi5-aikit — Obico ML Hailo Home Assistant Addon

Home Assistant **spaghetti / print-failure detection** addon for a **Raspberry Pi 5**
with a **Hailo-8 / Hailo-8L** AI accelerator (Raspberry Pi AI HAT+).

It runs a port of the **Obico YOLOv5** model (1 class: `failure`) on the Hailo
device using the two compiled HEFs (`obico_part1.hef` / `obico_part2.hef`) plus
the host-side `decode.onnx` head.

The addon is **self-contained**: it discovers the user's `camera.*` entities and
lets you pick one from a built-in web UI, then polls that camera's snapshot,
runs inference and publishes the result back to HA (`binary_sensor.obico_failure`
and `sensor.obico_confidence`). No companion integration is required.

## Architecture

```
camera (HA entity)
   │  snapshot via HA REST API (Supervisor token)
   ▼
HA addon (Flask :3333)
   └── camera worker (daemon thread) → detect → publish to HA
        │
        ▼
lib/hailo.py  (HailoNet)
   ├── obico_part1.hef   backbone+neck  → 199 [1,1024,13,13] + 202 [1,64,26,26]
   ├── host bridge       space-to-depth + concat → 214 [1,1280,13,13]
   ├── obico_part2.hef   detection tail → 218 [1,30,13,13]
   └── decode.onnx       host-side YOLOv5 head → boxes + confs → NMS
```

Hailo-8 vs Hailo-8L are distinguished at runtime (`_detect_arch`) and the
corresponding HEF set is loaded automatically:

| Device      | Files loaded                              |
|-------------|-------------------------------------------|
| Hailo-8     | `obico_part1.hef`  + `obico_part2.hef`    |
| Hailo-8L    | `obico_part1_8l.hef` + `obico_part2_8l.hef` |

Both HEF sets ship in `app/model/` and are committed via Git LFS (`*.hef`).

## Layout

| Path | Purpose |
|------|---------|
| `Dockerfile` | single self-contained addon image (HailoRT + onnxruntime + OpenCV + s6-overlay + Flask app) |
| `config.yaml` | addon descriptor; `devices: /dev/hailo0`, options `max_fps`, `camera_entity`, `interval`, `threshold` |
| `app/lib/hailo.py`, `meta.py` | Hailo inference runtime (from obico-server) |
| `app/lib/ha.py` | HA/Supervisor connectivity (token, cameras, snapshot, state publish) |
| `app/lib/config_store.py` | persist camera-selection config to `/data` |
| `app/lib/detection_model.py` | `load_net` → `HailoNet` (`.hef` only) |
| `app/model/` | `obico_part*.hef`, `decode.onnx`, `model.meta`, `names` |
| `app/server.py` | Flask app: camera-selection web UI + background detection worker |
| `rootfs/` | s6 service lifecycle (exports config options as env vars) |
| `hailo_assets/` | optional offline HailoRT binaries (`hailort_<V>_arm64.deb` + wheel) |
| `hailo_assets/licenses/` | HailoRT redistribution licenses (MIT + LGPL-2.1), shipped inside the image |

## Build

The addon is built with a **single Dockerfile** that starts from a stock
`python:3.11-slim-bookworm` and provisions HailoRT, ONNX Runtime and OpenCV
headless itself. No separate base image or registry push is needed.

HailoRT provisioning reads `hailo_assets/`:

- if `hailort_<V>_arm64.deb` and the matching
  `hailort-<V>-cp311-cp311-linux_aarch64.whl` are present → **offline** build;
- if they are absent → the HailoRT binaries are **downloaded** from
  `https://dev-public.hailo.ai/2025_04/` at build time (online), then cached in
  `hailo_assets/` for next time.

HailoRT is redistributed inside the image under its own terms; the required
license texts (MIT for libhailort/pyhailort/hailortcli, LGPL-2.1-or-later for
the hailonet GStreamer plugin) live in `hailo_assets/licenses/` and are copied
to `/usr/share/licenses/hailort/` in the final image.

```bash
# Direct build (any arm64 host):
docker build --build-arg BUILD_ARCH=aarch64 -t obico-ml-hailo .
```

## Build & install (local addon on HAOS RPi5)

Prerequisites on the host (RPi5):

- the Hailo device is present (`/dev/hailo0`) and HailoRT is installed;
- the `hailort_service` daemon is running and listening on
  `/tmp/hailort_uds.sock` (the addon mounts it; otherwise the device RPC fails).

To install as a local addon:

1. Copy this repo to a folder reachable by HAOS and add it as a **local addon**
   (or add the repository and install `Obico ML (Hailo)`).
2. In the addon configuration set `max_fps` (default `1`); increase it to at
   most `2` if you want more responsive (un-throttled) detection while sharing
   the Hailo with other processes.

## Camera selection (self-contained)

On HAOS the addon talks to Home Assistant through the **Supervisor API** (the
`SUPERVISOR_TOKEN` is injected automatically and exchanged for a HA core access
token — no manual token needed). It enumerates your `camera.*` entities over
the HA REST API, pulls the selected camera's snapshot, runs detection, and
publishes the outcome back to HA.

- **Web UI** (default port `3333`, or the addon's *Open Web UI* button): pick a
  camera from the dropdown, set interval & threshold, and hit *Start detection*.
  The choice is saved to `/data` and survives restarts.
- **Config options**: `camera_entity` (entity id, e.g. `camera.front_door`),
  `interval` (seconds, default `10`), `threshold` (default `0.2`). These map to
  the same fields as the web UI.
- **Published entities**:
  - `binary_sensor.obico_failure` → `on`/`off` (spaghetti detected), attributes
    carry `detections`, `avg_confidence`, `camera`.
  - `sensor.obico_confidence` → average confidence (%), unit `%`.

The background worker runs in a daemon thread; its current state is available at
`GET /status`.

## Endpoints (port 3333)

- `GET /` → camera-selection web UI
- `GET /api/cameras` → `{"ok": true, "cameras": [...camera.* ids]}`
- `GET/POST /api/config` → read/update `camera_entity`, `interval`, `threshold`
- `POST /api/start`, `POST /api/stop` → start/stop the detection worker
- `GET /status` → config, HA connectivity and last detection state
- `GET /hc/` → `ok` (health)

`MAX_FPS` (the `max_fps` addon option) is a **last-frame-wins** throttle on the
shared Hailo: surplus requests reuse the previous frame's detections instead of
running inference, leaving the accelerator free for other consumers.

## Publish to GHCR

Build and push the image manually (e.g. from the RPi5):

```bash
# Build on an arm64 host
docker build --build-arg BUILD_ARCH=aarch64 -t obico-ml-hailo:4.21 .

# Optionally use the offline HailoRT assets instead of downloading at build time
cp hailo_assets/hailort_4.21.0_arm64.deb hailo_assets/hailort-4.21.0-cp311-cp311-linux_aarch64.whl .
docker build --build-arg BUILD_ARCH=aarch64 -t obico-ml-hailo:4.21 .

# Tag and push
docker tag obico-ml-hailo:4.21 ghcr.io/mpeex/obico-ha-app:4.21
docker tag obico-ml-hailo:4.21 ghcr.io/mpeex/obico-ha-app:latest
docker push ghcr.io/mpeex/obico-ha-app:4.21
docker push ghcr.io/mpeex/obico-ha-app:latest
```

(Requires `docker login ghcr.io` with a token that has `write:packages`.) The
tag tracks the HailoRT version (`4.21`); `config.yaml` is left untouched.

## License

AGPL-3.0. Based on https://github.com/TheSpaghettiDetective/obico-server
(Hailo runtime) and https://github.com/nobodyguy/obico_ml_ha_addon (addon structure).

The redistributed HailoRT binaries keep their own licenses (MIT and
LGPL-2.1-or-later); the texts ship in `hailo_assets/licenses/` and inside the
image under `/usr/share/licenses/hailort/`.
