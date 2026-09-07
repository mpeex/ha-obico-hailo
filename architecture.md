# Architecture

This document details the internals of the **Obico ML (Hailo)** addon: the
inference pipeline, the REST API, and how the addon is built and published. For
an executive summary, install and usage, see [`README.md`](README.md).

## Model pipeline

```
camera (HA entity)
   │  snapshot via HA REST API (Supervisor token)
   ▼
HA addon (Flask :3333)
   └── camera worker (daemon thread) → detect → annotated stream + /status
        │
        ▼
lib/hailo.py  (HailoNet)
   ├── obico_part1.hef   backbone+neck  → 199 [1,1024,13,13] + 202 [1,64,26,26]
   ├── host bridge       space-to-depth + concat → 214 [1,1280,13,13]
   ├── obico_part2.hef   detection tail → 218 [1,30,13,13]
   └── decode.onnx       host-side YOLOv5 head → boxes + confs → NMS
```

Hailo-8 vs Hailo-8L are distinguished at runtime (`_detect_arch` in
`app/lib/hailo.py`) and the matching HEF set is loaded automatically:

| Device      | Files loaded                              |
|-------------|-------------------------------------------|
| Hailo-8     | `obico_part1.hef`  + `obico_part2.hef`    |
| Hailo-8L    | `obico_part1_8l.hef` + `obico_part2_8l.hef` |

Both HEF sets ship in `app/model/` and are committed via Git LFS (`*.hef`,
see `.gitattributes`). The host-side `decode.onnx` YOLOv5 head runs on
ONNX Runtime.

## Camera snapshots

On HA OS the addon talks to Home Assistant through the **Supervisor API** (the
`SUPERVISOR_TOKEN` is injected automatically and exchanged for a HA core access
token — no manual token needed). It pulls the configured camera's snapshot over
the HA REST API, runs detection, and serves the annotated frame + state to the
companion integration. Snapshots prefer the camera `entity_picture` (camera
proxy) with retries and backoff, falling back to the core REST
`/api/camera/{id}/snapshot` endpoint.

## Configuration & the companion integration

The addon has **no options**: `config.yaml` declares `options: {}` and
`schema: {}`. It starts idle and is driven entirely by the companion
integration (`custom_components/obico_ml_hailo/`):

- The integration owns `url`, `camera_entity` (native HA `entity` selector on
  the `camera.*` domain), `detection_interval` (int, min `1`) and `threshold`
  in its options flow.
- Every time the integration is (re)loaded and a camera is configured, it
  pushes the worker via `POST /api/start` (idempotent) — this re-enables
  detection after HA restarts and applies option changes to a running worker.
- `switch.obico_ml_detection` stops/starts the worker within a session.

Registered entities:

- `binary_sensor.obico_failure` (device class `problem`) → `on` when a failure
  is detected, with `detections` / `avg_confidence` / `camera` attributes
- `sensor.obico_confidence` → average confidence (%), unit `%`
- `camera.obico_ml_detection_camera` → the annotated frame from the addon
- `switch.obico_ml_detection` → start/stop the addon's detection worker

Legacy config entries are migrated automatically: the old `interval` option
becomes `detection_interval` and the `camera.your_entity` placeholder is
cleared so the camera is re-selected from the dropdown.

## REST API (port 3333)

The addon API listens on `3333`; the integration reaches it through the
Supervisor network gateway (`http://172.30.32.1:3333`) — the addon's own
hostname is **not** resolvable from HA core. The port is also published on the
host, so keep the network trusted or firewall it.

- `GET/POST /api/config` → read/update `detection_interval`, `threshold`
- `POST /api/start`, `POST /api/stop` → start/stop the detection worker
  (`/api/start` also receives `camera_entity`, `detection_interval` and
  `threshold` and applies them to the worker)
- `GET /status` → config and last detection state
- `GET /last_image` → latest annotated frame (JPEG)
- `GET /hc/` → `ok` (health)

`detection_interval` (seconds, minimum `1`) is the single detection knob: every
`detection_interval` seconds the addon captures a frame, runs one inference on
the Hailo and refreshes the annotated stream. A minimum of 1 s guarantees the
device is shared fairly with other processes on the same Hailo.

## Repository layout

| Path | Purpose |
|------|---------|
| `Dockerfile` | single self-contained addon image (HailoRT + onnxruntime + OpenCV + s6-overlay + Flask app) |
| `config.yaml` | addon descriptor; `devices: /dev/hailo0`, no options |
| `build.yaml` | builder base-image mapping (`python:3.11-slim-bookworm` per arch) |
| `repository.yaml` | HA addon repository descriptor |
| `.github/workflows/ci.yaml` | publish images to GHCR via `home-assistant/builder` |
| `app/lib/hailo.py`, `meta.py` | Hailo inference runtime (from obico-server) |
| `app/lib/ha.py` | HA/Supervisor connectivity (token, camera snapshot) |
| `app/lib/config_store.py` | persists worker config (`detection_interval`, `threshold`) to `/data` |
| `app/lib/detection_model.py` | `load_net` → `HailoNet` (`.hef` only) |
| `app/model/` | `obico_part*.hef`, `decode.onnx`, `model.meta`, `names` (git LFS) |
| `app/server.py` | Flask app: detection worker + REST API |
| `rootfs/` | s6 service lifecycle (the `obico` service runs gunicorn) |
| `custom_components/obico_ml_hailo/` | companion Home Assistant integration |
| `hailo_assets/` | optional offline HailoRT binaries (gitignored) + license texts |
| `hailo_assets/licenses/` | HailoRT redistribution licenses (MIT + LGPL-2.1), shipped inside the image |

## Build

The addon is built with a **single self-contained Dockerfile** starting from a
stock `python:3.11-slim-bookworm`; it provisions HailoRT, ONNX Runtime and
OpenCV headless itself — no separate prebuilt base image needed.

HailoRT provisioning reads `hailo_assets/`. Expected file names depend on
`BUILD_ARCH`:

| `BUILD_ARCH` | `.deb` | `.whl` |
|--------------|--------|--------|
| `aarch64` | `hailort_<V>_arm64.deb` | `hailort-<V>-cp311-cp311-linux_aarch64.whl` |
| `amd64` | `hailort_<V>_amd64.deb` | `hailort-<V>-cp311-cp311-linux_x86_64.whl` |

- if the matching files are present → **offline** build;
- if they are absent → the HailoRT binaries are **downloaded** from
  `https://dev-public.hailo.ai/2025_04/` at build time (online), then cached in
  `hailo_assets/` for next time.

HailoRT is redistributed inside the image under its own terms; the required
license texts (MIT for libhailort/pyhailort/hailortcli, LGPL-2.1-or-later for
the hailonet GStreamer plugin) live in `hailo_assets/licenses/` and are copied
to `/usr/share/licenses/hailort/` in the final image.

Build manually on any host:

```bash
docker build --build-arg BUILD_ARCH=aarch64 -t obico-ml-hailo .   # arm64 host
docker build --build-arg BUILD_ARCH=amd64 -t obico-ml-hailo .     # x86_64 host
```

## Install as a local addon (HA OS)

Prerequisites on the host:

- the Hailo device is present (`/dev/hailo0`) and HailoRT is installed;
- the `hailort_service` daemon is running and listening on
  `/tmp/hailort_uds.sock` (the addon mounts it; otherwise the device RPC fails).

Then copy this repo to a folder reachable by HAOS and add it as a **local
addon** (or add the repository and install *Obico ML (Hailo)*), install the
companion integration, and configure it via the options flow as described in
the README.

## Publish to GHCR

The public build path is the GitHub workflow `.github/workflows/ci.yaml`, which
uses `home-assistant/builder` and pushes per-arch images to
`ghcr.io/mpeex/obico_ml_hailo_addon-<arch>:<version>` on push to `main` / on
release.

Local cross-compile (docker buildx) for a quick iteration is also possible:

```bash
docker login ghcr.io
docker buildx create --name obico-builder --use --bootstrap
docker buildx build \
  --platform linux/arm64 \
  --build-arg BUILD_ARCH=aarch64 \
  --tag ghcr.io/mpeex/obico_ml_hailo_addon-aarch64:0.16 \
  --push .
docker buildx build \
  --platform linux/amd64 \
  --build-arg BUILD_ARCH=amd64 \
  --tag ghcr.io/mpeex/obico_ml_hailo_addon-amd64:0.16 \
  --push .
```

The image tag (<version>, from `config.yaml`) is the Docker tag Supervisor
pulls; the HailoRT version stays pinned inside the Dockerfile (`4.21.0`).

For a multi-arch index, combine the per-arch tags into one manifest
(requires a token with `write:packages`):

```bash
docker manifest create ghcr.io/mpeex/obico_ml_hailo_addon:0.16 \
  ghcr.io/mpeex/obico_ml_hailo_addon-aarch64:0.16 \
  ghcr.io/mpeex/obico_ml_hailo_addon-amd64:0.16
docker manifest push ghcr.io/mpeex/obico_ml_hailo_addon:0.16
```

A developer-local `build-push.sh` wrapper used during development is
**gitignored** and not part of the published repository; the commands above are
its equivalent.