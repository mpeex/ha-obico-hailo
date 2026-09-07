# Obico ML Hailo Home Assistant Addon

[![](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fmpeex%2Fspaghetti-rpi5-aikit)

Home Assistant **spaghetti / print-failure detection** addon for a **Raspberry Pi 5**
with a **Hailo-8 / Hailo-8L** AI accelerator (Raspberry Pi AI HAT+).

It should work also on x64 architectures, altough it is not tested.

It runs a port of the **Obico YOLOv5** model (1 class: `failure`) on the Hailo
device using the two compiled HEFs (`obico_part1.hef` / `obico_part2.hef`) plus
the host-side `decode.onnx` head.

The bundle comes in two parts:

- an **addon** that runs the Obico model on the Hailo, pulls the configured
  camera's snapshot from HA and exposes an annotated stream + detection state
  over a small REST API (port `3333`);
- a **companion integration** (`custom_components/obico_ml_hailo`) that mirrors
  the addon's state into registered HA entities (`binary_sensor`,
  `sensor`, `camera`, `switch`). The camera is picked from a dropdown in the
  integration's options flow (HA `entity` selector, domain `camera.*`).

## Architecture

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
| `config.yaml` | addon descriptor; `devices: /dev/hailo0`, no options (the companion integration drives camera/interval/threshold) |
| `build.yaml` | builder base-image mapping (`python:3.11-slim-bookworm` per arch) |
| `repository.yaml` | HA addon repository descriptor (name, url, maintainer) |
| `build-push.sh` | cross-compile (docker buildx) + push to `ghcr.io/mpeex/obico_ml_hailo_addon-<arch>:0.16` |
| `app/lib/hailo.py`, `meta.py` | Hailo inference runtime (from obico-server) |
| `app/lib/ha.py` | HA/Supervisor connectivity (token, camera snapshot) |
| `app/lib/config_store.py` | persist camera-selection config to `/data` |
| `app/lib/detection_model.py` | `load_net` → `HailoNet` (`.hef` only) |
| `app/model/` | `obico_part*.hef`, `decode.onnx`, `model.meta`, `names` |
| `app/server.py` | Flask app: detection worker + REST API |
| `rootfs/` | s6 service lifecycle (exports config options as env vars) |
| `hailo_assets/` | optional offline HailoRT binaries (`hailort_<V>_<arch>.deb` + wheel, arch derived from `BUILD_ARCH`) |
| `hailo_assets/licenses/` | HailoRT redistribution licenses (MIT + LGPL-2.1), shipped inside the image |

## Build

The addon is built with a **single Dockerfile** that starts from a stock
`python:3.11-slim-bookworm` and provisions HailoRT, ONNX Runtime and OpenCV
headless itself. No separate base image or registry push is needed.

HailoRT provisioning reads `hailo_assets/`. The expected file names depend on
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

```bash
# Direct build (any arm64 host):
docker build --build-arg BUILD_ARCH=aarch64 -t obico-ml-hailo .

# Direct build on x86_64:
docker build --build-arg BUILD_ARCH=amd64 -t obico-ml-hailo .
```

## Build & install (local addon on HAOS RPi5)

Prerequisites on the host (RPi5):

- the Hailo device is present (`/dev/hailo0`) and HailoRT is installed;
- the `hailort_service` daemon is running and listening on
  `/tmp/hailort_uds.sock` (the addon mounts it; otherwise the device RPC fails).

To install as a local addon:

1. Copy this repo to a folder reachable by HAOS and add it as a **local addon**
   (or add the repository and install `Obico ML (Hailo)`).
2. Install the **companion integration** (below): it is the single place where
   the camera, detection interval and threshold are configured, and it drives
   the addon through its REST API.

## Camera selection

On HAOS the addon talks to Home Assistant through the **Supervisor API** (the
`SUPERVISOR_TOKEN` is injected automatically and exchanged for a HA core access
token — no manual token needed). It pulls the configured camera's snapshot over
the HA REST API, runs detection, and serves the annotated frame + state to the
companion integration.

The camera is chosen from a dropdown in the integration's options flow (an HA
`entity` selector filtered on the `camera.*` domain, so no custom UI is needed).
The addon itself has **no options**: it starts idle and detection only runs when
the integration pushes a camera via `POST /api/start`.

## Companion integration (registered entities)

To get the detection result as **real HA entities** (registered under a device,
with unique ids and friendly names), install the **`Obico ML (Hailo)`
integration** shipped in `custom_components/obico_ml_hailo/`:

1. Copy `custom_components/obico_ml_hailo/` into
   `config/custom_components/obico_ml_hailo/` on your HA instance
   (e.g. with the Samba share or the SSH add-on).
2. Restart Home Assistant (Settings → System → Restart).
3. Settings → Devices & Services → **Add Integration** → search *Obico ML (Hailo)*.
4. Configure:
   - **URL** → `http://172.30.32.1:3333` (the addon's API port, published on the
     Supervisor network gateway so HA core reaches it without DNS; the addon's
     own hostname is not resolvable from HA core).
   - **Camera entity** → pick from the dropdown; only `camera.*` entities are
     listed.
   - **Detection interval** (min `1`) / **Threshold** → mirror the former
     addon options (`1` / `0.2`). When a camera is configured, detection
     (re)starts automatically on every integration load.

The integration mirrors the addon's camera worker — it does **not** touch the
camera or the Hailo itself (the addon's own worker does the frame capture and
inference). The integration (re)starts detection automatically on every load
whenever a camera is configured, and the switch below stops/starts the worker
within a session. It exposes:

- `binary_sensor.obico_failure` (device class `problem`) → `on` when the model
  finds a failure, with `detections` / `avg_confidence` / `camera` attributes
- `sensor.obico_confidence` → average confidence (%), unit `%`
- `camera.obico_ml_detection_camera` → the annotated frame from the addon
- `switch.obico_ml_detection` → start/stop the addon's detection worker

## Endpoints (port 3333)

The addon API listens on `3333`; the integration talks to it over the
Supervisor network (`http://172.30.32.1:3333`):

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

The addon has no options and stays idle until the integration starts it. The
integration enforces the same minimum (`1`) and always pushes camera, interval,
threshold and start/stop through the addon API.

## Publish to GHCR

Cross-compile and publish with the wrapper script (uses `docker buildx`,
so `linux/arm64` can be built from any host — including a non-arm64 one):

```bash
docker login ghcr.io
./build-push.sh                 # linux/arm64, tag 0.16
./build-push.sh 0.16 linux/amd64
```

`build-push.sh` sets the right `BUILD_ARCH` from the platform (`aarch64` for
`linux/arm64`, `amd64` for `linux/amd64`), so the matching HailoRT .deb/.whl
names are derived automatically and the final image is tagged and pushed as
`ghcr.io/mpeex/obico_ml_hailo_addon-<arch>:0.16` (the `<arch>` suffix matching
the `{arch}` placeholder in `config.yaml`).

For an offline build, drop the 4.21.0 `.deb`/`.whl` matching your arch into
`hailo_assets/` first (they are gitignored); the build uses them instead of
downloading:

- aarch64 → `hailort_4.21.0_arm64.deb` + `hailort-4.21.0-cp311-cp311-linux_aarch64.whl`
- amd64 → `hailort_4.21.0_amd64.deb` + `hailort-4.21.0-cp311-cp311-linux_x86_64.whl`

For a multi-arch manifest, run the script once per platform (docker buildx
builds and pushes each arch with the same tag, still combinable manually):

```bash
# arm64 variant (already pushed by ./build-push.sh ... linux/arm64)
# amd64 variant comes from running it with linux/amd64
# any host
docker manifest create ghcr.io/mpeex/obico_ml_hailo_addon:0.16 \
  ghcr.io/mpeex/obico_ml_hailo_addon-aarch64:0.16 \
  ghcr.io/mpeex/obico_ml_hailo_addon-amd64:0.16
docker manifest push ghcr.io/mpeex/obico_ml_hailo_addon:0.16
```

(Requires `docker login ghcr.io` with a token that has `write:packages`.) The
image version (`0.16`, from `config.yaml`) is the Docker tag Supervisor pulls;
the HailoRT version stays pinned inside the Dockerfile (`4.21.0`).

## License

AGPL-3.0. Based on https://github.com/TheSpaghettiDetective/obico-server
(Hailo runtime) and https://github.com/nobodyguy/obico_ml_ha_addon (addon structure).

The redistributed HailoRT binaries keep their own licenses (MIT and
LGPL-2.1-or-later); the texts ship in `hailo_assets/licenses/` and inside the
image under `/usr/share/licenses/hailort/`.
