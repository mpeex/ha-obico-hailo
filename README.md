# Obico ML (Hailo) — Home Assistant Addon

[![](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fmpeex%2Fspaghetti-rpi5-aikit)

Spaghetti / 3D-print **failure detection** for Home Assistant, running the
Obico YOLOv5 model on a **Hailo-8 / Hailo-8L** AI accelerator (Raspberry Pi AI
HAT+).

## What it is

- A Home Assistant **addon** (built for `aarch64`, also published for `amd64`)
  that captures a camera snapshot, runs inference on the Hailo and serves an
  annotated stream + detection state over a small REST API on port `3333`.
- A **companion integration** that mirrors the addon state into real HA
  entities.

All configuration lives in the integration's options flow — the addon itself
has no options.

## Features

- Real-time failure detection (`failure`, 1 class) with an annotated stream.
- Camera chosen from a native HA `camera.*` entity dropdown; configurable
  `detection_interval` (seconds, min `1`) and `threshold`.
- Detection (re)starts automatically whenever the integration is loaded and a
  camera is configured; a manual switch stops/starts the worker at any time.
- Entities: `binary_sensor.obico_failure`, `sensor.obico_confidence`,
  `camera.obico_ml_detection_camera`, `switch.obico_ml_detection`.

## Quick start

1. Add this repository to Home Assistant (button above) and install the
   **Obico ML (Hailo)** addon.
2. Copy `custom_components/obico_ml_hailo/` into
   `config/custom_components/obico_ml_hailo/` and restart Home Assistant.
3. Settings → Devices & Services → Add Integration → *Obico ML (Hailo)* →
   URL `http://172.30.32.1:3333`, pick a camera, set interval/threshold.

## Requirements

- Raspberry Pi 5 (or x64) with a Hailo-8 / Hailo-8L AI HAT+, HailoRT installed
  and `hailort_service` running so `/dev/hailo0` is present.
- Home Assistant core ≥ `2026.9.0`.

## Documents

- [`architecture.md`](architecture.md) — model pipeline, REST API, build and
  publish details.
- [`CHANGELOG.md`](CHANGELOG.md)
- [`LICENSE`](LICENSE) — AGPL-3.0 plus the licenses of all redistributed
  third-party components.

## License

AGPL-3.0. Based on
[obico-server](https://github.com/TheSpaghettiDetective/obico-server) and
[obico_ml_ha_addon](https://github.com/nobodyguy/obico_ml_ha_addon). See
[`LICENSE`](LICENSE) for the complete list of redistributed components (HailoRT,
s6-overlay, bashio, tempio, Python packages).