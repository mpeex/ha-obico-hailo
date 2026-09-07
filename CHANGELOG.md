# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.16] - 2026-09-07

First public release.

### Added

- Home Assistant **addon** (aarch64 + amd64) that runs the Obico YOLOv5
  failure-detection model on a Hailo-8 / Hailo-8L (RPi AI HAT+) from compiled
  HEFs, using the host-side `decode.onnx` YOLOv5 head.
- Detects a single class (`failure`) from any `camera.*` entity in Home
  Assistant: the addon captures the camera snapshot via the Supervisor API,
  runs inference on the Hailo and serves an annotated stream + detection state
  over a small REST API (port `3333`).
- **Companion integration** (`custom_components/obico_ml_hailo`) mirroring the
  addon state into registered entities: `binary_sensor.obico_failure`,
  `sensor.obico_confidence`, `camera.obico_ml_detection_camera` and
  `switch.obico_ml_detection`.
- All configuration lives in the integration's options flow: camera picked from
  a native HA `entity` dropdown (`camera.*`), a single `detection_interval`
  knob (seconds, minimum `1`, default `1`) and a `threshold`. The addon itself
  has no options and starts idle.
- Detection (re)starts automatically every time the integration is (re)loaded
  and a camera is configured (addon boot, HA core restart, options save); the
  switch stops/starts the worker within a session.
- Automatic migration of legacy configuration entries to the current format
  (`interval` → `detection_interval`, clearing of the old camera placeholder).
- Min Home Assistant core `2026.9.0`.

### Security

- The addon API port is published on the host; the README instructs to keep
  the network trusted or firewall the port.