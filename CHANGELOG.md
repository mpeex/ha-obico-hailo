# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.16]

### Changed

- Detection now (re)starts automatically every time the integration is
  (re)loaded and a camera is configured: after an addon boot, a HA core restart
  or an options save the worker is (re)started with the current options
  (posting `/api/start` is idempotent). With no camera configured the addon
  stays idle.

### Removed

- Integration option `auto_start`: it was hard to grasp and no longer needed —
  the "restart on load" behaviour is now always-on whenever a camera is set
  (`switch.obico_ml_detection` still stops/starts the worker within a session).

## [0.15]

### Added

- Integration option `auto_start`: restarts detection whenever the integration
  is (re)loaded (addon boot, HA core restart, options save).
- The integration now re-pushes its options to the addon on every setup, so a
  running worker picks up `detection_interval` / `threshold` / camera changes
  immediately (posting `/api/start` is idempotent).

### Removed

- All addon options (`camera_entity`, `auto_start`, `detection_interval`,
  `threshold`): the addon now starts idle and is driven entirely by the
  companion integration (camera dropdown, detection interval, threshold and
  auto-start all live in the integration's options flow).

## [0.14]

### Changed

- Detection is driven by a single `detection_interval` option (seconds,
  minimum `1`, default `1`) replacing the `max_fps` throttle and the old
  `interval` option.
- Companion integration (0.14): renamed `interval` to `detection_interval`
  (minimum `1`, default `1`) in the options flow and in the `/api/start`
  payload.
- Integration options now pick the camera from a native HA dropdown (`entity`
  selector filtered on the `camera.*` domain, default empty); no custom UI.
- Legacy integration entries are auto-migrated on startup: the old `interval`
  option becomes `detection_interval` and the `camera.your_entity` placeholder
  is cleared so the camera is re-selected from the dropdown.
- Addon declares a minimum Home Assistant core of `2026.9.0`.

### Fixed

- Addon boot after removing Ingress/nginx: the rootfs still shipped a stale
  `s6-rc.d/nginx` service stub and a `user` bundle marker pointing at it, so
  `s6-rc-compile` aborted (`undefined service name nginx`). Both leftovers are
  removed from the rootfs and the s6-rc tree is re-verified at build time.

### Removed

- `max_fps` option and the `FrameRateLimiter` code path.
- Self-contained mode: Flask camera-selection web UI, camera enumeration
  (`/api/cameras`, `list_cameras`) and the addon's direct REST state publishing
  (`binary_sensor.obico_failure` / `sensor.obico_confidence` are now mirrored
  exclusively through the companion integration).
- HA Ingress / nginx reverse proxy (the integration talks to the addon directly
  on port `3333`).

## [0.13]

### Changed

- Camera snapshots now use the HA `entity_picture` (camera proxy) with retries
  and backoff, falling back to the core REST `/api/camera/{id}/snapshot`
  endpoint; the removed `get_stream_source` service is no longer used.
- Annotated stream always shows a status banner: `NO FAILURE` (green) or
  `FAILURE xx.x%` (red) plus one red box per detected object.
- Web UI, addon and integration options align with the new camera handling.

## [0.11]

### Fixed

- `ports` schema accepts only host port numbers.

## [0.10]

### Added

- Fallback to the camera's RTSP/stream for entities without reliable stills.

## [0.9]

### Changed

- Port `3333` is published on the Supervisor gateway so the HA core can reach
  the addon API.

## [0.8]

### Added

- Companion integration mirrors the addon worker and publishes registered
  entities; `auto_start` option.
- The detection worker auto-starts on boot when `auto_start` is enabled and a
  camera is configured.