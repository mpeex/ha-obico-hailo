# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.13] — unreleased

### Changed

- Detection is now driven by a single `detection_interval` option (seconds,
  minimum `1`, default `1`) replacing the `max_fps` throttle and the old
  `interval` option.
- Companion integration (1.2.0): renamed `interval` to `detection_interval`
  (minimum `1`, default `1`) in the options flow and in the `/api/start`
  payload.
- Addon declares a minimum Home Assistant core of `2026.9.0`.

### Removed

- `max_fps` option and the `FrameRateLimiter` code path.

## [0.12]

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