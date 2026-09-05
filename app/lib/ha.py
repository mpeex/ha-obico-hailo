#!/usr/bin/env python
"""Home Assistant / Supervisor connectivity for the Obico ML (Hailo) addon.

The addon is a container running on HAOS. It reaches
  - the Supervisor  at  http://supervisor          (SUPERVISOR_TOKEN env)
  - the HA core API at  http://supervisor/core/api  (SUPERVISOR_TOKEN env)

This lets the addon be self-contained: it discovers the user's camera.*
entities, fetches snapshots and publishes detection results back to HA
entities without depending on a companion integration.
"""

import logging
import os
import time

import requests

_LOGGER = logging.getLogger(__name__)

SUPERVISOR_URL = os.environ.get("HA_SUPERVISOR_URL", "http://supervisor")


def supervisor_token():
    """The Supervisor token injected into the addon container."""
    return os.environ.get("SUPERVISOR_TOKEN")


def ha_available():
    """True when running under HAOS with Supervisor token present."""
    return bool(supervisor_token())


def ha_request(method, path, **kwargs):
    """Call the HA core REST API through the Supervisor proxy.

    `path` is relative, e.g. "api/states" or "/api/states". Returns a
    requests.Response or raises on connection / token failure.
    """
    token = supervisor_token()
    if not token:
        raise RuntimeError("No SUPERVISOR_TOKEN available")
    url = (
        path
        if path.startswith("http")
        else f"{SUPERVISOR_URL}/core/{path.lstrip('/')}"
    )
    resp = requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=15,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def list_cameras():
    """Return a sorted list of camera.* entity ids.

    Includes all camera.* entities, regardless of state, so that the UI
    dropdown can populate even when cameras report 'unavailable'/'unknown'
    during startup.  Diagnostic logging helps spot filtering issues in
    future runs.
    """
    states = ha_request("GET", "api/states").json()
    cameras = sorted(
        s["entity_id"]
        for s in states
        if s["entity_id"].startswith("camera.")
    )
    _LOGGER.debug(
        "list_cameras: %d camera.* entities found: %s",
        len(cameras),
        cameras,
    )
    return cameras


def camera_entity_picture(camera_entity):
    """Absolute HA URL of the camera entity_picture (snapshot endpoint)."""
    states = ha_request("GET", "api/states").json()
    for s in states:
        if s["entity_id"] == camera_entity:
            pic = (s.get("attributes") or {}).get("entity_picture")
            if pic:
                return f"{SUPERVISOR_URL}/core{pic}"
            return None
    return None


def fetch_camera_image(camera_entity):
    """Return the raw JPEG bytes of `camera_entity`'s current snapshot.

    Strategy (in order):
      1. HA camera_proxy snapshot (from the entity's attribute). Cheap and
         works for most cameras, but some (e.g. Bambu Lab P1/S1) return 500
         when HA cannot produce a still on demand.
      2. The camera's stream source (REST camera.get_stream_source): open the
         RTSP/HLS URL and grab one frame with OpenCV. Works for cameras whose
         still endpoint is unreliable, as long as we have LAN access to the
         stream.
    """
    pic = camera_entity_picture(camera_entity)
    if pic:
        token = supervisor_token()
        last_err = None
        for attempt in (1, 2, 3):
            try:
                resp = requests.get(
                    pic, headers={"Authorization": f"Bearer {token}"}, timeout=10
                )
                resp.raise_for_status()
                _LOGGER.debug("camera_proxy snapshot OK (%d bytes)", len(resp.content))
                return resp.content
            except Exception as err:
                last_err = err
                _LOGGER.warning(
                    "camera_proxy snapshot attempt %d failed: %s", attempt, err
                )
                time.sleep(1.0)
        _LOGGER.warning(
            "camera_proxy snapshot failed (%s); falling back to stream source",
            last_err,
        )

    url = get_stream_source(camera_entity)
    if url:
        return _grab_stream_frame(url)

    raise RuntimeError(
        f"Camera {camera_entity} provides neither a snapshot nor a stream source"
    )


def get_stream_source(camera_entity):
    """Return the camera's stream URL (RTSP/HLS), or None."""
    try:
        resp = ha_request(
            "POST",
            "api/services/camera/get_stream_source",
            json={"entity_id": camera_entity},
        )
        result = resp.json().get("result")
        if result:
            _LOGGER.debug("stream source for %s: %s", camera_entity, result)
        return result or None
    except Exception as err:
        _LOGGER.warning("get_stream_source failed for %s: %s", camera_entity, err)
        return None


def _grab_stream_frame(url):
    """Open a stream URL and return the first decoded frame as JPEG bytes."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(url)
    try:
        frame = None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.2)
        if frame is None:
            raise RuntimeError(f"no frame decoded from {url}")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return np.asarray(buf).tobytes()
    finally:
        cap.release()


def set_state(entity_id, state, attributes=None):
    """Create/update a HA entity state via the REST API."""
    ha_request(
        "POST",
        f"api/states/{entity_id}",
        json={"state": state, "attributes": attributes or {}},
    )
