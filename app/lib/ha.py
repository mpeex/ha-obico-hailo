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
      1. HA proxy snapshot from the entity's `entity_picture` attribute
         (camera_proxy/image_proxy registered by HA), with retries/backoff.
      2. REST snapshot endpoint /api/camera/{entity_id}/snapshot as a fallback
         for cameras that don't publish an entity_picture.

    Raises RuntimeError with a diagnostic message (snapshot empty, entity
    missing, endpoint failure) when no image could be fetched.
    """
    token = supervisor_token()
    pic = camera_entity_picture(camera_entity)
    if pic:
        last_err = None
        for attempt in range(1, 5):
            try:
                resp = requests.get(
                    pic, headers={"Authorization": f"Bearer {token}"}, timeout=10
                )
                resp.raise_for_status()
                if not resp.content:
                    raise RuntimeError("HA returned an empty image (camera source idle)")
                _LOGGER.debug("camera proxy snapshot OK (%d bytes)", len(resp.content))
                return resp.content
            except Exception as err:
                last_err = err
                _LOGGER.warning("snapshot attempt %d failed: %s", attempt, err)
                if attempt < 4:
                    time.sleep(1.5 if attempt < 3 else 2.5)
        _LOGGER.warning(
            "entity_picture snapshot failed (%s); trying REST snapshot", last_err
        )

    try:
        resp = ha_request("GET", f"api/camera/{camera_entity}/snapshot")
        resp.raise_for_status()
        if not resp.content:
            raise RuntimeError("HA returned an empty image (camera source idle)")
        _LOGGER.debug("REST snapshot OK (%d bytes)", len(resp.content))
        return resp.content
    except Exception as err:
        raise RuntimeError(
            f"Camera {camera_entity} unavailable: no snapshot ({err})"
        ) from err


def set_state(entity_id, state, attributes=None):
    """Create/update a HA entity state via the REST API."""
    ha_request(
        "POST",
        f"api/states/{entity_id}",
        json={"state": state, "attributes": attributes or {}},
    )
