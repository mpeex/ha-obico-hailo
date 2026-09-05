#!/usr/bin/env python
"""Home Assistant / Supervisor connectivity for the Obico ML (Hailo) addon.

The addon is a container running on HAOS. It reaches
  - the Supervisor  at  http://supervisor          (SUPERVISOR_TOKEN env)
  - the HA core API at  http://supervisor/core     (access token obtained from
    GET /addons/self/info -> .data.access_token)

This lets the addon be self-contained: it discovers the user's camera.*
entities, fetches snapshots and publishes detection results back to HA
entities without depending on a companion integration.
"""

import logging
import os
import threading

import requests

_LOGGER = logging.getLogger(__name__)

SUPERVISOR_URL = os.environ.get("HA_SUPERVISOR_URL", "http://supervisor")

_token_lock = threading.Lock()
_cached_token = None


def supervisor_token():
    """The Supervisor token injected into the addon container."""
    return os.environ.get("SUPERVISOR_TOKEN")


def ha_token():
    """Long-lived HA core API token, retrieved from the Supervisor and cached.

    The Supervisor exposes the addon's own HA access token at
    GET /addons/self/info -> .data.access_token.
    """
    global _cached_token
    if _cached_token:
        return _cached_token
    with _token_lock:
        if _cached_token:
            return _cached_token
        token = supervisor_token()
        if not token:
            _LOGGER.warning("SUPERVISOR_TOKEN not set - not running under HAOS")
            return None
        try:
            resp = requests.get(
                f"{SUPERVISOR_URL}/addons/self/info",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            resp.raise_for_status()
            _cached_token = resp.json()["data"]["access_token"]
            return _cached_token
        except Exception as err:
            _LOGGER.error("Failed to fetch HA access token: %s", err)
            return None


def ha_available():
    """True when we have both a Supervisor token and a HA core token."""
    return bool(supervisor_token() and ha_token())


def ha_request(method, path, **kwargs):
    """Call the HA core REST API through the Supervisor proxy.

    `path` is relative, e.g. "/api/states" or "api/states". Returns a
    requests.Response or raises on connection / token failure.
    """
    token = ha_token()
    if not token:
        raise RuntimeError("No HA access token available")
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
    states = ha_request("GET", "/api/states").json()
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
    states = ha_request("GET", "/api/states").json()
    for s in states:
        if s["entity_id"] == camera_entity:
            pic = (s.get("attributes") or {}).get("entity_picture")
            if pic:
                return f"{SUPERVISOR_URL}/core{pic}"
            return None
    return None


def fetch_camera_image(camera_entity):
    """Return the raw JPEG bytes of `camera_entity`'s current snapshot."""
    pic = camera_entity_picture(camera_entity)
    if not pic:
        raise RuntimeError(
            f"Camera {camera_entity} does not provide an image URL"
        )
    token = ha_token()
    resp = requests.get(
        pic, headers={"Authorization": f"Bearer {token}"}, timeout=15
    )
    resp.raise_for_status()
    return resp.content


def set_state(entity_id, state, attributes=None):
    """Create/update a HA entity state via the REST API."""
    ha_request(
        "POST",
        f"/api/states/{entity_id}",
        json={"state": state, "attributes": attributes or {}},
    )
