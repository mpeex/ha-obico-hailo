#!/usr/bin/env python
"""Persist the addon's camera-selection configuration.

Stored as JSON in /data (the HAOS addon data dir, which is writable).
Falls back to in-memory storage when /data is not available (local dev).
"""

import json
import logging
import os
import threading

_LOGGER = logging.getLogger(__name__)

DATA_DIR = os.environ.get("OBICO_DATA_DIR", "/data")
CONFIG_FILE = os.path.join(DATA_DIR, "obico_config.json")

DEFAULTS = {
    "camera_entity": "",
    "interval": 10,
    "threshold": 0.2,
}

_lock = threading.Lock()
_memory = dict(DEFAULTS)


def _data_dir_available():
    try:
        return os.path.isdir(DATA_DIR) and os.access(DATA_DIR, os.W_OK)
    except Exception:
        return False


def load_config():
    """Return the current config dict (persisted or env override merged)."""
    cfg = dict(DEFAULTS)
    # Environment overrides take precedence (set by the s6 run script).
    if os.environ.get("CAMERA_ENTITY", ""):
        cfg["camera_entity"] = os.environ["CAMERA_ENTITY"]
    if os.environ.get("INTERVAL", ""):
        try:
            cfg["interval"] = int(os.environ["INTERVAL"])
        except ValueError:
            pass
    if os.environ.get("THRESHOLD", ""):
        try:
            cfg["threshold"] = float(os.environ["THRESHOLD"])
        except ValueError:
            pass
    if os.environ.get("CAMERA_ENTITY", "") or os.environ.get("INTERVAL") or os.environ.get("THRESHOLD"):
        return cfg

    with _lock:
        if _data_dir_available():
            try:
                with open(CONFIG_FILE, "r") as fh:
                    stored = json.load(fh)
                cfg.update({k: stored[k] for k in cfg if k in stored})
                return cfg
            except FileNotFoundError:
                pass
            except Exception as err:
                _LOGGER.warning("Failed to read config file: %s", err)
        else:
            cfg.update(_memory)
        return cfg


def save_config(partial):
    """Merge `partial` over the current config and persist it."""
    cfg = load_config()
    for key in cfg:
        if key in partial and partial[key] not in (None, ""):
            cfg[key] = partial[key]
    with _lock:
        if _data_dir_available():
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(CONFIG_FILE, "w") as fh:
                    json.dump(cfg, fh, indent=2)
            except Exception as err:
                _LOGGER.error("Failed to write config file: %s", err)
        else:
            _memory.update(cfg)
    return cfg
