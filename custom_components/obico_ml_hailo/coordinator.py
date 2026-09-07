from datetime import timedelta

import aiohttp
import base64
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import ATTR_ERROR_DETECTED

_LOGGER = logging.getLogger(__name__)


class ObicoDataUpdateCoordinator(DataUpdateCoordinator):
    """Poll the addon's /status endpoint and mirror it as HA entities.

    The addon itself owns frame capture + inference (its camera worker, started
    via /api/start or auto_start), so the integration only mirrors results — it
    never talks to the camera or the Hailo directly.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        url: str,
        camera_entity: str,
        detection_interval: int,
        threshold: float,
        auto_start: bool = False,
    ):
        self._base_url = url.rstrip("/")
        self.camera_entity = camera_entity
        self._interval = detection_interval
        self._threshold = threshold
        self._auto_start = auto_start
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

        super().__init__(
            hass,
            _LOGGER,
            name="Obico ML",
            update_interval=timedelta(seconds=self._interval),
        )

    async def _async_update_data(self):
        """Fetch /status from the addon and normalize it for the entities."""
        async with self._session.get(self._base_url + "/status") as resp:
            resp.raise_for_status()
            data = await resp.json()

        state = data.get("state") or {}
        cfg = data.get("config") or {}
        detections = state.get("last_detections") or []

        return {
            "running": bool(state.get("running")),
            ATTR_ERROR_DETECTED: bool(detections),
            "avg_confidence": state.get("last_avg_confidence", 0.0),
            "last_error": state.get("last_error"),
            "last_update": state.get("last_update"),
            "camera": cfg.get("camera_entity", self.camera_entity),
        }

    async def async_set_running(self, running: bool):
        """Start/stop the addon's detection worker via its REST API."""
        if running:
            payload = {
                "camera_entity": self.camera_entity,
                "detection_interval": self._interval,
                "threshold": self._threshold,
            }
            async with self._session.post(
                self._base_url + "/api/start", json=payload
            ) as resp:
                resp.raise_for_status()
        else:
            async with self._session.post(self._base_url + "/api/stop") as resp:
                resp.raise_for_status()
        await self.async_request_refresh()

    async def async_fetch_last_image(self) -> bytes | None:
        """Return the latest annotated frame JPEG (or None)."""
        async with self._session.get(self._base_url + "/last_image") as resp:
            if resp.status != 200:
                return None
            return await resp.read()

    async def async_apply_config(self):
        """Push the current options to the addon.

        Re-posting `/api/start` is idempotent: it (re)applies the configured
        camera/interval/threshold to a worker that is already running, and
        re-enables detection after restarts when `auto_start` is set. Errors
        are logged, not raised, so setup never fails on a busy addon.
        """
        if not (
            self._auto_start or (self.data and self.data.get("running"))
        ):
            return
        try:
            await self.async_set_running(True)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Could not sync detection config to addon: %s", err)

    async def async_shutdown(self):
        """Close the HTTP session."""
        await self._session.close()
        await super().async_shutdown()