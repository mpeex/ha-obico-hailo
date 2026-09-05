from datetime import timedelta

import aiohttp
import base64
import logging

from homeassistant.components import camera
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import ATTR_ERROR_DETECTED

_LOGGER = logging.getLogger(__name__)


class ObicoDataUpdateCoordinator(DataUpdateCoordinator):
    def __init__(
        self,
        hass: HomeAssistant,
        url: str,
        camera_entity: str,
        interval: int,
        threshold: float,
    ):
        self._url = url.rstrip("/") + "/"
        self.camera_entity = camera_entity
        self._threshold = threshold
        self.api_enabled = True  # start enabled; the switch can turn it off

        super().__init__(
            hass,
            _LOGGER,
            name="Obico ML",
            update_interval=timedelta(seconds=interval),
        )

    async def _async_update_data(self):
        if not self.api_enabled:
            return None

        # Grab the current frame from the camera platform itself (no external
        # URL / auth relay needed - HA resolves entity pictures internally).
        try:
            image = await camera.async_get_image(self.hass, self.camera_entity)
            image_bytes = image.content
        except Exception as err:
            _LOGGER.warning("Failed to fetch camera snapshot: %s", err)
            raise

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "img": image_base64,
            "threshold": self._threshold,
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self._url + "detect/", json=payload) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"API call failed: {resp.status}")
                    data = await resp.json()
            except Exception as err:
                _LOGGER.error("Error fetching data from Obico API: %s", err)
                raise

        detections = data.get("detections", [])
        error_detected = detections != []

        avg_confidence = 0.0
        if detections:
            avg_confidence = (
                sum(float(d[1]) for d in detections) / len(detections)
            ) * 100
            avg_confidence = round(avg_confidence, 2)

        image_with_errors = None
        raw = data.get("image_with_detections")
        if raw:
            image_with_errors = base64.b64decode(raw)

        return {
            "image_with_errors": image_with_errors,
            ATTR_ERROR_DETECTED: error_detected,
            "avg_confidence": avg_confidence,
        }