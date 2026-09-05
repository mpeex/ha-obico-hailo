from homeassistant.components.camera import Camera
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Obico entities from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # Create and add the ObicoCamera entity
    async_add_entities([ObicoCamera(coordinator, entry)])


class ObicoCamera(CoordinatorEntity, Camera):
    """Camera entity showing the addon's latest annotated frame."""

    def __init__(self, coordinator, entry):
        """Initialize the camera entity."""
        CoordinatorEntity.__init__(self, coordinator)
        Camera.__init__(self)
        self._entry = entry
        self._attr_name = "Obico ML Detection Camera"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_camera"

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Return the latest annotated frame from the addon."""
        return await self.coordinator.async_fetch_last_image()

    @property
    def available(self) -> bool:
        """Available while the addon reports detections/frames."""
        return (
            self.coordinator.last_update_success
            and self.coordinator.data is not None
        )