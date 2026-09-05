from homeassistant.components.sensor import SensorEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Obico entities from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # Create and add the ObicoConfidenceSensor entity
    async_add_entities([ObicoConfidenceSensor(coordinator, entry)])


class ObicoConfidenceSensor(CoordinatorEntity, SensorEntity):
    """Sensor entity for average failure detection confidence."""

    def __init__(self, coordinator, entry):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_name = "Obico ML Failure Detection Confidence"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_failure_detection_confidence"
        self._attr_native_unit_of_measurement = "%"

    @property
    def native_value(self):
        """Value of the sensor in percent; None (N/A) while no detections."""
        if self.coordinator.data is None or not self.coordinator.last_update_success:
            return None
        return self.coordinator.data.get("avg_confidence", None)

    @property
    def available(self):
        """Return True if the sensor is available (i.e., data is valid)."""
        return self.coordinator.last_update_success and self.coordinator.data is not None