from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Obico entities from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]

    # Create and add the ObicoSwitch entity
    async_add_entities([ObicoSwitch(coordinator, entry)])


class ObicoSwitch(CoordinatorEntity, SwitchEntity):
    """A switch that starts/stops the addon's detection worker."""

    def __init__(self, coordinator, entry):
        """Initialize the switch."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_name = "Obico ML Detection"
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_api_communication"

    @property
    def is_on(self) -> bool:
        """Return True while the addon's camera worker is running."""
        if self.coordinator.data is None:
            return False
        return bool(self.coordinator.data.get("running"))

    async def async_turn_on(self, **kwargs) -> None:
        """Start detection on the addon."""
        await self.coordinator.async_set_running(True)

    async def async_turn_off(self, **kwargs) -> None:
        """Stop detection on the addon."""
        await self.coordinator.async_set_running(False)