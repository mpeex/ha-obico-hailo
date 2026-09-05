from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
from .const import DOMAIN, PLATFORMS
from .coordinator import ObicoDataUpdateCoordinator
import logging

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Legacy setup function. We use config flow for setup."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Obico detection from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Effective config = options (live-editable via the UI) over data.
    cfg = {**entry.data, **entry.options}

    # Import data from config entry
    coordinator = ObicoDataUpdateCoordinator(
        hass,
        url=cfg["url"],
        camera_entity=cfg["camera_entity"],
        interval=cfg["interval"],
        threshold=cfg["threshold"],
    )

    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as err:
        # Do not fail the whole setup if the addon is temporarily unreachable:
        # entities are still registered and just show as unavailable until the
        # next refresh succeeds.
        _LOGGER.warning("Initial Obico refresh failed: %s", err)

    # Store the coordinator so it can be accessed by entities
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()

    return unload_ok