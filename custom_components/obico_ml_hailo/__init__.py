from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType
from .const import DOMAIN, PLATFORMS, DEFAULT_AUTO_START
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
        url=cfg.get("url", ""),
        camera_entity=cfg.get("camera_entity", ""),
        detection_interval=cfg.get("detection_interval", 1),
        threshold=cfg.get("threshold", 0.2),
        auto_start=cfg.get("auto_start", DEFAULT_AUTO_START),
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

    # (Re)sync the addon worker with the current options: starts it if
    # auto_start is enabled, and applies option changes to a running worker.
    hass.async_create_task(coordinator.async_apply_config())

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()

    return unload_ok