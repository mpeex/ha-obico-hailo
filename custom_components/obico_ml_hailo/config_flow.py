import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector
from .const import (
    DOMAIN,
    DEFAULT_DETECTION_INTERVAL,
    DEFAULT_THRESHOLD,
    DEFAULT_URL,
)


def _camera_selector():
    return selector.EntitySelector(
        selector.EntitySelectorConfig(domain="camera")
    )


def _create_schema(config_entry=None):
    cfg = {**config_entry.data, **config_entry.options} if config_entry else {}
    return vol.Schema({
        vol.Required("url", default=cfg.get("url", DEFAULT_URL)): str,
        vol.Required("detection_interval", default=cfg.get("detection_interval", DEFAULT_DETECTION_INTERVAL)): vol.All(int, vol.Range(min=1)),
        vol.Required("camera_entity", default=cfg.get("camera_entity", "")): _camera_selector(),
        vol.Optional("threshold", default=cfg.get("threshold", DEFAULT_THRESHOLD)): float,
    })


class ObicoConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    @staticmethod
    async def async_migrate_entry(hass, config_entry):
        """Migrate legacy (v1) entries to the current schema.

        Renames the old `interval` option to `detection_interval` and clears
        the old `camera.your_entity` placeholder so the camera has to be chosen
        again from the native HA dropdown.
        """
        if config_entry.version < 2:
            new = {**config_entry.data, **config_entry.options}

            if "interval" in new and "detection_interval" not in new:
                try:
                    new["detection_interval"] = int(new.pop("interval"))
                except (TypeError, ValueError):
                    new.pop("interval", None)
            new.setdefault("detection_interval", DEFAULT_DETECTION_INTERVAL)

            cam = str(new.get("camera_entity", "") or "")
            if cam in ("camera.your_entity", "camera.Your camera entity"):
                cam = ""
            new["camera_entity"] = cam

            new.setdefault("url", DEFAULT_URL)
            new.setdefault("threshold", DEFAULT_THRESHOLD)
            new.pop("auto_start", None)

            hass.config_entries.async_update_entry(config_entry, data=new)

        return True

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            camera_id = user_input["camera_entity"].replace("camera.", "") or "default"
            await self.async_set_unique_id(f"{DOMAIN}_{camera_id}", raise_on_progress=False)
            return self.async_create_entry(title=f"Obico ML - {camera_id}", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=_create_schema()
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return ObicoOptionsFlow(config_entry)


class ObicoOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            self.hass.async_create_task(
                self.hass.config_entries.async_reload(self.config_entry.entry_id)
            )
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=_create_schema(self.config_entry)
        )
