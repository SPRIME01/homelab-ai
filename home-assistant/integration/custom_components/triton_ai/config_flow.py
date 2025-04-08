"""Config flow for TritonAI integration."""
import logging
import voluptuous as vol
from typing import Any, Dict, Optional

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_TRITON_URL,
    CONF_RAY_ADDRESS,
    CONF_MODELS,
    CONF_SENSOR_ANALYSIS_INTERVAL,
    CONF_LOG_LEVEL,
    DEFAULT_SENSOR_ANALYSIS_INTERVAL,
    DEFAULT_LOG_LEVEL,
)
from .triton_client import TritonClient

_LOGGER = logging.getLogger(__name__)

class TritonAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for TritonAI integration."""

    VERSION = 1

    async def async_step_user(self, user_input=None) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Validate the Triton URL by checking connectivity
            triton_client = TritonClient(user_input[CONF_TRITON_URL])
            success = await triton_client.initialize()

            if success:
                # Close the client
                await triton_client.close()

                # Create configuration entry
                return self.async_create_entry(
                    title="Triton AI Integration",
                    data=user_input,
                )
            else:
                errors["base"] = "cannot_connect"

        # Show configuration form
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TRITON_URL): cv.string,
                    vol.Required(CONF_RAY_ADDRESS): cv.string,
                    vol.Required(CONF_MODELS + ".text_generation"): cv.string,
                    vol.Required(CONF_MODELS + ".image_recognition"): cv.string,
                    vol.Required(CONF_MODELS + ".speech_recognition"): cv.string,
                    vol.Optional(
                        CONF_SENSOR_ANALYSIS_INTERVAL,
                        default=DEFAULT_SENSOR_ANALYSIS_INTERVAL,
                    ): cv.positive_int,
                    vol.Optional(
                        CONF_LOG_LEVEL,
                        default=DEFAULT_LOG_LEVEL,
                    ): vol.In(["debug", "info", "warning", "error", "critical"]),
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return TritonAIOptionsFlow(config_entry)

class TritonAIOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Handle options flow."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = {
            vol.Optional(
                CONF_SENSOR_ANALYSIS_INTERVAL,
                default=self.config_entry.options.get(
                    CONF_SENSOR_ANALYSIS_INTERVAL,
                    self.config_entry.data.get(
                        CONF_SENSOR_ANALYSIS_INTERVAL,
                        DEFAULT_SENSOR_ANALYSIS_INTERVAL,
                    ),
                ),
            ): cv.positive_int,
            vol.Optional(
                CONF_LOG_LEVEL,
                default=self.config_entry.options.get(
                    CONF_LOG_LEVEL,
                    self.config_entry.data.get(
                        CONF_LOG_LEVEL,
                        DEFAULT_LOG_LEVEL,
                    ),
                ),
            ): vol.In(["debug", "info", "warning", "error", "critical"]),
            vol.Optional(
                CONF_MODELS + ".text_generation",
                default=self.config_entry.options.get(
                    CONF_MODELS + ".text_generation",
                    self.config_entry.data.get(CONF_MODELS, {}).get("text_generation", ""),
                ),
            ): cv.string,
            vol.Optional(
                CONF_MODELS + ".image_recognition",
                default=self.config_entry.options.get(
                    CONF_MODELS + ".image_recognition",
                    self.config_entry.data.get(CONF_MODELS, {}).get("image_recognition", ""),
                ),
            ): cv.string,
            vol.Optional(
                CONF_MODELS + ".speech_recognition",
                default=self.config_entry.options.get(
                    CONF_MODELS + ".speech_recognition",
                    self.config_entry.data.get(CONF_MODELS, {}).get("speech_recognition", ""),
                ),
            ): cv.string,
        }

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(options),
        )
