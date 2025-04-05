"""Config flow for Anthropic integration."""
from __future__ import annotations

import logging
from typing import Any

# We don't need the anthropic package anymore
# Define our own APIError class for compatibility
class APIError(Exception):
    """API Error."""

    def __init__(self, message, type, status_code):
        """Initialize the error."""
        self.message = message
        self.type = type
        self.status_code = status_code
        super().__init__(message)
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
)

from .const import (
    ANTHROPIC_MODELS,
    CONF_CHAT_MODEL,
    CONF_CONTROL_HA,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED_SETTINGS,
    CONF_TEMPERATURE,
    DEFAULT_CONTROL_HA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_RECOMMENDED_SETTINGS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    AnthropicModels,
)

_LOGGER = logging.getLogger(__name__)


async def validate_api_key(hass: HomeAssistant, api_key: str) -> None:
    """Validate the API key by making a request."""
    # Instead of using the Anthropic client directly, we'll make a simple HTTP request
    # to validate the API key
    import aiohttp
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = {
        "model": AnthropicModels.default(),
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                text = await response.text()
                raise APIError(
                    message=f"API request failed: {text}",
                    type="api_error",
                    status_code=response.status,
                )


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Anthropic."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required(CONF_API_KEY): str}),
            )

        errors = {}

        try:
            await validate_api_key(self.hass, user_input[CONF_API_KEY])
        except APIError as err:
            _LOGGER.error("API error: %s", err)
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected error")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(title="Anthropic", data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({vol.Required(CONF_API_KEY): str}),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """Anthropic config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Define the options schema using translations
        options = {
            vol.Optional(
                CONF_PROMPT,
                default=self.config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
            ): TemplateSelector(),
            vol.Optional(
                CONF_CONTROL_HA,
                default=self.config_entry.options.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA),
            ): bool,
            vol.Optional(
                CONF_RECOMMENDED_SETTINGS,
                default=self.config_entry.options.get(
                    CONF_RECOMMENDED_SETTINGS, DEFAULT_RECOMMENDED_SETTINGS
                ),
            ): bool,
        }

        if not self.config_entry.options.get(
            CONF_RECOMMENDED_SETTINGS, DEFAULT_RECOMMENDED_SETTINGS
        ):
            options.update(
                {
                    vol.Optional(
                        CONF_CHAT_MODEL,
                        default=self.config_entry.options.get(
                            CONF_CHAT_MODEL, AnthropicModels.default()
                        ),
                    ): vol.In({model: model for model in ANTHROPIC_MODELS}),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=self.config_entry.options.get(
                            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=1,
                            max=4096,
                            step=1,
                            mode="box",
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=self.config_entry.options.get(
                            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.0,
                            max=1.0,
                            step=0.05,
                            mode="slider",
                        )
                    ),
                }
            )

        return self.async_show_form(step_id="init", data_schema=vol.Schema(options))