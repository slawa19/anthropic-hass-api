"""The Anthropic integration."""
from __future__ import annotations

import logging

# Define our own APIError class for compatibility
class APIError(Exception):
    """API Error."""

    def __init__(self, message, type, status_code):
        """Initialize the error."""
        self.message = message
        self.type = type
        self.status_code = status_code
        super().__init__(message)
from homeassistant.components import conversation as ha_conversation
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import (
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
from .conversation import AnthropicAgent

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Anthropic component."""
    hass.data[DOMAIN] = {}
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Anthropic from a config entry."""
    try:
        # Create a simple wrapper class that mimics the Anthropic client
        # but only implements the methods we need
        class AnthropicWrapper:
            """Simple wrapper for Anthropic API."""
            
            def __init__(self, api_key):
                """Initialize the wrapper."""
                self.api_key = api_key
                self.messages = self.Messages(api_key)
            
            class Messages:
                """Messages API wrapper."""
                
                def __init__(self, api_key):
                    """Initialize the messages wrapper."""
                    self.api_key = api_key
                
                def create(self, model, max_tokens, temperature, messages):
                    """Create a message."""
                    import requests
                    
                    url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                    data = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": messages,
                    }
                    
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code != 200:
                        raise APIError(
                            message=f"API request failed: {response.text}",
                            type="api_error",
                            status_code=response.status_code,
                        )
                    
                    # Parse the response and return a similar object to what the Anthropic client would return
                    response_data = response.json()
                    
                    # Create a response object with the same structure as the Anthropic client
                    class ResponseContent:
                        """Response content object."""
                        
                        def __init__(self, text):
                            """Initialize the response content."""
                            self.text = text
                    
                    class Response:
                        """Response object."""
                        
                        def __init__(self, content):
                            """Initialize the response."""
                            self.content = content
                    
                    # Extract the content from the response
                    content_text = response_data.get("content", [{}])[0].get("text", "")
                    
                    return Response([ResponseContent(content_text)])
        
        # Create our wrapper instead of the Anthropic client
        client = AnthropicWrapper(entry.data[CONF_API_KEY])
    except Exception as err:
        _LOGGER.error("Failed to initialize Anthropic client: %s", err)
        raise ConfigEntryNotReady from err

    # Set default options if not already set
    if not entry.options:
        options = {
            CONF_PROMPT: DEFAULT_PROMPT,
            CONF_CONTROL_HA: DEFAULT_CONTROL_HA,
            CONF_RECOMMENDED_SETTINGS: DEFAULT_RECOMMENDED_SETTINGS,
            CONF_CHAT_MODEL: AnthropicModels.default(),
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        }
        hass.config_entries.async_update_entry(entry, options=options)

    agent = AnthropicAgent(hass, entry, client)
    ha_conversation.async_set_agent(hass, entry, agent)
    hass.data[DOMAIN][entry.entry_id] = agent

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, []):
        hass.data[DOMAIN].pop(entry.entry_id)
        ha_conversation.async_unset_agent(hass, entry)

    return unload_ok