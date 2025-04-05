"""Support for Anthropic conversation."""
from __future__ import annotations

import logging
from typing import Literal, Dict, List, Any

# Define our own APIError class for compatibility
class APIError(Exception):
    """API Error."""

    def __init__(self, message, type, status_code):
        """Initialize the error."""
        self.message = message
        self.type = type
        self.status_code = status_code
        super().__init__(message)

# Define a type for message parameters
MessageParam = Dict[str, Any]

from homeassistant.components import conversation as ha_conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent, template
from homeassistant.util import ulid

from .const import (
    CONF_API_KEY,
    CONF_CHAT_MODEL,
    CONF_CONTROL_HA,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED_SETTINGS,
    CONF_TEMPERATURE,
    DEFAULT_CONTROL_HA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    AnthropicModels,
)

_LOGGER = logging.getLogger(__name__)


class AnthropicAgent(ha_conversation.AbstractConversationAgent):
    """Anthropic conversation agent."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: Any,  # Use Any for the client type
    ) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[MessageParam]] = {}
        self.client = client
        self.prompt = entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        self.model = entry.options.get(
            CONF_CHAT_MODEL, AnthropicModels.default()
        )
        self.max_tokens = entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        self.temperature = entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        self.control_ha = entry.options.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)
        self.recommended_settings = entry.options.get(CONF_RECOMMENDED_SETTINGS, True)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process a sentence."""
        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                prompt = self._async_generate_prompt()
            except template.TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                return ha_conversation.ConversationResult(
                    conversation_id=conversation_id,
                    response=ha_conversation.ConversationResponse(
                        speech={"plain": {"speech": f"Sorry, I had a problem processing your request: {err}", "extra_data": None}},
                        intent=intent.IntentResponse(language=user_input.language),
                    ),
                )

            messages = [
                {"role": "system", "content": prompt},
            ]
            self.history[conversation_id] = messages

        if self.control_ha:
            try:
                ha_response = await ha_conversation.async_converse(
                    self.hass,
                    user_input.text,
                    user_input.conversation_id,
                    user_input.language,
                    user_input.agent_id,
                    user_input.context,
                    ha_conversation.ConversationEngine.HOME_ASSISTANT,
                )
                if ha_response.response.intent.response_type in (
                    intent.IntentResponseType.ACTION_DONE,
                    intent.IntentResponseType.SUCCESS,
                ):
                    # If Home Assistant handled the intent, add it to the context
                    if entities := ha_response.response.intent.slots.get(
                        ATTR_ENTITY_ID, {}
                    ).get("value"):
                        if isinstance(entities, list):
                            entities_text = ", ".join(entities)
                        else:
                            entities_text = entities
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"{user_input.text}\n\n"
                                    f"Home Assistant was able to control: {entities_text}"
                                ),
                            }
                        )
                    else:
                        messages.append({"role": "user", "content": user_input.text})
                    messages.append(
                        {
                            "role": "assistant",
                            "content": ha_response.response.speech.get("plain", {}).get(
                                "speech", ""
                            ),
                        }
                    )
                    self.history[conversation_id] = messages
                    return ha_conversation.ConversationResult(
                        conversation_id=conversation_id, response=ha_response.response
                    )
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error handling Home Assistant conversation")

        messages.append({"role": "user", "content": user_input.text})

        model = self.model
        max_tokens = self.max_tokens
        temperature = self.temperature

        if self.recommended_settings:
            model = AnthropicModels.default()
            max_tokens = DEFAULT_MAX_TOKENS
            temperature = DEFAULT_TEMPERATURE

        try:
            # Call the create method directly with parameters
            # Our wrapper class handles the API call
            response = await self.hass.async_add_executor_job(
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages
                )
            )
        except APIError as err:
            _LOGGER.error("Error calling Anthropic API: %s", err)
            return ha_conversation.ConversationResult(
                conversation_id=conversation_id,
                response=ha_conversation.ConversationResponse(
                    speech={"plain": {"speech": f"Sorry, I had a problem processing your request: {err}", "extra_data": None}},
                    intent=intent.IntentResponse(language=user_input.language),
                ),
            )

        messages.append(
            {"role": "assistant", "content": response.content[0].text}
        )
        self.history[conversation_id] = messages

        return ha_conversation.ConversationResult(
            conversation_id=conversation_id,
            response=ha_conversation.ConversationResponse(
                speech={"plain": {"speech": response.content[0].text, "extra_data": None}},
                intent=intent.IntentResponse(language=user_input.language),
            ),
        )

    def _async_generate_prompt(self) -> str:
        """Generate a prompt for the user."""
        return template.Template(self.prompt, self.hass).async_render(
            {
                "control_ha": self.control_ha,
            },
            parse_result=False,
        )