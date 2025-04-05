"""Support for Anthropic conversation."""
from __future__ import annotations

import logging
from typing import Literal, Dict, List, Any, Optional

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
from dataclasses import dataclass

# Create our own ConversationResponse class since it's not available in Home Assistant
@dataclass
class ConversationResponse:
    """Response from a conversation agent."""

    speech: dict
    intent: Any

    def as_dict(self):
        """Return a dict representation of the response."""
        return {
            "speech": self.speech,
            "intent": self.intent,
        }
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
        
    def _get_entity_states(self) -> str:
        """Get information about Home Assistant entities."""
        states = self.hass.states.async_all()
        
        # Filter to only include entities that are exposed to the conversation agent
        exposed_entities = []
        for state in states:
            # You might need to adjust this logic based on how entities are exposed in your system
            # This is a simplified version
            domain = state.entity_id.split('.')[0]
            if domain in ['sensor', 'binary_sensor', 'light', 'switch', 'climate', 'media_player']:
                exposed_entities.append(state)
        
        if not exposed_entities:
            return "No entities are currently exposed to the conversation agent."
        
        # Format entity information
        entity_info = []
        for state in exposed_entities:
            entity_id = state.entity_id
            friendly_name = state.attributes.get('friendly_name', entity_id)
            current_state = state.state
            
            entity_info.append(f"{friendly_name} ({entity_id}): {current_state}")
        
        return "\n".join(entity_info)
        
    async def _call_service(self, domain, service, entity_id, **service_data):
        """Call a Home Assistant service to control a device."""
        try:
            # Get the friendly name of the entity
            state = self.hass.states.get(entity_id)
            friendly_name = state.attributes.get("friendly_name", entity_id) if state else entity_id
            
            # Log the technical details
            _LOGGER.debug("Calling service %s.%s on %s with data %s", domain, service, entity_id, service_data)
            
            # Call the service
            await self.hass.services.async_call(
                domain,
                service,
                {"entity_id": entity_id, **service_data},
                blocking=True,
            )
            
            # Generate a user-friendly response based on the domain and service
            if domain == "light":
                if service == "turn_on":
                    return True, f"Я включил свет {friendly_name}."
                elif service == "turn_off":
                    return True, f"Я выключил свет {friendly_name}."
            elif domain == "switch":
                if service == "turn_on":
                    return True, f"Я включил {friendly_name}."
                elif service == "turn_off":
                    return True, f"Я выключил {friendly_name}."
            elif domain == "climate":
                if service == "turn_on":
                    return True, f"Я включил {friendly_name}."
                elif service == "turn_off":
                    return True, f"Я выключил {friendly_name}."
                elif service == "set_temperature":
                    temp = service_data.get("temperature", "")
                    return True, f"Я установил температуру {friendly_name} на {temp}°C."
                elif service == "set_hvac_mode":
                    mode = service_data.get("hvac_mode", "")
                    if mode == "heat":
                        return True, f"Я включил режим обогрева на {friendly_name}."
                    elif mode == "cool":
                        return True, f"Я включил режим охлаждения на {friendly_name}."
                    else:
                        return True, f"Я изменил режим {friendly_name} на {mode}."
            
            # Default response if no specific message is defined
            return True, f"Я выполнил вашу команду для {friendly_name}."
        except Exception as err:
            _LOGGER.error("Error calling service %s.%s: %s", domain, service, err)
            return False, f"Извините, я не смог выполнить команду для {entity_id}: {err}"
            
    async def _process_command(self, text: str) -> tuple[bool, str]:
        """Process a command to control a device."""
        text = text.lower()
        _LOGGER.debug("Processing command: %s", text)
        
        # Check for common device control patterns
        if "turn on" in text or "включи" in text or "включить" in text:
            # First check for climate devices (air conditioners, heaters)
            if "air conditioner" in text or "ac" in text or "кондиционер" in text:
                for entity_id in self.hass.states.async_entity_ids("climate"):
                    entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                    _LOGGER.debug("Checking climate entity: %s (%s)", entity_id, entity_name)
                    if entity_name and (entity_name in text or entity_id.lower() in text or "air conditioner" in text or "кондиционер" in text):
                        _LOGGER.debug("Found matching climate entity: %s", entity_id)
                        # Turn on climate device
                        return await self._call_service("climate", "turn_on", entity_id)
            
            # Check for other devices
            for domain in ["light", "switch", "fan", "media_player"]:
                entities = self.hass.states.async_entity_ids(domain)
                for entity_id in entities:
                    entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                    _LOGGER.debug("Checking %s entity: %s (%s)", domain, entity_id, entity_name)
                    if entity_name and (entity_name in text or entity_id.lower() in text):
                        _LOGGER.debug("Found matching %s entity: %s", domain, entity_id)
                        return await self._call_service(domain, "turn_on", entity_id)
        
        elif "turn off" in text or "выключи" in text or "выключить" in text:
            # First check for climate devices
            if "air conditioner" in text or "ac" in text or "кондиционер" in text:
                for entity_id in self.hass.states.async_entity_ids("climate"):
                    entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                    if entity_name and (entity_name in text or entity_id.lower() in text or "air conditioner" in text or "кондиционер" in text):
                        return await self._call_service("climate", "turn_off", entity_id)
            
            # Check for other devices
            for domain in ["light", "switch", "fan", "media_player"]:
                entities = self.hass.states.async_entity_ids(domain)
                for entity_id in entities:
                    entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                    if entity_name and (entity_name in text or entity_id.lower() in text):
                        return await self._call_service(domain, "turn_off", entity_id)
        
        # Climate control - set temperature
        elif "set temperature" in text or "установи температуру" in text:
            for entity_id in self.hass.states.async_entity_ids("climate"):
                entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                if entity_name and (entity_name in text or entity_id.lower() in text or "air conditioner" in text or "кондиционер" in text):
                    # Try to extract temperature
                    import re
                    temp_match = re.search(r'(\d+)(?:\s*degrees|\s*°C|\s*градусов)?', text)
                    if temp_match:
                        temperature = int(temp_match.group(1))
                        return await self._call_service("climate", "set_temperature", entity_id, temperature=temperature)
        
        # Climate control - set mode
        elif "heat" in text or "обогрев" in text:
            for entity_id in self.hass.states.async_entity_ids("climate"):
                entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                if entity_name and (entity_name in text or entity_id.lower() in text or "air conditioner" in text or "кондиционер" in text):
                    return await self._call_service("climate", "set_hvac_mode", entity_id, hvac_mode="heat")
        
        elif "cool" in text or "охлаждение" in text:
            for entity_id in self.hass.states.async_entity_ids("climate"):
                entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "").lower()
                if entity_name and (entity_name in text or entity_id.lower() in text or "air conditioner" in text or "кондиционер" in text):
                    return await self._call_service("climate", "set_hvac_mode", entity_id, hvac_mode="cool")
        
        # Add more command patterns as needed
        
        # Log all available entities for debugging
        _LOGGER.debug("Available entities:")
        for domain in ["climate", "light", "switch", "fan", "media_player"]:
            entities = self.hass.states.async_entity_ids(domain)
            for entity_id in entities:
                entity_name = self.hass.states.get(entity_id).attributes.get("friendly_name", "")
                _LOGGER.debug("  %s: %s (%s)", domain, entity_id, entity_name)
        
        return False, "I couldn't understand the command or find a matching device to control."

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL
        
    def _get_system_prompt(self, language: str) -> str:
        """Get the system prompt for the specified language."""
        # Use the template prompt, but add language-specific instructions
        prompt = self._async_generate_prompt()
        
        # Add information about Home Assistant entities
        entity_states = self._get_entity_states()
        prompt += "\n\nHere is information about the current state of Home Assistant entities:\n" + entity_states
        prompt += "\n\nYou can use this information to answer questions about the state of devices and sensors."
        prompt += "\n\nYou can also control devices by responding to user commands like 'turn on the light', 'turn off the air conditioner', or 'set temperature to 22 degrees'. When you receive such commands, you will attempt to control the devices directly and respond in a natural, conversational way about what you did."
        
        # Add language-specific instructions
        if language and language.lower().startswith("ru"):
            prompt += "\n\nПожалуйста, отвечайте на русском языке, когда пользователь задает вопросы на русском."
            prompt += "\n\nВы можете использовать информацию о состоянии устройств Home Assistant, приведенную выше, чтобы отвечать на вопросы пользователя."
            prompt += "\n\nВы также можете управлять устройствами, отвечая на команды пользователя, такие как 'включи свет', 'выключи кондиционер' или 'установи температуру на 22 градуса'. Когда вы получаете такие команды, вы будете пытаться управлять устройствами напрямую и отвечать естественным, разговорным языком о том, что вы сделали."
        elif language and language.lower().startswith("en"):
            prompt += "\n\nPlease respond in English when the user asks questions in English."
            prompt += "\n\nYou can use the Home Assistant entity state information provided above to answer the user's questions."
            prompt += "\n\nYou can also control devices by responding to user commands like 'turn on the light', 'turn off the air conditioner', or 'set temperature to 22 degrees'. When you receive such commands, you will attempt to control the devices directly and respond in a natural, conversational way about what you did."
            
        return prompt

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
                # Get language-specific prompt
                prompt = self._get_system_prompt(user_input.language)
            except template.TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                # Create a response with the correct structure
                return ha_conversation.ConversationResult(
                    conversation_id=conversation_id,
                    response=ConversationResponse(
                        speech={"plain": {"speech": f"Sorry, I had a problem processing your request: {err}", "extra_data": None}},
                        intent=intent.IntentResponse(language=user_input.language),
                    ),
                )

            # Store the system prompt separately
            system_prompt = prompt
            
            # Initialize messages without the system prompt
            messages = []
            self.history[conversation_id] = messages
            
            # Log the conversation start
            _LOGGER.debug(
                "Starting conversation %s with language %s and text: %s",
                conversation_id,
                user_input.language,
                user_input.text,
            )

        # Check if this is a command to control a device
        if self.control_ha:
            success, message = await self._process_command(user_input.text)
            if success:
                # If the command was successful, add it to the conversation history
                messages.append({"role": "user", "content": user_input.text})
                messages.append({"role": "assistant", "content": message})
                self.history[conversation_id] = messages
                
                return ha_conversation.ConversationResult(
                    conversation_id=conversation_id,
                    response=ConversationResponse(
                        speech={"plain": {"speech": message, "extra_data": None}},
                        intent=intent.IntentResponse(language=user_input.language),
                    ),
                )

        # If not a command or command failed, proceed with normal conversation
        messages.append({"role": "user", "content": user_input.text})

        model = self.model
        max_tokens = self.max_tokens
        temperature = self.temperature

        if self.recommended_settings:
            model = AnthropicModels.default()
            max_tokens = DEFAULT_MAX_TOKENS
            temperature = DEFAULT_TEMPERATURE

        try:
            # Log the API request
            _LOGGER.debug(
                "Calling Anthropic API with model=%s, max_tokens=%s, temperature=%s, system=%s, messages=%s",
                model,
                max_tokens,
                temperature,
                system_prompt,
                messages,
            )
            
            # Call the create method directly with parameters, including the system parameter
            # Our wrapper class handles the API call
            response = await self.hass.async_add_executor_job(
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                )
            )
            
            # Log the API response
            _LOGGER.debug("Received response from Anthropic API: %s", response)
        except APIError as err:
            _LOGGER.error("Error calling Anthropic API: %s", err)
            return ha_conversation.ConversationResult(
                conversation_id=conversation_id,
                response=ConversationResponse(
                    speech={"plain": {"speech": f"Sorry, I had a problem processing your request: {err}", "extra_data": None}},
                    intent=intent.IntentResponse(language=user_input.language),
                ),
            )

        # Safely extract the response text
        response_text = ""
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                if hasattr(response.content[0], "text"):
                    response_text = response.content[0].text
        
        # If we couldn't extract text, provide a fallback
        if not response_text:
            response_text = "I'm sorry, I couldn't generate a proper response."
            _LOGGER.warning("Could not extract text from Anthropic response")
        
        # Add the assistant's response to the conversation history
        messages.append(
            {"role": "assistant", "content": response_text}
        )
        
        # Make sure to store the updated history for this conversation ID
        self.history[conversation_id] = messages
        
        # Log the conversation history for debugging
        _LOGGER.debug(
            "Conversation history for %s: %s",
            conversation_id,
            self.history[conversation_id]
        )

        return ha_conversation.ConversationResult(
            conversation_id=conversation_id,
            response=ConversationResponse(
                speech={"plain": {"speech": response_text, "extra_data": None}},
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