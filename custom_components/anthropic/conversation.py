"""Support for Anthropic conversation."""
from __future__ import annotations

import logging
import json
from typing import Literal, Dict, List, Any, Optional, cast

try:
    from voluptuous_openapi import convert
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False
    
    def convert(schema, custom_serializer=None):
        """Fallback convert function when voluptuous_openapi is not available."""
        if hasattr(schema, "to_dict"):
            return schema.to_dict()
        if hasattr(schema, "__dict__"):
            return schema.__dict__
        return {"type": "object", "properties": {}}

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
from homeassistant.const import ATTR_ENTITY_ID, MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, template, llm
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
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


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
            # Include more domains to provide more comprehensive information
            domain = state.entity_id.split('.')[0]
            if domain in [
                'sensor', 'binary_sensor', 'light', 'switch', 'climate',
                'media_player', 'fan', 'cover', 'lock', 'alarm_control_panel',
                'vacuum', 'camera', 'scene', 'script', 'automation'
            ]:
                exposed_entities.append(state)
        
        if not exposed_entities:
            return "No entities are currently exposed to the conversation agent."
        
        # Format entity information with more details
        entity_info = []
        
        # Group entities by domain for better organization
        entities_by_domain = {}
        for state in exposed_entities:
            domain = state.entity_id.split('.')[0]
            if domain not in entities_by_domain:
                entities_by_domain[domain] = []
            entities_by_domain[domain].append(state)
        
        # Format each domain's entities
        for domain, entities in entities_by_domain.items():
            entity_info.append(f"\n{domain.upper()} DEVICES:")
            
            for state in entities:
                entity_id = state.entity_id
                friendly_name = state.attributes.get('friendly_name', entity_id)
                current_state = state.state
                
                # Add more details based on domain
                details = []
                
                # Add common attributes if available
                if 'unit_of_measurement' in state.attributes:
                    details.append(f"{current_state} {state.attributes['unit_of_measurement']}")
                elif domain == 'climate':
                    if 'current_temperature' in state.attributes:
                        details.append(f"Current: {state.attributes['current_temperature']}°")
                    if 'temperature' in state.attributes:
                        details.append(f"Set to: {state.attributes['temperature']}°")
                    if 'hvac_action' in state.attributes:
                        details.append(f"Action: {state.attributes['hvac_action']}")
                elif domain in ['light', 'fan']:
                    if 'brightness' in state.attributes:
                        # Convert brightness (0-255) to percentage
                        brightness_pct = round(state.attributes['brightness'] / 255 * 100)
                        details.append(f"Brightness: {brightness_pct}%")
                
                # Format the line
                if details:
                    entity_info.append(f"  {friendly_name} ({entity_id}): {current_state} ({', '.join(details)})")
                else:
                    entity_info.append(f"  {friendly_name} ({entity_id}): {current_state}")
        
        return "\n".join(entity_info)
        
    async def _call_service(self, domain, service, entity_id, **service_data):
        """Call a Home Assistant service to control a device."""
        try:
            # Get the friendly name of the entity
            state = self.hass.states.get(entity_id)
            if not state:
                _LOGGER.warning("Entity %s not found", entity_id)
                return False, f"Entity {entity_id} not found"
                
            friendly_name = state.attributes.get("friendly_name", entity_id)
            current_state = state.state
            
            # Log the technical details
            _LOGGER.info("Calling service %s.%s on %s (current state: %s) with data %s", 
                        domain, service, entity_id, current_state, service_data)
            
            # Call the service
            try:
                await self.hass.services.async_call(
                    domain,
                    service,
                    {"entity_id": entity_id, **service_data},
                    blocking=True,
                )
                _LOGGER.info("Service call successful for %s.%s on %s", domain, service, entity_id)
                return True, f"Successfully called {domain}.{service} on {friendly_name}"
            except Exception as service_err:
                _LOGGER.error("Service call failed for %s.%s on %s: %s", domain, service, entity_id, service_err)
                return False, f"Service call failed: {service_err}"
        except Exception as err:
            _LOGGER.error("Error calling service %s.%s: %s", domain, service, err)
            return False, f"Error: {err}"
            
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
        
        return prompt

    async def async_process(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process a sentence."""
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.APIInstance | None = None
        
        # Check if Home Assistant control is enabled and LLM_HASS_API is set
        # control_ha is the option that determines if the model can control devices
        # CONF_LLM_HASS_API is the specific API that will be used (usually "assist")
        # If "No control" is selected, options.get(CONF_LLM_HASS_API) will be None
        if self.control_ha and options.get(CONF_LLM_HASS_API):
            try:
                # Create context for LLM API with information about the user's request
                llm_context = llm.LLMContext(
                    platform=DOMAIN,  # Integration name (anthropic)
                    context=user_input.context,  # Conversation context
                    user_prompt=user_input.text,  # User request text
                    language=user_input.language,  # Request language
                    assistant=ha_conversation.DOMAIN,  # Assistant domain (conversation)
                    device_id=user_input.device_id,  # Device ID from which the request came
                )
                
                # Get API instance for interacting with Home Assistant
                # This will allow the model to get information about device states and control them
                llm_api = await llm.async_get_api(
                    self.hass,
                    options[CONF_LLM_HASS_API],  # API ID (usually "assist")
                    llm_context,
                )
            except HomeAssistantError as err:
                _LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return ha_conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
        
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
                        intent=intent_response,
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

        # Add the user's message to the conversation history
        # Format the content as a list with a text object, as required by the Anthropic API
        # For Anthropic API v2, each message must have content as an array of content blocks
        # See: https://docs.anthropic.com/claude/reference/messages-create
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input.text}]
        })

        model = self.model
        max_tokens = self.max_tokens
        temperature = self.temperature

        if self.recommended_settings:
            model = AnthropicModels.default()
            max_tokens = DEFAULT_MAX_TOKENS
            temperature = DEFAULT_TEMPERATURE

        # Prepare tools for the model if LLM API is available
        # Tools allow the model to interact with Home Assistant:
        # - Get information about device states
        # - Control devices (turn on/off lights, adjust thermostats, etc.)
        # - Perform other actions in Home Assistant
        tools = None
        if llm_api and llm_api.tools:
            tools = []
            for tool in llm_api.tools:
                # Each tool has a name, description, and input parameter schema
                # For example, the "turn_on" tool can turn on a light and requires an "entity_id" parameter
                try:
                    # Use convert function to make the schema JSON serializable
                    # This is the same approach used by the OpenAI integration
                    parameters = convert(tool.parameters, custom_serializer=llm_api.custom_serializer if hasattr(llm_api, "custom_serializer") else None)
                    
                    # Format tools according to Anthropic API requirements
                    # https://docs.anthropic.com/claude/reference/messages-create
                    tools.append({
                        "name": tool.name,  # Tool name (e.g., "turn_on", "get_state")
                        "description": tool.description or "",  # Tool description
                        "input_schema": parameters,  # Input parameter schema (JSON Schema)
                        "type": "custom"  # Use "custom" as the tool type
                    })
                except Exception as err:
                    _LOGGER.error("Error processing tool %s: %s", tool.name, err)

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                # Update system prompt with current entity states
                # This ensures the model has the latest information about device states
                updated_system_prompt = self._get_system_prompt(user_input.language)
                
                # Log the API request
                _LOGGER.debug(
                    "Calling Anthropic API with model=%s, max_tokens=%s, temperature=%s, system=%s, messages=%s, tools=%s",
                    model,
                    max_tokens,
                    temperature,
                    updated_system_prompt,
                    messages,
                    tools,
                )
                
                # Prepare API call parameters
                api_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": updated_system_prompt,  # Use updated system prompt with current entity states
                    "messages": messages,
                }
                
                # Only add tools if they are available
                if tools:
                    api_params["tools"] = tools
                
                # Call the create method with the prepared parameters
                response = await self.client.messages.create(**api_params)
                
                # Log the API response
                _LOGGER.debug("Received response from Anthropic API: %s", response)
            except APIError as err:
                _LOGGER.error("Error calling Anthropic API: %s", err)
                return ha_conversation.ConversationResult(
                    conversation_id=conversation_id,
                    response=ConversationResponse(
                        speech={"plain": {"speech": f"Sorry, I had a problem processing your request: {err}", "extra_data": None}},
                        intent=intent_response,
                    ),
                )

            # Safely extract the response text and tool calls
            response_text = ""
            tool_calls = []
            
            # Handle different response formats
            if hasattr(response, "content") and response.content:
                # Try to extract content from structured response
                for content_item in response.content:
                    if hasattr(content_item, "type"):
                        if content_item.type == "text":
                            response_text = content_item.text
                        elif content_item.type == "tool_use":
                            # Store the original tool_use for later reference
                            # We need to keep the original ID to match with tool_result
                            tool_calls.append({
                                "id": content_item.id,  # This will be used as tool_use_id in the response
                                "name": content_item.name,
                                "input": content_item.input,
                                "type": "custom"  # Use "custom" as the tool type
                            })
                    elif isinstance(content_item, dict):
                        if content_item.get("type") == "text":
                            response_text = content_item.get("text", "")
                        elif content_item.get("type") == "tool_use":
                            # Store the original tool_use for later reference
                            # We need to keep the original ID to match with tool_result
                            tool_calls.append({
                                "id": content_item.get("id", ""),
                                "name": content_item.get("name", ""),
                                "input": content_item.get("input", {}),
                                "type": "custom"  # Use "custom" as the tool type
                            })
            
            # If we couldn't extract text, provide a fallback
            if not response_text and not tool_calls:
                response_text = "I'm sorry, I couldn't generate a proper response."
                _LOGGER.warning("Could not extract text from Anthropic response: %s", response)
            
            # Add the assistant's response to the conversation history
            # Format the content as a list with a text object, as required by the Anthropic API
            # For Anthropic API v2, each message must have content as an array of content blocks
            # See: https://docs.anthropic.com/claude/reference/messages-create
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            })
            
            # If the model decided to use tools to fulfill the user's request,
            # process these tool calls
            if tool_calls and llm_api:
                _LOGGER.info("Processing %d tool calls", len(tool_calls))
                tool_results = []
                
                # Process each tool call in sequence
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]  # Tool name (e.g., "turn_on")
                    tool_args = tool_call["input"]  # Tool arguments (e.g., {"entity_id": "light.living_room"})
                    tool_id = tool_call["id"]  # Unique ID of the tool call
                    
                    # Log the tool call for debugging
                    _LOGGER.info("Calling tool %s with args %s", tool_name, tool_args)
                    
                    try:
                        # Make sure tool_args is a dictionary
                        if not isinstance(tool_args, dict):
                            if isinstance(tool_args, str):
                                try:
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    tool_args = {"entity_id": tool_args}  # Try using as entity_id
                            else:
                                tool_args = {"entity_id": str(tool_args)}  # Try using as entity_id
                        
                        # Validate required parameters for common tools
                        if tool_name == "HassTurnOn" or tool_name == "HassTurnOff":
                            if "entity_id" not in tool_args:
                                # If entity_id is missing, try to find it in other fields
                                if "domain" in tool_args and "device" in tool_args:
                                    # Try to construct entity_id from domain and device
                                    tool_args["entity_id"] = f"{tool_args['domain']}.{tool_args['device']}"
                                elif len(tool_args) == 1:
                                    # If there's only one parameter, try using it as entity_id
                                    key, value = next(iter(tool_args.items()))
                                    tool_args = {"entity_id": value}
                            
                            # Check if entity_id is a wildcard or contains multiple entities
                            if "entity_id" in tool_args and (
                                tool_args["entity_id"] == "all" or
                                isinstance(tool_args["entity_id"], list) or
                                "," in str(tool_args["entity_id"])
                            ):
                                # Cannot target all devices
                                _LOGGER.warning("Cannot target all devices or multiple entities at once")
                                tool_response = {"error": "Service handler cannot target all devices"}
                                continue
                        
                        _LOGGER.debug("Processed tool args: %s", tool_args)
                        
                        tool_input = llm.ToolInput(
                            tool_name=tool_name,
                            tool_args=cast(dict[str, Any], tool_args),
                        )
                        tool_response = await llm_api.async_call_tool(tool_input)
                        _LOGGER.info("Tool response: %s", tool_response)
                    except Exception as err:
                        _LOGGER.error("Error calling tool %s: %s", tool_name, err)
                        tool_response = {"error": str(err)}
                    
                    # Format the tool result according to the API expectations
                    tool_result = {
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "content": json.dumps(tool_response),
                    }
                    tool_results.append(tool_result)
                
                # Add tool results to the conversation history
                if tool_results:
                    # We need to check if there are corresponding tool_use blocks in the previous message
                    # According to the error, each tool_result must have a corresponding tool_use in the previous message
                    
                    # Get the previous message (assistant's message)
                    if len(messages) >= 1 and messages[-1]["role"] == "assistant":
                        assistant_message = messages[-1]
                        # Check if the assistant's message has tool_use blocks
                        has_tool_use = False
                        tool_use_ids = []
                        
                        if "content" in assistant_message and isinstance(assistant_message["content"], list):
                            for content_item in assistant_message["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "tool_use":
                                    has_tool_use = True
                                    if "id" in content_item:
                                        tool_use_ids.append(content_item["id"])
                        
                        if has_tool_use:
                            # Format tool results according to Anthropic API requirements
                            formatted_content = []
                            for result in tool_results:
                                # Only include results for tools that were actually called
                                if result["tool_call_id"] in tool_use_ids:
                                    # The content must be a string, not an object
                                    content_str = result["content"]
                                    if not isinstance(content_str, str):
                                        # If it's not already a string, convert it to a JSON string
                                        content_str = json.dumps(content_str)
                                        
                                    formatted_content.append({
                                        "type": "tool_result",
                                        "tool_use_id": result["tool_call_id"],
                                        "content": content_str
                                    })
                            
                            if formatted_content:
                                # Only add the message if we have valid tool results
                                messages.append({"role": "user", "content": formatted_content})
                                _LOGGER.debug("Added tool results to conversation: %s", formatted_content)
                                # Continue the conversation to get the final response
                                continue
                        else:
                            _LOGGER.warning("Cannot add tool results because there are no tool_use blocks in the previous message")
                            # Don't continue the loop, just break
                            break
                    else:
                        _LOGGER.warning("Cannot add tool results because there is no assistant message")
                        # Don't continue the loop, just break
                        break
                    
                    # Add tool results as a user message
                    messages.append({"role": "user", "content": formatted_content})
                    _LOGGER.debug("Added tool results to conversation: %s", formatted_content)
                    # Continue the conversation to get the final response
                    continue
            
            # If no tool calls or we've processed them all, we're done
            break
        
        # Make sure to store the updated history for this conversation ID
        self.history[conversation_id] = messages
        
        # Log the conversation history for debugging
        _LOGGER.debug(
            "Conversation history for %s: %s",
            conversation_id,
            self.history[conversation_id]
        )

        # Set the response text
        intent_response.async_set_speech(response_text)
        
        return ha_conversation.ConversationResult(
            conversation_id=conversation_id,
            response=intent_response,
        )

    def _async_generate_prompt(self) -> str:
        """Generate a prompt for the user."""
        return template.Template(self.prompt, self.hass).async_render(
            {
                "control_ha": self.control_ha,
            },
            parse_result=False,
        )
