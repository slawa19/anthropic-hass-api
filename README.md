# Anthropic-hass-api

A Home Assistant custom component that integrates Anthropic Claude with Home Assistant, allowing you to use Claude as a voice assistant with your API key.

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Version](https://img.shields.io/badge/Version-1.3.0-brightgreen.svg)](https://github.com/slawa19/anthropic-hass-api/releases/tag/1.3.0)

## What's New in Version 1.3.0

- **Updated Model Catalog**: Support for the latest Claude 4.6 and Claude 4.5 model families (Opus, Sonnet, Haiku). Default model is now Claude Sonnet 4.5
- **TTS-Optimized Prompts**: Completely rewritten system prompts optimized for text-to-speech synthesis — responses are ready to be read aloud without post-processing
- **Dual Response Modes**: Default brief mode for quick answers, with a detailed explanation mode activated on explicit request
- **Locale-Specific Prompts**: English and Russian prompts are now independently tailored — Russian includes transliteration rules and "градусов тепла/мороза" phrasing, English uses natural conventions ("degrees", "minus")
- **Anthropic API Compliance**: Fixed tool format (removed invalid `type` field) and message structure (combined content array for assistant messages with tool calls)
- **Improved Error Handling**: Added exception handling for unexpected API errors with specific error messages and fallback response processing

## Features

- **Multilingual Support**: Automatically adapts to your Home Assistant interface language (English and Russian supported)
- **Use Anthropic Claude as a conversation agent in Home Assistant**
- **Control your Home Assistant devices using natural language**
- **Configure Claude's behavior with custom instructions**
- **Choose from different Claude models** (Claude 4.6 Opus, Claude 4.5 Opus/Sonnet/Haiku, Claude 4 Sonnet, Claude 3.5 Haiku)
- **Adjust parameters like temperature and max tokens**
- **Direct API Integration**: Uses direct HTTP requests to the Anthropic API for better compatibility

## Installation

### HACS (Home Assistant Community Store)

1. Make sure you have [HACS](https://hacs.xyz/) installed
2. Go to HACS > Integrations
3. Click the three dots in the top right corner and select "Custom repositories"
4. Add this repository URL: `https://github.com/slawa19/anthropic-hass-api`
5. Select "Integration" as the category
6. Click "ADD"
7. Search for "Anthropic" in the integrations tab
8. Click "Install"
9. Restart Home Assistant

### Manual Installation

1. Download or clone this repository
2. Copy the `custom_components/anthropic` directory to your Home Assistant `custom_components` directory
3. Restart Home Assistant

## Configuration

1. Go to Settings > Devices & Services
2. Click "Add Integration"
3. Search for "Anthropic"
4. Follow the configuration steps:
   - Enter your Anthropic API key (get one at https://console.anthropic.com/settings/keys)
   - Configure options like instructions, model, and parameters

## API Key

This integration requires an Anthropic API key. To get one:

1. Sign up or log in at [Anthropic Console](https://console.anthropic.com/)
2. Enable billing with a valid credit card on the [plans page](https://console.anthropic.com/settings/plans)
3. Visit the [API Keys page](https://console.anthropic.com/settings/keys) to generate an API key

**Note:** This is a paid service. Monitor your costs in the [Anthropic portal](https://console.anthropic.com/settings/cost).

## Legal Notice

This integration uses the official Anthropic API. Please note that the Anthropic API is intended for B2B and not for individual use, [more info here](https://support.anthropic.com/en/articles/8987200-can-i-use-the-claude-api-for-individual-use). Therefore, this integration is intended for commercial uses.

## Usage

Once configured, you can interact with Claude through:

- The conversation panel in Home Assistant
- Voice assistants that use Home Assistant's conversation agent
- The conversation.process service

## Exposed Entities

Claude can only control or provide information about entities that are exposed to it. To manage which entities Claude can access:

1. Go to Settings > Voice Assistants
2. Select the "Exposed Entities" tab
3. Toggle entities you want Claude to be able to control

## Configuration Options

- **Custom Instructions for Claude**: Instructions for the AI on how it should respond to your requests
- **Allow Claude to Control Home Assistant**: Enable/disable Claude's ability to control your devices
- **Use Recommended Model & Settings**: Quickly set up with optimal defaults or customize advanced settings
- **Claude Model**: Choose between different Claude models (when recommended settings are disabled)
- **Maximum Response Length**: Control how verbose Claude's responses are (when recommended settings are disabled)
- **Response Creativity**: Adjust the balance between focused/deterministic and creative/varied responses (when recommended settings are disabled)

## Multilingual Support

The integration automatically adapts to your Home Assistant interface language:
- If your Home Assistant interface is set to English, the integration will display in English
- If your Home Assistant interface is set to Russian, the integration will display in Russian
- For other languages, English will be used as the default

## Default Prompt Architecture

The integration uses a layered prompt system that combines a base system prompt with dynamic entity state information. The final system prompt sent to the Claude API is assembled at runtime from multiple sources.

### Prompt Components

The system prompt is built from three distinct parts:

1. **Base Prompt** (`base_prompt`) — The core instructions that define Claude's persona, TTS rules, response modes, and Home Assistant interaction guidelines. It is a Jinja2 template that conditionally includes device-control instructions based on the `control_ha` setting.

2. **Entity Info Header** (`entity_info`) — A short label that introduces the entity state data block (e.g., *"Here is information about the current state of Home Assistant entities:"*).

3. **Data Instructions** (`data_instructions`) — Rules that tell Claude how to use the provided entity data when answering questions (e.g., always use the most current data, state when information is missing).

### Where Prompts Are Defined

| Source | Purpose |
|--------|---------|
| `const.py` → `DEFAULT_PROMPT` | Hardcoded Russian fallback prompt. Used when translations fail to load or as the default value in the config UI. |
| `translations/en.json` → `system_prompts` | English locale prompts (`base_prompt`, `entity_info`, `data_instructions`). |
| `translations/ru.json` → `system_prompts` | Russian locale prompts with locale-specific rules (transliteration, "градусов тепла/мороза" — degrees of warmth/frost — phrasing). |
| Config entry options → `CONF_PROMPT` | User-provided custom prompt set through the integration options UI. |

### Prompt Resolution Flow

When a conversation starts, the agent resolves the system prompt in `_get_system_prompt(language)`:

```
1. Is a custom prompt configured (differs from DEFAULT_PROMPT)?
   ├─ YES → Use the custom prompt as-is (skip translations entirely)
   └─ NO  → Continue to step 2

2. Load translations for the user's HA language (e.g., "en" or "ru")
   ├─ Found "system_prompts.base_prompt" → Use it as the template
   └─ Not found → Fall back to English translations
       └─ English also missing → Fall back to DEFAULT_PROMPT (Russian)

3. Render the base_prompt Jinja2 template with { control_ha: true/false }
   └─ When control_ha is true, device-control instructions are included
   └─ When control_ha is false, those sections are omitted

4. Load entity_info and data_instructions from the same translations

5. Build the entity state block from currently exposed HA entities
   (grouped by domain: lights, switches, sensors, etc.)

6. Assemble the final prompt:
   ┌─────────────────────────────────┐
   │ Rendered base_prompt            │
   │                                 │
   │ {entity_info header}            │
   │ {entity states by domain}       │
   │                                 │
   │ {data_instructions}             │
   └─────────────────────────────────┘
```

### The `control_ha` Variable

The base prompt template uses a Jinja2 conditional `{%- if control_ha %}...{%- endif %}` to include or exclude device-control instructions. The value of `control_ha` is derived from the **LLM API** dropdown in the options UI:

- **"Assist" selected** → `control_ha = true` — Claude receives tool-use instructions, TTS formatting rules, and response mode guidelines.
- **"No control" selected** → `control_ha = false` — Claude acts as a general assistant without device-control capabilities.

For backward compatibility, if no LLM API option is set (legacy installs), the old `control_ha` boolean config value is used as a fallback.

### Custom Prompt Behavior

Users can provide a custom prompt via **Settings → Devices & Services → Anthropic → Configure → Custom Instructions for Claude**. When a custom prompt is set:

- The translation-based prompt loading is **completely bypassed**.
- The custom prompt is used as-is without the `entity_info` and `data_instructions` sections being appended.
- The custom prompt field supports Jinja2 templates, so users can include `{%- if control_ha %}` blocks if needed.

To restore the default behavior, clear the custom prompt field or reset it to the default value shown in the UI.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.