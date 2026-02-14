# Release Notes for v1.3.0

## Overview

This release brings updated Claude model support (4.6/4.5 series), TTS-optimized prompts for speech-ready output, and critical Anthropic API compliance fixes.

## What's New

### Updated Model Catalog
- Added Claude 4.6 Opus (`claude-opus-4-6`)
- Added Claude 4.5 Opus, Sonnet, and Haiku
- Default model is now Claude Sonnet 4.5
- Removed deprecated Claude 3.x models and legacy variants

### TTS-Optimized Prompts
- System prompts completely rewritten for text-to-speech synthesis
- Responses are ready to be read aloud without post-processing
- Symbols, emojis, and special characters are replaced with full words
- Temperature and humidity values formatted for natural speech

### Dual Response Modes
- **Default (Brief)**: Short, result-only answers optimized for voice output
- **Explanation**: Detailed technical report when user explicitly asks for details

### Locale-Specific Prompts
- English prompt uses natural English conventions ("degrees", "minus", `#` for numbers)
- Russian prompt includes transliteration rules and locale-specific phrasing ("градусов тепла/мороза", "№")

### Anthropic API Compliance Fixes
- Removed invalid `type` field from tool definitions
- Fixed assistant messages to combine text and tool_use blocks in a single content array
- Improved type validation — skips validation when using recommended settings

### Error Handling Improvements
- Added exception handling for unexpected API errors
- Added fallback for response processing failures
- Made error messages more specific and actionable

## Upgrading

To upgrade to version 1.3.0:

1. Go to HACS > Integrations
2. Find "Anthropic" in your installed integrations
3. Click on it and select "Upgrade"
4. Restart Home Assistant

## Contributors

- @slawa19

## Feedback

If you encounter any issues or have suggestions for improvements, please [open an issue](https://github.com/slawa19/anthropic-hass-api/issues) on GitHub.