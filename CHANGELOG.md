# Changelog

## [1.3.0] - 2026-02-14

### Added
- Support for Claude 4.6 Opus model
- Support for Claude 4.5 models: Opus, Sonnet, Haiku
- TTS-optimized system prompts for speech-ready output
- Dual response modes: brief (default) and detailed explanation on request
- Locale-specific prompt customizations (English and Russian)

### Changed
- Default model changed from Claude Sonnet 4 to Claude Sonnet 4.5
- System prompts rewritten for TTS synthesis — responses ready to be read aloud
- English prompt uses natural conventions ("degrees", "minus", "#")
- Russian prompt uses locale-specific rules (transliteration, "градусов тепла/мороза", "№")

### Fixed
- Removed invalid `type` field from Anthropic API tool definitions
- Fixed assistant messages to combine text and tool_use blocks in single content array
- Fixed English locale prompt to remove Russian-specific transliteration rules
- Improved error handling for unexpected API errors

### Removed
- Deprecated Claude 3.x models (Opus, Sonnet, Haiku, 3.5 Sonnet legacy variants)
- Removed Claude Opus 4 and Claude 3.7 Sonnet

## [1.2.1] - 2025-04-06

### Fixed
- Fixed syntax error in conversation_agent.py that was causing issues with Python 3.13
- Fixed indentation issues in the _async_generate_prompt method
- Improved multilingual support to properly handle language switching
- Enhanced translation system to better handle fallbacks to English

## [1.2.0] - 2025-04-06

### Added
- Enhanced error handling throughout the integration
- Added caching for translations to improve performance
- Added additional logging for better troubleshooting

### Fixed
- Fixed critical issue with brightness attribute processing that could cause conversation failures
- Fixed potential infinite recursion in system prompt generation
- Fixed error handling in translation system
- Improved entity state processing to handle unexpected attribute values

### Changed
- Refactored error handling in entity state processing for better stability
- Improved robustness of the translation system
- Enhanced overall stability with multi-level error handling

## [1.1.0] - 2025-03-15

### Added
- Support for Claude 3.5 Sonnet model
- Improved multilingual support with Russian translations
- Enhanced entity state reporting

### Changed
- Optimized API request handling
- Improved conversation context management

## [1.0.0] - 2025-02-20

### Added
- Initial release
- Support for Claude 3 models (Opus, Sonnet, Haiku)
- Custom instructions configuration
- Home Assistant device control capability
- Adjustable parameters (temperature, max tokens)
- Basic multilingual support