# Home Assistant AI Integration Tests

This directory contains test scripts to validate the integration between Home Assistant and your AI inference architecture. These tests ensure that the end-to-end functionality of voice commands, sensor data analysis, notification generation, and automation triggering works correctly.

## Test Overview

The test suite (`test_integration.py`) validates the following integration points:

1. **Voice Command Processing Pipeline**
   - Wake word detection
   - Speech-to-text conversion
   - Natural language understanding
   - Home Assistant command execution

2. **Sensor Data Analysis**
   - Retrieving sensor data from Home Assistant
   - Processing data with AI models
   - Generating insights and notifications

3. **AI-Driven Notifications**
   - Context-aware notification generation
   - Personalized message creation
   - Priority determination
   - Delivery through Home Assistant

4. **Automation Triggering**
   - AI-based event detection
   - Condition evaluation
   - Action execution
   - State verification

## Prerequisites

Before running the tests, ensure that:

1. Home Assistant is running and accessible
2. The Triton Inference Server is deployed with required models
3. MQTT broker is configured and running
4. Required environment variables are set:
   ```bash
   export HASS_TOKEN="your_long_lived_access_token"
   export HASS_API_PASSWORD="your_api_password"  # If needed
   export MQTT_USERNAME="your_mqtt_username"
   export MQTT_PASSWORD="your_mqtt_password"
   ```

## Required Models

The test suite expects the following models to be available on your Triton Inference Server:

1. **Wake Word Detection** (`porcupine`)
   - Input: Audio data
   - Output: Detection confidence score

2. **Speech-to-Text** (`whisper-base`)
   - Input: Audio data
   - Output: Transcribed text

3. **Natural Language Understanding** (`llama2-7b-chat-q4`)
   - Input: Text query
   - Output: Intent and entities JSON

4. **Text-to-Speech** (`coqui-tts`)
   - Input: Text
   - Output: Audio data

5. **Sensor Analysis Model** (`sensor_analysis`)
   - Input: JSON with sensor readings
   - Output: Analysis results JSON

6. **Notification Generation** (`notification_generator`)
   - Input: Event and context data
   - Output: Enhanced notification JSON

## Test Data

Create a `test_data` directory with the following files:

- `turn_on_lights.wav` - Audio sample saying "Turn on the living room lights"
- `check_temperature.wav` - Audio sample saying "What's the temperature in the living room?"

You can record these samples yourself or use text-to-speech software to generate them.

## Running the Tests

### From the Command Line

Run the test suite using pytest:

```bash
# Navigate to the homelab-ai directory
cd /home/sprime01/homelab/homelab-ai

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run the tests
python -m pytest home-assistant/tests/test_integration.py -v
```

### Test Configuration

The test configuration is defined in `TEST_CONFIG` within the test file. You can override settings by:

1. Editing the `TEST_CONFIG` dictionary in the script
2. Setting environment variables
3. Creating a config.json file and passing it via the command line

### Command Line Options

```bash
python -m pytest home-assistant/tests/test_integration.py -v \
  --html=test-report.html \
  --config-file=my_config.json
```

## Test Results

Test results are saved to:
- Console output with detailed logs
- JSON files with raw test results in the `test_results` directory
- HTML report for visual inspection

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Home Assistant URL and token
   - Check Triton server is accessible
   - Confirm MQTT broker is running

2. **Model Inference Errors**
   - Ensure models are loaded and ready in Triton
   - Check model input/output format matches test expectations
   - Examine Triton server logs for model loading issues

3. **Home Assistant Integration Errors**
   - Verify entity IDs match those in your Home Assistant instance
   - Check permissions for your access token
   - Confirm MQTT topics are properly set up

### Logging

Logs are written to both the console and `ha_integration_test.log`. Increase verbosity by modifying the log level in the script if needed.

## Extending the Tests

### Adding New Test Cases

1. Add new test methods to the `TestHomeAssistantAIIntegration` class
2. Include new test data in the `TEST_CONFIG` dictionary
3. Create helper methods for specialized test operations

### Testing Custom Models

Update the `triton.models` section in `TEST_CONFIG` to reference your custom models:

```python
"triton": {
    "url": "localhost:8000",
    "models": {
        "wake_word": "your_custom_wake_word_model",
        "stt": "your_custom_stt_model",
        "tts": "your_custom_tts_model",
        "nlu": "your_custom_nlu_model"
    }
}
```

## CI/CD Integration

These tests can be integrated into your CI/CD pipeline by:

1. Adding them to your GitHub Actions workflow
2. Setting up necessary secrets for authentication
3. Deploying a test environment with Home Assistant and Triton
4. Running the tests against the deployed environment

Example GitHub Actions step:

```yaml
- name: Run Home Assistant AI Integration Tests
  run: |
    python -m pytest home-assistant/tests/test_integration.py -v --html=test-report.html
  env:
    HASS_TOKEN: ${{ secrets.HASS_TOKEN }}
    MQTT_USERNAME: ${{ secrets.MQTT_USERNAME }}
    MQTT_PASSWORD: ${{ secrets.MQTT_PASSWORD }}
```

## Architecture

