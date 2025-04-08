# Setting Up the Triton AI Integration for Home Assistant

This guide explains how to set up the Triton AI integration for Home Assistant, connecting it to Triton Inference Server and Ray.

## Prerequisites

- Home Assistant instance (Core, OS, or Supervised)
- NVIDIA Triton Inference Server running in your homelab
- Ray cluster running in your homelab
- AI models deployed to Triton Inference Server
  - Text generation model (e.g., llama2-7b)
  - Image recognition model (e.g., yolov5)
  - Speech recognition model (e.g., whisper-medium)

## Installation

### Method 1: Using HACS (Recommended)

1. Make sure you have [HACS](https://hacs.xyz/) installed
2. Add this repository as a custom repository in HACS:
   - Go to HACS > Integrations
   - Click the three dots in the top right corner
   - Select "Custom repositories"
   - Add the repository URL: `https://github.com/yourusername/homelab-ai`
   - Select category: "Integration"
   - Click "Add"
3. Find and install "Triton AI" in HACS
4. Restart Home Assistant

### Method 2: Manual Installation

1. Copy the `custom_components/triton_ai` directory to your Home Assistant configuration directory
   ```bash
   cp -r custom_components/triton_ai /path/to/your/homeassistant/config/custom_components/
   ```
2. Restart Home Assistant

## Configuration

### Using the UI

1. Go to Configuration > Integrations
2. Click the "+" button to add a new integration
3. Search for "Triton AI" and select it
4. Enter the following information:
   - Triton URL: URL of your Triton Inference Server (e.g., `http://triton-inference-server.ai.svc.cluster.local:8000`)
   - Ray Address: Address of your Ray cluster (e.g., `ray://ray-head.ai.svc.cluster.local:10001`)
   - Text generation model: Name of your text generation model (e.g., `llama2-7b`)
   - Image recognition model: Name of your image recognition model (e.g., `yolov5`)
   - Speech recognition model: Name of your speech recognition model (e.g., `whisper-medium`)
   - Sensor analysis interval: How often to analyze sensor data (in minutes)
   - Log level: Logging verbosity

### Using configuration.yaml

You can also configure the integration in your `configuration.yaml`:

```yaml
triton_ai:
  triton_url: http://triton-inference-server.ai.svc.cluster.local:8000
  ray_address: ray://ray-head.ai.svc.cluster.local:10001
  models:
    text_generation: llama2-7b
    image_recognition: yolov5
    speech_recognition: whisper-medium
  sensor_analysis_interval: 30  # minutes
  log_level: info
```

## Available Services

The integration provides the following services:

### triton_ai.run_inference

Run inference on a Triton model with custom inputs.

```yaml
service: triton_ai.run_inference
data:
  model_name: "llama2-7b"
  inputs:
    prompt: "Generate a haiku about smart homes"
  parameters:
    max_tokens: 100
    temperature: 0.7
```

### triton_ai.analyze_sensor

Analyze a specific sensor for anomalies and generate predictions.

```yaml
service: triton_ai.analyze_sensor
data:
  entity_id: sensor.living_room_temperature
```

### triton_ai.generate_text

Generate text using the configured LLM.

```yaml
service: triton_ai.generate_text
data:
  prompt: "Summarize the current home state"
  max_tokens: 200
  temperature: 0.5
```

### triton_ai.analyze_image

Analyze an image using computer vision models.

```yaml
service: triton_ai.analyze_image
data:
  image_path: "/config/www/camera_snapshot.jpg"
  model_name: "yolov5"
  task: "detection"  # or "classification"
```

### triton_ai.process_voice_command

Process an audio file as a voice command.

```yaml
service: triton_ai.process_voice_command
data:
  audio_path: "/config/www/voice_command.wav"
  language: "en"
```

## Events

The integration fires the following events:

### triton_ai_inference_result
Fired when an inference request completes.

### triton_ai_anomaly_detected
Fired when an anomaly is detected in sensor data.

### triton_ai_prediction_updated
Fired when a new prediction is generated for a sensor.

### triton_ai_text_generation_result
Fired when text generation completes.

### triton_ai_image_analysis_result
Fired when image analysis completes.

### triton_ai_voice_command_result
Fired when voice command processing completes.

## Example Automations

See the [example automations](../config/automations/ai_automations.yaml) for ideas on how to use the integration.

## Troubleshooting

If you encounter issues with the integration, try these steps:

1. Check the logs in Configuration > Logs, filtering for "triton_ai"
2. Verify connectivity to the Triton server and Ray cluster
3. Ensure the configured models are available in your Triton server
4. Check resource usage on your Jetson AGX Orin to ensure there's enough GPU memory

For more detailed troubleshooting, see [troubleshooting.md](troubleshooting.md).
