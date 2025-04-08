# Triton AI Integration API Reference

This document provides a detailed reference for the Triton AI Integration API, including services, events, and entity attributes.

## Services

### triton_ai.run_inference

Run inference using a Triton model with custom inputs.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model_name | string | Yes | Name of the model to use |
| inputs | object | Yes | Dictionary of input name to values |
| outputs | list | No | Optional list of output names to request |
| parameters | object | No | Optional parameters for the model |

#### Example

```yaml
service: triton_ai.run_inference
data:
  model_name: "llama2-7b"
  inputs:
    prompt: "Generate a poem about smart homes"
  parameters:
    max_tokens: 100
    temperature: 0.7
```

### triton_ai.analyze_sensor

Analyze a sensor for anomalies and generate predictions.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| entity_id | string | Yes | Entity ID of the sensor to analyze |

#### Example

```yaml
service: triton_ai.analyze_sensor
data:
  entity_id: sensor.living_room_temperature
```

### triton_ai.generate_text

Generate text using the LLM model.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| prompt | string | Yes | Text prompt for generation |
| model_name | string | No | Model name (default: from config) |
| max_tokens | integer | No | Maximum tokens to generate |
| temperature | float | No | Temperature for sampling (0-1) |

#### Example

```yaml
service: triton_ai.generate_text
data:
  prompt: "Write a welcome message for my smart home"
  max_tokens: 150
  temperature: 0.8
```

### triton_ai.analyze_image

Analyze an image using computer vision models.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_path | string | Yes | Path to the image file |
| model_name | string | No | Model name (default: from config) |
| task | string | No | Task type: "detection" or "classification" |

#### Example

```yaml
service: triton_ai.analyze_image
data:
  image_path: "/config/www/camera_snapshot.jpg"
  model_name: "yolov5"
  task: "detection"
```

### triton_ai.process_voice_command

Process an audio file as a voice command.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| audio_path | string | Yes | Path to the audio file |
| language | string | No | Language code (default: "en") |

#### Example

```yaml
service: triton_ai.process_voice_command
data:
  audio_path: "/config/www/voice_command.wav"
  language: "en"
```

## Events

### triton_ai_inference_result

Fired when an inference request completes.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| model_name | string | Name of the model used |
| status | string | Status of the inference request |
| inference_time | float | Time taken for inference in seconds |
| outputs | object | Dictionary of outputs (if successful) |
| error | string | Error message (if failed) |

### triton_ai_anomaly_detected

Fired when an anomaly is detected in sensor data.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| entity_id | string | Entity ID of the sensor |
| score | float | Anomaly score (0-1) |
| threshold | float | Threshold used for detection |
| description | string | Description of the anomaly |

### triton_ai_prediction_updated

Fired when a new prediction is generated for a sensor.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| entity_id | string | Entity ID of the sensor |
| forecast | list | List of predicted values |
| timestamps | list | List of timestamps for predictions |

### triton_ai_text_generation_result

Fired when text generation completes.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| prompt | string | Original prompt |
| model_name | string | Name of the model used |
| generated_text | string | Generated text |
| status | string | Status of the generation |
| inference_time | float | Time taken for generation |
| error | string | Error message (if failed) |

### triton_ai_image_analysis_result

Fired when image analysis completes.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| image_path | string | Path to the analyzed image |
| model_name | string | Name of the model used |
| task | string | Analysis task (detection/classification) |
| result | object | Analysis results |
| status | string | Status of the analysis |
| inference_time | float | Time taken for analysis |
| error | string | Error message (if failed) |

### triton_ai_voice_command_result

Fired when voice command processing completes.

#### Event Data

| Field | Type | Description |
|-------|------|-------------|
| audio_path | string | Path to the audio file |
| transcribed_text | string | Transcribed text |
| intent | object | Extracted intent and entities |
| status | string | Status of processing |
| stt_time | float | Time taken for speech-to-text |
| nlu_time | float | Time taken for intent extraction |
| error | string | Error message (if failed) |

## Entities

### Sensor Entities

#### Anomaly Score Sensors

For each tracked sensor, an anomaly score sensor is created with the following attributes:

| Attribute | Description |
|-----------|-------------|
| source_entity | Source entity ID |
| threshold | Anomaly detection threshold |
| description | Description of detected anomaly |
| last_updated | Timestamp of last update |

The state represents the anomaly score from 0 to 1.

#### Prediction Sensors

For numeric sensors, prediction sensors are created with the following attributes:

| Attribute | Description |
|-----------|-------------|
| source_entity | Source entity ID |
| forecast | List of predicted values |
| forecast_timestamps | List of timestamps for predictions |
| last_updated | Timestamp of last update |

The state represents the next predicted value.

### Binary Sensor Entities

#### Anomaly Detection Sensors

For each tracked sensor, an anomaly detection binary sensor is created:

| Attribute | Description |
|-----------|-------------|
| source_entity | Source entity ID |
| score | Current anomaly score |
| threshold | Detection threshold |
| description | Description of detected anomaly |
| last_detected | Timestamp of last detection |

The state is ON when an anomaly is detected, OFF otherwise.

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| triton_url | string | Required | URL of the Triton Inference Server |
| ray_address | string | Required | Address of the Ray cluster |
| models.text_generation | string | Required | Name of the text generation model |
| models.image_recognition | string | Required | Name of the image recognition model |
| models.speech_recognition | string | Required | Name of the speech recognition model |
| sensor_analysis_interval | integer | 30 | Interval for sensor analysis (minutes) |
| log_level | string | "info" | Log level (debug/info/warning/error/critical) |

## HTTP API

When using the Home Assistant HTTP API, you can call the integration's services as follows:

```http
POST /api/services/triton_ai/generate_text
Content-Type: application/json
Authorization: Bearer YOUR_LONG_LIVED_ACCESS_TOKEN

{
  "prompt": "Write a welcome message for my smart home",
  "max_tokens": 150,
  "temperature": 0.8
}
```

The result will be available via the events interface or websocket API.
