"""Constants for the TritonAI integration."""

DOMAIN = "triton_ai"

# Configuration
CONF_TRITON_URL = "triton_url"
CONF_RAY_ADDRESS = "ray_address"
CONF_MODELS = "models"
CONF_SENSOR_ANALYSIS_INTERVAL = "sensor_analysis_interval"
CONF_LOG_LEVEL = "log_level"

# Default values
DEFAULT_SENSOR_ANALYSIS_INTERVAL = 30  # minutes
DEFAULT_LOG_LEVEL = "info"

# Platform names
PLATFORMS = ["sensor", "binary_sensor", "switch", "light", "climate"]

# Service names
SERVICE_RUN_INFERENCE = "run_inference"
SERVICE_ANALYZE_SENSOR = "analyze_sensor"
SERVICE_GENERATE_TEXT = "generate_text"
SERVICE_ANALYZE_IMAGE = "analyze_image"
SERVICE_PROCESS_VOICE_COMMAND = "process_voice_command"

# Event names
EVENT_INFERENCE_RESULT = "triton_ai_inference_result"
EVENT_ANOMALY_DETECTED = "triton_ai_anomaly_detected"
EVENT_PREDICTION_UPDATED = "triton_ai_prediction_updated"

# Entity attributes
ATTR_MODEL_NAME = "model_name"
ATTR_INFERENCE_TIME = "inference_time"
ATTR_CONFIDENCE = "confidence"
ATTR_PREDICTION = "prediction"
ATTR_ANOMALY_SCORE = "anomaly_score"
ATTR_ANOMALY_THRESHOLD = "anomaly_threshold"
