import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
from pydantic import BaseModel, Field, validator
import yaml
import tensorflow as tf
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("input-validation")

class InputValidationConfig(BaseModel):
    """Configuration for input validation per model type."""
    model_name: str
    model_type: str  # text, image, tabular, etc.
    max_input_length: Optional[int] = None
    max_tokens: Optional[int] = None
    max_image_size: Optional[Dict[str, int]] = None  # {width: 1024, height: 1024}
    max_tabular_rows: Optional[int] = None
    allowed_input_keys: List[str] = []
    blocked_terms: List[str] = []
    content_filter_level: str = "medium"  # none, low, medium, high
    detect_adversarial: bool = False
    schema: Optional[Dict[str, Any]] = None
    tokenizer_name: Optional[str] = None

class InputValidator:
    """
    Input validator for AI models served by Triton Inference Server.
    Provides functionality for schema validation, content filtering,
    input sanitization, size/complexity limits, and adversarial example detection.
    """

    def __init__(self, config_path: str):
        """
        Initialize the validator with configurations for different model types.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.configs = self._load_configs(config_path)
        self.model_configs = {}

        # Initialize tokenizers for text models that need them
        self.tokenizers = {}
        for config in self.configs:
            self.model_configs[config.model_name] = config
            if config.model_type == "text" and config.tokenizer_name:
                try:
                    self.tokenizers[config.model_name] = AutoTokenizer.from_pretrained(config.tokenizer_name)
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer {config.tokenizer_name}: {e}")
                    self.tokenizers[config.model_name] = None

        # Compile regex patterns for blocked terms once to improve performance
        self.blocked_patterns = {}
        for config in self.configs:
            if config.blocked_terms:
                patterns = [re.compile(term, re.IGNORECASE) for term in config.blocked_terms]
                self.blocked_patterns[config.model_name] = patterns

    def _load_configs(self, config_path: str) -> List[InputValidationConfig]:
        """Load and parse the configuration file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            configs = []
            for model_config in config_data.get('models', []):
                configs.append(InputValidationConfig(**model_config))
            return configs
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def validate(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inputs for the specified model.

        Args:
            model_name: Name of the target model
            inputs: Dictionary of input data

        Returns:
            Validated and sanitized inputs

        Raises:
            ValueError: If inputs fail validation
        """
        if model_name not in self.model_configs:
            raise ValueError(f"No validation configuration found for model: {model_name}")

        config = self.model_configs[model_name]

        # 1. Schema validation
        self._validate_schema(model_name, inputs, config)

        # 2. Content filtering for harmful inputs
        self._filter_harmful_content(model_name, inputs, config)

        # 3. Sanitize text to prevent prompt injection
        if config.model_type == "text":
            inputs = self._sanitize_text_inputs(model_name, inputs, config)

        # 4. Size and complexity limits
        self._check_size_limits(model_name, inputs, config)

        # 5. Detect adversarial examples
        if config.detect_adversarial:
            self._detect_adversarial_examples(model_name, inputs, config)

        logger.info(f"Input validation passed for model {model_name}")
        return inputs

    def _validate_schema(self, model_name: str, inputs: Dict[str, Any], config: InputValidationConfig):
        """Validate inputs against the schema defined in the configuration."""
        # Check for required keys
        if config.allowed_input_keys:
            for key in inputs:
                if key not in config.allowed_input_keys:
                    raise ValueError(f"Unexpected input key: {key}. Allowed keys: {config.allowed_input_keys}")

        # Validate against schema if defined
        if config.schema:
            # Create a dynamic Pydantic model based on the schema
            class InputSchema(BaseModel):
                pass

            for field_name, field_spec in config.schema.items():
                setattr(InputSchema, field_name, (field_spec.get('type'), Field(**{k:v for k,v in field_spec.items() if k != 'type'})))

            try:
                validated_inputs = InputSchema(**inputs)
                logger.debug(f"Schema validation passed for model {model_name}")
            except Exception as e:
                logger.error(f"Schema validation failed: {e}")
                raise ValueError(f"Input schema validation failed: {e}")

    def _filter_harmful_content(self, model_name: str, inputs: Dict[str, Any], config: InputValidationConfig):
        """Filter potentially harmful content from inputs."""
        if config.content_filter_level == "none":
            return

        # Skip filtering for non-text inputs
        if config.model_type != "text":
            return

        # Check for blocked terms
        if model_name in self.blocked_patterns:
            patterns = self.blocked_patterns[model_name]
            for key, value in inputs.items():
                if isinstance(value, str):
                    for pattern in patterns:
                        if pattern.search(value):
                            term = pattern.pattern
                            logger.warning(f"Blocked term detected in input: {term}")
                            raise ValueError(f"Input contains blocked content")

    def _sanitize_text_inputs(self, model_name: str, inputs: Dict[str, Any], config: InputValidationConfig) -> Dict[str, Any]:
        """Sanitize text inputs to prevent prompt injection attacks."""
        sanitized_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, str):
                # Remove control characters that might be used for prompt injection
                sanitized = re.sub(r'[\x00-\x1F\x7F]', '', value)

                # Remove potential delimiter tokens that could be used to confuse the model
                sanitized = re.sub(r'(<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>|<\|endofprompt\|>)', '', sanitized)

                # Check for potential prompt injection patterns
                if re.search(r'(ignore|forget|disregard).*previous.*instructions', sanitized, re.IGNORECASE):
                    logger.warning(f"Potential prompt injection detected in input to {model_name}")
                    raise ValueError("Suspicious prompt pattern detected")

                sanitized_inputs[key] = sanitized
            else:
                sanitized_inputs[key] = value

        return sanitized_inputs

    def _check_size_limits(self, model_name: str, inputs: Dict[str, Any], config: InputValidationConfig):
        """Check that inputs do not exceed configured size and complexity limits."""
        # Check text length limits
        if config.model_type == "text":
            for key, value in inputs.items():
                if isinstance(value, str):
                    # Check character length
                    if config.max_input_length and len(value) > config.max_input_length:
                        raise ValueError(f"Input text exceeds maximum length of {config.max_input_length} characters")

                    # Check token length if tokenizer is available
                    if config.max_tokens and model_name in self.tokenizers and self.tokenizers[model_name]:
                        tokens = self.tokenizers[model_name].encode(value)
                        if len(tokens) > config.max_tokens:
                            raise ValueError(f"Input text exceeds maximum of {config.max_tokens} tokens")

        # Check image size limits
        elif config.model_type == "image" and config.max_image_size:
            for key, value in inputs.items():
                if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                    height, width = value.shape[0], value.shape[1]
                    max_height = config.max_image_size.get('height')
                    max_width = config.max_image_size.get('width')

                    if max_height and height > max_height:
                        raise ValueError(f"Image height ({height}) exceeds maximum of {max_height} pixels")
                    if max_width and width > max_width:
                        raise ValueError(f"Image width ({width}) exceeds maximum of {max_width} pixels")

        # Check tabular data limits
        elif config.model_type == "tabular" and config.max_tabular_rows:
            for key, value in inputs.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > config.max_tabular_rows:
                    raise ValueError(f"Tabular data exceeds maximum of {config.max_tabular_rows} rows")

    def _detect_adversarial_examples(self, model_name: str, inputs: Dict[str, Any], config: InputValidationConfig):
        """Detect potential adversarial examples in the input data."""
        if config.model_type == "image":
            for key, value in inputs.items():
                if isinstance(value, np.ndarray) and len(value.shape) >= 3:
                    # Convert to tensor for TF operations if needed
                    image = tf.convert_to_tensor(value) if not isinstance(value, tf.Tensor) else value

                    # Simple adversarial detection based on statistical properties
                    # This is a basic check - production systems would use more sophisticated methods

                    # Check for unusual pixel value distributions
                    pixel_mean = tf.reduce_mean(image)
                    pixel_std = tf.math.reduce_std(image)

                    # Extremely low variance could indicate adversarial examples
                    if pixel_std < 0.01:
                        logger.warning(f"Potential adversarial image detected: unusually low variance")
                        raise ValueError("Potential adversarial image detected")

                    # Check for unusual edge patterns
                    if len(image.shape) == 3 and image.shape[2] in [1, 3]:
                        # Convert to grayscale if color
                        if image.shape[2] == 3:
                            gray = tf.image.rgb_to_grayscale(image)
                        else:
                            gray = image

                        # Simple edge detection
                        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
                        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
                        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
                        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

                        # Add batch dimension and convert to correct format
                        gray_batch = tf.expand_dims(gray, 0)

                        # Apply edge detection
                        edges_x = tf.nn.conv2d(gray_batch, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
                        edges_y = tf.nn.conv2d(gray_batch, sobel_y, strides=[1, 1, 1, 1], padding='SAME')

                        # Calculate edge magnitude
                        edges = tf.sqrt(tf.square(edges_x) + tf.square(edges_y))
                        edge_mean = tf.reduce_mean(edges)

                        # Unusually high or low edge values could indicate adversarial examples
                        if edge_mean > 1.0 or edge_mean < 0.01:
                            logger.warning(f"Potential adversarial image detected: unusual edge patterns")
                            raise ValueError("Potential adversarial image detected")

def create_validator(config_path: str) -> InputValidator:
    """Factory function to create an input validator."""
    return InputValidator(config_path)
