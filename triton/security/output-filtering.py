"""
Output Filtering Module for Triton Inference Server

This module provides output filtering functionality for AI models:
1. Content safety checks for generated text
2. Filtering of potentially harmful or inappropriate outputs
3. PII detection and redaction
4. Confidence thresholds for model outputs
5. Logging of filtered outputs for review

The module is designed to be configurable for different sensitivity levels.
"""

import re
import json
import logging
import os
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Optional imports - install as needed
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Configure logging
logger = logging.getLogger("output_filter")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# File handler for filtered outputs
filtered_outputs_dir = os.environ.get("FILTERED_OUTPUTS_DIR", "/var/log/triton/filtered_outputs")
os.makedirs(filtered_outputs_dir, exist_ok=True)
file_handler = logging.FileHandler(f"{filtered_outputs_dir}/filtered_outputs.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class SensitivityLevel(Enum):
    """Sensitivity levels for content filtering"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class OutputFilter:
    """Main class for filtering model outputs"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the output filter with configuration.

        Args:
            config_path: Path to configuration file (JSON)
        """
        # Default configuration
        self.config = {
            "sensitivity_level": SensitivityLevel.MEDIUM.name,
            "harmful_content": {
                "enabled": True,
                "blocked_terms": [
                    "hack", "exploit", "vulnerability", "attack",
                    "malware", "virus", "ransomware"
                ],
                "custom_regex_patterns": []
            },
            "pii_detection": {
                "enabled": True,
                "entities_to_redact": [
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS",
                    "CREDIT_CARD", "US_SSN", "US_PASSPORT", "LOCATION"
                ]
            },
            "confidence_threshold": {
                "enabled": True,
                "min_threshold": 0.7
            },
            "logging": {
                "enabled": True,
                "include_original": False,
                "include_filtered_reason": True
            }
        }

        # Load custom configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self._merge_configs(custom_config)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading configuration: {e}")

        # Initialize PII detection if enabled
        self.pii_analyzer = None
        self.pii_anonymizer = None
        if self.config["pii_detection"]["enabled"]:
            if PRESIDIO_AVAILABLE:
                self.pii_analyzer = AnalyzerEngine()
                self.pii_anonymizer = AnonymizerEngine()
            else:
                logger.warning("Microsoft Presidio not available. PII detection will use basic patterns.")

        # Initialize NLP model if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("Loaded spaCy NLP model for advanced content analysis")
            except:
                logger.warning("Could not load spaCy model. Using basic text analysis.")

        # Compile regex patterns for harmful content
        self.harmful_patterns = []
        if self.config["harmful_content"]["enabled"]:
            # Compile built-in patterns
            terms = self.config["harmful_content"]["blocked_terms"]
            if terms:
                term_pattern = r'\b(' + '|'.join(re.escape(term) for term in terms) + r')\b'
                self.harmful_patterns.append(re.compile(term_pattern, re.IGNORECASE))

            # Compile custom patterns
            for pattern in self.config["harmful_content"]["custom_regex_patterns"]:
                try:
                    self.harmful_patterns.append(re.compile(pattern))
                except re.error:
                    logger.error(f"Invalid regex pattern: {pattern}")

        logger.info(f"Output filter initialized with sensitivity level: {self.config['sensitivity_level']}")

    def _merge_configs(self, custom_config: Dict[str, Any]) -> None:
        """Merge custom configuration with default configuration"""
        for key, value in custom_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    for subkey, subvalue in value.items():
                        if subkey in self.config[key]:
                            self.config[key][subkey] = subvalue
                else:
                    self.config[key] = value

    def filter_output(self,
                      output: Any,
                      model_name: str,
                      confidence: Optional[float] = None) -> Tuple[Any, bool, Optional[str]]:
        """
        Filter the model output based on configured rules.

        Args:
            output: The output from the model (text or structured data)
            model_name: Name of the model that generated the output
            confidence: Confidence score if available

        Returns:
            Tuple containing:
            - Filtered output (or original if no issues)
            - Boolean indicating if output was modified
            - Reason for filtering (if applicable)
        """
        original_output = output
        was_filtered = False
        filter_reason = None

        # Convert output to string if it's not already
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output)
        else:
            output_str = str(output)

        # Check confidence threshold
        if (self.config["confidence_threshold"]["enabled"] and
            confidence is not None and
            confidence < self.config["confidence_threshold"]["min_threshold"]):
            filter_reason = f"Low confidence score: {confidence}"
            was_filtered = True

            # For low confidence, we might want to add a disclaimer rather than blocking
            sensitivity = SensitivityLevel[self.config["sensitivity_level"]]
            if sensitivity == SensitivityLevel.HIGH:
                output = "Content filtered due to low confidence."
            else:
                output = f"{output}\n\nNote: This response was generated with lower confidence ({confidence:.2f})."

        # Apply harmful content filtering
        if self.config["harmful_content"]["enabled"] and not (was_filtered and sensitivity == SensitivityLevel.HIGH):
            harmful_result, harmful_reason = self._check_harmful_content(output_str)
            if harmful_result:
                filter_reason = harmful_reason
                was_filtered = True

                sensitivity = SensitivityLevel[self.config["sensitivity_level"]]
                if sensitivity == SensitivityLevel.HIGH:
                    output = "Content filtered due to potentially harmful content."
                else:
                    # For medium sensitivity, we might redact just the problematic parts
                    output = self._redact_harmful_content(output_str)

        # Apply PII detection and redaction
        if self.config["pii_detection"]["enabled"] and not (was_filtered and sensitivity == SensitivityLevel.HIGH):
            output, pii_detected = self._redact_pii(output_str if isinstance(output, str) else str(output))
            if pii_detected:
                was_filtered = True
                if not filter_reason:
                    filter_reason = "PII detected and redacted"

        # Log the filtering event if something was filtered
        if was_filtered and self.config["logging"]["enabled"]:
            self._log_filtered_output(
                model_name=model_name,
                original=original_output if self.config["logging"]["include_original"] else None,
                filtered=output,
                reason=filter_reason if self.config["logging"]["include_filtered_reason"] else None,
                confidence=confidence
            )

        # If the output was originally structured data, try to convert it back
        if isinstance(original_output, (dict, list)) and isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                # If we can't parse it back to JSON, keep it as a string
                pass

        return output, was_filtered, filter_reason

    def _check_harmful_content(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains harmful content.

        Args:
            text: The text to check

        Returns:
            Tuple of (contains_harmful, reason)
        """
        # Check against regex patterns
        for pattern in self.harmful_patterns:
            match = pattern.search(text)
            if match:
                return True, f"Matched harmful pattern: {match.group()}"

        # Use NLP model for more advanced checking if available
        if self.nlp:
            doc = self.nlp(text)

            # Check for specific entities that might indicate harmful content
            if any(ent.label_ in ["ORG", "PRODUCT"] and ent.text.lower() in ["malware", "exploit", "virus"] for ent in doc.ents):
                return True, "Contains references to harmful software"

            # Check for imperative verbs related to harmful actions
            imperative_harmful_verbs = ["hack", "attack", "exploit", "steal", "breach"]
            if any(token.lemma_ in imperative_harmful_verbs and token.dep_ in ["ROOT"] for token in doc):
                return True, "Contains instructions for harmful actions"

        return False, None

    def _redact_harmful_content(self, text: str) -> str:
        """
        Redact harmful content from text.

        Args:
            text: The text to redact

        Returns:
            Redacted text
        """
        redacted_text = text
        for pattern in self.harmful_patterns:
            redacted_text = pattern.sub("[REDACTED]", redacted_text)
        return redacted_text

    def _redact_pii(self, text: str) -> Tuple[str, bool]:
        """
        Detect and redact personally identifiable information.

        Args:
            text: The text to process

        Returns:
            Tuple of (redacted_text, pii_detected)
        """
        if not text:
            return text, False

        # Use Presidio if available (more accurate)
        if PRESIDIO_AVAILABLE and self.pii_analyzer and self.pii_anonymizer:
            # Get entities to redact from config
            entities_to_redact = self.config["pii_detection"]["entities_to_redact"]

            # Analyze the text
            results = self.pii_analyzer.analyze(
                text=text,
                entities=entities_to_redact,
                language='en'
            )

            # If entities were found, anonymize them
            if results:
                anonymized_text = self.pii_anonymizer.anonymize(
                    text=text,
                    analyzer_results=results
                ).text
                return anonymized_text, True
        else:
            # Basic regex patterns for common PII
            patterns = {
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "phone": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
                "credit_card": r'\b(?:\d{4}[- ]?){3}\d{4}\b',
                "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            }

            modified = False
            redacted_text = text

            for pii_type, pattern in patterns.items():
                if re.search(pattern, redacted_text):
                    redacted_text = re.sub(pattern, f"[REDACTED {pii_type.upper()}]", redacted_text)
                    modified = True

            return redacted_text, modified

        return text, False

    def _log_filtered_output(self,
                           model_name: str,
                           filtered: Any,
                           reason: Optional[str] = None,
                           original: Optional[Any] = None,
                           confidence: Optional[float] = None) -> None:
        """Log filtered output for review"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "model_name": model_name,
            "filtered_output": filtered,
            "filter_reason": reason,
            "confidence": confidence
        }

        if original is not None:
            log_entry["original_output"] = original

        # Log to file
        log_file_path = os.path.join(filtered_outputs_dir, f"filtered_{datetime.now().strftime('%Y%m%d')}.json")
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also log to standard logger
        logger.warning(f"Output filtered from model {model_name}: {reason}")


class TritonOutputFilter:
    """Integration with Triton Inference Server"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with configuration"""
        self.output_filter = OutputFilter(config_path)

    def process_inference_response(self,
                                 response: Dict[str, Any],
                                 model_name: str) -> Dict[str, Any]:
        """
        Process and filter a response from Triton Inference Server.

        Args:
            response: The response from Triton
            model_name: Name of the model

        Returns:
            Filtered response
        """
        # Extract confidence if available in the response
        confidence = None
        if "model_confidence" in response:
            confidence = response["model_confidence"]
        elif "outputs" in response and isinstance(response["outputs"], list):
            # Try to find confidence in outputs
            for output in response["outputs"]:
                if "confidence" in output or "scores" in output:
                    confidence_field = "confidence" if "confidence" in output else "scores"
                    confidence_data = output[confidence_field]
                    if isinstance(confidence_data, (list, tuple)) and confidence_data:
                        confidence = max(confidence_data)  # Take max confidence
                    else:
                        confidence = confidence_data
                    break

        # Handle different response structures
        filtered_response = response.copy()

        if "outputs" in response and isinstance(response["outputs"], list):
            filtered_outputs = []
            for output in response["outputs"]:
                if "data" in output:
                    filtered_data, was_filtered, reason = self.output_filter.filter_output(
                        output["data"],
                        model_name,
                        confidence
                    )

                    filtered_output = output.copy()
                    filtered_output["data"] = filtered_data

                    # Add filtering metadata if applicable
                    if was_filtered:
                        filtered_output["was_filtered"] = True
                        filtered_output["filter_reason"] = reason

                    filtered_outputs.append(filtered_output)
                else:
                    filtered_outputs.append(output)

            filtered_response["outputs"] = filtered_outputs

        elif "output" in response:
            # Single output case
            filtered_output, was_filtered, reason = self.output_filter.filter_output(
                response["output"],
                model_name,
                confidence
            )

            filtered_response["output"] = filtered_output
            if was_filtered:
                filtered_response["was_filtered"] = True
                filtered_response["filter_reason"] = reason

        return filtered_response


def create_output_filter(config_path: Optional[str] = None) -> TritonOutputFilter:
    """Factory function to create an output filter for Triton"""
    return TritonOutputFilter(config_path)
