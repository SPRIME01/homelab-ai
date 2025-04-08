import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger("config")

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self._load_config()

        # Extract config sections
        self.general = self.config_data.get("general", {})
        self.home_assistant = self.config_data.get("home_assistant", {})
        self.triton = self.config_data.get("triton", {})
        self.models = self.config_data.get("models", {})
        self.notification_settings = self.config_data.get("notification_settings", {})

        # Validate config
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    return config or {}
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "general": {
                "polling_interval": 30,
                "log_level": "INFO"
            },
            "home_assistant": {
                "url": "http://homeassistant.local:8123",
                "token": "YOUR_LONG_LIVED_ACCESS_TOKEN",
                "websocket_url": "ws://homeassistant.local:8123/api/websocket"
            },
            "triton": {
                "url": "localhost:8001",
                "model_repository": "/models",
                "timeout_ms": 30000
            },
            "models": {
                "summarizer": {
                    "name": "event_summarizer",
                    "version": "1"
                },
                "prioritizer": {
                    "name": "notification_prioritizer",
                    "version": "1"
                },
                "personalizer": {
                    "name": "content_personalizer",
                    "version": "1"
                },
                "nl_generator": {
                    "name": "nl_generator",
                    "version": "1"
                }
            },
            "notification_settings": {
                "titles": {
                    "1": "Info",
                    "3": "Important",
                    "5": "Urgent"
                },
                "channels": {
                    "1": "general",
                    "3": "important",
                    "5": "critical"
                },
                "ttl": {
                    "1": 3600,
                    "3": 7200,
                    "5": 86400
                },
                "persistent_threshold": 3
            }
        }

    def _validate_config(self):
        """Validate configuration values"""
        # Check required fields
        if not self.home_assistant.get("url"):
            raise ValueError("Home Assistant URL not configured")
        if not self.home_assistant.get("token"):
            raise ValueError("Home Assistant token not configured")
        if not self.triton.get("url"):
            raise ValueError("Triton URL not configured")

        # Set defaults if missing
        if "polling_interval" not in self.general:
            self.general["polling_interval"] = 30

        logger.info("Configuration validated")
