import logging
import numpy as np
import json
from typing import Dict, Any, List

from triton_client import TritonClient

logger = logging.getLogger("notification_prioritizer")

class NotificationPrioritizer:
    """Prioritizes notifications based on importance"""

    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self.model_name = "notification_prioritizer"
        self.model_version = "1"

        # Priority levels:
        # 1: Low - informational only
        # 2: Medium-low - minor alerts
        # 3: Medium - important information
        # 4: Medium-high - requires attention soon
        # 5: High - urgent, requires immediate attention

    async def prioritize(
        self,
        message: str,
        context: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> int:
        """Determine the priority level of a notification"""
        try:
            # Extract relevant data from context
            entities = self._extract_entity_data(context)
            history = self._extract_history_data(context)

            # Check for high-priority domains
            high_priority_domains = ["alarm_control_panel", "smoke", "water_leak",
                                    "gas", "lock", "person", "camera"]

            # Quick rules-based priority for critical situations
            for entity_id, data in entities.items():
                domain = entity_id.split(".")[0]
                state = data.get("state", "")

                # Security and safety events get high priority
                if domain in high_priority_domains:
                    if (domain == "alarm_control_panel" and state in ["triggered", "arming"]) or \
                       (domain == "smoke" and state == "detected") or \
                       (domain == "water_leak" and state == "detected") or \
                       (domain == "gas" and state == "detected") or \
                       (domain == "lock" and state == "unlocked"):
                        return 5

            # Prepare input for model
            input_dict = {
                "message": message,
                "entities": entities,
                "history": history or {}
            }

            # Convert to JSON string
            input_json = json.dumps(input_dict)

            # Prepare input tensor for the model
            input_data = np.array([input_json], dtype=np.object_)

            # Run inference using Triton
            inputs = {"json_input": input_data}
            result = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                version=self.model_version
            )

            # Extract the priority value (should be an int from 1-5)
            priority = int(result["priority_output"][0])

            # Clamp to valid range
            priority = max(1, min(5, priority))

            logger.debug(f"Calculated priority {priority} for message: {message[:50]}...")
            return priority

        except Exception as e:
            logger.error(f"Error determining notification priority: {e}")
            # Default to medium priority
            return 3

    def _extract_entity_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant entity data from context"""
        if not context or "entities" not in context:
            return {}

        entities = {}
        for entity_id, data in context["entities"].items():
            entities[entity_id] = {
                "state": data.get("state"),
                "last_changed": data.get("last_changed"),
                "attributes": {
                    k: v for k, v in data.get("attributes", {}).items()
                    if k in ["device_class", "friendly_name", "unit_of_measurement",
                             "battery_level", "tampered", "temperature", "humidity"]
                }
            }

        return entities

    def _extract_history_data(self, context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract relevant historical data from context"""
        if not context or "history" not in context:
            return {}

        return context["history"]
