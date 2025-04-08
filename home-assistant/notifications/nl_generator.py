import logging
import numpy as np
import json
from typing import Dict, Any, List

from triton_client import TritonClient

logger = logging.getLogger("nl_generator")

class NaturalLanguageGenerator:
    """Generates natural language descriptions of sensor data and events"""

    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self.model_name = "nl_generator"
        self.model_version = "1"

    async def generate_description(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate natural language description of sensor data and events"""
        try:
            # For simple messages with little context, return as-is
            if not context or not context.get("entities"):
                return message

            # Extract relevant data for the model
            entities = self._extract_entity_data(context)
            areas = context.get("areas", {})

            # If we have history, include it for better context
            history = None
            if "history" in context:
                history = self._extract_history_summary(context["history"])

            # Prepare input for model
            input_dict = {
                "message": message,
                "entities": entities,
                "areas": areas,
                "history": history
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

            # Extract the generated description
            description = result["description_output"][0]

            # If the model returns bytes, decode to string
            if isinstance(description, bytes):
                description = description.decode("utf-8")

            logger.debug(f"Generated description: {description[:50]}...")
            return description

        except Exception as e:
            logger.error(f"Error generating natural language description: {e}")
            # Return the original message as fallback
            return message

    def _extract_entity_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant entity data from context"""
        if not context or "entities" not in context:
            return {}

        entities = {}
        for entity_id, data in context["entities"].items():
            # Get friendly name
            friendly_name = data.get("attributes", {}).get("friendly_name", entity_id)

            # Get area name if available
            area_id = None
            area_name = None
            if "entity_area_map" in context.get("areas", {}):
                area_id = context["areas"]["entity_area_map"].get(entity_id)

            if area_id and area_id in context.get("areas", {}).get("areas", {}):
                area_name = context["areas"]["areas"][area_id].get("name", "Unknown Area")

            # Create simplified entity data
            entities[entity_id] = {
                "name": friendly_name,
                "state": data.get("state"),
                "area": area_name,
                "attributes": data.get("attributes", {})
            }

        return entities

    def _extract_history_summary(self, history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create a summary of historical data"""
        if not history:
            return None

        summary = {}
        for entity_id, states in history.items():
            if not states:
                continue

            # Get the frequency of states
            state_counts = {}
            for state_data in states:
                state = state_data["state"]
                state_counts[state] = state_counts.get(state, 0) + 1

            # Find the most common state
            most_common = max(state_counts.items(), key=lambda x: x[1])

            # Get the current state (most recent)
            current = states[0]["state"]

            # Get the previous state (2nd most recent if available)
            previous = states[1]["state"] if len(states) > 1 else None

            summary[entity_id] = {
                "most_common": most_common[0],
                "current": current,
                "previous": previous,
                "changes": len(states)
            }

        return summary
