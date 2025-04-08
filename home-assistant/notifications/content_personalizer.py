import logging
import numpy as np
import json
from typing import Dict, Any, List
from datetime import datetime

from triton_client import TritonClient

logger = logging.getLogger("content_personalizer")

class ContentPersonalizer:
    """Personalizes notification content based on user preferences"""

    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self.model_name = "content_personalizer"
        self.model_version = "1"

    async def personalize(
        self,
        message: str,
        priority: int,
        user_preferences: Dict[str, Any]
    ) -> str:
        """Personalize notification content based on user preferences"""
        try:
            # Check if we should use AI personalization or simple rule-based
            # For low priority notifications, we can avoid model inference
            if priority <= 2:
                return self._simple_personalize(message, priority, user_preferences)

            # For higher priority, use the AI model for personalization

            # Prepare input for the model
            current_time = datetime.now().strftime("%H:%M:%S")
            current_day = datetime.now().strftime("%A")

            # Create a simplified version of user preferences
            simple_prefs = {}
            for user_id, user_data in user_preferences.items():
                simple_prefs[user_id] = {
                    "name": user_data.get("name", "User"),
                    "notification_settings": user_data.get("notification_settings", {})
                }

            # Prepare input data
            input_dict = {
                "message": message,
                "priority": priority,
                "user_preferences": simple_prefs,
                "context": {
                    "time": current_time,
                    "day": current_day
                }
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

            # Extract the personalized message
            personalized_message = result["message_output"][0]

            # If the model returns bytes, decode to string
            if isinstance(personalized_message, bytes):
                personalized_message = personalized_message.decode("utf-8")

            logger.debug(f"Personalized message: {personalized_message[:50]}...")
            return personalized_message

        except Exception as e:
            logger.error(f"Error personalizing notification content: {e}")
            # Return the original message as fallback
            return self._simple_personalize(message, priority, user_preferences)

    def _simple_personalize(
        self,
        message: str,
        priority: int,
        user_preferences: Dict[str, Any]
    ) -> str:
        """Simple rule-based personalization as fallback"""
        # Add priority indicator
        priority_prefixes = {
            1: "Info: ",
            2: "Notice: ",
            3: "Important: ",
            4: "Attention: ",
            5: "URGENT: "
        }

        prefix = priority_prefixes.get(priority, "")

        # For high priority, add emphasis
        if priority >= 4:
            return f"{prefix}{message.upper()}"
        else:
            return f"{prefix}{message}"
