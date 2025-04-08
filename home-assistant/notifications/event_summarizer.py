import logging
import numpy as np
import json
from typing import Dict, Any, List

from triton_client import TritonClient

logger = logging.getLogger("event_summarizer")

class EventSummarizer:
    """Summarizes multiple related events into a concise description"""

    def __init__(self, triton_client: TritonClient):
        self.triton_client = triton_client
        self.model_name = "event_summarizer"
        self.model_version = "1"

    async def summarize_events(self, events: List[Dict[str, Any]]) -> str:
        """Summarize a list of related events into a concise description"""
        try:
            # For single events, no need to summarize
            if len(events) <= 1:
                return events[0].get("message", "Event occurred")

            # Prepare input for the model
            event_texts = []
            for event in events:
                # Format the event as text
                entity_id = event.get("entity_id", "unknown")
                state = event.get("state", "changed")
                attributes = event.get("attributes", {})

                # Create a readable text representation
                event_text = f"Entity {entity_id} changed to {state}"

                # Add important attributes
                if "friendly_name" in attributes:
                    event_text = f"{attributes['friendly_name']} {state}"

                event_texts.append(event_text)

            # Join with newlines for the model
            input_text = "\n".join(event_texts)

            # Prepare input tensor for the model
            input_data = np.array([input_text], dtype=np.object_)

            # Run inference using Triton
            inputs = {"text_input": input_data}
            result = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                version=self.model_version
            )

            # Extract the summary text
            summary = result["summary_output"][0]

            # If the model returns bytes, decode to string
            if isinstance(summary, bytes):
                summary = summary.decode("utf-8")

            logger.debug(f"Summarized {len(events)} events into: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing events: {e}")
            # Return a basic summary as fallback
            return f"Multiple events ({len(events)}) occurred"

    def _format_entity_name(self, entity_id: str) -> str:
        """Format entity ID into a more readable name"""
        if not entity_id:
            return "unknown device"

        parts = entity_id.split(".")
        if len(parts) != 2:
            return entity_id

        domain = parts[0]
        name = parts[1].replace("_", " ")

        return f"{name} {domain}"
