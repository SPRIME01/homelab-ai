import asyncio
import logging
import json
from typing import Dict, Any, Optional
import numpy as np

from .triton_client import TritonClient

logger = logging.getLogger("nlu")

class NaturalLanguageUnderstanding:
    """Natural language understanding using a local LLM on Triton."""

    def __init__(self, triton_url: str, model_name: str = "llama",
                 system_prompt: Optional[str] = None):
        """
        Initialize the NLU module.

        Args:
            triton_url: URL of the Triton Inference Server
            model_name: Name of the LLM model in Triton
            system_prompt: System prompt for the LLM
        """
        self.triton_client = TritonClient(triton_url)
        self.model_name = model_name

        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = (
                "You are a home assistant AI. Interpret the user's request and "
                "convert it into structured intent. Extract entities such as device names, "
                "locations, and desired states. Format your response as JSON with 'intent', "
                "'entities', and 'confidence' fields."
            )

        self.system_prompt = system_prompt

    async def understand(self, text: str) -> Dict[str, Any]:
        """
        Process text input through LLM to extract intents and entities.

        Args:
            text: User input text

        Returns:
            Dictionary containing intent, entities, and confidence score
        """
        try:
            # Prepare prompt with system context and user input
            prompt = f"{self.system_prompt}\n\nUser request: {text}\n\nStructured intent:"

            # Convert prompt to input for Triton
            inputs = {
                "text": np.array([prompt], dtype=np.object_)
            }

            # Parameters for inference
            params = {
                "max_tokens": 256,
                "temperature": 0.2,
                "top_p": 0.9,
                "response_format": "json_object"
            }

            # Send inference request to Triton
            result = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                parameters=params
            )

            # Extract and parse JSON response
            if "text" in result:
                llm_response = result["text"][0].decode("utf-8")

                # Extract JSON portion from response if needed
                if "```json" in llm_response:
                    json_str = llm_response.split("```json")[1].split("```")[0].strip()
                elif "{" in llm_response:
                    # Find the first { and last }
                    start = llm_response.find("{")
                    end = llm_response.rfind("}")
                    if start >= 0 and end >= 0:
                        json_str = llm_response[start:end+1]
                    else:
                        json_str = llm_response
                else:
                    json_str = llm_response

                try:
                    # Parse JSON response
                    nlu_result = json.loads(json_str)

                    # Ensure required fields are present
                    if "intent" not in nlu_result:
                        nlu_result["intent"] = "unknown"
                    if "entities" not in nlu_result:
                        nlu_result["entities"] = {}
                    if "confidence" not in nlu_result:
                        nlu_result["confidence"] = 0.8

                    return nlu_result

                except json.JSONDecodeError:
                    logger.error(f"Failed to parse NLU response as JSON: {llm_response}")
                    return {
                        "intent": "error",
                        "entities": {},
                        "confidence": 0.0,
                        "raw_response": llm_response
                    }
            else:
                logger.error(f"Unexpected response format: {result}")
                return {"intent": "error", "entities": {}, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error in NLU processing: {e}")
            return {"intent": "error", "entities": {}, "confidence": 0.0}

    async def cleanup(self):
        """Clean up resources."""
        await self.triton_client.cleanup()
