import asyncio
import logging
import json
import yaml
from typing import Dict, Any, List
import aiohttp
from pathlib import Path

logger = logging.getLogger("intent_resolver")

class IntentResolver:
    """Resolve intents to Home Assistant actions."""

    def __init__(self, ha_url: str, ha_token: str, intent_mappings_path: str):
        """
        Initialize the intent resolver.

        Args:
            ha_url: URL of the Home Assistant instance
            ha_token: Long-lived access token for Home Assistant
            intent_mappings_path: Path to intent mappings YAML file
        """
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.intent_mappings_path = intent_mappings_path
        self.headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json"
        }
        self.intent_mappings = self._load_intent_mappings()

    def _load_intent_mappings(self) -> Dict:
        """Load intent mappings from YAML file."""
        try:
            with open(self.intent_mappings_path, 'r') as f:
                mappings = yaml.safe_load(f)
            logger.info(f"Loaded {len(mappings)} intent mappings")
            return mappings
        except Exception as e:
            logger.error(f"Failed to load intent mappings: {e}")
            return {}

    async def resolve_intent(self, nlu_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve the NLU result to a Home Assistant action.

        Args:
            nlu_result: Result from NLU with intent and entities

        Returns:
            Dictionary with action result and response text
        """
        try:
            intent = nlu_result.get("intent", "unknown")
            entities = nlu_result.get("entities", {})

            # Check if we have a mapping for this intent
            if intent in self.intent_mappings:
                mapping = self.intent_mappings[intent]

                # Get service to call
                domain = mapping.get("domain")
                service = mapping.get("service")

                if not domain or not service:
                    return {
                        "success": False,
                        "response": "I'm not sure how to handle that request."
                    }

                # Format service data using entities
                service_data = mapping.get("service_data", {}).copy()

                # Replace placeholders with entity values
                for key, value in service_data.items():
                    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                        entity_key = value[2:-2].strip()
                        if entity_key in entities:
                            service_data[key] = entities[entity_key]

                # Extra entity processing for special cases
                if "entity_id" in service_data and "area" in entities:
                    # Try to map area to entity_id
                    area = entities["area"].lower()
                    await self._enhance_entity_id_with_area(service_data, area, domain)

                # Call Home Assistant service
                result = await self._call_ha_service(domain, service, service_data)

                # Generate response based on mapping and result
                response_template = mapping.get("response_template", "I've processed your request.")
                response_text = self._format_response(response_template, entities, result)

                return {
                    "success": True,
                    "response": response_text,
                    "action": {
                        "domain": domain,
                        "service": service,
                        "service_data": service_data,
                        "result": result
                    }
                }
            else:
                # Try to handle generic intent through conversation API
                return await self._handle_unknown_intent(nlu_result)

        except Exception as e:
            logger.error(f"Error resolving intent: {e}")
            return {
                "success": False,
                "response": "I encountered an error processing your request. Please try again."
            }

    async def _call_ha_service(self, domain: str, service: str, service_data: Dict) -> Dict:
        """Call Home Assistant service and return the result."""
        try:
            endpoint = f"{self.ha_url}/api/services/{domain}/{service}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=self.headers,
                    json=service_data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        text = await response.text()
                        logger.error(f"Error calling HA service: {response.status} {text}")
                        return {"error": f"Status {response.status}"}

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling HA service: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error calling HA service: {e}")
            return {"error": str(e)}

    async def _enhance_entity_id_with_area(self, service_data: Dict, area: str, domain: str):
        """Try to find entities in the specified area and domain."""
        try:
            # Get states from Home Assistant
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ha_url}/api/states",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        states = await response.json()

                        # Filter entities by domain and area
                        matching_entities = []
                        for entity in states:
                            entity_id = entity["entity_id"]
                            if entity_id.startswith(f"{domain}.") and "area_id" in entity["attributes"]:
                                # Check if this entity belongs to the requested area
                                entity_area = await self._get_area_name(entity["attributes"]["area_id"])
                                if entity_area and area in entity_area.lower():
                                    matching_entities.append(entity_id)

                        if matching_entities:
                            # If we have multiple entities, use them all
                            if isinstance(service_data["entity_id"], list):
                                service_data["entity_id"].extend(matching_entities)
                            else:
                                service_data["entity_id"] = matching_entities
        except Exception as e:
            logger.error(f"Error enhancing entity_id with area: {e}")

    async def _get_area_name(self, area_id: str) -> str:
        """Get the name of an area from its ID."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ha_url}/api/areas/{area_id}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        area_data = await response.json()
                        return area_data.get("name", "")
            return ""
        except Exception as e:
            logger.error(f"Error getting area name: {e}")
            return ""

    async def _handle_unknown_intent(self, nlu_result: Dict) -> Dict:
        """Handle unknown intents using the conversation API."""
        try:
            async with aiohttp.ClientSession() as session:
                text = nlu_result.get("original_text", "")
                if not text:
                    text = " ".join([nlu_result.get("intent", "help"),
                                    str(nlu_result.get("entities", {}))])

                async with session.post(
                    f"{self.ha_url}/api/conversation/process",
                    headers=self.headers,
                    json={"text": text}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": result.get("speech", {}).get("plain", {}).get("speech",
                                       "I'm not sure how to handle that request.")
                        }

            return {
                "success": False,
                "response": "I'm not sure how to handle that request."
            }

        except Exception as e:
            logger.error(f"Error handling unknown intent: {e}")
            return {
                "success": False,
                "response": "I encountered an error processing your request."
            }

    def _format_response(self, template: str, entities: Dict, result: Dict) -> str:
        """Format response template with entity values and result info."""
        response = template

        # Replace entity placeholders
        for key, value in entities.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in response:
                response = response.replace(placeholder, str(value))

        # Replace result placeholders (simple version)
        if "{{result}}" in response:
            response = response.replace("{{result}}", str(result))

        return response
