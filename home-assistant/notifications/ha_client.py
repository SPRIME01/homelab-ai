import aiohttp
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ha_client")

class HomeAssistantClient:
    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("url")
        self.token = config.get("token")
        self.websocket_url = config.get("websocket_url")
        self.session = None
        self.ws = None
        self.ws_id = 1
        self.ws_connected = False
        self.last_event_id = None
        self.subscribed_events = False

    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.token}"}
            )
        return self.session

    async def _get_websocket(self):
        """Get or create WebSocket connection"""
        if self.ws is None or self.ws.closed:
            session = await self._get_session()
            self.ws = await session.ws_connect(self.websocket_url)
            self.ws_connected = False
            self.ws_id = 1

            # Authenticate
            auth_result = await self._ws_authenticate()
            if not auth_result:
                raise ConnectionError("Failed to authenticate with Home Assistant WebSocket API")

        return self.ws

    async def _ws_authenticate(self) -> bool:
        """Authenticate with the WebSocket API"""
        ws = self.ws

        # Wait for auth required message
        msg = await ws.receive_json()
        if msg["type"] != "auth_required":
            logger.error(f"Unexpected message when connecting to HA: {msg}")
            return False

        # Send auth message
        await ws.send_json({
            "type": "auth",
            "access_token": self.token
        })

        # Wait for auth_ok or auth_invalid
        msg = await ws.receive_json()
        if msg["type"] == "auth_ok":
            logger.info("Successfully authenticated with Home Assistant")
            self.ws_connected = True
            return True
        else:
            logger.error(f"Authentication failed: {msg}")
            return False

    async def _ws_call(self, msg_type: str, payload: Dict = None) -> Dict:
        """Make a call to the WebSocket API"""
        ws = await self._get_websocket()

        # Create the message
        msg = {
            "id": self.ws_id,
            "type": msg_type
        }
        if payload:
            msg.update(payload)

        # Send the message
        await ws.send_json(msg)

        # Wait for the response
        response = await ws.receive_json()
        while response.get("id") != self.ws_id:
            # This is an event or other message, not our response
            response = await ws.receive_json()

        self.ws_id += 1
        return response

    async def get_latest_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get latest events from Home Assistant"""
        try:
            session = await self._get_session()

            # Get events from REST API
            url = f"{self.url}/api/events"
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error getting events: {response.status} - {error_text}")
                    return []

                events = await response.json()

            # Filter events we've already seen
            if self.last_event_id:
                new_events = []
                for event in events:
                    if event["id"] > self.last_event_id:
                        new_events.append(event)
                events = new_events

            # Save the latest event id we've seen
            if events:
                self.last_event_id = max(e["id"] for e in events)

            return events[-limit:] if len(events) > limit else events

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error while getting events: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []

    async def get_related_sensor_data(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get related sensor data for a set of events"""
        try:
            # Extract entity ids from events
            entity_ids = set()
            domains = set()

            for event in events:
                if "entity_id" in event:
                    entity_ids.add(event["entity_id"])
                    domains.add(event["entity_id"].split(".")[0])

            # Add related entities based on domain
            session = await self._get_session()
            url = f"{self.url}/api/states"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error getting states: {response.status}")
                    return {}

                all_states = await response.json()

            # Find related entities
            related_entities = {}
            for state in all_states:
                state_entity_id = state["entity_id"]
                domain = state_entity_id.split(".")[0]

                # Add if it's directly mentioned in events
                if state_entity_id in entity_ids:
                    related_entities[state_entity_id] = state

                # Add if it's in the same domain as our events
                elif domain in domains:
                    # Only add if it's in the same area
                    area_id = state["context"].get("area_id")
                    for entity_id in entity_ids:
                        event_entity = next((s for s in all_states if s["entity_id"] == entity_id), None)
                        if event_entity and event_entity["context"].get("area_id") == area_id:
                            related_entities[state_entity_id] = state
                            break

            # Format the context data
            context = {
                "entities": related_entities,
                "areas": await self._get_areas_for_entities(list(related_entities.keys())),
                "timestamp": time.time()
            }

            # Add insights if available
            history = await self._get_history_for_entities(list(entity_ids), limit=10)
            if history:
                context["history"] = history

            return context

        except Exception as e:
            logger.error(f"Error getting related sensor data: {e}")
            return {}

    async def _get_areas_for_entities(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Get area information for a list of entity IDs"""
        try:
            session = await self._get_session()

            # Get all areas
            url = f"{self.url}/api/config/area_registry"
            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                areas = await response.json()

            # Get entity registry to map entities to areas
            url = f"{self.url}/api/config/entity_registry"
            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                entities = await response.json()

            # Create mapping of entity_id to area_id
            entity_area_map = {}
            for entity in entities:
                if entity["entity_id"] in entity_ids and entity.get("area_id"):
                    entity_area_map[entity["entity_id"]] = entity["area_id"]

            # Get area details for all relevant areas
            area_ids = set(entity_area_map.values())
            area_details = {}
            for area in areas:
                if area["area_id"] in area_ids:
                    area_details[area["area_id"]] = area

            return {
                "entity_area_map": entity_area_map,
                "areas": area_details
            }

        except Exception as e:
            logger.error(f"Error getting area information: {e}")
            return {}

    async def _get_history_for_entities(self, entity_ids: List[str], limit: int = 20) -> Dict[str, List]:
        """Get historical data for a list of entity IDs"""
        try:
            if not entity_ids:
                return {}

            session = await self._get_session()

            # Build the request
            entity_filter = ",".join(entity_ids)
            url = f"{self.url}/api/history/period?filter_entity_id={entity_filter}&end_time=now&minimal_response"

            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                history_data = await response.json()

            # Format the data
            history = {}
            for entity_history in history_data:
                if not entity_history:
                    continue

                entity_id = entity_history[0]["entity_id"]
                states = []

                for state in entity_history[:limit]:
                    states.append({
                        "state": state["state"],
                        "timestamp": state["last_changed"]
                    })

                history[entity_id] = states

            return history

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return {}

    async def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from Home Assistant"""
        try:
            session = await self._get_session()

            # Get users
            url = f"{self.url}/api/config/person"
            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                persons = await response.json()

            # Get notification settings
            preferences = {}
            for person in persons:
                person_id = person["id"]
                preferences[person_id] = {
                    "name": person["name"],
                    "notification_settings": {
                        "mobile": True,
                        "web": True,
                        "quiet_hours": {
                            "enabled": False,
                            "start": "22:00:00",
                            "end": "07:00:00"
                        },
                        "priority_threshold": 3
                    }
                }

            return preferences

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}

    async def send_notification(self, notification: Dict[str, Any]):
        """Send a notification to Home Assistant"""
        try:
            session = await self._get_session()

            # Format the notification data
            service_data = {
                "message": notification["message"]
            }

            if "title" in notification:
                service_data["title"] = notification["title"]

            if "data" in notification:
                service_data["data"] = notification["data"]

            # Call the notify service
            url = f"{self.url}/api/services/notify/notify"
            async with session.post(url, json=service_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error sending notification: {response.status} - {error_text}")
                    return False

                logger.info(f"Notification sent: {notification['message'][:50]}...")
                return True

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    async def close(self):
        """Close connections"""
        if self.ws:
            await self.ws.close()
            self.ws = None

        if self.session:
            await self.session.close()
            self.session = None
