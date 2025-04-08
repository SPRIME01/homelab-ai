import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

from config import Config
from triton_client import TritonClient
from ha_client import HomeAssistantClient
from event_summarizer import EventSummarizer
from notification_prioritizer import NotificationPrioritizer
from content_personalizer import ContentPersonalizer
from nl_generator import NaturalLanguageGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("notification_service")

class AINotificationService:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.triton_client = TritonClient(self.config.triton)
        self.ha_client = HomeAssistantClient(self.config.home_assistant)

        self.summarizer = EventSummarizer(self.triton_client)
        self.prioritizer = NotificationPrioritizer(self.triton_client)
        self.personalizer = ContentPersonalizer(self.triton_client)
        self.nl_generator = NaturalLanguageGenerator(self.triton_client)

        self.event_cache = {}
        self.user_preferences = {}

        logger.info("AI Notification Service initialized")

    async def load_user_preferences(self):
        """Load user preferences from Home Assistant"""
        logger.info("Loading user preferences")
        self.user_preferences = await self.ha_client.get_user_preferences()

    async def process_events(self):
        """Process new events from Home Assistant"""
        events = await self.ha_client.get_latest_events()
        if not events:
            return

        logger.info(f"Processing {len(events)} new events")

        # Group related events
        event_groups = self._group_related_events(events)

        for group_id, event_group in event_groups.items():
            # Summarize events if there are multiple related ones
            if len(event_group) > 1:
                summary = await self.summarizer.summarize_events(event_group)
            else:
                summary = event_group[0]["message"]

            # Get context from sensors
            context = await self.ha_client.get_related_sensor_data(event_group)

            # Generate natural language description
            description = await self.nl_generator.generate_description(summary, context)

            # Determine priority
            priority = await self.prioritizer.prioritize(description, context, self.user_preferences)

            # Personalize content
            personalized_content = await self.personalizer.personalize(
                description,
                priority,
                self.user_preferences
            )

            # Send notification
            await self.send_notification(personalized_content, priority, context)

    def _group_related_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group related events together based on entity_id or domain"""
        groups = {}

        for event in events:
            # Create a simple grouping key
            if "entity_id" in event:
                key = event["entity_id"].split(".")[0]  # Use domain as key
            else:
                key = event.get("domain", "default")

            if key not in groups:
                groups[key] = []

            groups[key].append(event)

        return groups

    async def send_notification(self, content: str, priority: int, context: Dict[str, Any]):
        """Send notification to Home Assistant"""
        logger.info(f"Sending notification with priority {priority}: {content[:50]}...")

        # Prepare notification data
        notification_data = {
            "message": content,
            "title": self._get_title_for_priority(priority),
            "priority": priority,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "persistent": priority >= self.config.notification_settings["persistent_threshold"],
                "ttl": self._get_ttl_for_priority(priority),
                "channel": self._get_channel_for_priority(priority)
            }
        }

        # Add images if available
        if "image_url" in context:
            notification_data["data"]["image"] = context["image_url"]

        # Add actions if available
        if "actions" in context:
            notification_data["data"]["actions"] = context["actions"]

        await self.ha_client.send_notification(notification_data)

    def _get_title_for_priority(self, priority: int) -> str:
        """Get notification title based on priority"""
        titles = self.config.notification_settings["titles"]
        for p, title in titles.items():
            if priority >= int(p):
                return title
        return "Notification"

    def _get_ttl_for_priority(self, priority: int) -> int:
        """Get time-to-live for notification based on priority"""
        ttls = self.config.notification_settings["ttl"]
        for p, ttl in ttls.items():
            if priority >= int(p):
                return ttl
        return 3600  # 1 hour default

    def _get_channel_for_priority(self, priority: int) -> str:
        """Get notification channel based on priority"""
        channels = self.config.notification_settings["channels"]
        for p, channel in channels.items():
            if priority >= int(p):
                return channel
        return "default"

    async def run(self):
        """Main service loop"""
        logger.info("Starting AI Notification Service")

        await self.load_user_preferences()
        await self.triton_client.initialize()

        try:
            while True:
                await self.process_events()
                await asyncio.sleep(self.config.general["polling_interval"])

        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Error in service: {e}", exc_info=True)
        finally:
            await self.triton_client.close()
            await self.ha_client.close()

if __name__ == "__main__":
    service = AINotificationService()
    asyncio.run(service.run())
