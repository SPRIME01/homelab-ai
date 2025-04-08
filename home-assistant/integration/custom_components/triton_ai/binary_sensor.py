"""Binary sensor platform for the TritonAI integration."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    EVENT_ANOMALY_DETECTED,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_platform(
    hass: HomeAssistant,
    config: Dict[str, Any],
    async_add_entities: AddEntitiesCallback,
    discovery_info=None,
) -> None:
    """Set up the TritonAI binary sensor platform."""
    if discovery_info is None:
        return

    # Get sensor analysis service
    sensor_analysis = hass.data[DOMAIN]["services"].get("sensor_analysis")
    if not sensor_analysis:
        _LOGGER.error("Sensor analysis service not available")
        return

    # Create anomaly detection binary sensors for tracked sensors
    entities = []
    for entity_id in sensor_analysis.tracked_sensors:
        anomaly_sensor = AnomalyDetectionSensor(hass, entity_id)
        entities.append(anomaly_sensor)

    if entities:
        async_add_entities(entities)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the TritonAI binary sensors from a config entry."""
    await async_setup_platform(
        hass, {}, async_add_entities, {"config_entry": config_entry}
    )

class AnomalyDetectionSensor(BinarySensorEntity):
    """Binary sensor for anomaly detection."""

    def __init__(self, hass: HomeAssistant, entity_id: str):
        """Initialize the anomaly detection binary sensor."""
        self.hass = hass
        self._source_entity_id = entity_id
        self._attr_name = f"{entity_id.split('.')[-1]} Anomaly"
        self._attr_unique_id = f"{entity_id}_anomaly"
        self._attr_is_on = False
        self._attr_device_class = BinarySensorDeviceClass.PROBLEM
        self._attr_icon = "mdi:alert-circle"
        self._attr_extra_state_attributes = {
            "source_entity": entity_id,
            "score": 0.0,
            "threshold": 0.5,
            "description": "",
            "last_detected": None
        }

        # Register event listener
        @callback
        def handle_anomaly_event(event):
            """Handle anomaly detection event."""
            if event.data.get("entity_id") == self._source_entity_id:
                score = event.data.get("score", 0.0)
                threshold = event.data.get("threshold", 0.5)
                # Set state based on whether score exceeds threshold
                self._attr_is_on = score > threshold

                self._attr_extra_state_attributes.update({
                    "score": score,
                    "threshold": threshold,
                    "description": event.data.get("description", ""),
                })

                # Update last_detected timestamp if anomaly is detected
                if self._attr_is_on:
                    self._attr_extra_state_attributes["last_detected"] = datetime.now().isoformat()

                self.async_write_ha_state()

        hass.bus.async_listen(EVENT_ANOMALY_DETECTED, handle_anomaly_event)
