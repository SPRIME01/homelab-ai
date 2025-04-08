"""Sensor platform for the TritonAI integration."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change
from homeassistant.helpers.typing import StateType

from .const import (
    DOMAIN,
    ATTR_ANOMALY_SCORE,
    ATTR_PREDICTION,
    EVENT_PREDICTION_UPDATED,
    EVENT_ANOMALY_DETECTED,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_platform(
    hass: HomeAssistant,
    config: Dict[str, Any],
    async_add_entities: AddEntitiesCallback,
    discovery_info=None,
) -> None:
    """Set up the TritonAI sensor platform."""
    if discovery_info is None:
        return

    # Get sensor analysis service
    sensor_analysis = hass.data[DOMAIN]["services"].get("sensor_analysis")
    if not sensor_analysis:
        _LOGGER.error("Sensor analysis service not available")
        return

    # Create entities for tracked sensors
    entities = []
    for entity_id in sensor_analysis.tracked_sensors:
        # Create anomaly score sensor
        anomaly_sensor = AnomalyScoreSensor(hass, entity_id)
        entities.append(anomaly_sensor)

        # Create prediction sensor if numeric
        state = hass.states.get(entity_id)
        if state and _is_numeric_sensor(state):
            prediction_sensor = PredictionSensor(hass, entity_id)
            entities.append(prediction_sensor)

    if entities:
        async_add_entities(entities)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the TritonAI sensors from a config entry."""
    await async_setup_platform(
        hass, {}, async_add_entities, {"config_entry": config_entry}
    )

def _is_numeric_sensor(state) -> bool:
    """Check if a sensor state is numeric."""
    try:
        float(state.state)
        return True
    except (ValueError, TypeError):
        return False

class AnomalyScoreSensor(SensorEntity):
    """Sensor for anomaly detection score."""

    def __init__(self, hass: HomeAssistant, entity_id: str):
        """Initialize the anomaly score sensor."""
        self.hass = hass
        self._source_entity_id = entity_id
        self._attr_name = f"{entity_id.split('.')[-1]} Anomaly Score"
        self._attr_unique_id = f"{entity_id}_anomaly_score"
        self._attr_native_value = 0.0
        self._attr_device_class = None
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_native_unit_of_measurement = None
        self._attr_icon = "mdi:chart-bell-curve"
        self._attr_extra_state_attributes = {
            "source_entity": entity_id,
            "threshold": 0.5,
            "last_updated": None
        }

        # Register event listener
        @callback
        def handle_anomaly_event(event):
            """Handle anomaly detection event."""
            if event.data.get("entity_id") == self._source_entity_id:
                self._attr_native_value = event.data.get("score", 0.0)
                self._attr_extra_state_attributes.update({
                    "threshold": event.data.get("threshold", 0.5),
                    "description": event.data.get("description", ""),
                    "last_updated": datetime.now().isoformat()
                })
                self.async_write_ha_state()

        hass.bus.async_listen(EVENT_ANOMALY_DETECTED, handle_anomaly_event)

class PredictionSensor(SensorEntity):
    """Sensor for future value prediction."""

    def __init__(self, hass: HomeAssistant, entity_id: str):
        """Initialize the prediction sensor."""
        self.hass = hass
        self._source_entity_id = entity_id
        self._attr_name = f"{entity_id.split('.')[-1]} Prediction"
        self._attr_unique_id = f"{entity_id}_prediction"
        self._attr_native_value = None

        # Try to get unit and device class from source sensor
        source_state = hass.states.get(entity_id)
        if source_state:
            self._attr_native_unit_of_measurement = source_state.attributes.get("unit_of_measurement")
            self._attr_device_class = source_state.attributes.get("device_class")
            self._attr_state_class = source_state.attributes.get("state_class")

        self._attr_icon = "mdi:crystal-ball"
        self._attr_extra_state_attributes = {
            "source_entity": entity_id,
            "forecast": [],
            "forecast_timestamps": [],
            "last_updated": None
        }

        # Register event listener
        @callback
        def handle_prediction_event(event):
            """Handle prediction update event."""
            if event.data.get("entity_id") == self._source_entity_id:
                forecast = event.data.get("forecast", [])
                timestamps = event.data.get("timestamps", [])

                if forecast and len(forecast) > 0:
                    # Set the first prediction as the state
                    self._attr_native_value = forecast[0]

                    self._attr_extra_state_attributes.update({
                        "forecast": forecast,
                        "forecast_timestamps": timestamps,
                        "last_updated": datetime.now().isoformat()
                    })
                    self.async_write_ha_state()

        hass.bus.async_listen(EVENT_PREDICTION_UPDATED, handle_prediction_event)
