"""Service for analyzing sensor data using AI models."""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import numpy as np

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.event import async_track_state_change
from homeassistant.util import dt as dt_util

from .triton_client import TritonClient
from .ray_manager import RayTaskManager
from .const import (
    EVENT_ANOMALY_DETECTED,
    EVENT_PREDICTION_UPDATED,
    ATTR_ANOMALY_SCORE,
    ATTR_PREDICTION,
)

_LOGGER = logging.getLogger(__name__)

# Supported sensor domains for analysis
SUPPORTED_DOMAINS = [
    "sensor",
    "binary_sensor",
    "climate",
    "weather",
    "device_tracker"
]

# Analysis models
ANOMALY_DETECTION_MODEL = "sensor_anomaly"
FORECASTING_MODEL = "sensor_forecast"

class SensorAnalysisService:
    """Service for analyzing sensor data using AI models."""

    def __init__(
        self,
        hass: HomeAssistant,
        triton_client: TritonClient,
        ray_manager: RayTaskManager,
        history_window: int = 24  # hours
    ):
        """Initialize the sensor analysis service.

        Args:
            hass: Home Assistant instance
            triton_client: Triton client for inference
            ray_manager: Ray manager for distributed tasks
            history_window: Hours of history to use for analysis
        """
        self.hass = hass
        self.triton_client = triton_client
        self.ray_manager = ray_manager
        self.history_window = history_window
        self.tracked_sensors = set()
        self.sensor_data = {}
        self.analysis_results = {}

    async def initialize(self) -> bool:
        """Initialize the sensor analysis service."""
        # Check if required models are available
        models_available = True

        anomaly_metadata = await self.triton_client.get_model_metadata(ANOMALY_DETECTION_MODEL)
        if not anomaly_metadata:
            _LOGGER.warning("Anomaly detection model not available")
            models_available = False

        forecast_metadata = await self.triton_client.get_model_metadata(FORECASTING_MODEL)
        if not forecast_metadata:
            _LOGGER.warning("Forecasting model not available")
            models_available = False

        # Set up state tracking for relevant entities
        @callback
        def sensor_state_change(entity_id, old_state, new_state):
            """Handle sensor state changes."""
            if new_state is None or old_state is None:
                return

            # Skip if value hasn't changed
            if new_state.state == old_state.state:
                return

            # Store sensor data
            try:
                # Try to convert value to float for numerical sensors
                value = float(new_state.state)
                timestamp = dt_util.as_timestamp(new_state.last_updated)

                if entity_id not in self.sensor_data:
                    self.sensor_data[entity_id] = []

                self.sensor_data[entity_id].append({
                    "timestamp": timestamp,
                    "value": value,
                    "attributes": dict(new_state.attributes)
                })

                # Limit history length
                max_age = dt_util.as_timestamp(
                    dt_util.utcnow() - timedelta(hours=self.history_window)
                )
                self.sensor_data[entity_id] = [
                    item for item in self.sensor_data[entity_id]
                    if item["timestamp"] >= max_age
                ]

                # Schedule analysis for this sensor
                asyncio.create_task(self.analyze_sensor(entity_id))

            except ValueError:
                # Not a numerical sensor, store as-is
                pass

        # Find and track relevant sensors
        for entity_id in self.hass.states.async_entity_ids():
            domain = entity_id.split(".", 1)[0]
            if domain in SUPPORTED_DOMAINS:
                self.tracked_sensors.add(entity_id)

        # Set up state tracking
        async_track_state_change(
            self.hass,
            list(self.tracked_sensors),
            sensor_state_change
        )

        _LOGGER.info("Initialized sensor analysis service tracking %s sensors",
                    len(self.tracked_sensors))

        return models_available

    async def analyze_sensor(self, entity_id: str) -> Dict[str, Any]:
        """Analyze a specific sensor for anomalies and future predictions.

        Args:
            entity_id: Entity ID of the sensor to analyze

        Returns:
            Dictionary with analysis results
        """
        # Skip if we don't have data for this sensor
        if entity_id not in self.sensor_data or not self.sensor_data[entity_id]:
            return {"status": "no_data"}

        # Skip if sensor doesn't have enough data points
        sensor_data = self.sensor_data[entity_id]
        if len(sensor_data) < 10:  # Need at least 10 data points
            return {"status": "insufficient_data"}

        # Extract features
        try:
            # Extract timestamps and values
            timestamps = np.array([item["timestamp"] for item in sensor_data])
            values = np.array([item["value"] for item in sensor_data])

            # Normalize time to hours since start
            start_time = timestamps[0]
            times_hours = (timestamps - start_time) / 3600.0

            # Check if sensor has constant values
            if np.std(values) < 1e-6:
                # Skip analysis for constant sensors
                return {"status": "constant_values"}

            # Run anomaly detection
            anomaly_result = await self._detect_anomalies(entity_id, times_hours, values)

            # Run forecasting
            forecast_result = await self._generate_forecast(entity_id, times_hours, values)

            # Combine results
            result = {
                "entity_id": entity_id,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "anomaly": anomaly_result,
                "forecast": forecast_result
            }

            # Store results
            self.analysis_results[entity_id] = result

            # Fire events if needed
            if anomaly_result.get("anomaly_detected", False):
                self.hass.bus.async_fire(EVENT_ANOMALY_DETECTED, {
                    "entity_id": entity_id,
                    "score": anomaly_result.get("score", 0),
                    "threshold": anomaly_result.get("threshold", 0),
                    "description": anomaly_result.get("description", "")
                })

            if forecast_result.get("values") is not None:
                self.hass.bus.async_fire(EVENT_PREDICTION_UPDATED, {
                    "entity_id": entity_id,
                    "forecast": forecast_result.get("values", []),
                    "timestamps": forecast_result.get("timestamps", [])
                })

            return result

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error analyzing sensor %s: %s", entity_id, err)
            return {"status": "error", "error": str(err)}

    async def analyze_all(self) -> Dict[str, Any]:
        """Analyze all tracked sensors."""
        results = {}

        for entity_id in self.tracked_sensors:
            if entity_id in self.sensor_data and len(self.sensor_data[entity_id]) >= 10:
                results[entity_id] = await self.analyze_sensor(entity_id)

        return results

    async def _detect_anomalies(
        self,
        entity_id: str,
        times: np.ndarray,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Detect anomalies in sensor data.

        Args:
            entity_id: Entity ID of the sensor
            times: Time values in hours (normalized)
            values: Sensor values

        Returns:
            Dictionary with anomaly detection results
        """
        # Prepare inputs for the model
        inputs = {
            "times": times.astype(np.float32),
            "values": values.astype(np.float32),
            "entity_id": np.array([entity_id], dtype=np.object_)
        }

        # Run inference for anomaly detection
        result = await self.triton_client.infer(
            model_name=ANOMALY_DETECTION_MODEL,
            inputs=inputs
        )

        if result["status"] == "success" and "outputs" in result:
            outputs = result["outputs"]

            # Extract outputs
            scores = outputs.get("anomaly_scores", np.array([0.0])).flatten()
            threshold = float(outputs.get("threshold", np.array([0.5])).item())
            is_anomaly = scores[-1] > threshold

            # Get anomaly description if available
            description = ""
            if "description" in outputs and outputs["description"].size > 0:
                description = outputs["description"][0]
                if isinstance(description, bytes):
                    description = description.decode("utf-8")

            return {
                "anomaly_detected": is_anomaly,
                "score": float(scores[-1]),
                "threshold": threshold,
                "description": description,
                "recent_scores": scores[-5:].tolist() if len(scores) > 5 else scores.tolist()
            }
        else:
            _LOGGER.error("Anomaly detection failed for %s: %s",
                         entity_id, result.get("error", "unknown error"))
            return {"anomaly_detected": False, "error": result.get("error", "unknown error")}

    async def _generate_forecast(
        self,
        entity_id: str,
        times: np.ndarray,
        values: np.ndarray
    ) -> Dict[str, Any]:
        """Generate forecast for sensor values.

        Args:
            entity_id: Entity ID of the sensor
            times: Time values in hours (normalized)
            values: Sensor values

        Returns:
            Dictionary with forecast results
        """
        # Prepare inputs for the model
        inputs = {
            "times": times.astype(np.float32),
            "values": values.astype(np.float32),
            "entity_id": np.array([entity_id], dtype=np.object_),
            "forecast_hours": np.array([24.0], dtype=np.float32)  # 24-hour forecast
        }

        # Run inference for forecasting
        result = await self.triton_client.infer(
            model_name=FORECASTING_MODEL,
            inputs=inputs
        )

        if result["status"] == "success" and "outputs" in result:
            outputs = result["outputs"]

            # Extract outputs
            forecast_values = outputs.get("forecast_values", np.array([])).flatten()
            forecast_times = outputs.get("forecast_times", np.array([])).flatten()

            # Convert times back to timestamps
            start_time = times[0]
            forecast_timestamps = (forecast_times * 3600.0) + start_time

            return {
                "values": forecast_values.tolist(),
                "timestamps": [datetime.fromtimestamp(ts).isoformat()
                              for ts in forecast_timestamps],
                "horizon_hours": 24.0
            }
        else:
            _LOGGER.error("Forecasting failed for %s: %s",
                         entity_id, result.get("error", "unknown error"))
            return {"values": None, "error": result.get("error", "unknown error")}

    async def close(self):
        """Clean up resources."""
        self.tracked_sensors.clear()
        self.sensor_data.clear()
        self.analysis_results.clear()
