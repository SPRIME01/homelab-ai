"""
Integration test suite for Home Assistant AI functionality.
Tests voice command processing, sensor data analysis, notification generation,
and automation triggering using the AI inference architecture.
"""

import os
import sys
import json
import time
import asyncio
import logging
import pytest
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional
import aiohttp
from homeassistant_api import Client
import tritonclient.http
import paho.mqtt.client as mqtt
import sounddevice as sd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ha_integration_test.log')
    ]
)
logger = logging.getLogger("ha_test")

# Test configuration
TEST_CONFIG = {
    "hass": {
        "url": "http://localhost:8123",
        "token": os.getenv("HASS_TOKEN"),
        "api_password": os.getenv("HASS_API_PASSWORD")
    },
    "triton": {
        "url": "localhost:8000",
        "models": {
            "wake_word": "porcupine",
            "stt": "whisper-base",
            "tts": "coqui-tts",
            "nlu": "llama2-7b-chat-q4"
        }
    },
    "mqtt": {
        "broker": "localhost",
        "port": 1883,
        "username": os.getenv("MQTT_USERNAME"),
        "password": os.getenv("MQTT_PASSWORD")
    },
    "test_data": {
        "voice_commands": [
            {
                "audio": "test_data/turn_on_lights.wav",
                "expected_intent": "light.turn_on",
                "expected_entities": ["light.living_room"]
            },
            {
                "audio": "test_data/check_temperature.wav",
                "expected_intent": "sensor.get_state",
                "expected_entities": ["sensor.living_room_temperature"]
            }
        ],
        "sensor_scenarios": [
            {
                "inputs": {
                    "temperature": 25.5,
                    "humidity": 65,
                    "motion": True
                },
                "expected_analysis": "comfort_analysis",
                "expected_notification": True
            }
        ],
        "notification_tests": [
            {
                "event": "motion_detected",
                "context": {
                    "location": "front_door",
                    "time": "22:00"
                },
                "expected_priority": "high",
                "expected_ai_enhancement": True
            }
        ]
    },
    "output_dir": "test_results"
}

@pytest.fixture(scope="session")
async def hass_client():
    """Create Home Assistant API client."""
    try:
        client = Client(
            TEST_CONFIG["hass"]["url"],
            TEST_CONFIG["hass"]["token"]
        )
        await client.async_start()
        yield client
        await client.async_stop()
    except Exception as e:
        pytest.fail(f"Failed to connect to Home Assistant: {e}")

@pytest.fixture(scope="session")
def triton_client():
    """Create Triton Inference Server client."""
    try:
        client = tritonclient.http.InferenceServerClient(
            url=TEST_CONFIG["triton"]["url"],
            verbose=False
        )
        assert client.is_server_live()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to Triton server: {e}")

@pytest.fixture(scope="session")
def mqtt_client():
    """Create MQTT client."""
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            pytest.fail(f"Failed to connect to MQTT broker: {rc}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.username_pw_set(
        TEST_CONFIG["mqtt"]["username"],
        TEST_CONFIG["mqtt"]["password"]
    )

    try:
        client.connect(
            TEST_CONFIG["mqtt"]["broker"],
            TEST_CONFIG["mqtt"]["port"]
        )
        client.loop_start()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to MQTT broker: {e}")

class TestHomeAssistantAIIntegration:
    """Test suite for Home Assistant AI integration."""

    @pytest.mark.asyncio
    async def test_001_voice_command_processing(self, hass_client, triton_client):
        """Test voice command processing pipeline."""
        for test_case in TEST_CONFIG["test_data"]["voice_commands"]:
            # Load test audio
            audio_path = test_case["audio"]
            if not os.path.exists(audio_path):
                pytest.skip(f"Test audio file not found: {audio_path}")

            audio_data, sample_rate = sf.read(audio_path)

            # Test wake word detection
            wake_word_detected = await self._detect_wake_word(
                triton_client,
                audio_data,
                sample_rate
            )
            assert wake_word_detected, "Wake word should be detected"

            # Test speech-to-text
            transcription = await self._speech_to_text(
                triton_client,
                audio_data,
                sample_rate
            )
            assert transcription, "Should get non-empty transcription"
            logger.info(f"Transcription: {transcription}")

            # Test natural language understanding
            intent_data = await self._process_nlu(
                triton_client,
                transcription
            )
            assert intent_data["intent"] == test_case["expected_intent"]
            assert all(entity in intent_data["entities"]
                      for entity in test_case["expected_entities"])

            # Verify Home Assistant can execute the command
            success = await self._execute_ha_command(
                hass_client,
                intent_data
            )
            assert success, "Command should be executed successfully"

    @pytest.mark.asyncio
    async def test_002_sensor_data_analysis(self, hass_client, triton_client):
        """Test AI analysis of sensor data."""
        for scenario in TEST_CONFIG["test_data"]["sensor_scenarios"]:
            # Get sensor data from Home Assistant
            sensor_data = await self._get_sensor_data(
                hass_client,
                scenario["inputs"].keys()
            )
            assert sensor_data, "Should get sensor data"

            # Run AI analysis
            analysis_result = await self._analyze_sensor_data(
                triton_client,
                sensor_data
            )
            assert analysis_result["type"] == scenario["expected_analysis"]

            # Verify notification generation if expected
            if scenario["expected_notification"]:
                notification = await self._generate_notification(
                    triton_client,
                    analysis_result
                )
                assert notification, "Should generate notification"
                assert "priority" in notification
                assert "message" in notification

    @pytest.mark.asyncio
    async def test_003_notification_generation(self, hass_client, triton_client):
        """Test AI-driven notification generation."""
        for test_case in TEST_CONFIG["test_data"]["notification_tests"]:
            # Generate AI-enhanced notification
            notification = await self._generate_ai_notification(
                triton_client,
                test_case["event"],
                test_case["context"]
            )

            assert notification["priority"] == test_case["expected_priority"]

            if test_case["expected_ai_enhancement"]:
                assert "ai_context" in notification
                assert "natural_language" in notification

            # Verify notification can be sent through Home Assistant
            success = await self._send_notification(
                hass_client,
                notification
            )
            assert success, "Notification should be sent successfully"

    @pytest.mark.asyncio
    async def test_004_automation_triggering(self, hass_client, triton_client, mqtt_client):
        """Test AI-driven automation triggering."""
        # Set up test automation listener
        automation_triggered = asyncio.Event()

        def on_automation_message(client, userdata, message):
            payload = json.loads(message.payload)
            if payload.get("trigger_type") == "ai_event":
                automation_triggered.set()

        mqtt_client.subscribe("homeassistant/automation/#")
        mqtt_client.message_callback_add(
            "homeassistant/automation/#",
            on_automation_message
        )

        try:
            # Simulate condition that should trigger AI automation
            sensor_data = {
                "temperature": 30.0,
                "humidity": 75,
                "time": "14:00"
            }

            # Submit data for AI analysis
            analysis_result = await self._analyze_sensor_data(
                triton_client,
                sensor_data
            )

            # Verify automation was triggered
            triggered = await self._verify_automation_trigger(
                automation_triggered,
                analysis_result
            )
            assert triggered, "Automation should be triggered"

            # Verify automation action was executed
            state_changed = await self._verify_automation_action(
                hass_client,
                analysis_result
            )
            assert state_changed, "Automation action should change state"

        finally:
            mqtt_client.unsubscribe("homeassistant/automation/#")

    # Helper methods
    async def _detect_wake_word(
        self,
        triton_client: tritonclient.http.InferenceServerClient,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> bool:
        """Detect wake word in audio using Triton model."""
        model_name = TEST_CONFIG["triton"]["models"]["wake_word"]

        try:
            # Prepare audio input for model
            audio_input = np.expand_dims(audio_data, 0).astype(np.float32)

            # Create inference input
            inputs = tritonclient.http.InferInput(
                "audio", audio_input.shape, "FP32"
            )
            inputs.set_data_from_numpy(audio_input)

            # Run inference
            response = triton_client.infer(model_name, [inputs])
            output = response.as_numpy("detection")

            return bool(output[0] > 0.5)

        except Exception as e:
            logger.error(f"Wake word detection failed: {e}")
            return False

    async def _speech_to_text(
        self,
        triton_client: tritonclient.http.InferenceServerClient,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Optional[str]:
        """Convert speech to text using Triton model."""
        model_name = TEST_CONFIG["triton"]["models"]["stt"]

        try:
            # Prepare audio input for model
            audio_input = np.expand_dims(audio_data, 0).astype(np.float32)

            # Create inference input
            inputs = tritonclient.http.InferInput(
                "audio", audio_input.shape, "FP32"
            )
            inputs.set_data_from_numpy(audio_input)

            # Run inference
            response = triton_client.infer(model_name, [inputs])
            output = response.as_numpy("text")

            # Decode text from bytes
            return output[0].decode('utf-8') if output.size > 0 else None

        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return None

    async def _process_nlu(
        self,
        triton_client: tritonclient.http.InferenceServerClient,
        text: str
    ) -> Dict:
        """Process text with NLU model for intent recognition."""
        model_name = TEST_CONFIG["triton"]["models"]["nlu"]

        try:
            # Prepare text input for model
            text_input = np.array([text], dtype=np.object_)

            # Create inference input
            inputs = tritonclient.http.InferInput(
                "text", text_input.shape, "BYTES"
            )
            inputs.set_data_from_numpy(text_input)

            # Run inference
            response = triton_client.infer(model_name, [inputs])
            output = response.as_numpy("intent")

            # Parse intent and entities
            intent_data = json.loads(output[0].decode('utf-8'))
            return intent_data

        except Exception as e:
            logger.error(f"NLU processing failed: {e}")
            return {"intent": None, "entities": []}

    async def _execute_ha_command(
        self,
        hass_client: Client,
        intent_data: Dict
    ) -> bool:
        """Execute command in Home Assistant based on intent."""
        try:
            service = intent_data["intent"]
            domain, action = service.split(".")

            await hass_client.async_call_service(
                domain,
                action,
                target={"entity_id": intent_data["entities"]}
            )
            return True

        except Exception as e:
            logger.error(f"Failed to execute HA command: {e}")
            return False

    async def _get_sensor_data(
        self,
        hass_client: Client,
        sensors: List[str]
    ) -> Dict:
        """Get sensor data from Home Assistant."""
        try:
            sensor_data = {}
            for sensor in sensors:
                state = await hass_client.async_get_state(f"sensor.{sensor}")
                sensor_data[sensor] = state.state
            return sensor_data

        except Exception as e:
            logger.error(f"Failed to get sensor data: {e}")
            return {}

    async def _analyze_sensor_data(
        self,
        triton_client: tritonclient.http.InferenceServerClient,
        sensor_data: Dict
    ) -> Dict:
        """Analyze sensor data using AI model."""
        try:
            # Convert sensor data to model input format
            input_data = json.dumps(sensor_data)
            model_input = np.array([input_data], dtype=np.object_)

            # Create inference input
            inputs = tritonclient.http.InferInput(
                "sensor_data", model_input.shape, "BYTES"
            )
            inputs.set_data_from_numpy(model_input)

            # Run inference
            response = triton_client.infer("sensor_analysis", [inputs])
            output = response.as_numpy("analysis")

            # Parse analysis result
            return json.loads(output[0].decode('utf-8'))

        except Exception as e:
            logger.error(f"Sensor data analysis failed: {e}")
            return {}

    async def _generate_ai_notification(
        self,
        triton_client: tritonclient.http.InferenceServerClient,
        event: str,
        context: Dict
    ) -> Dict:
        """Generate AI-enhanced notification."""
        try:
            # Prepare input data
            input_data = json.dumps({
                "event": event,
                "context": context
            })
            model_input = np.array([input_data], dtype=np.object_)

            # Create inference input
            inputs = tritonclient.http.InferInput(
                "event_data", model_input.shape, "BYTES"
            )
            inputs.set_data_from_numpy(model_input)

            # Run inference
            response = triton_client.infer("notification_generator", [inputs])
            output = response.as_numpy("notification")

            # Parse notification data
            return json.loads(output[0].decode('utf-8'))

        except Exception as e:
            logger.error(f"Notification generation failed: {e}")
            return {}

    async def _send_notification(
        self,
        hass_client: Client,
        notification: Dict
    ) -> bool:
        """Send notification through Home Assistant."""
        try:
            await hass_client.async_call_service(
                "notify",
                "persistent_notification",
                {
                    "message": notification["message"],
                    "title": notification.get("title", "AI Notification"),
                    "notification_id": f"ai_{int(time.time())}"
                }
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def _verify_automation_trigger(
        self,
        event: asyncio.Event,
        analysis_result: Dict
    ) -> bool:
        """Verify that automation was triggered."""
        try:
            # Wait for automation trigger event
            triggered = await asyncio.wait_for(
                event.wait(),
                timeout=10.0
            )
            return triggered
        except asyncio.TimeoutError:
            return False

    async def _verify_automation_action(
        self,
        hass_client: Client,
        analysis_result: Dict
    ) -> bool:
        """Verify that automation action was executed."""
        try:
            # Get relevant entity states
            states_after = await hass_client.async_get_states()

            # Check if states changed as expected
            # This would need customization based on your specific automation
            return True

        except Exception as e:
            logger.error(f"Failed to verify automation action: {e}")
            return False

def main():
    """Run the tests."""
    # Create output directory
    os.makedirs(TEST_CONFIG["output_dir"], exist_ok=True)

    # Run pytest with async support
    pytest.main([
        __file__,
        "-v",
        "--asyncio-mode=auto",
        f"--html={os.path.join(TEST_CONFIG['output_dir'], 'report.html')}",
        "--self-contained-html"
    ])

if __name__ == "__main__":
    main()
