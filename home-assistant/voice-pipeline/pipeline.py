import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional
import json

from .config import PipelineConfig
from .wake_word import WakeWordDetector
from .stt import SpeechToText
from .nlu import NaturalLanguageUnderstanding
from .intent_resolver import IntentResolver
from .tts import TextToSpeech
from .audio import AudioManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_pipeline")

class VoiceAssistantPipeline:
    """Main pipeline that orchestrates the voice assistant components."""

    def __init__(self, config_path: str):
        """Initialize the voice assistant pipeline with the given configuration."""
        self.config = PipelineConfig.from_file(config_path)

        # Initialize audio manager
        self.audio_manager = AudioManager(
            input_device=self.config.audio.input_device,
            output_device=self.config.audio.output_device,
            sample_rate=self.config.audio.sample_rate,
            chunk_size=self.config.audio.chunk_size
        )

        # Initialize pipeline components
        self.wake_word = WakeWordDetector(
            access_key=self.config.wake_word.porcupine_key,
            model_path=self.config.wake_word.model_path,
            sensitivity=self.config.wake_word.sensitivity
        )

        self.stt = SpeechToText(
            triton_url=self.config.triton.url,
            model_name=self.config.stt.model_name,
            sample_rate=self.config.audio.sample_rate
        )

        self.nlu = NaturalLanguageUnderstanding(
            triton_url=self.config.triton.url,
            model_name=self.config.nlu.model_name,
            system_prompt=self.config.nlu.system_prompt
        )

        self.intent_resolver = IntentResolver(
            ha_url=self.config.home_assistant.url,
            ha_token=self.config.home_assistant.token,
            intent_mappings_path=self.config.intent.mappings_path
        )

        self.tts = TextToSpeech(
            triton_url=self.config.triton.url,
            model_name=self.config.tts.model_name,
            sample_rate=self.config.audio.sample_rate
        )

        logger.info("Voice assistant pipeline initialized")

    async def start(self):
        """Start the voice assistant pipeline."""
        logger.info("Starting voice assistant pipeline")

        try:
            while True:
                # Step 1: Listen for wake word
                logger.info("Listening for wake word...")
                await self.wake_word.listen(self.audio_manager)
                logger.info("Wake word detected!")

                # Play an acknowledgement sound
                await self.audio_manager.play_audio(self.config.audio.ack_sound_path)

                # Step 2: Record audio for command
                logger.info("Listening for command...")
                audio_data = await self.audio_manager.record_command(
                    timeout=self.config.audio.command_timeout
                )

                # Step 3: Convert speech to text
                logger.info("Converting speech to text...")
                text = await self.stt.transcribe(audio_data)
                if not text:
                    logger.info("No speech detected")
                    continue

                logger.info(f"Transcribed: '{text}'")

                # Step 4: Perform natural language understanding
                logger.info("Performing NLU...")
                nlu_result = await self.nlu.understand(text)

                # Step 5: Resolve intent to Home Assistant action
                logger.info("Resolving intent...")
                action_result = await self.intent_resolver.resolve_intent(nlu_result)

                # Step 6: Generate response
                response_text = action_result.get("response", "I've processed your request.")
                logger.info(f"Response: '{response_text}'")

                # Step 7: Convert response to speech
                logger.info("Converting response to speech...")
                speech_audio = await self.tts.synthesize(response_text)

                # Step 8: Play the response
                await self.audio_manager.play_audio_data(speech_audio)

                # Brief pause before listening again
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Stopping voice assistant pipeline")
        except Exception as e:
            logger.exception(f"Error in voice assistant pipeline: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        logger.info("Cleaning up resources")
        await self.wake_word.cleanup()
        await self.audio_manager.cleanup()

def main():
    """Entry point for the voice assistant pipeline."""
    import os

    config_path = os.environ.get(
        "VOICE_PIPELINE_CONFIG",
        str(Path(__file__).parent / "config" / "pipeline.yaml")
    )

    pipeline = VoiceAssistantPipeline(config_path)
    asyncio.run(pipeline.start())

if __name__ == "__main__":
    main()
