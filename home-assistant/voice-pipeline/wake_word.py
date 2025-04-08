import asyncio
import logging
import numpy as np
from typing import List, Optional
import pvporcupine

from .audio import AudioManager

logger = logging.getLogger("wake_word")

class WakeWordDetector:
    """Wake word detection using Porcupine."""

    def __init__(self, access_key: str, model_path: Optional[str] = None, sensitivity: float = 0.5):
        """
        Initialize the wake word detector.

        Args:
            access_key: Porcupine access key
            model_path: Path to custom wake word model (optional)
            sensitivity: Detection sensitivity (0-1)
        """
        self.access_key = access_key
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.porcupine = None
        self.is_listening = False

    async def _init_porcupine(self):
        """Initialize the Porcupine wake word engine."""
        try:
            # Use default keywords if no custom model provided
            keywords = ["jarvis", "hey_jetson"] if not self.model_path else None
            keyword_paths = [self.model_path] if self.model_path else None

            # Run Porcupine initialization in a thread pool as it's CPU-intensive
            loop = asyncio.get_running_loop()
            self.porcupine = await loop.run_in_executor(
                None,
                lambda: pvporcupine.create(
                    access_key=self.access_key,
                    keywords=keywords,
                    keyword_paths=keyword_paths,
                    sensitivities=[self.sensitivity]
                )
            )
            logger.info(f"Porcupine initialized with sample rate: {self.porcupine.sample_rate}")

        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise

    async def listen(self, audio_manager: AudioManager) -> bool:
        """
        Listen for the wake word using the provided audio manager.

        Args:
            audio_manager: AudioManager instance for recording audio

        Returns:
            True if wake word detected, False otherwise
        """
        if not self.porcupine:
            await self._init_porcupine()

        self.is_listening = True

        try:
            # Ensure the audio manager is configured with the right sample rate
            if audio_manager.sample_rate != self.porcupine.sample_rate:
                logger.warning(
                    f"Audio manager sample rate ({audio_manager.sample_rate}) "
                    f"doesn't match Porcupine sample rate ({self.porcupine.sample_rate})"
                )

            # Start streaming audio for wake word detection
            async for audio_chunk in audio_manager.stream_audio():
                if not self.is_listening:
                    break

                # Process audio chunk with Porcupine (run in thread pool)
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.porcupine.process(audio_chunk.flatten().astype(np.int16))
                )

                if result >= 0:
                    logger.info(f"Wake word detected (keyword index: {result})")
                    return True

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False

        return False

    async def stop(self):
        """Stop listening for wake word."""
        self.is_listening = False

    async def cleanup(self):
        """Clean up Porcupine resources."""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
