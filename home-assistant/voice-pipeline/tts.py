import asyncio
import logging
import numpy as np
from typing import Dict, Optional
import aiohttp

from .triton_client import TritonClient

logger = logging.getLogger("tts")

class TextToSpeech:
    """Text-to-speech synthesis using Triton Inference Server."""

    def __init__(self, triton_url: str, model_name: str = "fastpitch-waveglow",
                 sample_rate: int = 22050):
        """
        Initialize the text-to-speech module.

        Args:
            triton_url: URL of the Triton Inference Server
            model_name: Name of the TTS model in Triton
            sample_rate: Sample rate for output audio
        """
        self.triton_client = TritonClient(triton_url)
        self.model_name = model_name
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text to speech using TTS model on Triton.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array
        """
        try:
            # Prepare input tensor
            inputs = {
                "text": np.array([text], dtype=np.object_)
            }

            # Optional parameters
            params = {
                "voice_id": "en-US-neural2-F",  # Can be parameterized
                "speaking_rate": 1.0,
                "pitch": 0.0
            }

            # Send inference request to Triton
            result = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                parameters=params
            )

            # Extract audio from response
            if "audio" in result:
                audio_data = result["audio"]

                # Ensure audio is in the correct format (float32 in range [-1, 1])
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                    audio_data = np.clip(audio_data / 32768.0, -1.0, 1.0)

                return audio_data
            else:
                logger.error(f"Unexpected response format: {result}")
                return np.zeros(0, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return np.zeros(0, dtype=np.float32)

    async def cleanup(self):
        """Clean up resources."""
        await self.triton_client.cleanup()
