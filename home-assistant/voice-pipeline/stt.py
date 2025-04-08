import asyncio
import logging
import numpy as np
from typing import Dict, Optional, Union
import aiohttp
import json

from .triton_client import TritonClient

logger = logging.getLogger("stt")

class SpeechToText:
    """Speech-to-text conversion using Whisper on Triton Inference Server."""

    def __init__(self, triton_url: str, model_name: str = "whisper", sample_rate: int = 16000):
        """
        Initialize the speech-to-text module.

        Args:
            triton_url: URL of the Triton Inference Server
            model_name: Name of the Whisper model in Triton
            sample_rate: Sample rate of the audio
        """
        self.triton_client = TritonClient(triton_url)
        self.model_name = model_name
        self.sample_rate = sample_rate

    async def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text using Whisper on Triton.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text
        """
        try:
            # Prepare audio input (ensure float32 normalized to [-1, 1])
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Create input tensor
            inputs = {
                "audio": audio_data,
                "sample_rate": np.array([self.sample_rate], dtype=np.int32)
            }

            # Optional parameters
            params = {
                "language": "en",  # Can be parameterized
                "task": "transcribe",
                "beam_size": 5
            }

            # Send inference request to Triton
            result = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                parameters=params
            )

            # Extract text from response
            if "text" in result:
                return result["text"].decode("utf-8").strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""

        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return ""

    async def cleanup(self):
        """Clean up resources."""
        await self.triton_client.cleanup()
