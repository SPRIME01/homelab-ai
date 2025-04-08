import asyncio
import logging
import numpy as np
from typing import AsyncGenerator, List, Optional, Tuple
import wave
import sounddevice as sd
from pathlib import Path

logger = logging.getLogger("audio")

class AudioManager:
    """Manager for audio recording and playback."""

    def __init__(self, input_device: Optional[int] = None,
                output_device: Optional[int] = None,
                sample_rate: int = 16000,
                chunk_size: int = 512):
        """
        Initialize the audio manager.

        Args:
            input_device: Input device ID (None for default)
            output_device: Output device ID (None for default)
            sample_rate: Sample rate for recording/playback
            chunk_size: Chunk size for streaming
        """
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self._stream_task = None

        # Optionally log available devices
        self._log_available_devices()

    def _log_available_devices(self):
        """Log available audio devices for debugging."""
        try:
            devices = sd.query_devices()
            logger.info(f"Available audio devices:\n{devices}")

            if self.input_device is not None:
                input_info = sd.query_devices(self.input_device, 'input')
                logger.info(f"Selected input device: {input_info}")

            if self.output_device is not None:
                output_info = sd.query_devices(self.output_device, 'output')
                logger.info(f"Selected output device: {output_info}")

        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")

    async def stream_audio(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio from the input device in chunks.

        Yields:
            Audio chunks as numpy arrays
        """
        loop = asyncio.get_running_loop()

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")

            # Make a copy of the data for this chunk
            data = indata.copy()

            # Add to the queue
            try:
                loop.call_soon_threadsafe(lambda: asyncio.create_task(queue.put(data)))
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

        # Create queue for audio chunks
        queue = asyncio.Queue()

        try:
            # Start the audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.input_device,
                channels=1,
                dtype='int16',
                callback=callback
            ):
                logger.debug("Audio stream started")

                while True:
                    # Get audio chunk from queue
                    chunk = await queue.get()
                    yield chunk

        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
            raise

    async def record_command(self, timeout: float = 5.0,
                           silence_threshold: float = 0.02,
                           silence_duration: float = 1.0) -> np.ndarray:
        """
        Record audio until silence is detected or timeout occurs.

        Args:
            timeout: Maximum recording time in seconds
            silence_threshold: RMS threshold for silence detection
            silence_duration: Silence duration to stop recording (seconds)

        Returns:
            Recorded audio as numpy array
        """
        chunks = []
        silence_chunks = 0
        silence_chunks_threshold = int(silence_duration * self.sample_rate / self.chunk_size)

        try:
            start_time = asyncio.get_event_loop().time()

            async for chunk in self.stream_audio():
                chunks.append(chunk)

                # Check if this chunk is silence
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                if rms < silence_threshold:
                    silence_chunks += 1
                else:
                    silence_chunks = 0

                # Stop if we've detected enough silence
                if silence_chunks >= silence_chunks_threshold:
                    logger.debug("Stopping recording due to silence")
                    break

                # Check timeout
                if (asyncio.get_event_loop().time() - start_time) > timeout:
                    logger.debug("Stopping recording due to timeout")
                    break

            # Combine all chunks
            if chunks:
                return np.vstack(chunks)
            else:
                return np.array([], dtype=np.int16)

        except Exception as e:
            logger.error(f"Error recording command: {e}")
            return np.array([], dtype=np.int16)

    async def play_audio(self, file_path: str) -> bool:
        """
        Play audio from a file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if playback successful, False otherwise
        """
        try:
            with wave.open(file_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                channels = wf.getnchannels()
                if channels > 1:
                    data = data.reshape(-1, channels)

                return await self.play_audio_data(data, sample_rate)

        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            return False

    async def play_audio_data(self, audio_data: np.ndarray,
                            sample_rate: Optional[int] = None) -> bool:
        """
        Play audio from numpy array.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio (None to use default)

        Returns:
            True if playback successful, False otherwise
        """
        if audio_data.size == 0:
            logger.warning("Empty audio data, nothing to play")
            return False

        if sample_rate is None:
            sample_rate = self.sample_rate

        try:
            # Create an event to signal completion
            done_event = asyncio.Event()

            # Define the callback to signal when done
            def callback(outdata, frames, time, status):
                if status:
                    logger.warning(f"Audio output status: {status}")

                # Signal completion when the last chunk is played
                if frames == 0:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(done_event.set)

            # Start playing in a separate thread
            loop = asyncio.get_running_loop()

            # Normalize audio data if needed
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                    # Convert from [-1, 1] float to int16
                    audio_data = (audio_data * 32767).astype(np.int16)

            # Ensure data is in the right format
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)

            # Play the audio
            with sd.OutputStream(
                samplerate=sample_rate,
                device=self.output_device,
                channels=audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
                dtype='int16',
                callback=callback
            ) as stream:
                await loop.run_in_executor(
                    None,
                    lambda: stream.write(audio_data)
                )

                # Wait for playback to complete
                await done_event.wait()

            return True

        except Exception as e:
            logger.error(f"Error playing audio data: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        pass
