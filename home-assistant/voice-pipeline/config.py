import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AudioConfig:
    input_device: Optional[int]
    output_device: Optional[int]
    sample_rate: int
    chunk_size: int
    command_timeout: float
    ack_sound_path: str


@dataclass
class WakeWordConfig:
    porcupine_key: str
    model_path: str
    sensitivity: float
    keywords: List[str]


@dataclass
class TritonConfig:
    url: str
    timeout: float
    concurrency: int


@dataclass
class SttConfig:
    model_name: str
    language: str
    max_audio_secs: float


@dataclass
class NluConfig:
    model_name: str
    system_prompt: str
    max_tokens: int
    temperature: float


@dataclass
class IntentConfig:
    mappings_path: str
    confidence_threshold: float


@dataclass
class TtsConfig:
    model_name: str
    voice_id: str
    speed: float


@dataclass
class HomeAssistantConfig:
    url: str
    token: str
    verify_ssl: bool


@dataclass
class PipelineConfig:
    audio: AudioConfig
    wake_word: WakeWordConfig
    triton: TritonConfig
    stt: SttConfig
    nlu: NluConfig
    intent: IntentConfig
    tts: TtsConfig
    home_assistant: HomeAssistantConfig

    @classmethod
    def from_file(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Fill in environment variables for sensitive data
        if "PORCUPINE_KEY" in os.environ:
            config_data["wake_word"]["porcupine_key"] = os.environ["PORCUPINE_KEY"]
        if "HOME_ASSISTANT_TOKEN" in os.environ:
            config_data["home_assistant"]["token"] = os.environ["HOME_ASSISTANT_TOKEN"]

        # Create config objects
        audio_config = AudioConfig(**config_data["audio"])
        wake_word_config = WakeWordConfig(**config_data["wake_word"])
        triton_config = TritonConfig(**config_data["triton"])
        stt_config = SttConfig(**config_data["stt"])
        nlu_config = NluConfig(**config_data["nlu"])
        intent_config = IntentConfig(**config_data["intent"])
        tts_config = TtsConfig(**config_data["tts"])
        home_assistant_config = HomeAssistantConfig(**config_data["home_assistant"])

        return cls(
            audio=audio_config,
            wake_word=wake_word_config,
            triton=triton_config,
            stt=stt_config,
            nlu=nlu_config,
            intent=intent_config,
            tts=tts_config,
            home_assistant=home_assistant_config
        )
