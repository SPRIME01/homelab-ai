"""
Voice Assistant Pipeline for Home Assistant with Triton Inference Server.

This package implements a voice assistant that integrates Home Assistant with
Triton Inference Server for efficient AI inference on Jetson AGX Orin.
"""

from .pipeline import VoiceAssistantPipeline
from .wake_word import WakeWordDetector
from .stt import SpeechToText
from .nlu import NaturalLanguageUnderstanding
from .intent_resolver import IntentResolver
from .tts import TextToSpeech
from .audio import AudioManager

__version__ = "0.1.0"
