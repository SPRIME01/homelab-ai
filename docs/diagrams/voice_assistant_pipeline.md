# Voice Assistant Pipeline

This diagram shows the voice assistant pipeline that integrates Home Assistant with Triton Inference Server, including all major components and data flows.

## System Architecture Overview

```mermaid
flowchart TB
    %% User interactions and physical devices
    User([User]) <-->|Voice Commands\nResponses| Microphone[Microphone Array]
    User <-->|Visual Feedback| HomeUI[Home Assistant UI]
    Speaker[Speakers] -->|Audio Output| User

    subgraph IoT["Home Environment"]
        Microphone
        Speaker
        Devices[Smart Home Devices]
    end

    subgraph HomeAssistant["Home Assistant"]
        HACore[Core]
        VoiceIntegration[Voice Integration]
        IntentProcessor[Intent Processor]
        EntityRegistry[Entity Registry]
        ResponseGenerator[Response Generator]
        HACore <--> EntityRegistry
        HACore <--> IntentProcessor
        HACore <--> ResponseGenerator
        VoiceIntegration <--> IntentProcessor
        IntentProcessor <--> ResponseGenerator
    end

    subgraph Kubernetes["Kubernetes Cluster"]
        subgraph TritonInference["Triton Inference Server"]
            subgraph WakeWord["Wake Word Detection"]
                PorcupineModel[Porcupine Model]
            end

            subgraph STT["Speech-to-Text"]
                WhisperModel[Whisper Model]
            end

            subgraph NLU["Natural Language Understanding"]
                LLMModel[LLM Model\nQuantized LLama2]
                IntentClassifier[Intent Classification]
                EntityExtractor[Entity Extraction]

                LLMModel --> IntentClassifier
                LLMModel --> EntityExtractor
            end

            subgraph TTS["Text-to-Speech"]
                CoquiModel[Coqui TTS Model]
            end
        end

        subgraph RayCluster["Ray Cluster"]
            AudioPostprocessing[Audio Post-processing]
            Caching[Inference Result Caching]
            ResourceScheduler[Resource Scheduler]
        end
    end

    %% Data Flow Connections
    Microphone -->|Raw Audio| VoiceIntegration
    VoiceIntegration -->|Audio Stream| PorcupineModel

    PorcupineModel -->|Wake Word Detected| WhisperModel
    PorcupineModel -->|Wake Word Events| VoiceIntegration

    WhisperModel -->|Transcribed Text| LLMModel
    WhisperModel -->|Text| VoiceIntegration

    LLMModel -->|Processed Text| IntentProcessor
    IntentClassifier -->|Intent| IntentProcessor
    EntityExtractor -->|Entities| IntentProcessor

    IntentProcessor -->|Commands| HACore
    HACore -->|Actions| Devices

    ResponseGenerator -->|Response Text| CoquiModel
    CoquiModel -->|Synthesized Speech| AudioPostprocessing
    AudioPostprocessing -->|Processed Audio| Speaker

    HACore -->|State Updates| HomeUI

    %% Ray Integrations
    RayCluster -.->|Resource Management| TritonInference
    AudioPostprocessing -.->|Utilizes| ResourceScheduler
    Caching -.->|Improves| TritonInference

    %% Styling
    classDef user fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef physical fill:#d7e3fc,stroke:#333,stroke-width:1px
    classDef homeAssistant fill:#d0f4de,stroke:#333,stroke-width:1px
    classDef triton fill:#ffdab9,stroke:#333,stroke-width:1px
    classDef ray fill:#ffe6e6,stroke:#333,stroke-width:1px
    classDef models fill:#ffcc99,stroke:#333,stroke-width:1px
    classDef ui fill:#f2ffe6,stroke:#333,stroke-width:1px

    class User user
    class Microphone,Speaker,Devices physical
    class HomeAssistant,HACore,VoiceIntegration,IntentProcessor,EntityRegistry,ResponseGenerator homeAssistant
    class TritonInference,WakeWord,STT,NLU,TTS triton
    class RayCluster,AudioPostprocessing,Caching,ResourceScheduler ray
    class PorcupineModel,WhisperModel,LLMModel,IntentClassifier,EntityExtractor,CoquiModel models
    class HomeUI ui
```

## Sequence Diagram for Voice Assistant Interaction

```mermaid
sequenceDiagram
    actor User
    participant MA as Microphone Array
    participant WW as Wake Word Detection
    participant STT as Speech-to-Text
    participant NLU as Natural Language Understanding
    participant IP as Intent Processor
    participant HA as Home Assistant Core
    participant TTS as Text-to-Speech
    participant SP as Speakers

    User->>MA: Speaks wake word + command
    MA->>WW: Streams audio buffer
    WW->>WW: Detects wake word
    activate WW
    WW->>STT: Forwards audio after wake word
    deactivate WW

    activate STT
    STT->>STT: Transcribes speech to text
    STT->>NLU: Sends transcribed text
    deactivate STT

    activate NLU
    NLU->>NLU: Analyzes text<br/>Identifies intent and entities
    NLU->>IP: Sends structured intent + entities
    deactivate NLU

    activate IP
    IP->>IP: Maps to Home Assistant intent
    IP->>HA: Sends command with entities
    deactivate IP

    activate HA
    HA->>HA: Executes action on devices
    HA->>TTS: Sends response text
    deactivate HA

    activate TTS
    TTS->>TTS: Synthesizes speech
    TTS->>SP: Streams audio response
    deactivate TTS

    SP->>User: Plays audio response

    Note over User,SP: Complete interaction cycle: ~2-3 seconds
```

## Component Detail View

```mermaid
flowchart LR
    %% Wake Word Detection Detail
    subgraph WakeWord["Wake Word Detection (Porcupine)"]
        AudioBuffer[Audio Buffer<br/>Sliding Window]
        FeatureExtraction[MFCC<br/>Feature Extraction]
        PorcupineEngine[Porcupine Engine<br/>Keyword Spotting]
        ThresholdDetection[Confidence<br/>Thresholding]

        AudioBuffer --> FeatureExtraction
        FeatureExtraction --> PorcupineEngine
        PorcupineEngine --> ThresholdDetection
    end

    %% STT Detail
    subgraph SpeechToText["Speech-to-Text (Whisper)"]
        AudioSegmenter[Audio Segmenter]
        SpectrogramConv[Spectrogram<br/>Conversion]
        WhisperEncoder[Whisper Encoder]
        WhisperDecoder[Whisper Decoder]

        AudioSegmenter --> SpectrogramConv
        SpectrogramConv --> WhisperEncoder
        WhisperEncoder --> WhisperDecoder
    end

    %% NLU Detail
    subgraph NLU["Natural Language Understanding"]
        TokenizerLLM[Tokenization]
        LLMInference[LLM Inference<br/>Context-Aware]
        IntentExtract[Intent Parser]
        EntityRecognizer[Entity<br/>Recognition]

        TokenizerLLM --> LLMInference
        LLMInference --> IntentExtract
        LLMInference --> EntityRecognizer
    end

    %% Home Assistant Intent Processing
    subgraph IntentProc["Intent Processing"]
        IntentMapper[Intent Schema<br/>Mapping]
        EntityResolver[Entity<br/>Resolution]
        ValidationLogic[Validation Logic]
        ContextBuilder[Context<br/>Building]

        IntentMapper --> EntityResolver
        EntityResolver --> ValidationLogic
        ValidationLogic --> ContextBuilder
    end

    %% TTS Detail
    subgraph TextToSpeech["Text-to-Speech (Coqui)"]
        TextNormalizer[Text Normalizer]
        Vocoder[Neural Vocoder]
        AudioPostproc[Audio<br/>Post-processing]

        TextNormalizer --> Vocoder
        Vocoder --> AudioPostproc
    end

    %% Connections between major components
    WakeWord -->|"Wake Detection Event"| SpeechToText
    SpeechToText -->|"Text Transcript"| NLU
    NLU -->|"Structured Intent"| IntentProc
    IntentProc -->|"HA Service Call"| HAExecution[Home Assistant<br/>Action Execution]
    HAExecution -->|"Response Text"| TextToSpeech

    %% Styling
    classDef blue fill:#d7e3fc,stroke:#333,stroke-width:1px
    classDef green fill:#d0f4de,stroke:#333,stroke-width:1px
    classDef orange fill:#ffdab9,stroke:#333,stroke-width:1px
    classDef red fill:#ffe6e6,stroke:#333,stroke-width:1px

    class WakeWord,AudioBuffer,FeatureExtraction,PorcupineEngine,ThresholdDetection blue
    class SpeechToText,AudioSegmenter,SpectrogramConv,WhisperEncoder,WhisperDecoder green
    class NLU,TokenizerLLM,LLMInference,IntentExtract,EntityRecognizer orange
    class IntentProc,IntentMapper,EntityResolver,ValidationLogic,ContextBuilder,HAExecution red
    class TextToSpeech,TextNormalizer,Vocoder,AudioPostproc blue
```

## Performance and Optimization Considerations

- **End-to-end Latency**: The pipeline is optimized to keep total latency under 2 seconds from wake word detection to response
- **Model Quantization**: All models are quantized (INT8/FP16) for Jetson AGX Orin's hardware accelerators
- **Batch Processing**: Audio is processed in overlapping chunks for continuous recognition
- **Caching**: Common intents and responses are cached to reduce inference time
- **Resource Management**: Ray manages GPU memory allocation between different inference tasks
- **Offline Operation**: The entire pipeline operates locally without cloud dependencies

## Integration Points

- **Home Assistant**: Via custom component and WebSocket API
- **Triton Inference Server**: gRPC API for efficient inference
- **Ray Cluster**: For distributed resource management and task scheduling
- **Audio Hardware**: Via ALSA/PulseAudio interfaces
- **Smart Home Devices**: Through Home Assistant integrations
