import asyncio
import os
import argparse
from pathlib import Path

from .pipeline import VoiceAssistantPipeline

def main():
    """Entry point for the voice assistant pipeline."""
    parser = argparse.ArgumentParser(description="Home Assistant Voice Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get(
            "VOICE_PIPELINE_CONFIG",
            str(Path(__file__).parent / "config" / "pipeline.yaml")
        ),
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Create and start the pipeline
    pipeline = VoiceAssistantPipeline(args.config)
    asyncio.run(pipeline.start())

if __name__ == "__main__":
    main()
