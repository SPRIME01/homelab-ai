"""
Example usage of the Ray Resource Management system for AI inference.
This script demonstrates how to use the system for different AI workloads.
"""
import ray
import asyncio
import numpy as np
import time
import logging
from typing import Dict, List
import os
import argparse

# Import our modules
from main import get_ray_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example_usage")

class AIWorkloadExamples:
    """Examples of AI workloads using the Ray Resource Management system."""

    def __init__(self):
        """Initialize the example class."""
        self.manager = None

    async def initialize(self):
        """Initialize the Ray Manager."""
        self.manager = get_ray_manager()
        await self.manager.initialize()
        logger.info("Ray Manager initialized")

    async def shutdown(self):
        """Shutdown the Ray Manager."""
        if self.manager:
            await self.manager.shutdown()
            logger.info("Ray Manager shut down")

    async def run_text_generation(self, prompt: str, model_name: str = "llm"):
        """
        Run text generation using a language model.

        Args:
            prompt: Text prompt
            model_name: Name of the language model

        Returns:
            Generated text
        """
        logger.info(f"Running text generation with prompt: '{prompt[:50]}...'")

        # Convert prompt to model input format
        # This is a simplified example - adjust based on your model's requirements
        inputs = {
            "text": np.array([prompt], dtype=np.object_)
        }

        # Run inference with high priority (interactive)
        result = await self.manager.submit_inference_task(
            model_name=model_name,
            inputs=inputs,
            model_type="llm_medium",
            priority="interactive",
            wait=True,
            timeout=30.0
        )

        # Extract generated text from model output
        # This will depend on your model's output format
        if result and "outputs" in result:
            generated_text = result["outputs"].get("text", [""])[0]
            return generated_text
        else:
            logger.error("Failed to get text generation result")
            return None

    async def run_image_classification(self, image_path: str, model_name: str = "resnet"):
        """
        Run image classification on an image.

        Args:
            image_path: Path to image file
            model_name: Name of the vision model

        Returns:
            Classification result
        """
        try:
            # Simple image loading (you might want to use PIL or OpenCV)
            import cv2
            logger.info(f"Running image classification on: {image_path}")

            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Resize and normalize
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            # Convert BGR to RGB
            image = image[:, :, ::-1]
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            # Transpose to NCHW format
            image = np.transpose(image, (0, 3, 1, 2))

            # Run inference
            inputs = {"input": image}
            result = await self.manager.submit_inference_task(
                model_name=model_name,
                inputs=inputs,
                model_type="vision_model",
                priority="batch_inference",
                wait=True,
                timeout=10.0
            )

            # Process result
            if result and "outputs" in result:
                # Extract class predictions
                logits = result["outputs"].get("output", [])
                if len(logits) > 0:
                    # Get top class
                    class_id = np.argmax(logits[0])
                    confidence = float(logits[0][class_id])
                    return {"class_id": int(class_id), "confidence": confidence}

            logger.error("Failed to get classification result")
            return None

        except Exception as e:
            logger.error(f"Error in image classification: {e}")
            return None

    async def run_speech_recognition(self, audio_path: str, model_name: str = "whisper"):
        """
        Run speech recognition on an audio file.

        Args:
            audio_path: Path to audio file
            model_name: Name of the speech model

        Returns:
            Transcribed text
        """
        try:
        await manager.shutdown()
        logger.info("Examples completed")


if __name__ == "__main__":
    asyncio.run(run_examples())
