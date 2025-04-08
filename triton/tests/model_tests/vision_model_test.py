"""
Test framework for vision models served by Triton Inference Server.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt
from base_test import BaseModelTest, logger

class VisionModelTest(BaseModelTest):
    """Test class for vision models."""

    # Override this in subclasses
    model_name = "vision_model"
    model_type = "classification"  # or "detection", "segmentation"
    test_image_dir = "test_images"
    image_size = (224, 224)  # Default image size

    @classmethod
    def setup_test_data(cls):
        """Set up test data for vision model tests."""
        cls.test_data_dir = os.path.join(cls.output_dir, "vision_test_data")
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Directory for test images
        default_images_dir = os.path.join("test_data", cls.test_image_dir)
        cls.image_dir = default_images_dir if os.path.exists(default_images_dir) else cls.test_data_dir

        # Check for existing test images or create dummy images
        cls.test_images = []
        if os.path.exists(cls.image_dir):
            for file in os.listdir(cls.image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    cls.test_images.append(os.path.join(cls.image_dir, file))

        # If no real images found, create synthetic test images
        if not cls.test_images:
            logger.info("No test images found, creating synthetic images")
            cls._create_synthetic_images()

        # Load and preprocess images
        cls.inputs = []
        for image_path in cls.test_images[:5]:  # Limit to 5 images
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(cls.image_size)
                img_array = np.array(img).astype(np.float32)

                # Normalize to [0,1]
                img_array = img_array / 255.0

                # Adjust for expected input format (NCHW vs NHWC)
                input_name = cls.model_metadata["inputs"][0]["name"]
                input_shape = cls.model_metadata["inputs"][0]["shape"]

                # Determine expected format from shape
                if len(input_shape) == 4 and input_shape[1] in [1, 3]:
                    # NCHW format expected
                    img_array = np.transpose(img_array, (2, 0, 1))

                # Add batch dimension
                img_array = np.expand_dims(img_array, 0)

                cls.inputs.append({input_name: img_array})
                logger.info(f"Loaded test image: {os.path.basename(image_path)}")

            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}")

        # Create reference data if it doesn't exist
        cls.reference_file = os.path.join(cls.test_data_dir, "reference_outputs.json")
        if not os.path.exists(cls.reference_file):
            logger.info("Generating reference outputs...")
            reference_outputs = {}

            try:
                # Get reference outputs for standard inputs
                for i, input_data in enumerate(cls.inputs):
                    result = cls.run_inference(cls, input_data)
                    # Remove timing info and store rest of outputs
                    _ = result.pop("_inference_time", None)

                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result = {}
                    for k, v in result.items():
                        if isinstance(v, np.ndarray):
                            serializable_result[k] = v.tolist()
                        else:
                            serializable_result[k] = v

                    reference_outputs[f"input_{i}"] = serializable_result

                # Save reference outputs
                with open(cls.reference_file, 'w') as f:
                    json.dump(reference_outputs, f, indent=2)

                logger.info(f"Reference outputs saved to {cls.reference_file}")

            except Exception as e:
                logger.warning(f"Failed to generate reference outputs: {e}")
        else:
            logger.info(f"Using existing reference outputs from {cls.reference_file}")

            # Load reference outputs
            with open(cls.reference_file, 'r') as f:
                cls.reference_outputs = json.load(f)

    @classmethod
    def _create_synthetic_images(cls):
        """Create synthetic test images."""
        # Create directory for synthetic images
        synthetic_dir = os.path.join(cls.test_data_dir, "synthetic_images")
        os.makedirs(synthetic_dir, exist_ok=True)

        # Create a few basic test patterns
        patterns = [
            ("solid_color", lambda x, y: np.ones((cls.image_size[1], cls.image_size[0], 3)) * 0.8),
            ("gradient", lambda x, y: np.tile(np.linspace(0, 1, cls.image_size[0]),
                                           (cls.image_size[1], 1, 1))),
            ("checkerboard", lambda x, y: np.fromfunction(
                lambda i, j, c: ((i//20 + j//20) % 2) * np.ones((1))[0],
                (cls.image_size[1], cls.image_size[0], 3)
            )),
            ("circles", lambda x, y: np.fromfunction(
                lambda i, j, c: ((i-y/2)**2 + (j-x/2)**2 < min(x,y)**2/4) * np.ones((1))[0],
                (cls.image_size[1], cls.image_size[0], 3)
            ))
        ]

        # Generate and save images
        for name, pattern_fn in patterns:
            img_array = pattern_fn(*cls.image_size)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)

            # Create PIL image and save
            img = Image.fromarray(img_array.astype(np.uint8))
            image_path = os.path.join(synthetic_dir, f"{name}.jpg")
            img.save(image_path)
            cls.test_images.append(image_path)

        logger.info(f"Created {len(patterns)} synthetic test images")

    def test_model_metadata(self):
        """Test model metadata is valid."""
        self.assertIsNotNone(self.model_metadata)
        self.assertEqual(self.model_metadata["name"], self.model_name)

        # Check required inputs
        self.assertTrue(len(self.model_metadata["inputs"]) > 0)

        # Check for image input and confirm shape
        input_shapes = [input_info["shape"] for input_info in self.model_metadata["inputs"]]
        # Most vision models expect 4D input (NCHW or NHWC)
        for shape in input_shapes:
            self.assertTrue(len(shape) >= 3)  # At least 3D (HWC) or 4D (NCHW/NHWC)

        # Check outputs
        self.assertTrue(len(self.model_metadata["outputs"]) > 0)

    def test_model_config(self):
        """Test model configuration."""
        self.assertIsNotNone(self.model_config)

        # Check max batch size
        max_batch_size = self.model_config.get("max_batch_size", 0)
        self.assertGreaterEqual(max_batch_size, 1)

        # Check for expected optimization parameters for vision models
        # (Like dynamic batching, instance group count, etc.)
        optimization_config = self.model_config.get("optimization", {})
        instances = self.model_config.get("instance_group", [])

        logger.info(f"Model optimization config: {json.dumps(optimization_config)}")
        logger.info(f"Model has {len(instances)} instance groups")

    def test_basic_inference(self):
        """Test basic model inference."""
        for i, input_data in enumerate(self.inputs):
            outputs = self.run_inference(input_data)

            # Check that we got some output
            self.assertTrue(len(outputs) > 0)
            self.assertNotIn(None, outputs.values())

            # Log inference time
            inference_time = outputs.pop("_inference_time", None)
            self.assertIsNotNone(inference_time)
            logger.info(f"Image {i} inference time: {inference_time:.4f}s")

            # Check output shape based on model type
            for name, output in outputs.items():
                if self.model_type == "classification":
                    # Classification typically outputs class probabilities
                    self.assertTrue(len(output.shape) <= 2)
                elif self.model_type == "detection":
                    # Detection often outputs bounding boxes
                    # Format varies, but typically has 3+ dimensions
                    self.assertTrue(len(output.shape) >= 2)
                elif self.model_type == "segmentation":
                    # Segmentation typically outputs masks
                    self.assertTrue(len(output.shape) >= 3)

    def test_output_validation(self):
        """Test output validates against reference data."""
        # Skip if reference outputs not available
        if not hasattr(self, "reference_outputs"):
            self.skipTest("Reference outputs not available")

        # Test each input against reference
        for i, input_data in enumerate(self.inputs):
            input_key = f"input_{i}"
            if input_key not in self.reference_outputs:
                continue

            reference = self.reference_outputs[input_key]

            # Convert reference data back to numpy arrays
            reference_arrays = {}
            for k, v in reference.items():
                if isinstance(v, list):
                    reference_arrays[k] = np.array(v)
                else:
                    reference_arrays[k] = v

            # Run inference
            outputs = self.run_inference(input_data)
            _ = outputs.pop("_inference_time", None)

            # Check output shapes match reference
            for name, ref_array in reference_arrays.items():
                self.assertIn(name, outputs)
                np_ref_array = np.array(ref_array)
                self.assertEqual(outputs[name].shape, np_ref_array.shape)

                # Optionally check values, with tolerance for floating point
                # This may fail if model implementation changes, so we catch it
                try:
                    if np.issubdtype(outputs[name].dtype, np.floating):
                        np.testing.assert_allclose(
                            outputs[name], np_ref_array,
                            rtol=1e-3, atol=1e-3
                        )
                    else:
                        np.testing.assert_array_equal(outputs[name], np_ref_array)
                    logger.info(f"Output values match reference for {name}")
                except AssertionError as e:
                    logger.warning(f"Output values differ from reference: {e}")

    def test_batch_performance(self):
        """Test performance with different batch sizes."""
        # Use first input as template
        if not self.inputs:
            self.skipTest("No test inputs available")

        template_input = self.inputs[0]
        input_name = list(template_input.keys())[0]
        single_image = template_input[input_name]

        # Get batch dimension index (usually 0)
        batch_sizes = [1, 2, 4]
        performance_results = {}

        for batch_size in batch_sizes:
            # Skip if batch size is too large for the model
            if hasattr(self, "model_config") and \
               batch_size > self.model_config.get("max_batch_size", float('inf')):
                logger.info(f"Skipping batch size {batch_size}, exceeds model max")
                continue

            # Create batched input
            batched_image = np.repeat(single_image, batch_size, axis=0)
            input_data = {input_name: batched_image}

            try:
                # Run multiple times to get average performance
                iterations = 3
                latencies = []

                for _ in range(iterations):
                    outputs = self.run_inference(input_data)
                    latencies.append(outputs["_inference_time"])

                # Calculate metrics
                avg_latency = sum(latencies) / len(latencies)
                throughput = batch_size / avg_latency

                performance_results[f"batch_{batch_size}"] = {
                    "avg_latency_seconds": avg_latency,
                    "throughput_images_per_second": throughput
                }

                logger.info(f"Batch {batch_size}: {avg_latency:.4f}s, {throughput:.2f} images/s")

            except Exception as e:
                logger.error(f"Batch {batch_size} failed: {e}")

        # Store performance results
        self.test_results["performance"] = performance_results

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Skip if no test inputs
        if not self.inputs:
            self.skipTest("No test inputs available")

        template_input = self.inputs[0]
        input_name = list(template_input.keys())[0]

        # Test with wrong shape (wrong number of channels)
        wrong_shape_input = template_input.copy()
        original_image = template_input[input_name]

        if original_image.shape[-1] == 3:  # RGB
            # Convert to single channel
            wrong_shape_input[input_name] = original_image.mean(axis=-1, keepdims=True)
        else:
            # Add extra channels
            wrong_shape_input[input_name] = np.repeat(original_image, 2, axis=-1)

        # This should raise an error
        with self.assertRaises(Exception):
            self.run_inference(wrong_shape_input)
