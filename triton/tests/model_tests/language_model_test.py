"""
Test framework for language models served by Triton Inference Server.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Union
from base_test import BaseModelTest, logger

class LanguageModelTest(BaseModelTest):
    """Test class for language models."""

    # Override this in subclasses
    model_name = "llm"
    test_inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing briefly."
    ]
    max_tokens = 50
    temperature = 0.7

    @classmethod
    def setup_test_data(cls):
        """Set up test data for language model tests."""
        cls.test_data_dir = os.path.join(cls.output_dir, "language_test_data")
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Create test inputs
        cls.inputs = []
        for i, text in enumerate(cls.test_inputs):
            input_data = {
                "text": np.array([text], dtype=np.object_),
                "max_tokens": np.array([cls.max_tokens], dtype=np.int32),
                "temperature": np.array([cls.temperature], dtype=np.float32)
            }
            cls.inputs.append(input_data)

        # Expected shapes validation inputs
        # Include an invalid shape for error testing
        cls.valid_shapes = {
            "batch_1": {"text": np.array(["Test"], dtype=np.object_)},
            "batch_4": {"text": np.array(["T1", "T2", "T3", "T4"], dtype=np.object_)}
        }

        cls.invalid_shapes = {
            "wrong_shape": {"text": np.array([["nested", "array"]], dtype=np.object_)},
        }

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
                            if v.dtype.kind == 'S' or v.dtype.kind == 'U':  # Handle string arrays
                                serializable_result[k] = v.tolist()
                            else:
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

    def test_model_metadata(self):
        """Test model metadata is valid."""
        self.assertIsNotNone(self.model_metadata)
        self.assertEqual(self.model_metadata["name"], self.model_name)

        # Check required inputs
        input_names = [inp["name"] for inp in self.model_metadata["inputs"]]
        self.assertIn("text", input_names)

        # Check outputs
        self.assertTrue(len(self.model_metadata["outputs"]) > 0)

    def test_model_config(self):
        """Test model configuration."""
        self.assertIsNotNone(self.model_config)

        # Check model platform
        platform = self.model_config.get("platform", "")
        expected_platforms = ["pytorch", "onnxruntime", "tensorflow", "tensorrt"]
        self.assertIn(platform.lower(), [p.lower() for p in expected_platforms])

        # Check max batch size
        max_batch_size = self.model_config.get("max_batch_size", 0)
        self.assertGreaterEqual(max_batch_size, 1)

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
            logger.info(f"Input {i} inference time: {inference_time:.4f}s")

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
                    # Determine the dtype based on the content
                    if isinstance(v[0], str):
                        reference_arrays[k] = np.array(v, dtype=np.object_)
                    elif isinstance(v[0], int):
                        reference_arrays[k] = np.array(v, dtype=np.int32)
                    elif isinstance(v[0], float):
                        reference_arrays[k] = np.array(v, dtype=np.float32)
                    else:
                        reference_arrays[k] = np.array(v)
                else:
                    reference_arrays[k] = v

            # Run inference
            outputs = self.run_inference(input_data)
            _ = outputs.pop("_inference_time", None)

            # Check output shapes match reference
            for name, ref_array in reference_arrays.items():
                self.assertIn(name, outputs)
                self.assertEqual(outputs[name].shape, tuple(np.array(ref_array).shape))

    def test_input_shape_validation(self):
        """Test input shape validation."""
        # Test valid shapes
        for name, input_data in self.valid_shapes.items():
            try:
                self.validate_input_shapes(input_data)
                logger.info(f"Valid shape test passed: {name}")
            except ValueError as e:
                self.fail(f"Valid shape {name} failed validation: {e}")

        # Test invalid shapes
        for name, input_data in self.invalid_shapes.items():
            try:
                self.validate_input_shapes(input_data)
                self.fail(f"Invalid shape {name} passed validation")
            except ValueError:
                # This is expected
                logger.info(f"Invalid shape correctly rejected: {name}")

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with missing required input
        missing_input = {}
        with self.assertRaises(Exception):
            self.run_inference(missing_input)

        # Test with invalid input name
        invalid_input = {"invalid_name": np.array(["test"], dtype=np.object_)}
        with self.assertRaises(Exception):
            self.run_inference(invalid_input)

    def test_batch_performance(self):
        """Test performance with different batch sizes."""
        batch_sizes = [1, 2, 4]
        performance_results = {}

        for batch_size in batch_sizes:
            # Skip if batch size is too large for the model
            if hasattr(self, "model_config") and \
               batch_size > self.model_config.get("max_batch_size", float('inf')):
                logger.info(f"Skipping batch size {batch_size}, exceeds model max")
                continue

            # Create batched input
            batched_text = np.array(["Test prompt"] * batch_size, dtype=np.object_)
            input_data = {
                "text": batched_text,
                "max_tokens": np.array([10] * batch_size, dtype=np.int32),
                "temperature": np.array([0.5] * batch_size, dtype=np.float32)
            }

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
                    "throughput_samples_per_second": throughput
                }

                logger.info(f"Batch {batch_size}: {avg_latency:.4f}s, {throughput:.2f} samples/s")

            except Exception as e:
                logger.error(f"Batch {batch_size} failed: {e}")

        # Store performance results
        self.test_results["performance"] = performance_results
