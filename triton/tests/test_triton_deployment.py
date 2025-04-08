"""
Comprehensive test suite for validating Triton Inference Server deployment on Kubernetes.
Tests server availability, model loading, inference requests for different model types,
performance benchmarking, and error handling.
"""

import os
import sys
import time
import json
import logging
import pytest
import numpy as np
import requests
import unittest
import argparse
from typing import Dict, List, Tuple, Optional, Union
import tritonclient.http
import tritonclient.grpc
from kubernetes import client, config
import kubetest
from kubetest.client import TestClient
from PIL import Image
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('triton_deployment_test.log')
    ]
)
logger = logging.getLogger("triton_test")

# Test configuration (can be overridden via CLI arguments)
DEFAULT_CONFIG = {
    "namespace": "ai",
    "service_name": "triton-inference-server",
    "http_port": 8000,
    "grpc_port": 8001,
    "metrics_port": 8002,
    "test_timeout": 60,  # seconds
    "test_models": {
        "language": ["llama2-7b-q4", "gpt-j-6b", "bert-base"],
        "vision": ["yolov8n", "resnet50", "detection-model"],
        "speech": ["whisper-base", "tts-model"]
    },
    "test_image_path": "test_data/dog.jpg",
    "test_text": "The quick brown fox jumps over the lazy dog.",
    "test_audio_path": "test_data/speech_sample.wav",
    "benchmarking": {
        "batch_sizes": [1, 2, 4, 8],
        "concurrency": [1, 2, 4, 8],
        "iterations": 100,
        "warmup_iterations": 10
    },
    "model_repo_path": "/models",
    "output_dir": "results",
    "port_forward": True  # Use port forwarding for local testing
}

class TritonDeploymentTest(unittest.TestCase):
    """Test suite for validating Triton Inference Server deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and connections."""
        cls.config = DEFAULT_CONFIG.copy()

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Triton Inference Server deployment test')
        parser.add_argument('--namespace', type=str, help='Kubernetes namespace')
        parser.add_argument('--service', type=str, help='Triton service name')
        parser.add_argument('--http-port', type=int, help='HTTP port for Triton')
        parser.add_argument('--grpc-port', type=int, help='gRPC port for Triton')
        parser.add_argument('--port-forward', action='store_true', help='Use port forwarding')
        parser.add_argument('--no-port-forward', dest='port_forward', action='store_false')
        parser.add_argument('--config-file', type=str, help='Path to config JSON file')
        args, _ = parser.parse_known_args()

        # Load config from file if specified
        if args.config_file and os.path.exists(args.config_file):
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                cls.config.update(file_config)

        # Override with command line arguments
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                if arg_name in cls.config:
                    cls.config[arg_name] = arg_value

        # Create output directory
        os.makedirs(cls.config["output_dir"], exist_ok=True)

        # Setup Kubernetes client
        try:
            config.load_kube_config()
        except:
            config.load_incluster_config()

        cls.k8s_client = client.CoreV1Api()
        cls.test_client = TestClient(namespace=cls.config["namespace"])

        # Set up port forwarding if required
        cls.pf = None
        if cls.config["port_forward"]:
            from kubetest.client import PodPortForward
            logger.info(f"Setting up port forwarding for {cls.config['service_name']}")

            # Find service endpoints
            service = cls.k8s_client.read_namespaced_service(
                name=cls.config["service_name"],
                namespace=cls.config["namespace"]
            )

            # Get pods for service
            pods = cls.k8s_client.list_namespaced_pod(
                namespace=cls.config["namespace"],
                label_selector=f"app={cls.config['service_name']}"
            )

            if not pods.items:
                raise ValueError(f"No pods found for service {cls.config['service_name']}")

            # Setup port forwarding on first pod
            target_pod = pods.items[0]
            logger.info(f"Setting up port forwarding to pod {target_pod.metadata.name}")
            cls.pf = PodPortForward(
                cls.k8s_client, target_pod,
                ports=[cls.config["http_port"], cls.config["grpc_port"], cls.config["metrics_port"]]
            )
            cls.pf.start()

            # Update URLs for local port-forwarded access
            cls.http_url = f"localhost:{cls.config['http_port']}"
            cls.grpc_url = f"localhost:{cls.config['grpc_port']}"
            cls.metrics_url = f"localhost:{cls.config['metrics_port']}"
        else:
            # Get service IP for direct access
            service = cls.k8s_client.read_namespaced_service(
                name=cls.config["service_name"],
                namespace=cls.config["namespace"]
            )

            if service.spec.type == "LoadBalancer" and service.status.load_balancer.ingress:
                service_ip = service.status.load_balancer.ingress[0].ip
                cls.http_url = f"{service_ip}:{cls.config['http_port']}"
                cls.grpc_url = f"{service_ip}:{cls.config['grpc_port']}"
                cls.metrics_url = f"{service_ip}:{cls.config['metrics_port']}"
            else:
                # Use cluster IP as fallback
                service_ip = service.spec.cluster_ip
                cls.http_url = f"{service_ip}:{cls.config['http_port']}"
                cls.grpc_url = f"{service_ip}:{cls.config['grpc_port']}"
                cls.metrics_url = f"{service_ip}:{cls.config['metrics_port']}"

        logger.info(f"Triton HTTP URL: {cls.http_url}")
        logger.info(f"Triton gRPC URL: {cls.grpc_url}")
        logger.info(f"Triton Metrics URL: {cls.metrics_url}")

        # Set up Triton clients
        cls.http_client = tritonclient.http.InferenceServerClient(
            url=cls.http_url, verbose=False
        )
        cls.grpc_client = tritonclient.grpc.InferenceServerClient(
            url=cls.grpc_url, verbose=False
        )

        # Wait for server to become available
        server_ready = False
        max_retries = 10
        retry_count = 0

        while not server_ready and retry_count < max_retries:
            try:
                server_ready = cls.http_client.is_server_ready()
                if server_ready:
                    logger.info("Triton server is ready")
                    break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Waiting for server to be ready ({retry_count}/{max_retries}): {str(e)}")
                time.sleep(5)

        if not server_ready:
            logger.error("Triton server not ready after maximum retries")
            raise ConnectionError("Could not connect to Triton server")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        if cls.pf:
            logger.info("Stopping port forwarding")
            cls.pf.stop()

    def setUp(self):
        """Set up test case."""
        logger.info(f"\n{'=' * 80}\nRunning test: {self._testMethodName}\n{'=' * 80}")

    def test_001_server_health(self):
        """Test basic server health and readiness."""
        # Test server is live
        is_live = self.http_client.is_server_live()
        self.assertTrue(is_live, "Server should be live")

        # Test server is ready
        is_ready = self.http_client.is_server_ready()
        self.assertTrue(is_ready, "Server should be ready")

        # Test server metadata
        metadata = self.http_client.get_server_metadata()
        logger.info(f"Server metadata: {metadata}")
        self.assertIn("name", metadata, "Server metadata should include 'name'")
        self.assertIn("version", metadata, "Server metadata should include 'version'")

        # Test metrics endpoint
        try:
            metrics_response = requests.get(f"http://{self.metrics_url}/metrics")
            self.assertEqual(metrics_response.status_code, 200, "Metrics endpoint should return 200")
            logger.info("Metrics endpoint is accessible")
        except Exception as e:
            logger.error(f"Failed to access metrics endpoint: {e}")
            self.fail(f"Failed to access metrics endpoint: {e}")

    def test_002_model_repository(self):
        """Test model repository configuration and status."""
        # Get repository index
        repo_index = self.http_client.get_model_repository_index()
        logger.info(f"Found {len(repo_index)} models in repository")

        # Check if we have any models
        self.assertTrue(len(repo_index) > 0, "Model repository should not be empty")

        # Log available models
        available_models = []
        for model in repo_index:
            model_name = model["name"]
            available_models.append(model_name)
            logger.info(f"Found model: {model_name}")

            # Check model metadata
            try:
                metadata = self.http_client.get_model_metadata(model_name)
                self.assertIn("name", metadata, f"Model {model_name} metadata should include 'name'")
                logger.info(f"Model {model_name} metadata: {metadata}")
            except Exception as e:
                logger.warning(f"Could not get metadata for model {model_name}: {e}")

        self._available_models = available_models

        # Save available models to output
        with open(os.path.join(self.config["output_dir"], "available_models.json"), "w") as f:
            json.dump(available_models, f, indent=2)

    def test_003_model_readiness(self):
        """Test that models are ready for inference."""
        if not hasattr(self, "_available_models"):
            self.test_002_model_repository()

        # Check if all models are ready
        ready_models = []
        not_ready_models = []

        for model_name in self._available_models:
            try:
                is_ready = self.http_client.is_model_ready(model_name)
                if is_ready:
                    ready_models.append(model_name)
                    logger.info(f"Model {model_name} is ready")
                else:
                    not_ready_models.append(model_name)
                    logger.warning(f"Model {model_name} is not ready")
            except Exception as e:
                not_ready_models.append(model_name)
                logger.error(f"Error checking readiness for model {model_name}: {e}")

        self.assertTrue(len(ready_models) > 0, "At least one model should be ready")

        if not_ready_models:
            logger.warning(f"Some models are not ready: {', '.join(not_ready_models)}")

        # Save ready/not-ready models to output
        with open(os.path.join(self.config["output_dir"], "model_readiness.json"), "w") as f:
            json.dump({
                "ready_models": ready_models,
                "not_ready_models": not_ready_models
            }, f, indent=2)

        self._ready_models = ready_models

    def _find_model_by_type(self, model_type: str) -> Optional[str]:
        """Find a model by type from available and ready models."""
        # First try models from configuration
        if not hasattr(self, "_ready_models"):
            self.test_003_model_readiness()

        for model in self.config["test_models"].get(model_type, []):
            if model in self._ready_models:
                return model

        # If none found, try to guess from model name
        model_type_keywords = {
            "language": ["gpt", "llama", "bert", "t5", "text", "nlp", "llm"],
            "vision": ["yolo", "resnet", "efficientnet", "detect", "class", "segment", "vision"],
            "speech": ["whisper", "wav2vec", "speech", "tts", "stt", "audio"]
        }

        keywords = model_type_keywords.get(model_type, [])
        for model in self._ready_models:
            for keyword in keywords:
                if keyword.lower() in model.lower():
                    return model

        logger.warning(f"No {model_type} model found in ready models")
        return None

    def test_004_language_model_inference(self):
        """Test inference with a language model."""
        model_name = self._find_model_by_type("language")
        if not model_name:
            self.skipTest("No suitable language model available")

        try:
            # Get model metadata to determine input/output names
            metadata = self.http_client.get_model_metadata(model_name)
            config = self.http_client.get_model_config(model_name)

            # Extract input and output information
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]

            # Sample text input
            test_text = self.config["test_text"]
            logger.info(f"Running inference on language model {model_name} with text: {test_text}")

            # Prepare input based on model requirements
            # This is simplified - in reality we would need to handle tokenization properly based on model type
            input_data = np.array([test_text], dtype=np.object_)

            # Create HTTP input
            inputs = tritonclient.http.InferInput(input_name, input_data.shape, "BYTES")
            inputs.set_data_from_numpy(input_data)

            # Run inference
            start_time = time.time()
            response = self.http_client.infer(model_name, [inputs])
            elapsed = time.time() - start_time

            # Get output
            output = response.as_numpy(output_name)

            logger.info(f"Language model inference completed in {elapsed:.4f} seconds")
            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Output sample: {output[:100] if output.size > 0 else output}")

            # Basic validation - just ensure we got some output
            self.assertTrue(output.size > 0, "Language model should produce non-empty output")

            # Save results
            with open(os.path.join(self.config["output_dir"], f"{model_name}_language_results.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "input_text": test_text,
                    "output_sample": output.tolist() if hasattr(output, "tolist") else str(output),
                    "latency_seconds": elapsed
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Language model inference failed: {e}")
            self.fail(f"Language model inference failed: {e}")

    def test_005_vision_model_inference(self):
        """Test inference with a vision model."""
        model_name = self._find_model_by_type("vision")
        if not model_name:
            self.skipTest("No suitable vision model available")

        try:
            # Get model metadata to determine input/output names
            metadata = self.http_client.get_model_metadata(model_name)
            config = self.http_client.get_model_config(model_name)

            # Extract input and output information
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]

            # Load test image or create a dummy one
            if os.path.exists(self.config["test_image_path"]):
                img = Image.open(self.config["test_image_path"])
                img = img.resize((224, 224))  # Resize to common input size
                img_array = np.array(img).astype(np.float32)

                # Normalize image (common preprocessing)
                img_array = img_array / 255.0

                # Reshape to model's expected format (assume NCHW)
                if img_array.shape[-1] == 3:  # HWC format
                    img_array = img_array.transpose(2, 0, 1)  # Convert to CHW
                img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            else:
                logger.warning(f"Test image {self.config['test_image_path']} not found, using random data")
                img_array = np.random.rand(1, 3, 224, 224).astype(np.float32)

            logger.info(f"Running inference on vision model {model_name} with image of shape {img_array.shape}")

            # Create HTTP input
            inputs = tritonclient.http.InferInput(input_name, img_array.shape, "FP32")
            inputs.set_data_from_numpy(img_array)

            # Run inference
            start_time = time.time()
            response = self.http_client.infer(model_name, [inputs])
            elapsed = time.time() - start_time

            # Get output
            output = response.as_numpy(output_name)

            logger.info(f"Vision model inference completed in {elapsed:.4f} seconds")
            logger.info(f"Output shape: {output.shape}")

            # For classification models, try to interpret top result
            if len(output.shape) <= 2:  # Likely class probabilities
                top_idx = np.argmax(output, axis=-1)
                logger.info(f"Top class index: {top_idx}")
            else:
                logger.info(f"Output summary: min={output.min()}, max={output.max()}, mean={output.mean()}")

            # Basic validation - just ensure we got some output
            self.assertTrue(output.size > 0, "Vision model should produce non-empty output")

            # Save results
            results_file = os.path.join(self.config["output_dir"], f"{model_name}_vision_results.json")
            with open(results_file, "w") as f:
                result_data = {
                    "model_name": model_name,
                    "input_shape": img_array.shape,
                    "output_shape": output.shape,
                    "latency_seconds": elapsed
                }

                # Add model-specific output interpretation
                if len(output.shape) <= 2:  # Classification results
                    result_data["top_class"] = int(top_idx[0]) if hasattr(top_idx, "__len__") else int(top_idx)
                    top_k = 5
                    if output.shape[-1] >= top_k:
                        top_k_indices = np.argsort(output[0])[-top_k:][::-1]
                        top_k_values = output[0][top_k_indices]
                        result_data["top_k"] = [
                            {"class": int(idx), "probability": float(val)}
                            for idx, val in zip(top_k_indices, top_k_values)
                        ]
                else:
                    # For detection/segmentation, just add summary stats
                    result_data["output_stats"] = {
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "mean": float(output.mean())
                    }

                json.dump(result_data, f, indent=2)

        except Exception as e:
            logger.error(f"Vision model inference failed: {e}")
            self.fail(f"Vision model inference failed: {e}")

    def test_006_speech_model_inference(self):
        """Test inference with a speech model."""
        model_name = self._find_model_by_type("speech")
        if not model_name:
            self.skipTest("No suitable speech model available")

        try:
            # Get model metadata to determine input/output names
            metadata = self.http_client.get_model_metadata(model_name)
            config = self.http_client.get_model_config(model_name)

            # Extract input and output information
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]

            # Load test audio or create a dummy one
            if os.path.exists(self.config["test_audio_path"]):
                import soundfile as sf
                audio_data, sample_rate = sf.read(self.config["test_audio_path"])
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = audio_data[:, 0]  # Convert to mono if stereo
                audio_array = audio_data.astype(np.float32)
            else:
                logger.warning(f"Test audio {self.config['test_audio_path']} not found, using random data")
                # Create 4 seconds of random audio at 16kHz
                audio_array = np.random.randn(16000 * 4).astype(np.float32)
                audio_array = audio_array / np.abs(audio_array).max()  # Normalize

            # Add batch dimension
            audio_array = np.expand_dims(audio_array, 0)

            logger.info(f"Running inference on speech model {model_name} with audio of shape {audio_array.shape}")

            # Create HTTP input
            inputs = tritonclient.http.InferInput(input_name, audio_array.shape, "FP32")
            inputs.set_data_from_numpy(audio_array)

            # Run inference
            start_time = time.time()
            response = self.http_client.infer(model_name, [inputs])
            elapsed = time.time() - start_time

            # Get output
            output = response.as_numpy(output_name)

            logger.info(f"Speech model inference completed in {elapsed:.4f} seconds")
            logger.info(f"Output shape: {output.shape}")

            # Basic validation - just ensure we got some output
            self.assertTrue(output.size > 0, "Speech model should produce non-empty output")

            # Save results
            with open(os.path.join(self.config["output_dir"], f"{model_name}_speech_results.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "input_shape": audio_array.shape,
                    "output_shape": output.shape,
                    "latency_seconds": elapsed
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Speech model inference failed: {e}")
            self.fail(f"Speech model inference failed: {e}")

    def test_007_batch_inference(self):
        """Test batch inference capabilities with a model."""
        # Use any ready model for batch testing
        if not hasattr(self, "_ready_models"):
            self.test_003_model_readiness()

        if not self._ready_models:
            self.skipTest("No ready models available")

        model_name = self._ready_models[0]

        try:
            # Get model metadata to determine input/output names
            metadata = self.http_client.get_model_metadata(model_name)
            config = self.http_client.get_model_config(model_name)

            # Extract input and output information
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]
            input_shape = metadata["inputs"][0]["shape"]

            logger.info(f"Testing batch inference on model {model_name}")

            # Determine input data type
            dtype_str = metadata["inputs"][0]["datatype"]
            if dtype_str == "FP32":
                dtype = np.float32
                input_data = np.random.rand(1, *input_shape[1:]).astype(dtype)
            elif dtype_str == "INT32":
                dtype = np.int32
                input_data = np.random.randint(0, 100, size=(1, *input_shape[1:])).astype(dtype)
            elif dtype_str == "BYTES":
                input_data = np.array(["Test data for batch inference"], dtype=np.object_)
            else:
                # Default to float32
                dtype = np.float32
                input_data = np.random.rand(1, *input_shape[1:]).astype(dtype)

            # Test different batch sizes
            batch_results = []
            for batch_size in self.config["benchmarking"]["batch_sizes"]:
                if batch_size > 1:
                    # Create batched input by repeating single input
                    batch_input = np.repeat(input_data, batch_size, axis=0)
                else:
                    batch_input = input_data

                # Create HTTP input
                inputs = tritonclient.http.InferInput(input_name, batch_input.shape, dtype_str)
                inputs.set_data_from_numpy(batch_input)

                # Run inference
                start_time = time.time()
                response = self.http_client.infer(model_name, [inputs])
                elapsed = time.time() - start_time

                # Get output
                output = response.as_numpy(output_name)

                # Record results
                logger.info(f"Batch size {batch_size}: inference completed in {elapsed:.4f} seconds")
                logger.info(f"Throughput: {batch_size / elapsed:.2f} samples/second")

                batch_results.append({
                    "batch_size": batch_size,
                    "latency_seconds": elapsed,
                    "throughput_samples_per_second": batch_size / elapsed,
                    "output_shape": list(output.shape)
                })

            # Save batch results
            with open(os.path.join(self.config["output_dir"], f"{model_name}_batch_results.json"), "w") as f:
                json.dump(batch_results, f, indent=2)

            # Create throughput vs batch size plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                [r["batch_size"] for r in batch_results],
                [r["throughput_samples_per_second"] for r in batch_results],
                'o-', linewidth=2
            )
            plt.title(f'Throughput vs Batch Size for {model_name}')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.grid(True)
            plt.savefig(os.path.join(self.config["output_dir"], f"{model_name}_batch_throughput.png"))

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            self.fail(f"Batch inference failed: {e}")

    def test_008_concurrent_inference(self):
        """Test concurrent inference capabilities."""
        # Use any ready model for concurrency testing
        if not hasattr(self, "_ready_models"):
            self.test_003_model_readiness()

        if not self._ready_models:
            self.skipTest("No ready models available")

        model_name = self._ready_models[0]

        try:
            # Get model metadata
            metadata = self.http_client.get_model_metadata(model_name)
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]
            input_shape = metadata["inputs"][0]["shape"]
            dtype_str = metadata["inputs"][0]["datatype"]

            if dtype_str == "FP32":
                dtype = np.float32
                input_data = np.random.rand(1, *input_shape[1:]).astype(dtype)
            elif dtype_str == "INT32":
                dtype = np.int32
                input_data = np.random.randint(0, 100, size=(1, *input_shape[1:])).astype(dtype)
            elif dtype_str == "BYTES":
                input_data = np.array(["Test data for concurrent inference"], dtype=np.object_)
            else:
                # Default to float32
                dtype = np.float32
                input_data = np.random.rand(1, *input_shape[1:]).astype(dtype)

            logger.info(f"Testing concurrent inference on model {model_name}")

            import concurrent.futures

            def run_inference():
                """Run a single inference request."""
                client = tritonclient.http.InferenceServerClient(url=self.http_url, verbose=False)
                inputs = tritonclient.http.InferInput(input_name, input_data.shape, dtype_str)
                inputs.set_data_from_numpy(input_data)

                start = time.time()
                response = client.infer(model_name, [inputs])
                elapsed = time.time() - start

                return elapsed

            # Test different concurrency levels
            concurrency_results = []
            for concurrency in self.config["benchmarking"]["concurrency"]:
                logger.info(f"Running inference with concurrency {concurrency}")

                # Create thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    # Submit inference requests
                    futures = [executor.submit(run_inference) for _ in range(concurrency)]

                    # Wait for all to complete
                    latencies = []
                    for future in concurrent.futures.as_completed(futures):
                        latency = future.result()
                        latencies.append(latency)

                # Calculate statistics
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
                throughput = concurrency / avg_latency

                logger.info(f"Concurrency {concurrency}: avg={avg_latency:.4f}s, p95={p95_latency:.4f}s, throughput={throughput:.2f} req/s")

                concurrency_results.append({
                    "concurrency": concurrency,
                    "avg_latency_seconds": avg_latency,
                    "max_latency_seconds": max_latency,
                    "min_latency_seconds": min_latency,
                    "p95_latency_seconds": p95_latency,
                    "throughput_requests_per_second": throughput
                })

            # Save concurrency results
            with open(os.path.join(self.config["output_dir"], f"{model_name}_concurrency_results.json"), "w") as f:
                json.dump(concurrency_results, f, indent=2)

            # Create throughput vs concurrency plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                [r["concurrency"] for r in concurrency_results],
                [r["throughput_requests_per_second"] for r in concurrency_results],
                'o-', linewidth=2
            )
            plt.title(f'Throughput vs Concurrency for {model_name}')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Throughput (requests/second)')
            plt.grid(True)
            plt.savefig(os.path.join(self.config["output_dir"], f"{model_name}_concurrency_throughput.png"))

        except Exception as e:
            logger.error(f"Concurrent inference test failed: {e}")
            self.fail(f"Concurrent inference test failed: {e}")

    def test_009_error_handling(self):
        """Test error handling capabilities of the server."""
        # Test with non-existent model
        try:
            self.http_client.get_model_metadata("non_existent_model")
            self.fail("Should have raised exception for non-existent model")
        except Exception as e:
            logger.info(f"Expected error for non-existent model: {e}")

        # Test with existing model but malformed input (if any model is available)
        if not hasattr(self, "_ready_models") or not self._ready_models:
            self.test_003_model_readiness()

        if not self._ready_models:
            logger.warning("No ready models available to test error handling with malformed input")
            return

        model_name = self._ready_models[0]
        metadata = self.http_client.get_model_metadata(model_name)
        input_name = metadata["inputs"][0]["name"]

        try:
            # Create input with wrong shape
            wrong_shape_data = np.zeros([1, 1], dtype=np.float32)
            inputs = tritonclient.http.InferInput(input_name, wrong_shape_data.shape, "FP32")
            inputs.set_data_from_numpy(wrong_shape_data)

            self.http_client.infer(model_name, [inputs])
            self.fail("Should have raised exception for malformed input")
        except Exception as e:
            logger.info(f"Expected error for malformed input: {e}")

        # Test with wrong input name
        try:
            wrong_name_data = np.zeros([1, 1], dtype=np.float32)
            inputs = tritonclient.http.InferInput("wrong_input_name", wrong_name_data.shape, "FP32")
            inputs.set_data_from_numpy(wrong_name_data)

            self.http_client.infer(model_name, [inputs])
            self.fail("Should have raised exception for wrong input name")
        except Exception as e:
            logger.info(f"Expected error for wrong input name: {e}")

    def test_010_generate_summary_report(self):
        """Generate a summary report of all tests."""
        # If we get here, the server is generally working
        report = {
            "timestamp": time.time(),
            "server": {
                "url": self.http_url,
                "status": "operational" if self.http_client.is_server_live() else "not operational"
            },
            "models": {}
        }

        # Add model information
        try:
            repo_index = self.http_client.get_model_repository_index()

            for model in repo_index:
                model_name = model["name"]
                is_ready = False

                try:
                    is_ready = self.http_client.is_model_ready(model_name)
                except:
                    pass

                report["models"][model_name] = {
                    "status": "ready" if is_ready else "not ready"
                }

                # Add inference results if available
                for model_type in ["language", "vision", "speech"]:
                    result_file = os.path.join(self.config["output_dir"], f"{model_name}_{model_type}_results.json")
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            report["models"][model_name][f"{model_type}_results"] = json.load(f)

                # Add batch results if available
                batch_file = os.path.join(self.config["output_dir"], f"{model_name}_batch_results.json")
                if os.path.exists(batch_file):
                    with open(batch_file, 'r') as f:
                        report["models"][model_name]["batch_results"] = json.load(f)

                # Add concurrency results if available
                concurrency_file = os.path.join(self.config["output_dir"], f"{model_name}_concurrency_results.json")
                if os.path.exists(concurrency_file):
                    with open(concurrency_file, 'r') as f:
                        report["models"][model_name]["concurrency_results"] = json.load(f)
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

        # Save summary report
        with open(os.path.join(self.config["output_dir"], "triton_test_summary.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Generate HTML report
        try:
            self._generate_html_report(report)
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")

    def _generate_html_report(self, report_data):
        """Generate an HTML report from the test data."""
        html_path = os.path.join(self.config["output_dir"], "triton_test_report.html")

        with open(html_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Triton Inference Server Deployment Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .status-ok { color: green; }
        .status-problem { color: red; }
        .chart-container { width: 100%; height: 400px; margin-bottom: 20px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Triton Inference Server Deployment Test Report</h1>
    <p><strong>Generated:</strong> """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <p><strong>Server URL:</strong> """ + report_data["server"]["url"] + """</p>
    <p><strong>Server Status:</strong> <span class="status-""" + ("ok" if report_data["server"]["status"] == "operational" else "problem") + """">""" + report_data["server"]["status"] + """</span></p>

    <h2>Model Summary</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Status</th>
            <th>Types</th>
            <th>Avg Latency</th>
        </tr>
""")

            # Add rows for each model
            for model_name, model_data in report_data["models"].items():
                model_types = []
                avg_latency = "N/A"

                # Determine model types
                for model_type in ["language", "vision", "speech"]:
                    if f"{model_type}_results" in model_data:
                        model_types.append(model_type)
                        if avg_latency == "N/A" and "latency_seconds" in model_data[f"{model_type}_results"]:
                            avg_latency = f"{model_data[f'{model_type}_results']['latency_seconds']:.4f}s"

                f.write(f"""
        <tr>
            <td>{model_name}</td>
            <td class="status-{'ok' if model_data['status'] == 'ready' else 'problem'}">{model_data['status']}</td>
            <td>{', '.join(model_types) if model_types else 'Unknown'}</td>
            <td>{avg_latency}</td>
        </tr>""")

            f.write("""
    </table>

    <h2>Detailed Results</h2>
""")

            # Add detailed results for each model
            for model_name, model_data in report_data["models"].items():
                f.write(f"""
    <h3>Model: {model_name}</h3>
""")

                # Add batch performance results if available
                if "batch_results" in model_data:
                    f.write("""
    <h4>Batch Performance</h4>
    <table>
        <tr>
            <th>Batch Size</th>
            <th>Latency (s)</th>
            <th>Throughput (samples/s)</th>
        </tr>
""")
                    for batch_result in model_data["batch_results"]:
                        f.write(f"""
        <tr>
            <td>{batch_result['batch_size']}</td>
            <td>{batch_result['latency_seconds']:.4f}</td>
            <td>{batch_result['throughput_samples_per_second']:.2f}</td>
        </tr>""")

                    f.write("""
    </table>

    <div class="chart-container">
        <canvas id="batchChart_{0}"></canvas>
    </div>
    <script>
        new Chart(document.getElementById('batchChart_{0}').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {1},
                datasets: [{{
                    label: 'Throughput vs Batch Size',
                    data: {2},
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Batch Size'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Throughput (samples/second)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
""".format(
    model_name.replace("-", "_"),
    [batch_result['batch_size'] for batch_result in model_data['batch_results']],
    [batch_result['throughput_samples_per_second'] for batch_result in model_data['batch_results']]
))

                # Add concurrency results if available
                if "concurrency_results" in model_data:
                    f.write("""
    <h4>Concurrency Performance</h4>
    <table>
        <tr>
            <th>Concurrency</th>
            <th>Avg Latency (s)</th>
            <th>P95 Latency (s)</th>
            <th>Throughput (req/s)</th>
        </tr>
""")
                    for concurrency_result in model_data["concurrency_results"]:
                        f.write(f"""
        <tr>
            <td>{concurrency_result['concurrency']}</td>
            <td>{concurrency_result['avg_latency_seconds']:.4f}</td>
            <td>{concurrency_result['p95_latency_seconds']:.4f}</td>
            <td>{concurrency_result['throughput_requests_per_second']:.2f}</td>
        </tr>""")

                    f.write("""
    </table>

    <div class="chart-container">
        <canvas id="concurrencyChart_{0}"></canvas>
    </div>
    <script>
        new Chart(document.getElementById('concurrencyChart_{0}').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {1},
                datasets: [{{
                    label: 'Throughput vs Concurrency',
                    data: {2},
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.1,
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Concurrency'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Throughput (requests/second)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
""".format(
    model_name.replace("-", "_") + "_conc",
    [r['concurrency'] for r in model_data['concurrency_results']],
    [r['throughput_requests_per_second'] for r in model_data['concurrency_results']]
))

            f.write("""
    <h2>Test Environment</h2>
    <p><strong>Kubernetes Namespace:</strong> """ + self.config["namespace"] + """</p>
    <p><strong>Service Name:</strong> """ + self.config["service_name"] + """</p>
    <p><strong>Port Forwarding:</strong> """ + str(self.config["port_forward"]) + """</p>

    <footer>
        <p>Generated by Triton Inference Server Deployment Test Suite</p>
    </footer>
</body>
</html>
""")

        logger.info(f"Generated HTML report at {html_path}")

def main():
    """Run the tests."""
    # Setup command-line argument parsing for standalone execution
    parser = argparse.ArgumentParser(description='Triton Inference Server Deployment Tester')
    parser.add_argument('--namespace', type=str, default='ai',
                      help='Kubernetes namespace where Triton is deployed')
    parser.add_argument('--service', type=str, default='triton-inference-server',
                      help='Kubernetes service name for Triton')
    parser.add_argument('--http-port', type=int, default=8000,
                      help='HTTP port for Triton')
    parser.add_argument('--grpc-port', type=int, default=8001,
                      help='gRPC port for Triton')
    parser.add_argument('--port-forward', action='store_true', default=True,
                      help='Use port forwarding to access Triton service')
    parser.add_argument('--no-port-forward', dest='port_forward', action='store_false',
                      help='Do not use port forwarding (direct access)')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to store test results')
    parser.add_argument('--config-file', type=str, default=None,
                      help='Path to config JSON file')
    parser.add_argument('--test-pattern', type=str, default='test_*',
                      help='Pattern to match test methods')

    args = parser.parse_args()

    # Update DEFAULT_CONFIG with command line args
    for arg_name, arg_value in vars(args).items():
        if arg_name in DEFAULT_CONFIG and arg_value is not None:
            DEFAULT_CONFIG[arg_name] = arg_value

    # Setup test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TritonDeploymentTest)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
