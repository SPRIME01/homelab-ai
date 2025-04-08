"""
Base test framework for Triton Inference Server models.
Provides common functionality for all model tests.
"""

import os
import sys
import json
import time
import yaml
import logging
import unittest
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Union
import tritonclient.http
import tritonclient.grpc
from kubernetes import client, config as k8s_config
import kubetest
from kubetest.client import TestClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('triton_model_test.log')
    ]
)
logger = logging.getLogger("triton_test")

class BaseModelTest(unittest.TestCase):
    """Base class for Triton model tests."""

    # Class variables to be set by child classes
    model_name = None
    model_version = ""  # Empty string means use latest version
    test_data_dir = "test_data"
    output_dir = "test_results"

    @classmethod
    def setUpClass(cls):
        """Set up resources before any tests run."""
        cls.setup_config()
        cls.setup_clients()
        cls.setup_test_data()

        # Create output directory
        os.makedirs(cls.output_dir, exist_ok=True)

        # Test results store
        cls.test_results = {
            "model": cls.model_name,
            "version": cls.model_version,
            "timestamp": time.time(),
            "tests": {}
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests have run."""
        # Save test results to file
        results_file = os.path.join(
            cls.output_dir,
            f"{cls.model_name}_test_results.json"
        )
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)

        logger.info(f"Test results saved to {results_file}")

        # Generate HTML report
        cls.generate_html_report()

        # Clean up client connections
        if hasattr(cls, 'http_client'):
            del cls.http_client
        if hasattr(cls, 'grpc_client'):
            del cls.grpc_client

    @classmethod
    def setup_config(cls):
        """Set up test configuration."""
        parser = argparse.ArgumentParser(description='Triton Model Test')
        parser.add_argument('--url', type=str, default="localhost:8000",
                          help='Triton server URL')
        parser.add_argument('--model', type=str, default=cls.model_name,
                          help='Model name to test')
        parser.add_argument('--version', type=str, default=cls.model_version,
                          help='Model version to test')
        parser.add_argument('--protocol', type=str, choices=['http', 'grpc'],
                          default='http', help='Protocol to use')
        parser.add_argument('--config', type=str, default=None,
                          help='Path to config YAML file')
        parser.add_argument('--output-dir', type=str, default=cls.output_dir,
                          help='Directory for test outputs')
        parser.add_argument('--kubernetes', action='store_true',
                          help='Use Kubernetes for testing')
        parser.add_argument('--namespace', type=str, default='ai',
                          help='Kubernetes namespace')
        parser.add_argument('--service', type=str, default='triton-inference-server',
                          help='Triton service name in Kubernetes')

        # Parse only known args to avoid conflicts with unittest arguments
        args, _ = parser.parse_known_args()

        # Override class variables with arguments
        if args.model:
            cls.model_name = args.model
        if args.version:
            cls.model_version = args.version
        if args.output_dir:
            cls.output_dir = args.output_dir

        # Load config from file if provided
        cls.config = {
            "url": args.url,
            "protocol": args.protocol,
            "kubernetes": args.kubernetes,
            "namespace": args.namespace,
            "service": args.service,
            "performance": {
                "max_latency_ms": 100,
                "min_throughput": 10,
                "batch_sizes": [1, 2, 4, 8],
                "concurrency": [1, 2, 4, 8],
                "iterations": 100
            }
        }

        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f)
                # Recursively update config with file values
                cls._deep_update(cls.config, file_config)

        # Ensure model name is set
        if not cls.model_name:
            raise ValueError("Model name must be specified")

        logger.info(f"Testing model: {cls.model_name}, version: {cls.model_version or 'latest'}")

    @classmethod
    def _deep_update(cls, d, u):
        """Deep update dictionary d with values from dictionary u."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                cls._deep_update(d[k], v)
            else:
                d[k] = v

    @classmethod
    def setup_clients(cls):
        """Set up Triton clients."""
        if cls.config.get("kubernetes", False):
            # Setup Kubernetes client
            try:
                k8s_config.load_kube_config()
            except:
                k8s_config.load_incluster_config()

            cls.k8s_client = client.CoreV1Api()
            cls.test_client = TestClient(namespace=cls.config["namespace"])

            # Set up port forwarding if needed
            # ... (code to set up port forwarding in Kubernetes)

        # Create HTTP client
        cls.http_client = tritonclient.http.InferenceServerClient(
            url=cls.config["url"], verbose=False
        )

        # Create gRPC client if needed
        if cls.config["protocol"] == "grpc":
            # Convert URL from HTTP format to gRPC format if needed
            grpc_url = cls.config["url"]
            if ":" in grpc_url and not grpc_url.endswith(":8001"):
                # Replace port if present (default HTTP 8000 -> gRPC 8001)
                host, port = grpc_url.split(":")
                grpc_url = f"{host}:8001"

            cls.grpc_client = tritonclient.grpc.InferenceServerClient(
                url=grpc_url, verbose=False
            )

        # Check server status
        is_live = cls.http_client.is_server_live()
        if not is_live:
            raise ConnectionError("Triton server is not live")

        # Check if model exists and is ready
        try:
            is_ready = cls.http_client.is_model_ready(
                cls.model_name, cls.model_version
            )
            if not is_ready:
                logger.warning(f"Model {cls.model_name} is not ready")
        except Exception as e:
            raise ValueError(f"Model {cls.model_name} not found: {e}")

        # Get model metadata
        cls.model_metadata = cls.http_client.get_model_metadata(
            cls.model_name, cls.model_version
        )

        # Get model config
        cls.model_config = cls.http_client.get_model_config(
            cls.model_name, cls.model_version
        )

        # Extract input/output info
        cls.input_names = [inp["name"] for inp in cls.model_metadata["inputs"]]
        cls.output_names = [out["name"] for out in cls.model_metadata["outputs"]]

        # Dictionary of input name to data type and shape
        cls.input_dtypes = {
            inp["name"]: inp["datatype"] for inp in cls.model_metadata["inputs"]
        }
        cls.input_shapes = {
            inp["name"]: inp["shape"] for inp in cls.model_metadata["inputs"]
        }

    @classmethod
    def setup_test_data(cls):
        """Set up test data - to be implemented by child classes."""
        pass

    @classmethod
    def generate_html_report(cls):
        """Generate HTML report from test results."""
        report_file = os.path.join(
            cls.output_dir,
            f"{cls.model_name}_test_report.html"
        )

        # Simple HTML report generation
        with open(report_file, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Triton Model Test Results: {cls.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Test Results for {cls.model_name}</h1>
    <div class="summary">
        <p><strong>Model Version:</strong> {cls.model_version or "latest"}</p>
        <p><strong>Timestamp:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cls.test_results["timestamp"]))}</p>
    </div>

    <h2>Test Summary</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Result</th>
            <th>Duration (s)</th>
            <th>Details</th>
        </tr>
""")

            # Add rows for each test
            for test_name, test_data in cls.test_results.get("tests", {}).items():
                result_class = "passed" if test_data.get("success", False) else "failed"
                result_text = "PASS" if test_data.get("success", False) else "FAIL"
                duration = test_data.get("duration", 0)
                details = test_data.get("details", "")

                f.write(f"""
        <tr>
            <td>{test_name}</td>
            <td class="{result_class}">{result_text}</td>
            <td>{duration:.4f}</td>
            <td>{details}</td>
        </tr>""")

            # Add performance results if available
            if "performance" in cls.test_results:
                perf_data = cls.test_results["performance"]

                f.write("""
    </table>

    <h2>Performance Results</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
""")

                for metric, value in perf_data.items():
                    f.write(f"""
        <tr>
            <td>{metric}</td>
            <td>{value}</td>
        </tr>""")

            f.write("""
    </table>
</body>
</html>
""")

        logger.info(f"HTML report generated: {report_file}")

    def setUp(self):
        """Set up for each test."""
        self.start_time = time.time()
        self.test_name = self.id().split('.')[-1]
        logger.info(f"Starting test: {self.test_name}")

    def tearDown(self):
        """Clean up after each test."""
        duration = time.time() - self.start_time
        success = self._outcome.success

        # Record test results
        self.test_results["tests"][self.test_name] = {
            "success": success,
            "duration": duration,
            "details": self.get_test_details()
        }

        result_str = "PASSED" if success else "FAILED"
        logger.info(f"Test {self.test_name}: {result_str} in {duration:.4f}s")

    def get_test_details(self):
        """Get additional test details - override in subclasses."""
        return ""

    # Utility methods for tests
    def run_inference(
        self,
        inputs_dict: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
        use_grpc: bool = False
    ) -> Dict[str, np.ndarray]:
        """Run inference with the given inputs.

        Args:
            inputs_dict: Dictionary of input name to numpy array
            output_names: List of output names to fetch (None for all)
            use_grpc: Whether to use gRPC instead of HTTP

        Returns:
            Dictionary of output name to numpy array
        """
        client = self.grpc_client if use_grpc else self.http_client

        # Set default output names if not specified
        if output_names is None:
            output_names = self.output_names

        # Create input objects
        inputs = []
        for name, array in inputs_dict.items():
            if name not in self.input_names:
                raise ValueError(f"Unknown input: {name}")

            dtype = self.input_dtypes[name]

            if use_grpc:
                inp = tritonclient.grpc.InferInput(name, array.shape, dtype)
                inp.set_data_from_numpy(array)
            else:
                inp = tritonclient.http.InferInput(name, array.shape, dtype)
                inp.set_data_from_numpy(array)

            inputs.append(inp)

        # Create output objects
        outputs = []
        for name in output_names:
            if use_grpc:
                outputs.append(tritonclient.grpc.InferRequestedOutput(name))
            else:
                outputs.append(tritonclient.http.InferRequestedOutput(name))

        # Run inference
        start_time = time.time()
        response = client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs
        )
        inference_time = time.time() - start_time

        # Extract outputs
        outputs_dict = {}
        for name in output_names:
            outputs_dict[name] = response.as_numpy(name)

        # Store inference time
        outputs_dict["_inference_time"] = inference_time

        return outputs_dict

    def validate_input_shapes(self, inputs_dict: Dict[str, np.ndarray]) -> bool:
        """Validate input shapes against model metadata.

        Args:
            inputs_dict: Dictionary of input name to numpy array

        Returns:
            True if valid, raises ValueError if invalid
        """
        for name, array in inputs_dict.items():
            if name not in self.input_names:
                raise ValueError(f"Unknown input: {name}")

            expected_shape = self.input_shapes[name]
            actual_shape = array.shape

            # Check if shapes are compatible (accounting for dynamic dimensions)
            if len(expected_shape) != len(actual_shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )

            for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
                # -1 in expected shape means dynamic dimension
                if exp != -1 and exp != act:
                    raise ValueError(
                        f"Shape mismatch for {name} at dimension {i}: "
                        f"expected {exp}, got {act}"
                    )

        return True

    def compare_outputs(
        self,
        actual: Dict[str, np.ndarray],
        expected: Dict[str, np.ndarray],
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """Compare actual outputs against expected outputs.

        Args:
            actual: Dictionary of output name to actual numpy array
            expected: Dictionary of output name to expected numpy array
            rtol: Relative tolerance for float comparison
            atol: Absolute tolerance for float comparison

        Returns:
            True if outputs match, False otherwise
        """
        for name, exp_array in expected.items():
            if name not in actual:
                logger.error(f"Missing output: {name}")
                return False

            act_array = actual[name]

            if act_array.shape != exp_array.shape:
                logger.error(
                    f"Shape mismatch for {name}: "
                    f"expected {exp_array.shape}, got {act_array.shape}"
                )
                return False

            if act_array.dtype != exp_array.dtype:
                logger.warning(
                    f"Dtype mismatch for {name}: "
                    f"expected {exp_array.dtype}, got {act_array.dtype}"
                )
                # Continue but with potential precision issues

            # For numeric arrays, use numpy's allclose
            if np.issubdtype(act_array.dtype, np.number) and \
               np.issubdtype(exp_array.dtype, np.number):
                if not np.allclose(act_array, exp_array, rtol=rtol, atol=atol):
                    logger.error(f"Values mismatch for {name}")
                    return False
            # For other types (strings, etc.), use equality
            else:
                if not np.array_equal(act_array, exp_array):
                    logger.error(f"Values mismatch for {name}")
                    return False

        return True
