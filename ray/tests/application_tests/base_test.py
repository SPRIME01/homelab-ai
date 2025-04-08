"""
Base testing framework for Ray applications following TDD principles.

This module provides the foundation for testing distributed AI workloads
on Ray, supporting task distribution, resource allocation, fault tolerance,
and service integration testing.
"""

import os
import sys
import time
import json
import yaml
import logging
import pytest
import unittest
import ray
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ray_application_test.log')
    ]
)
logger = logging.getLogger("ray_application_test")

class RayTestConfiguration:
    """Configuration manager for Ray application tests."""

    DEFAULT_CONFIG = {
        "ray": {
            "address": "auto",
            "namespace": "test",
            "runtime_env": {
                "pip": ["numpy", "pandas", "scikit-learn"]
            }
        },
        "resources": {
            "cpu_tests": [1, 2, 4],
            "gpu_tests": [0.25, 0.5, 1.0],
            "memory_tests": ["500MB", "1GB", "2GB"]
        },
        "fault_tolerance": {
            "worker_failure_simulation": True,
            "recovery_timeout": 30  # seconds
        },
        "performance": {
            "baseline_latency_ms": 100,
            "throughput_threshold": 10,
            "scaling_factor": 0.8,  # Expected performance when scaling
        },
        "output_dir": "test_results"
    }

    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file with defaults.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        config = cls.DEFAULT_CONFIG.copy()

        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                cls._deep_update(config, file_config)

        # Override with environment variables if set
        if "RAY_ADDRESS" in os.environ:
            config["ray"]["address"] = os.environ["RAY_ADDRESS"]
        if "RAY_NAMESPACE" in os.environ:
            config["ray"]["namespace"] = os.environ["RAY_NAMESPACE"]

        return config

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update a nested dictionary.

        Args:
            base_dict: Dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                RayTestConfiguration._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


class RayBaseTest(unittest.TestCase):
    """Base class for Ray application tests."""

    # Class variables to be overridden in child classes
    application_name = "ray_app"
    config_path = None

    @classmethod
    def setUpClass(cls):
        """Set up Ray cluster connection and test environment."""
        # Initialize configuration
        cls.config = RayTestConfiguration.load_config(cls.config_path)

        # Create output directory
        cls.output_dir = cls.config["output_dir"]
        os.makedirs(cls.output_dir, exist_ok=True)

        # Connect to Ray or start local Ray
        if not ray.is_initialized():
            try:
                ray.init(
                    address=cls.config["ray"]["address"],
                    namespace=cls.config["ray"]["namespace"],
                    runtime_env=cls.config["ray"]["runtime_env"],
                    ignore_reinit_error=True
                )
                logger.info(f"Connected to Ray at {cls.config['ray']['address']}")
            except Exception as e:
                logger.warning(f"Failed to connect to Ray cluster: {e}")
                logger.info("Starting local Ray cluster instead")
                ray.init(namespace=cls.config["ray"]["namespace"])

        # Store initial cluster state for comparison
        cls.initial_cluster_info = {
            "nodes": ray.nodes(),
            "available_resources": ray.available_resources(),
            "cluster_resources": ray.cluster_resources()
        }

        # Record test results
        cls.test_results = {
            "application": cls.application_name,
            "timestamp": time.time(),
            "tests": {},
            "performance": {},
            "ray_cluster": {
                "address": ray.get_runtime_context().get_address(),
                "nodes": len(cls.initial_cluster_info["nodes"]),
                "resources": cls.initial_cluster_info["cluster_resources"]
            }
        }

        logger.info(f"Test setup complete for {cls.application_name}")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        # Save test results
        results_file = os.path.join(cls.output_dir, f"{cls.application_name}_results.json")
        with open(results_file, "w") as f:
            json.dump(cls.test_results, f, indent=2)

        logger.info(f"Test results saved to {results_file}")

        # Generate HTML report
        cls._generate_html_report()

        # Don't shut down Ray - it might be an external cluster
        logger.info("Test teardown complete")

    def setUp(self):
        """Set up for each test."""
        self.start_time = time.time()
        self.test_name = self._testMethodName
        logger.info(f"Starting test: {self.test_name}")

    def tearDown(self):
        """Clean up after each test."""
        duration = time.time() - self.start_time
        success = self._outcome.success

        # Record test results
        self.test_results["tests"][self.test_name] = {
            "success": success,
            "duration": duration
        }

        status = "PASSED" if success else "FAILED"
        logger.info(f"Test {self.test_name} {status} in {duration:.2f}s")

    @classmethod
    def _generate_html_report(cls):
        """Generate HTML report from test results."""
        report_path = os.path.join(cls.output_dir, f"{cls.application_name}_report.html")

        with open(report_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Ray Application Test Results: {cls.application_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Ray Application Test Results: {cls.application_name}</h1>
    <div class="summary">
        <p><strong>Timestamp:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cls.test_results["timestamp"]))}</p>
        <p><strong>Ray Address:</strong> {cls.test_results["ray_cluster"]["address"]}</p>
        <p><strong>Cluster Nodes:</strong> {cls.test_results["ray_cluster"]["nodes"]}</p>
    </div>

    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
            <th>Duration (s)</th>
        </tr>
""")

            # Add rows for each test
            for test_name, test_data in cls.test_results["tests"].items():
                status = "PASSED" if test_data["success"] else "FAILED"
                status_class = "passed" if test_data["success"] else "failed"

                f.write(f"""
        <tr>
            <td>{test_name}</td>
            <td class="{status_class}">{status}</td>
            <td>{test_data["duration"]:.2f}</td>
        </tr>""")

            # Add performance results if available
            if cls.test_results.get("performance"):
                f.write("""
    </table>

    <h2>Performance Results</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
""")

                for metric, value in cls.test_results["performance"].items():
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

        logger.info(f"HTML report generated at {report_path}")

    # Utility methods for tests
    @contextmanager
    def measure_time(self, description: str = "Operation"):
        """
        Context manager for timing operations.

        Args:
            description: Description of the operation

        Yields:
            None
        """
        start_time = time.time()
        yield
        elapsed = time.time() - start_time
        logger.info(f"{description} took {elapsed:.4f} seconds")
        return elapsed

    def wait_for_nodes(self, min_nodes: int, timeout: int = 60) -> bool:
        """
        Wait for minimum number of nodes to be available.

        Args:
            min_nodes: Minimum number of nodes required
            timeout: Maximum time to wait in seconds

        Returns:
            True if condition met, False if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            nodes = ray.nodes()
            alive_nodes = [node for node in nodes if node.get("alive", False)]
            if len(alive_nodes) >= min_nodes:
                return True
            time.sleep(1)
        return False

    def verify_resources(self, required_resources: Dict[str, float]) -> bool:
        """
        Verify that required resources are available in cluster.

        Args:
            required_resources: Dictionary of resource name to quantity

        Returns:
            True if all resources are available, False otherwise
        """
        cluster_resources = ray.cluster_resources()
        for resource, quantity in required_resources.items():
            if resource not in cluster_resources:
                logger.warning(f"Resource {resource} not available in cluster")
                return False
            if cluster_resources[resource] < quantity:
                logger.warning(f"Insufficient {resource}: required {quantity}, available {cluster_resources[resource]}")
                return False
        return True

    def simulate_node_failure(self, recover: bool = True) -> None:
        """
        Simulate node failure in Ray cluster.

        Args:
            recover: Whether to simulate recovery after failure
        """
        logger.warning("Simulating node failure")
        # This is a simulation - in a real test, you might use Kubernetes API
        # to actually kill a pod, or use Ray's internals to simulate failure

        @ray.remote
        def system_info():
            import socket
            import os
            return {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "ray_id": ray.get_runtime_context().get_node_id()
            }

        # Get info from a worker
        try:
            node_info = ray.get(system_info.remote())
            logger.info(f"Simulating failure on {node_info['hostname']}")

            # Kill worker process (simulation only - doesn't actually kill)
            logger.info(f"Simulated worker failure for node {node_info['ray_id']}")

            if recover:
                # Wait for simulated recovery
                logger.info("Waiting for simulated recovery...")
                time.sleep(self.config["fault_tolerance"]["recovery_timeout"])
                logger.info("Simulated node recovery complete")
        except Exception as e:
            logger.error(f"Error in node failure simulation: {e}")

    def record_performance(self, metric: str, value: float) -> None:
        """
        Record a performance metric.

        Args:
            metric: Name of metric
            value: Value of metric
        """
        self.test_results["performance"][metric] = value
        logger.info(f"Performance metric {metric}: {value}")
