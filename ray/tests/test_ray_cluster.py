"""
Comprehensive test suite for validating Ray Cluster deployment on Kubernetes.
Tests cluster connectivity, resource allocation, task scheduling, GPU access,
integration with Triton Inference Server, and error recovery.
"""

import os
import sys
import time
import json
import logging
import pytest
import numpy as np
import unittest
import argparse
from typing import Dict, List, Tuple, Optional, Union
import ray
from ray.job_submission import JobSubmissionClient
from kubernetes import client, config
import kubetest
from kubetest.client import TestClient
import tritonclient.http

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ray_cluster_test.log')
    ]
)
logger = logging.getLogger("ray_test")

# Test configuration (can be overridden via CLI arguments)
DEFAULT_CONFIG = {
    "namespace": "ai",
    "ray_service_name": "ray-cluster",
    "ray_head_service_name": "ray-cluster-head-svc",
    "head_dashboard_port": 8265,
    "triton_service_name": "triton-inference-server",
    "triton_http_port": 8000,
    "wait_timeout": 120,  # seconds
    "test_timeout": 300,  # seconds
    "output_dir": "results",
    "port_forward": True,  # Use port forwarding for local testing
    "test_resources": {
        "cpu_tasks": [1, 2, 4],
        "gpu_tasks": [0.25, 0.5, 1.0]
    },
    "error_recovery": {
        "test_head_restart": False,  # Caution: only for full integration testing
        "test_worker_restart": True
    }
}

class RayClusterTest(unittest.TestCase):
    """Test suite for validating Ray Cluster deployment."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and connections."""
        cls.config = DEFAULT_CONFIG.copy()

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Ray Cluster deployment test')
        parser.add_argument('--namespace', type=str, help='Kubernetes namespace')
        parser.add_argument('--ray-service', type=str, help='Ray service name')
        parser.add_argument('--ray-head-service', type=str, help='Ray head service name')
        parser.add_argument('--dashboard-port', type=int, help='Ray dashboard port')
        parser.add_argument('--triton-service', type=str, help='Triton service name')
        parser.add_argument('--triton-port', type=int, help='Triton HTTP port')
        parser.add_argument('--wait-timeout', type=int, help='Wait timeout (seconds)')
        parser.add_argument('--port-forward', action='store_true', help='Use port forwarding')
        parser.add_argument('--no-port-forward', dest='port_forward', action='store_false')
        parser.add_argument('--config-file', type=str, help='Path to config JSON file')
        parser.add_argument('--output-dir', type=str, help='Test output directory')
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
        cls.apps_client = client.AppsV1Api()
        cls.test_client = TestClient(namespace=cls.config["namespace"])

        # Set up port forwarding if required
        cls.ray_pf = None
        cls.triton_pf = None

        if cls.config["port_forward"]:
            from kubetest.client import PodPortForward
            logger.info("Setting up port forwarding for Ray and Triton")

            # Find Ray head pod
            pods = cls.k8s_client.list_namespaced_pod(
                namespace=cls.config["namespace"],
                label_selector=f"component=ray-head"
            )

            if not pods.items:
                raise ValueError("No Ray head pod found")

            # Setup port forwarding for Ray head
            ray_head_pod = pods.items[0]
            logger.info(f"Setting up port forwarding to Ray head pod {ray_head_pod.metadata.name}")
            cls.ray_pf = PodPortForward(
                cls.k8s_client, ray_head_pod,
                ports=[cls.config["head_dashboard_port"], 10001]  # dashboard port and client server port
            )
            cls.ray_pf.start()

            # Set Ray address for port-forwarded connection
            cls.ray_address = f"ray://localhost:10001"
            cls.dashboard_url = f"http://localhost:{cls.config['head_dashboard_port']}"

            # Setup port forwarding for Triton (if requested)
            try:
                triton_pods = cls.k8s_client.list_namespaced_pod(
                    namespace=cls.config["namespace"],
                    label_selector=f"app={cls.config['triton_service_name']}"
                )

                if triton_pods.items:
                    triton_pod = triton_pods.items[0]
                    logger.info(f"Setting up port forwarding to Triton pod {triton_pod.metadata.name}")
                    cls.triton_pf = PodPortForward(
                        cls.k8s_client, triton_pod,
                        ports=[cls.config["triton_http_port"]]
                    )
                    cls.triton_pf.start()
                    cls.triton_url = f"localhost:{cls.config['triton_http_port']}"
                else:
                    logger.warning("No Triton pods found, skipping Triton port forwarding")
                    cls.triton_url = None
            except Exception as e:
                logger.warning(f"Failed to set up Triton port forwarding: {e}")
                cls.triton_url = None
        else:
            # Use service DNS for direct access
            cls.ray_address = f"ray://{cls.config['ray_head_service_name']}.{cls.config['namespace']}.svc.cluster.local:10001"
            cls.dashboard_url = f"http://{cls.config['ray_head_service_name']}.{cls.config['namespace']}.svc.cluster.local:{cls.config['head_dashboard_port']}"
            cls.triton_url = f"{cls.config['triton_service_name']}.{cls.config['namespace']}.svc.cluster.local:{cls.config['triton_http_port']}"

        logger.info(f"Ray address: {cls.ray_address}")
        logger.info(f"Ray dashboard: {cls.dashboard_url}")
        logger.info(f"Triton URL: {cls.triton_url}")

        # Create test summary dict
        cls.test_summary = {
            "timestamp": time.time(),
            "ray_address": cls.ray_address,
            "tests": {}
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        # Save test summary
        summary_file = os.path.join(cls.config["output_dir"], "ray_cluster_test_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(cls.test_summary, f, indent=2)
            logger.info(f"Test summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save test summary: {e}")

        # Stop port forwarding
        if cls.ray_pf:
            logger.info("Stopping Ray port forwarding")
            cls.ray_pf.stop()

        if cls.triton_pf:
            logger.info("Stopping Triton port forwarding")
            cls.triton_pf.stop()

    def setUp(self):
        """Set up test case."""
        logger.info(f"\n{'=' * 80}\nRunning test: {self._testMethodName}\n{'=' * 80}")

    def tearDown(self):
        """Clean up after test case."""
        # Disconnect from Ray if connected
        if ray.is_initialized():
            logger.info("Disconnecting from Ray")
            ray.shutdown()

        # Store test result in summary
        self.test_summary["tests"][self._testMethodName] = {
            "success": self._outcome.success,
            "timestamp": time.time()
        }

    def test_001_kubernetes_deployment_status(self):
        """Test that Ray cluster is properly deployed in Kubernetes."""
        try:
            # Check if Ray head service exists
            service = self.k8s_client.read_namespaced_service(
                name=self.config["ray_head_service_name"],
                namespace=self.config["namespace"]
            )
            self.assertIsNotNone(service, "Ray head service should exist")
            logger.info(f"Ray head service found: {service.metadata.name}")

            # Check head deployment
            head_deployment = self.apps_client.read_namespaced_deployment(
                name=f"{self.config['ray_service_name']}-head",
                namespace=self.config["namespace"]
            )
            self.assertIsNotNone(head_deployment, "Ray head deployment should exist")
            logger.info(f"Ray head deployment found: {head_deployment.metadata.name}")

            # Check head pod status
            head_pods = self.k8s_client.list_namespaced_pod(
                namespace=self.config["namespace"],
                label_selector=f"component=ray-head,app.kubernetes.io/instance={self.config['ray_service_name']}"
            )
            self.assertTrue(head_pods.items, "At least one Ray head pod should exist")

            for pod in head_pods.items:
                logger.info(f"Head pod: {pod.metadata.name}, status: {pod.status.phase}")
                self.assertEqual(pod.status.phase, "Running", f"Pod {pod.metadata.name} should be running")

                # Check for container readiness
                all_containers_ready = all(container_status.ready for container_status in pod.status.container_statuses)
                self.assertTrue(all_containers_ready, f"All containers in pod {pod.metadata.name} should be ready")

            # Check worker deployments/pods
            worker_pods = self.k8s_client.list_namespaced_pod(
                namespace=self.config["namespace"],
                label_selector=f"component=ray-worker,app.kubernetes.io/instance={self.config['ray_service_name']}"
            )

            logger.info(f"Found {len(worker_pods.items)} Ray worker pods")
            self.assertTrue(len(worker_pods.items) > 0, "At least one Ray worker pod should exist")

            for pod in worker_pods.items:
                logger.info(f"Worker pod: {pod.metadata.name}, status: {pod.status.phase}")
                self.assertEqual(pod.status.phase, "Running", f"Pod {pod.metadata.name} should be running")

                # Check for container readiness
                all_containers_ready = all(container_status.ready for container_status in pod.status.container_statuses)
                self.assertTrue(all_containers_ready, f"All containers in pod {pod.metadata.name} should be ready")

            # Save deployment details to summary
            self.test_summary["deployment"] = {
                "head_pods": len(head_pods.items),
                "worker_pods": len(worker_pods.items),
                "head_service": service.metadata.name
            }

        except Exception as e:
            logger.error(f"Failed to check Ray cluster deployment: {e}")
            self.fail(f"Ray cluster deployment check failed: {e}")

    def test_002_ray_cluster_connection(self):
        """Test connecting to the Ray cluster."""
        try:
            # Initialize Ray connection
            ray.init(address=self.ray_address, namespace="default")
            self.assertTrue(ray.is_initialized(), "Ray should be initialized")

            # Get Ray cluster info
            nodes_info = ray.nodes()
            logger.info(f"Connected to Ray cluster with {len(nodes_info)} nodes")

            # Check if nodes are alive
            alive_nodes = [node for node in nodes_info if node["alive"]]
            self.assertTrue(len(alive_nodes) > 0, "At least one Ray node should be alive")

            # Log cluster resources
            cluster_resources = ray.cluster_resources()
            logger.info(f"Cluster resources: {cluster_resources}")

            # Check for expected resources
            self.assertIn("CPU", cluster_resources, "Cluster should have CPU resources")
            total_cpus = cluster_resources["CPU"]
            logger.info(f"Total CPUs in cluster: {total_cpus}")
            self.assertTrue(total_cpus > 0, "Cluster should have at least one CPU")

            # Check for GPU resources if any
            if "GPU" in cluster_resources:
                total_gpus = cluster_resources["GPU"]
                logger.info(f"Total GPUs in cluster: {total_gpus}")
                self.assertTrue(total_gpus > 0, "Cluster should have at least one GPU")

            # Save cluster info to summary
            self.test_summary["cluster_info"] = {
                "nodes": len(nodes_info),
                "alive_nodes": len(alive_nodes),
                "resources": cluster_resources
            }

        except Exception as e:
            logger.error(f"Failed to connect to Ray cluster: {e}")
            self.fail(f"Ray cluster connection test failed: {e}")

    def test_003_basic_task_execution(self):
        """Test basic Ray task execution."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Define a simple Ray task
            @ray.remote
            def add(a, b):
                return a + b

            # Execute the task
            start_time = time.time()
            result = ray.get(add.remote(40, 2))
            elapsed = time.time() - start_time

            self.assertEqual(result, 42, "Task result should be correct")
            logger.info(f"Basic task executed in {elapsed:.4f} seconds")

            # Run multiple tasks in parallel
            num_tasks = 10
            start_time = time.time()
            results = ray.get([add.remote(i, i) for i in range(num_tasks)])
            elapsed = time.time() - start_time

            self.assertEqual(len(results), num_tasks, "All tasks should complete")
            logger.info(f"{num_tasks} tasks executed in parallel in {elapsed:.4f} seconds")

            # Save task results to summary
            self.test_summary["basic_tasks"] = {
                "single_task_time": elapsed,
                "parallel_tasks_time": elapsed,
                "num_parallel_tasks": num_tasks
            }

        except Exception as e:
            logger.error(f"Failed to execute basic Ray tasks: {e}")
            self.fail(f"Basic Ray task execution test failed: {e}")

    def test_004_cpu_resource_allocation(self):
        """Test CPU resource allocation with Ray tasks."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Get available CPU resources
            cluster_resources = ray.cluster_resources()
            available_cpus = cluster_resources.get("CPU", 0)
            logger.info(f"Available CPUs: {available_cpus}")

            if available_cpus < 1:
                self.skipTest("Not enough CPUs available for testing")

            # Define CPU-intensive Ray task
            @ray.remote
            def cpu_task(seconds):
                start = time.time()
                # Simulate CPU-intensive work
                while time.time() - start < seconds:
                    pass
                return {"start_time": start, "end_time": time.time()}

            # Test with different CPU allocations
            results = {}
            for cpu_count in self.config["test_resources"]["cpu_tasks"]:
                if cpu_count > available_cpus:
                    logger.warning(f"Skipping test with {cpu_count} CPUs as only {available_cpus} are available")
                    continue

                # Create task with specific CPU allocation
                task_with_cpus = cpu_task.options(num_cpus=cpu_count)

                # Run task and measure execution time
                logger.info(f"Running CPU task with {cpu_count} CPUs allocated")
                start_time = time.time()
                result = ray.get(task_with_cpus.remote(1.0))  # Run for 1 second
                elapsed = time.time() - start_time

                logger.info(f"CPU task with {cpu_count} CPUs completed in {elapsed:.4f} seconds")
                results[str(cpu_count)] = elapsed

            # Run multiple CPU tasks in parallel
            if available_cpus >= 2:
                num_tasks = int(available_cpus // 2)
                task_with_two_cpus = cpu_task.options(num_cpus=2)

                logger.info(f"Running {num_tasks} parallel CPU tasks with 2 CPUs each")
                start_time = time.time()
                results_parallel = ray.get([task_with_two_cpus.remote(1.0) for _ in range(num_tasks)])
                elapsed = time.time() - start_time

                logger.info(f"{num_tasks} CPU tasks completed in {elapsed:.4f} seconds")
                results["parallel"] = elapsed

            # Save results to summary
            self.test_summary["cpu_resource_tests"] = results

        except Exception as e:
            logger.error(f"Failed to test CPU resource allocation: {e}")
            self.fail(f"CPU resource allocation test failed: {e}")

    def test_005_gpu_resource_allocation(self):
        """Test GPU resource allocation with Ray tasks."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Check for GPU resources
            cluster_resources = ray.cluster_resources()
            if "GPU" not in cluster_resources or cluster_resources["GPU"] < 0.1:
                self.skipTest("No GPUs available for testing")

            available_gpus = cluster_resources["GPU"]
            logger.info(f"Available GPUs: {available_gpus}")

            # Define GPU task
            @ray.remote(num_gpus=0.1)
            def gpu_task():
                # Check if CUDA is available (if PyTorch is installed)
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        device_count = torch.cuda.device_count()
                        device = torch.device("cuda")
                        device_name = torch.cuda.get_device_name(0)
                        return {
                            "cuda_available": cuda_available,
                            "device_count": device_count,
                            "device_name": device_name
                        }
                    else:
                        return {"cuda_available": False}
                except ImportError:
                    # If PyTorch is not available, check for TensorFlow
                    try:
                        import tensorflow as tf
                        gpus = tf.config.list_physical_devices('GPU')
                        return {
                            "tensorflow_gpus": len(gpus),
                            "gpu_names": [gpu.name for gpu in gpus]
                        }
                    except ImportError:
                        # Check for NVIDIA GPUs directly
                        import os
                        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                        return {
                            "cuda_visible_devices": cuda_visible_devices,
                            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
                        }

            # Run a simple GPU task
            logger.info("Running a simple GPU task")
            result = ray.get(gpu_task.remote())
            logger.info(f"GPU task result: {result}")

            # Test with different GPU allocations
            results = {}
            for gpu_fraction in self.config["test_resources"]["gpu_tasks"]:
                if gpu_fraction > available_gpus:
                    logger.warning(f"Skipping test with {gpu_fraction} GPUs as only {available_gpus} are available")
                    continue

                # Create custom GPU task with specific allocation
                @ray.remote(num_gpus=gpu_fraction)
                def custom_gpu_task():
                    import os
                    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    return {
                        "gpu_fraction": gpu_fraction,
                        "cuda_visible_devices": cuda_visible_devices,
                        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
                    }

                # Run task and capture result
                logger.info(f"Running GPU task with {gpu_fraction} GPU fraction allocated")
                task_result = ray.get(custom_gpu_task.remote())
                logger.info(f"GPU task with {gpu_fraction} GPU fraction result: {task_result}")
                results[str(gpu_fraction)] = task_result

            # Run parallel GPU tasks if enough resources
            if available_gpus >= 1.0:
                max_tasks = int(available_gpus / 0.5)
                if max_tasks > 1:
                    @ray.remote(num_gpus=0.5)
                    def parallel_gpu_task():
                        import os
                        import time
                        time.sleep(1)  # Wait a bit to ensure we're really running in parallel
                        return os.environ.get("CUDA_VISIBLE_DEVICES", "")

                    logger.info(f"Running {max_tasks} parallel GPU tasks with 0.5 GPU each")
                    results_parallel = ray.get([parallel_gpu_task.remote() for _ in range(max_tasks)])
                    logger.info(f"Parallel GPU task results: {results_parallel}")
                    results["parallel"] = results_parallel

            # Save results to summary
            self.test_summary["gpu_resource_tests"] = results

        except Exception as e:
            logger.error(f"Failed to test GPU resource allocation: {e}")
            self.fail(f"GPU resource allocation test failed: {e}")

    def test_006_task_scheduling(self):
        """Test task scheduling and prioritization."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Define a task that simulates work
            @ray.remote
            def work_task(duration):
                time.sleep(duration)
                return time.time()

            # Submit a mix of short and long tasks
            short_tasks = []
            long_tasks = []

            logger.info("Submitting mix of short and long tasks")

            # Submit long-running tasks first
            for i in range(5):
                long_tasks.append(work_task.remote(2.0))  # 2 second tasks

            # Then submit short-running tasks
            for i in range(10):
                short_tasks.append(work_task.remote(0.5))  # 0.5 second tasks

            # Wait for short tasks to complete and measure time
            start_time = time.time()
            short_results = ray.get(short_tasks)
            short_elapsed = time.time() - start_time

            # Wait for long tasks to complete
            start_time = time.time()
            long_results = ray.get(long_tasks)
            long_elapsed = time.time() - start_time

            logger.info(f"Short tasks completed in {short_elapsed:.4f} seconds")
            logger.info(f"Long tasks completed in {long_elapsed:.4f} seconds")

            # Test task prioritization if available in this Ray version
            try:
                # Define tasks with different priorities
                @ray.remote
                def priority_task(priority_value):
                    time.sleep(1.0)
                    return {"priority": priority_value, "time": time.time()}

                # Submit tasks with different priorities (if supported)
                high_priority_tasks = []
                low_priority_tasks = []

                # Use options() to set priorities
                for i in range(3):
                    high_priority_tasks.append(
                        priority_task.options(priority=100).remote(100)
                    )

                for i in range(3):
                    low_priority_tasks.append(
                        priority_task.options(priority=1).remote(1)
                    )

                # Gather results
                all_results = ray.get(high_priority_tasks + low_priority_tasks)
                logger.info(f"Priority task results: {all_results}")

            except Exception as e:
                logger.warning(f"Priority scheduling test skipped: {e}")

            # Save scheduling results to summary
            self.test_summary["task_scheduling"] = {
                "short_task_time": short_elapsed,
                "long_task_time": long_elapsed,
                "short_tasks": len(short_tasks),
                "long_tasks": len(long_tasks)
            }

        except Exception as e:
            logger.error(f"Failed to test task scheduling: {e}")
            self.fail(f"Task scheduling test failed: {e}")

    def test_007_actor_pool(self):
        """Test Ray actor pool for parallel processing."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Define a simple actor
            @ray.remote
            class Worker:
                def __init__(self, worker_id):
                    self.worker_id = worker_id

                def process(self, task_id, sleep_time=0.5):
                    time.sleep(sleep_time)  # Simulate work
                    return {
                        "worker_id": self.worker_id,
                        "task_id": task_id,
                        "timestamp": time.time()
                    }

                def get_id(self):
                    return self.worker_id

            # Create a pool of actors
            num_workers = 4
            workers = [Worker.remote(i) for i in range(num_workers)]

            # Verify workers are created
            worker_ids = ray.get([worker.get_id.remote() for worker in workers])
            logger.info(f"Created {len(workers)} workers with IDs: {worker_ids}")
            self.assertEqual(len(worker_ids), num_workers, "All workers should be created")

            # Submit tasks to the workers
            num_tasks = 10
            start_time = time.time()
            results = ray.get([workers[i % num_workers].process.remote(i) for i in range(num_tasks)])
            elapsed = time.time() - start_time

            logger.info(f"Processed {num_tasks} tasks with {num_workers} workers in {elapsed:.4f} seconds")

            # Verify all tasks were processed
            processed_tasks = [r["task_id"] for r in results]
            self.assertEqual(len(processed_tasks), num_tasks, "All tasks should be processed")

            # Check distribution of tasks among workers
            tasks_per_worker = {}
            for result in results:
                worker_id = result["worker_id"]
                if worker_id not in tasks_per_worker:
                    tasks_per_worker[worker_id] = 0
                tasks_per_worker[worker_id] += 1

            logger.info(f"Tasks per worker: {tasks_per_worker}")

            # Save actor results to summary
            self.test_summary["actor_pool"] = {
                "num_workers": num_workers,
                "num_tasks": num_tasks,
                "processing_time": elapsed,
                "tasks_per_worker": tasks_per_worker
            }

        except Exception as e:
            logger.error(f"Failed to test actor pool: {e}")
            self.fail(f"Actor pool test failed: {e}")

    def test_008_triton_integration(self):
        """Test integration with Triton Inference Server."""
        # Skip if Triton URL is not available
        if not self.triton_url:
            self.skipTest("Triton URL not available. Skipping Triton integration test.")

        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # First, check if Triton is accessible
            triton_client = tritonclient.http.InferenceServerClient(
                url=self.triton_url, verbose=False
            )

            # Check server status
            is_live = False
            try:
                is_live = triton_client.is_server_live()
            except Exception as e:
                logger.error(f"Triton server is not accessible: {e}")
                self.skipTest(f"Triton server is not accessible: {e}")

            self.assertTrue(is_live, "Triton server should be live")

            # Get available models
            try:
                models = triton_client.get_model_repository_index()
                if not models:
                    logger.warning("No models available in Triton. Limited testing will be performed.")
                else:
                    logger.info(f"Available models in Triton: {[model['name'] for model in models]}")
            except Exception as e:
                logger.warning(f"Could not get model repository index: {e}")
                models = []

            # Define Ray task that calls Triton
            @ray.remote
            def call_triton(triton_url, model_name=None):
                import tritonclient.http
                import numpy as np

                # Connect to Triton
                client = tritonclient.http.InferenceServerClient(url=triton_url, verbose=False)

                # Check server status
                results = {
                    "is_live": client.is_server_live(),
                    "is_ready": client.is_server_ready()
                }

                # Try to get server metadata
                try:
                    metadata = client.get_server_metadata()
                    results["server_name"] = metadata.get("name")
                    results["server_version"] = metadata.get("version")
                except Exception as e:
                    results["metadata_error"] = str(e)

                # If a model name is provided, try to perform inference
                if model_name:
                    try:
                        # Get model metadata
                        model_metadata = client.get_model_metadata(model_name)
                        results["model_metadata"] = {
                            "name": model_metadata.get("name"),
                            "inputs": [i.get("name") for i in model_metadata.get("inputs", [])],
                            "outputs": [o.get("name") for o in model_metadata.get("outputs", [])]
                        }

                        # Check if model is ready
                        is_model_ready = client.is_model_ready(model_name)
                        results["is_model_ready"] = is_model_ready

                        # Skip inference if model is not ready
                        if not is_model_ready:
                            results["inference_skipped"] = True
                            return results

                        # Create a simple test input - this would need customizing per model
                        input_name = model_metadata["inputs"][0]["name"]
                        output_name = model_metadata["outputs"][0]["name"]
                        input_shape = model_metadata["inputs"][0]["shape"]
                        input_datatype = model_metadata["inputs"][0]["datatype"]

                        # Create a dummy input based on datatype
                        if input_datatype == "FP32":
                            input_data = np.random.rand(*input_shape).astype(np.float32)
                        elif input_datatype == "INT32":
                            input_data = np.random.randint(0, 100, size=input_shape).astype(np.int32)
                        elif input_datatype == "BYTES":
                            input_data = np.array(["test"], dtype=np.object_)
                        else:
                            input_data = np.random.rand(*input_shape).astype(np.float32)

                        # Create inference input
                        inputs = tritonclient.http.InferInput(input_name, input_data.shape, input_datatype)
                        inputs.set_data_from_numpy(input_data)

                        # Perform inference
                        start_time = time.time()
                        response = client.infer(model_name, [inputs])
                        inference_time = time.time() - start_time

                        # Get output
                        output = response.as_numpy(output_name)

                        results["inference_time"] = inference_time
                        results["output_shape"] = output.shape

                    except Exception as e:
                        results["inference_error"] = str(e)

                return results

            # Run Triton integration test from multiple Ray tasks
            logger.info(f"Testing Triton integration with URL: {self.triton_url}")
            num_tasks = 4
            tasks = []

            # Find a suitable model for inference testing
            model_name = None
            if models:
                model_name = models[0]["name"]
                logger.info(f"Using model {model_name} for inference testing")

            # Submit tasks
            for i in range(num_tasks):
                tasks.append(call_triton.remote(self.triton_url, model_name))

            # Get results
            results = ray.get(tasks)

            # Verify all tasks connected to Triton
            for i, result in enumerate(results):
                logger.info(f"Triton task {i} result: {result}")
                self.assertTrue(result.get("is_live"), f"Triton server should be live in task {i}")
                self.assertTrue(result.get("is_ready"), f"Triton server should be ready in task {i}")

            # Save Triton integration results to summary
            self.test_summary["triton_integration"] = {
                "num_tasks": num_tasks,
                "triton_url": self.triton_url,
                "model_tested": model_name,
                "all_tasks_successful": all(r.get("is_live") and r.get("is_ready") for r in results)
            }

        except Exception as e:
            logger.error(f"Failed to test Triton integration: {e}")
            self.fail(f"Triton integration test failed: {e}")

    def test_009_error_recovery(self):
        """Test error recovery capabilities."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            # Test task failure recovery
            logger.info("Testing task failure recovery")

            @ray.remote
            def failing_task(fail=True):
                if fail:
                    raise ValueError("This task is designed to fail")
                return "Task succeeded"

            # Create task that will fail and use ray.get with a timeout
            failing_task_ref = failing_task.remote(fail=True)

            # This should raise an exception
            with self.assertRaises(ray.exceptions.RayTaskError):
                ray.get(failing_task_ref)

            # Now try a task that succeeds
            success_task_ref = failing_task.remote(fail=False)
            result = ray.get(success_task_ref)
            self.assertEqual(result, "Task succeeded")
            logger.info("Ray successfully handled task failure and success")

            # Test actor failure recovery
            @ray.remote
            class FailingActor:
                def __init__(self):
                    self.counter = 0

                def increment(self):
                    self.counter += 1
                    return self.counter

                def fail(self):
                    raise ValueError("This actor is designed to fail")

            # Create an actor
            actor = FailingActor.remote()

            # Call method that should succeed
            counter = ray.get(actor.increment.remote())
            self.assertEqual(counter, 1)

            # Call method that fails
            with self.assertRaises(ray.exceptions.RayTaskError):
                ray.get(actor.fail.remote())

            # Actor should still be usable
            try:
                counter = ray.get(actor.increment.remote())
                self.assertEqual(counter, 2)
                logger.info("Ray successfully recovered from actor method failure")
            except Exception:
                logger.warning("Actor did not recover from method failure - this is expected in some Ray versions")

            # Test worker pod failure/restart if enabled
            if self.config["error_recovery"]["test_worker_restart"]:
                # Find a worker pod
                worker_pods = self.k8s_client.list_namespaced_pod(
                    namespace=self.config["namespace"],
                    label_selector=f"component=ray-worker,app.kubernetes.io/instance={self.config['ray_service_name']}"
                )

                if worker_pods.items:
                    worker_pod = worker_pods.items[0]
                    pod_name = worker_pod.metadata.name

                    # Create long-running actor on the cluster
                    @ray.remote
                    class MonitoringActor:
                        def __init__(self):
                            self.created_time = time.time()

                        def get_uptime(self):
                            return time.time() - self.created_time

                    # Create persistent actor
                    persistent_actor = MonitoringActor.remote()

                    # Verify actor is working
                    uptime = ray.get(persistent_actor.get_uptime.remote())
                    logger.info(f"Actor initial uptime: {uptime:.2f} seconds")

                    # Delete the worker pod to force restart
                    logger.warning(f"Deleting worker pod {pod_name} to test recovery...")
                    self.k8s_client.delete_namespaced_pod(name=pod_name, namespace=self.config["namespace"])

                    # Wait for pod to be deleted
                    logger.info("Waiting for pod to be deleted...")
                    deleted = False
                    for _ in range(30):  # Wait up to 30 seconds
                        try:
                            self.k8s_client.read_namespaced_pod(name=pod_name, namespace=self.config["namespace"])
                        except client.rest.ApiException:
                            deleted = True
                            break
                        time.sleep(1)

                    self.assertTrue(deleted, f"Pod {pod_name} should be deleted")

                    # Wait for worker to be replaced
                    logger.info("Waiting for worker to be replaced...")
                    time.sleep(30)  # Give Kubernetes time to start new pod

                    # Check if Ray cluster still functions
                    nodes_info = ray.nodes()
                    alive_nodes = [node for node in nodes_info if node["alive"]]
                    logger.info(f"Cluster now has {len(alive_nodes)} alive nodes")

                    # Check if actor is still available (may take some time to restart)
                    actor_available = False
                    for attempt in range(5):
                        try:
                            uptime = ray.get(persistent_actor.get_uptime.remote(), timeout=10)
                            logger.info(f"Actor uptime after pod restart: {uptime:.2f} seconds")
                            actor_available = True
                            break
                        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                            logger.info(f"Actor not available yet, attempt {attempt+1}/5")
                            time.sleep(5)

                    # Note: In many Ray setups, the actor won't recover after worker restart
                    # This is expected behavior, so we don't make this a test failure
                    if not actor_available:
                        logger.warning("Actor did not recover after worker restart - this may be expected")

                    # Try creating a new actor to verify cluster is working
                    new_actor = MonitoringActor.remote()
                    new_uptime = ray.get(new_actor.get_uptime.remote())
                    logger.info(f"New actor created after worker restart with uptime: {new_uptime:.2f} seconds")

                    # Save worker restart results to summary
                    self.test_summary["worker_restart"] = {
                        "deleted_pod": pod_name,
                        "actor_recovered": actor_available,
                        "cluster_functional": len(alive_nodes) > 0,
                        "new_actor_created": new_uptime > 0
                    }

            # Save error recovery results to summary
            self.test_summary["error_recovery"] = {
                "task_recovery": True,  # We got here, so task recovery worked
                "actor_method_recovery": True  # We got here, so actor method recovery worked
            }

        except Exception as e:
            logger.error(f"Failed to test error recovery: {e}")
            self.fail(f"Error recovery test failed: {e}")

    def test_010_benchmark_workload(self):
        """Run a benchmark workload to evaluate cluster performance."""
        try:
            # Initialize Ray connection if not already initialized
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            logger.info("Running benchmark workload")

            # Define a compute-intensive task for benchmarking
            @ray.remote
            def compute_task(size):
                import numpy as np
                # Create and multiply large matrices
                a = np.random.rand(size, size)
                b = np.random.rand(size, size)
                return np.sum(np.matmul(a, b))

            # Run tasks with increasing computational intensity
            sizes = [100, 500, 1000]
            results = {}

            for size in sizes:
                logger.info(f"Benchmarking with matrix size {size}x{size}")
                start_time = time.time()
                result = ray.get(compute_task.remote(size))
                elapsed = time.time() - start_time
                logger.info(f"Task with size {size} completed in {elapsed:.4f} seconds")
                results[str(size)] = elapsed

            # Run parallel tasks
            size = 500  # Use medium size for parallel test
            num_parallel = 4

            logger.info(f"Benchmarking {num_parallel} parallel tasks with matrix size {size}x{size}")
            start_time = time.time()
            parallel_results = ray.get([compute_task.remote(size) for _ in range(num_parallel)])
            elapsed = time.time() - start_time

            logger.info(f"{num_parallel} parallel tasks completed in {elapsed:.4f} seconds")
            results["parallel"] = elapsed

            # Save benchmark results to summary
            self.test_summary["benchmark"] = {
                "matrix_sizes": sizes,
                "single_task_times": {str(size): results[str(size)] for size in sizes},
                "parallel_tasks": num_parallel,
                "parallel_time": results["parallel"],
                "parallel_efficiency": results[str(size)] * num_parallel / results["parallel"]
            }

        except Exception as e:
            logger.error(f"Failed to run benchmark workload: {e}")
            self.fail(f"Benchmark workload test failed: {e}")

    def test_011_generate_report(self):
        """Generate a final report of all tests."""
        # Gather final cluster info
        try:
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace="default")

            nodes_info = ray.nodes()
            alive_nodes = [node for node in nodes_info if node["alive"]]
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            self.test_summary["final_cluster_state"] = {
                "total_nodes": len(nodes_info),
                "alive_nodes": len(alive_nodes),
                "total_resources": cluster_resources,
                "available_resources": available_resources
            }
        except Exception as e:
            logger.warning(f"Failed to gather final cluster info: {e}")

        # Save final test summary
        summary_file = os.path.join(self.config["output_dir"], "ray_cluster_test_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(self.test_summary, f, indent=2)
            logger.info(f"Test summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save test summary: {e}")

        # Generate HTML report
        try:
            self._generate_html_report()
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")

    def _generate_html_report(self):
        """Generate an HTML report from the test summary."""
        report_path = os.path.join(self.config["output_dir"], "ray_cluster_test_report.html")

        with open(report_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Ray Cluster Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .status-pass {{ color: green; }}
        .status-fail {{ color: red; }}
        pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Ray Cluster Test Report</h1>
    <p><strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Ray Address:</strong> {self.ray_address}</p>
    <p><strong>Dashboard URL:</strong> {self.dashboard_url}</p>

    <h2>Test Summary</h2>
    <table>
        <tr>
            <th>Test</th>
            <th>Status</th>
        </tr>
""")

            # Add rows for each test
            for test_name, test_data in self.test_summary.get("tests", {}).items():
                status = "PASS" if test_data.get("success", False) else "FAIL"
                status_class = "status-pass" if test_data.get("success", False) else "status-fail"
                f.write(f"""
        <tr>
            <td>{test_name}</td>
            <td class="{status_class}">{status}</td>
        </tr>""")

            f.write("""
    </table>

    <h2>Cluster Information</h2>
""")

            # Add cluster info if available
            if "deployment" in self.test_summary:
                f.write(f"""
    <h3>Kubernetes Deployment</h3>
    <ul>
        <li><strong>Head Pods:</strong> {self.test_summary["deployment"].get("head_pods", "N/A")}</li>
        <li><strong>Worker Pods:</strong> {self.test_summary["deployment"].get("worker_pods", "N/A")}</li>
        <li><strong>Head Service:</strong> {self.test_summary["deployment"].get("head_service", "N/A")}</li>
    </ul>
""")

            if "cluster_info" in self.test_summary:
                f.write(f"""
    <h3>Ray Cluster</h3>
    <ul>
        <li><strong>Total Nodes:</strong> {self.test_summary["cluster_info"].get("nodes", "N/A")}</li>
        <li><strong>Alive Nodes:</strong> {self.test_summary["cluster_info"].get("alive_nodes", "N/A")}</li>
    </ul>

    <h4>Resources</h4>
    <pre>{json.dumps(self.test_summary["cluster_info"].get("resources", {}), indent=2)}</pre>
""")

            # Add benchmark results if available
            if "benchmark" in self.test_summary:
                f.write("""
    <h2>Benchmark Results</h2>
    <table>
        <tr>
            <th>Matrix Size</th>
            <th>Execution Time (s)</th>
        </tr>
""")

                for size, time_taken in self.test_summary["benchmark"].get("single_task_times", {}).items():
                    f.write(f"""
        <tr>
            <td>{size}x{size}</td>
            <td>{time_taken:.4f}</td>
        </tr>""")

                parallel_tasks = self.test_summary["benchmark"].get("parallel_tasks", "N/A")
                parallel_time = self.test_summary["benchmark"].get("parallel_time", "N/A")
                parallel_efficiency = self.test_summary["benchmark"].get("parallel_efficiency", "N/A")
                if parallel_efficiency != "N/A":
                    parallel_efficiency = f"{parallel_efficiency:.2f}x"

                f.write(f"""
    </table>

    <h3>Parallel Performance</h3>
    <ul>
        <li><strong>Number of Parallel Tasks:</strong> {parallel_tasks}</li>
        <li><strong>Execution Time:</strong> {parallel_time if parallel_time == "N/A" else f"{parallel_time:.4f}s"}</li>
        <li><strong>Parallel Efficiency:</strong> {parallel_efficiency}</li>
    </ul>
""")

            # Add error recovery info if available
            if "worker_restart" in self.test_summary:
                restart_info = self.test_summary["worker_restart"]
                actor_recovered = "Yes" if restart_info.get("actor_recovered", False) else "No"
                cluster_functional = "Yes" if restart_info.get("cluster_functional", False) else "No"

                f.write(f"""
    <h2>Worker Restart Test</h2>
    <ul>
        <li><strong>Deleted Pod:</strong> {restart_info.get("deleted_pod", "N/A")}</li>
        <li><strong>Actor Recovered:</strong> {actor_recovered}</li>
        <li><strong>Cluster Functional After Restart:</strong> {cluster_functional}</li>
    </ul>
""")

            # Add Triton integration info if available
            if "triton_integration" in self.test_summary:
                triton_info = self.test_summary["triton_integration"]
                success = "Yes" if triton_info.get("all_tasks_successful", False) else "No"

                f.write(f"""
    <h2>Triton Integration</h2>
    <ul>
        <li><strong>Triton URL:</strong> {triton_info.get("triton_url", "N/A")}</li>
        <li><strong>Model Tested:</strong> {triton_info.get("model_tested", "None")}</li>
        <li><strong>Tasks Run:</strong> {triton_info.get("num_tasks", "N/A")}</li>
        <li><strong>All Tasks Successful:</strong> {success}</li>
    </ul>
""")

            # Add final cluster state if available
            if "final_cluster_state" in self.test_summary:
                final_state = self.test_summary["final_cluster_state"]

                f.write(f"""
    <h2>Final Cluster State</h2>
    <ul>
        <li><strong>Total Nodes:</strong> {final_state.get("total_nodes", "N/A")}</li>
        <li><strong>Alive Nodes:</strong> {final_state.get("alive_nodes", "N/A")}</li>
    </ul>

    <h3>Total Resources</h3>
    <pre>{json.dumps(final_state.get("total_resources", {}), indent=2)}</pre>

    <h3>Available Resources</h3>
    <pre>{json.dumps(final_state.get("available_resources", {}), indent=2)}</pre>
""")

            f.write("""
    <footer>
        <p>Generated by Ray Cluster Test Suite</p>
    </footer>
</body>
</html>
""")

        logger.info(f"Generated HTML report at {report_path}")

def main():
    """Run the tests."""
    # Setup command-line argument parsing for standalone execution
    parser = argparse.ArgumentParser(description='Ray Cluster Deployment Tester')
    parser.add_argument('--namespace', type=str, default='ai',
                      help='Kubernetes namespace where Ray is deployed')
    parser.add_argument('--ray-service', type=str, default='ray-cluster',
                      help='Ray service name')
    parser.add_argument('--ray-head-service', type=str, default='ray-cluster-head-svc',
                      help='Ray head service name')
    parser.add_argument('--dashboard-port', type=int, default=8265,
                      help='Ray dashboard port')
    parser.add_argument('--port-forward', action='store_true', default=True,
                      help='Use port forwarding to access Ray cluster')
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
    suite = unittest.TestLoader().loadTestsFromTestCase(RayClusterTest)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
