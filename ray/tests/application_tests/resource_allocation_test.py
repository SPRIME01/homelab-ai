"""
Test cases for Ray resource allocation.
"""

import time
import ray
import numpy as np
from typing import List, Dict, Any

from base_test import RayBaseTest, logger

class ResourceAllocationTest(RayBaseTest):
    """Test suite for Ray resource allocation."""

    application_name = "resource_allocation"

    def test_cpu_allocation(self):
        """Test CPU resource allocation."""
        # Get available CPUs
        cluster_resources = ray.cluster_resources()
        available_cpus = cluster_resources.get("CPU", 0)

        if available_cpus < 1:
            self.skipTest("No CPUs available for testing")

        logger.info(f"Testing CPU allocation with {available_cpus} available CPUs")

        # Try different CPU allocations
        cpu_counts = self.config["resources"]["cpu_tests"]
        cpu_results = {}

        for cpu_count in cpu_counts:
            if cpu_count > available_cpus:
                logger.info(f"Skipping test with {cpu_count} CPUs (only {available_cpus} available)")
                continue

            @ray.remote(num_cpus=cpu_count)
            def cpu_intensive_task():
                import time
                import os
                import psutil

                # Get process CPU info
                process = psutil.Process(os.getpid())

                # Perform CPU-intensive work
                start_time = time.time()
                result = 0
                duration = 1.0  # seconds
                while time.time() - start_time < duration:
                    result += sum(i*i for i in range(100000))

                # Get CPU utilization
                cpu_percent = process.cpu_percent()

                return {
                    "allocated_cpus": cpu_count,
                    "cpu_percent": cpu_percent,
                    "result": result
                }

            # Run task and measure
            with self.measure_time(f"CPU task with {cpu_count} CPUs") as elapsed:
                result = ray.get(cpu_intensive_task.remote())

            cpu_results[str(cpu_count)] = {
                "execution_time": elapsed,
                "cpu_percent": result["cpu_percent"]
            }

            logger.info(f"CPU {cpu_count} task: {elapsed:.2f}s, CPU usage: {result['cpu_percent']:.1f}%")

        # Record performance data
        self.test_results["performance"]["cpu_allocation"] = cpu_results

    def test_memory_allocation(self):
        """Test memory resource allocation."""
        # Parse memory strings like "1GB" to bytes
        def parse_memory_size(size_str):
            size = float(size_str[:-2])
            unit = size_str[-2:].upper()

            if unit == "MB":
                return int(size * 1024 * 1024)
            elif unit == "GB":
                return int(size * 1024 * 1024 * 1024)
            else:
                raise ValueError(f"Unsupported memory unit: {unit}")

        # Try different memory allocations
        memory_sizes = self.config["resources"]["memory_tests"]
        memory_results = {}

        for memory_size in memory_sizes:
            size_bytes = parse_memory_size(memory_size)
            array_elements = size_bytes // 8  # 8 bytes per float64

            @ray.remote(memory=size_bytes)
            def memory_intensive_task(size):
                import numpy as np

                # Allocate memory
                array = np.random.random(size).astype(np.float64)

                # Do something with the array
                result = float(np.sum(array))

                # Return size info
                return {
                    "allocated_size": size * 8,  # bytes
                    "array_shape": array.shape,
                    "checksum": result
                }

            try:
                # Run task and measure
                with self.measure_time(f"Memory task with {memory_size}") as elapsed:
                    result = ray.get(memory_intensive_task.remote(array_elements))

                memory_results[memory_size] = {
                    "execution_time": elapsed,
                    "array_elements": array_elements,
                    "reported_size": result["allocated_size"],
                }

                logger.info(f"Memory {memory_size} task: {elapsed:.2f}s, Elements: {array_elements}")

            except Exception as e:
                logger.error(f"Memory {memory_size} task failed: {e}")
                memory_results[memory_size] = {
                    "error": str(e)
                }

        # Record performance data
        self.test_results["performance"]["memory_allocation"] = memory_results

    def test_gpu_allocation(self):
        """Test GPU resource allocation if available."""
        # Check if GPUs are available
        cluster_resources = ray.cluster_resources()
        available_gpus = cluster_resources.get("GPU", 0)

        if available_gpus < 0.1:
            self.skipTest("No GPUs available for testing")

        logger.info(f"Testing GPU allocation with {available_gpus} available GPUs")

        # Try different GPU allocations
        gpu_fractions = self.config["resources"]["gpu_tests"]
        gpu_results = {}

        for gpu_fraction in gpu_fractions:
            if gpu_fraction > available_gpus:
                logger.info(f"Skipping test with {gpu_fraction} GPU (only {available_gpus} available)")
                continue

            @ray.remote(num_gpus=gpu_fraction)
            def gpu_task():
                # Try importing torch - if it fails, we'll use a simulation
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        # Create tensor on GPU
                        x = torch.rand(1000, 1000, device=device)
                        # Do matrix multiply
                        result = torch.matmul(x, x.T)
                        return {
                            "gpu_available": True,
                            "device_count": torch.cuda.device_count(),
                            "device_name": torch.cuda.get_device_name(0),
                            "tensor_sum": float(result.sum())
                        }
                    else:
                        return {"gpu_available": False, "error": "CUDA not available in torch"}
                except ImportError:
                    # If torch not available, check for TensorFlow
                    try:
                        import tensorflow as tf
                        if tf.config.list_physical_devices('GPU'):
                            gpus = tf.config.list_physical_devices('GPU')
                            tf.config.experimental.set_memory_growth(gpus[0], True)
                            # Create tensor on GPU
                            x = tf.random.normal([1000, 1000])
                            # Do matrix multiply
                            result = tf.matmul(x, x, transpose_b=True)
                            return {
                                "gpu_available": True,
                                "device_count": len(gpus),
                                "device_name": str(gpus[0]),
                                "tensor_sum": float(tf.reduce_sum(result))
                            }
                        else:
                            return {"gpu_available": False, "error": "GPU not available in TensorFlow"}
                    except ImportError:
                        # Neither torch nor TF available, just return environment info
                        import os
                        return {
                            "gpu_available": "CUDA_VISIBLE_DEVICES" in os.environ,
                            "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "error": "Neither PyTorch nor TensorFlow available"
                        }

            # Run task and measure
            with self.measure_time(f"GPU task with {gpu_fraction} GPU") as elapsed:
                result = ray.get(gpu_task.remote())

            gpu_results[str(gpu_fraction)] = {
                "execution_time": elapsed,
                "gpu_available": result.get("gpu_available", False),
                "details": result
            }

            logger.info(f"GPU {gpu_fraction} task: {elapsed:.2f}s, Available: {result.get('gpu_available', False)}")

        # Record performance data
        self.test_results["performance"]["gpu_allocation"] = gpu_results

    def test_custom_resources(self):
        """Test allocation of custom resources if defined."""
        # Check cluster for custom resources
        cluster_resources = ray.cluster_resources()
        custom_resources = {}

        for resource, quantity in cluster_resources.items():
            if resource not in ["CPU", "GPU", "memory", "object_store_memory"] and not resource.startswith("node:"):
                custom_resources[resource] = quantity

        if not custom_resources:
            self.skipTest("No custom resources available for testing")

        logger.info(f"Found custom resources: {custom_resources}")

        # Test each custom resource
        resource_results = {}

        for resource, total_quantity in custom_resources.items():
            # Test with half the available quantity
            test_quantity = total_quantity / 2

            @ray.remote
            def custom_resource_task(**resources):
                import time
                import os

                # Simulate work
                time.sleep(0.5)

                return {
                    "allocated_resources": resources,
                    "pid": os.getpid(),
                    "env": {k: v for k, v in os.environ.items()
                            if k.startswith("RAY_") or k.startswith("CUDA")}
                }

            # Set resource requirement dynamically
            task_with_resource = custom_resource_task.options(**{resource: test_quantity})

            # Run task and measure
            with self.measure_time(f"Task with {test_quantity} {resource}") as elapsed:
                result = ray.get(task_with_resource.remote())

            resource_results[resource] = {
                "execution_time": elapsed,
                "allocated": test_quantity,
                "details": result
            }

            logger.info(f"Custom resource {resource} task: {elapsed:.2f}s")

        # Record performance data
        self.test_results["performance"]["custom_resources"] = resource_results
