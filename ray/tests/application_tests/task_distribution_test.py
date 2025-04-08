"""
Test cases for Ray task distribution patterns.
"""

import time
import ray
import numpy as np
from typing import List, Dict, Any

from base_test import RayBaseTest, logger

class TaskDistributionTest(RayBaseTest):
    """Test suite for Ray task distribution patterns."""

    application_name = "task_distribution"

    def test_basic_task_distribution(self):
        """Test basic task distribution across workers."""
        @ray.remote
        def simple_task(task_id):
            import socket
            import time

            # Simulate work
            time.sleep(0.1)

            return {
                "task_id": task_id,
                "hostname": socket.gethostname(),
                "time": time.time()
            }

        # Submit a batch of tasks
        num_tasks = 50

        with self.measure_time(f"Submitting {num_tasks} tasks") as submit_time:
            results = ray.get([simple_task.remote(i) for i in range(num_tasks)])

        # Analyze distribution
        hosts = {}
        for result in results:
            hostname = result["hostname"]
            if hostname not in hosts:
                hosts[hostname] = 0
            hosts[hostname] += 1

        logger.info(f"Task distribution across {len(hosts)} hosts: {hosts}")

        # Verify distribution across hosts if we have multiple hosts
        if len(ray.nodes()) > 1:
            self.assertGreater(len(hosts), 1, "Tasks should be distributed across multiple hosts")

        # Record performance
        tasks_per_second = num_tasks / submit_time
        self.record_performance("tasks_per_second", tasks_per_second)

    def test_task_locality(self):
        """Test data locality in task distribution."""
        # Create data objects in Ray object store
        data_size_mb = 10  # Size in MB
        data_arrays = []

        # Create several large arrays
        for i in range(4):
            data = np.random.random((data_size_mb * 250000,)).astype(np.float32)  # ~10 MB
            data_arrays.append(ray.put(data))

        @ray.remote
        def process_local_data(data, task_id):
            import socket
            import time

            # Force loading data by calculating sum
            validation = float(np.sum(data))

            return {
                "task_id": task_id,
                "hostname": socket.gethostname(),
                "data_validation": validation,
                "time": time.time()
            }

        # Process each data object
        with self.measure_time("Processing data with locality"):
            results = ray.get([process_local_data.remote(data, i) for i, data in enumerate(data_arrays)])

        # Verify successful processing
        self.assertEqual(len(results), len(data_arrays), "All tasks should complete successfully")

    def test_task_dependencies(self):
        """Test tasks with dependencies between them."""
        @ray.remote
        def stage1(task_id):
            time.sleep(0.1)  # Simulate work
            return {"task_id": task_id, "stage": 1, "result": task_id * 10}

        @ray.remote
        def stage2(data):
            time.sleep(0.1)  # Simulate work
            return {"task_id": data["task_id"], "stage": 2, "result": data["result"] + 5}

        @ray.remote
        def stage3(data):
            time.sleep(0.1)  # Simulate work
            return {"task_id": data["task_id"], "stage": 3, "result": data["result"] * 2}

        # Create multi-stage pipeline
        num_pipelines = 20

        with self.measure_time(f"Running {num_pipelines} pipelines"):
            # Stage 1
            stage1_results = [stage1.remote(i) for i in range(num_pipelines)]

            # Stage 2
            stage2_results = [stage2.remote(result) for result in stage1_results]

            # Stage 3
            stage3_results = [stage3.remote(result) for result in stage2_results]

            # Get final results
            final_results = ray.get(stage3_results)

        # Verify results follow expected pattern
        for i, result in enumerate(final_results):
            expected = (i * 10 + 5) * 2
            self.assertEqual(result["result"], expected, f"Result for task {i} should be {expected}")

    def test_concurrent_tasks_scaling(self):
        """Test scaling with increasing concurrency levels."""
        @ray.remote
        def cpu_task(complexity):
            # Simulate CPU-bound work
            start = time.time()
            result = 0
            for i in range(complexity * 100000):
                result += i ** 2
            elapsed = time.time() - start
            return {"result": result, "time": elapsed}

        concurrency_levels = [1, 10, 20, 50, 100]
        scaling_results = {}

        # Test increasing concurrency
        for concurrency in concurrency_levels:
            with self.measure_time(f"Concurrency level {concurrency}") as elapsed:
                results = ray.get([cpu_task.remote(1) for _ in range(concurrency)])

            scaling_results[concurrency] = {
                "time": elapsed,
                "tasks_per_second": concurrency / elapsed
            }

        # Calculate scaling efficiency
        base_tps = scaling_results[1]["tasks_per_second"]
        for concurrency, data in scaling_results.items():
            if concurrency > 1:
                # Ideal scaling would be linear (base_tps * concurrency)
                ideal_tps = base_tps * concurrency
                scaling_efficiency = data["tasks_per_second"] / ideal_tps
                scaling_results[concurrency]["scaling_efficiency"] = scaling_efficiency

                # We expect reasonable scaling, though not perfect
                logger.info(f"Concurrency {concurrency}: efficiency {scaling_efficiency:.2f}")

        # Record detailed performance metrics
        self.test_results["performance"]["scaling_results"] = scaling_results
