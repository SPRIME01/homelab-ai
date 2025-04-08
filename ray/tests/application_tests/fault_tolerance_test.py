"""
Test cases for Ray fault tolerance.
"""

import time
import ray
import numpy as np
from typing import List, Dict, Any

from base_test import RayBaseTest, logger

class FaultToleranceTest(RayBaseTest):
    """Test suite for Ray fault tolerance."""

    application_name = "fault_tolerance"

    def test_task_failure_recovery(self):
        """Test recovery from task failures."""
        @ray.remote(max_retries=3)
        def flaky_task(fail_count, task_id):
            import time

            if ray.get_runtime_context().get_task_retry_count() < fail_count:
                # Simulate failure
                logger.info(f"Task {task_id} failing (retry {ray.get_runtime_context().get_task_retry_count() + 1}/{fail_count})")
                raise ValueError("Simulated task failure")

            # Task succeeds after fail_count retries
            return {
                "task_id": task_id,
                "retry_count": ray.get_runtime_context().get_task_retry_count(),
                "timestamp": time.time()
            }

        # Run tasks with different failure counts
        results = []
        num_tasks = 5

        with self.measure_time(f"Running {num_tasks} flaky tasks"):
            # Submit tasks that will fail 0, 1, 2 times
            tasks = [flaky_task.remote(i % 3, i) for i in range(num_tasks)]
            results = ray.get(tasks)

        # Verify all tasks eventually succeeded
        self.assertEqual(len(results), num_tasks, "All tasks should eventually succeed")

        # Check that retry_count matches expected failures
        for i, result in enumerate(results):
            expected_retries = i % 3
            self.assertEqual(
                result["retry_count"],
                expected_retries,
                f"Task {i} should have {expected_retries} retries"
            )

    def test_actor_failure_recovery(self):
        """Test recovery from actor failures."""
        # Define a stateful actor that can simulate failures
        @ray.remote(max_restarts=3)
        class FlakyActor:
            def __init__(self):
                self.state = 0
                self.call_count = 0

            def increment(self, amount=1):
                self.call_count += 1
                self.state += amount
                return self.state

            def get_state(self):
                return {
                    "state": self.state,
                    "call_count": self.call_count,
                    "restart_count": ray.get_runtime_context().get_actor_restart_count()
                }

            def crash(self):
                raise ValueError("Simulated actor failure")

        # Create actor
        actor = FlakyActor.remote()

        # Initial operations
        state1 = ray.get(actor.increment.remote(5))
        self.assertEqual(state1, 5, "Initial increment should work")

        # Crash the actor
        with self.assertRaises(ray.exceptions.RayActorError):
            ray.get(actor.crash.remote())

        # Allow time for restart
        time.sleep(1)

        # Actor should restart with clean state
        state2 = ray.get(actor.increment.remote(7))
        self.assertEqual(state2, 7, "Actor should restart with fresh state")

        restart_info = ray.get(actor.get_state.remote())
        logger.info(f"Actor restart info: {restart_info}")

        # Verify it tracked a restart
        self.assertEqual(restart_info["restart_count"], 1, "Actor should record one restart")

    def test_distributed_state_recovery(self):
        """Test recovery of distributed state."""
        # Create a simple key-value store
        @ray.remote
        class KeyValueStore:
            def __init__(self):
                self.data = {}
                self.backup_stores = []

            def set(self, key, value):
                self.data[key] = value
                # Replicate to backups
                for backup in self.backup_stores:
                    ray.get(backup._replicate.remote(key, value))
                return True

            def get(self, key):
                return self.data.get(key, None)

            def add_backup(self, backup):
                self.backup_stores.append(backup)

            # Method only called by primary
            def _replicate(self, key, value):
                self.data[key] = value
                return True

        # Create primary and backup
        primary = KeyValueStore.remote()
        backup = KeyValueStore.remote()

        # Link them
        ray.get(primary.add_backup.remote(backup))

        # Add some data
        ray.get(primary.set.remote("key1", "value1"))
        ray.get(primary.set.remote("key2", "value2"))

        # Verify primary works
        primary_val = ray.get(primary.get.remote("key1"))
        self.assertEqual(primary_val, "value1", "Primary should have the value")

        # Verify backup works
        backup_val = ray.get(backup.get.remote("key1"))
        self.assertEqual(backup_val, "value1", "Backup should have the value")

        # Simulate primary failure (just stop using it)
        logger.info("Simulating primary failure, switching to backup")

        # Create new value on backup (now acting as primary)
        ray.get(backup.set.remote("key3", "value3"))

        # Verify new value exists
        new_val = ray.get(backup.get.remote("key3"))
        self.assertEqual(new_val, "value3", "Backup should accept new values after primary failure")

    def test_load_balancing_with_failures(self):
        """Test load balancing with simulated failures."""
        if not self.config["fault_tolerance"]["worker_failure_simulation"]:
            self.skipTest("Worker failure simulation disabled in config")

        # Skip if we have less than 2 nodes
        if len([n for n in ray.nodes() if n.get("alive", False)]) < 2:
            self.skipTest("Need at least 2 nodes for load balancing test")

        @ray.remote
        def node_info_task():
            import socket
            import time

            # Simulate work
            time.sleep(0.1)

            return {
                "hostname": socket.gethostname(),
                "timestamp": time.time(),
                "pid": ray.get_runtime_context().get_node_id()
            }

        # First, get baseline distribution
        num_tasks = 50
        baseline_results = ray.get([node_info_task.remote() for _ in range(num_tasks)])

        # Count tasks per node
        baseline_distribution = {}
        for result in baseline_results:
            hostname = result["hostname"]
            if hostname not in baseline_distribution:
                baseline_distribution[hostname] = 0
            baseline_distribution[hostname] += 1

        logger.info(f"Baseline distribution: {baseline_distribution}")

        # Simulate a node failure
        self.simulate_node_failure(recover=True)

        # After "recovery", run another batch of tasks
        recovery_results = ray.get([node_info_task.remote() for _ in range(num_tasks)])

        # Count tasks per node after recovery
        recovery_distribution = {}
        for result in recovery_results:
            hostname = result["hostname"]
            if hostname not in recovery_distribution:
                recovery_distribution[hostname] = 0
            recovery_distribution[hostname] += 1

        logger.info(f"Recovery distribution: {recovery_distribution}")

        # Record results
        self.test_results["performance"]["load_balancing"] = {
            "baseline": baseline_distribution,
            "after_failure": recovery_distribution
        }
