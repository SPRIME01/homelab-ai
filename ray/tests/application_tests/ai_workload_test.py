"""
Test cases for AI workloads on Ray.
"""

import time
import ray
import numpy as np
from typing import List, Dict, Any

from base_test import RayBaseTest, logger

class AIWorkloadTest(RayBaseTest):
    """Test suite for distributed AI workloads."""

    application_name = "ai_workload"

    def test_distributed_data_processing(self):
        """Test distributed data preprocessing for AI workloads."""
        # Define data preprocessing task
        @ray.remote
        def preprocess_data_chunk(chunk_id, size):
            import numpy as np

            # Create synthetic data
            X = np.random.randn(size, 50)  # Features
            y = np.random.randint(0, 2, size)  # Labels

            # Simulate preprocessing
            # Standardize features
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            # One-hot encode labels
            y_onehot = np.zeros((size, 2))
            y_onehot[np.arange(size), y] = 1

            return {
                "chunk_id": chunk_id,
                "X_shape": X.shape,
                "y_shape": y_onehot.shape,
                "X_mean": float(np.mean(X)),
                "X_std": float(np.std(X)),
                "X_sample": X[:5].tolist() if size > 0 else []
            }

        # Create multiple chunks
        num_chunks = 10
        chunk_size = 10000

        with self.measure_time(f"Processing {num_chunks} data chunks"):
            results = ray.get([preprocess_data_chunk.remote(i, chunk_size)
                              for i in range(num_chunks)])

        # Verify all chunks were processed
        self.assertEqual(len(results), num_chunks, "All chunks should be processed")

        # Calculate throughput (samples/second)
        total_samples = num_chunks * chunk_size
        throughput = total_samples / (time.time() - self.start_time)

        logger.info(f"Data processing throughput: {throughput:.2f} samples/second")

        # Record performance metrics
        self.test_results["performance"]["data_processing"] = {
            "total_samples": total_samples,
            "throughput_samples_per_second": throughput
        }

    def test_distributed_training(self):
        """Test simulated distributed model training."""
        # Skip if scikit-learn is not available
        try:
            import sklearn
        except ImportError:
            self.skipTest("scikit-learn not available")

        # Define training task
        @ray.remote
        def train_model_shard(shard_id, size, complexity):
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            # Create synthetic data
            X = np.random.randn(size, 20)  # Features
            y = np.random.randint(0, 2, size)  # Binary labels

            # Create and train model
            start_time = time.time()
            model = RandomForestClassifier(n_estimators=complexity)
            model.fit(X, y)
            training_time = time.time() - start_time

            # Evaluate model
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)

            return {
                "shard_id": shard_id,
                "training_samples": size,
                "training_time_seconds": training_time,
                "accuracy": float(accuracy),
                "feature_importances": model.feature_importances_.tolist()
            }

        # Test with different configurations
        configs = [
            {"shards": 1, "samples_per_shard": 5000, "complexity": 10},
            {"shards": 4, "samples_per_shard": 5000, "complexity": 10},
            {"shards": 4, "samples_per_shard": 10000, "complexity": 20}
        ]

        training_results = []

        for config in configs:
            logger.info(f"Testing training with config: {config}")

            shards = config["shards"]
            samples = config["samples_per_shard"]
            complexity = config["complexity"]

            with self.measure_time(f"Training {shards} models with {samples} samples each") as elapsed:
                shard_results = ray.get([
                    train_model_shard.remote(i, samples, complexity)
                    for i in range(shards)
                ])

            # Calculate metrics
            total_samples = shards * samples
            throughput = total_samples / elapsed
            avg_accuracy = sum(r["accuracy"] for r in shard_results) / len(shard_results)

            training_results.append({
                "config": config,
                "throughput_samples_per_second": throughput,
                "elapsed_seconds": elapsed,
                "avg_accuracy": avg_accuracy
            })

            logger.info(f"Training throughput: {throughput:.2f} samples/second, accuracy: {avg_accuracy:.4f}")

        # Record performance metrics
        self.test_results["performance"]["distributed_training"] = training_results

    def test_inference_scaling(self):
        """Test inference scaling with batch size and concurrency."""
        # Define inference task
        @ray.remote
        def inference_task(batch_size, model_complexity):
            import numpy as np
            import time

            # Create synthetic input data
            inputs = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)

            # Simulate inference with increasing complexity
            start_time = time.time()

            # Simulated inference time roughly scales with batch_size * complexity
            compute_budget = batch_size * model_complexity * 0.0001

            # Simulate compute by waiting and doing some calculations
            result = 0
            deadline = start_time + compute_budget
            while time.time() < deadline:
                # Do some actual computation
                result += np.sum(np.random.randn(1000, 10) ** 2)

            elapsed = time.time() - start_time

            return {
                "batch_size": batch_size,
                "compute_budget": compute_budget,
                "actual_time": elapsed,
                "samples_per_second": batch_size / elapsed
            }

        # Test different batch sizes
        batch_sizes = [1, 4, 16, 64]
        model_complexity = 10  # Arbitrary complexity factor

        batch_results = []
        for batch_size in batch_sizes:
            with self.measure_time(f"Inference with batch size {batch_size}") as elapsed:
                result = ray.get(inference_task.remote(batch_size, model_complexity))

            batch_results.append(result)
            logger.info(f"Batch size {batch_size}: {result['samples_per_second']:.2f} samples/second")

        # Test different concurrency levels with fixed batch size
        batch_size = 16
        concurrency_levels = [1, 2, 4, 8]

        concurrency_results = []
        for concurrency in concurrency_levels:
            with self.measure_time(f"Inference with concurrency {concurrency}") as elapsed:
                results = ray.get([inference_task.remote(batch_size, model_complexity)
                                 for _ in range(concurrency)])

            total_samples = batch_size * concurrency
            throughput = total_samples / elapsed

            concurrency_results.append({
                "concurrency": concurrency,
                "total_samples": total_samples,
                "elapsed_seconds": elapsed,
                "throughput_samples_per_second": throughput
            })

            logger.info(f"Concurrency {concurrency}: {throughput:.2f} samples/second")

        # Record performance metrics
        self.test_results["performance"]["inference_scaling"] = {
            "batch_results": batch_results,
            "concurrency_results": concurrency_results
        }

    def test_distributed_hyperparameter_tuning(self):
        """Test simulated distributed hyperparameter tuning."""
        # Define task for training with specific hyperparameters
        @ray.remote
        def train_with_hyperparams(hp_config):
            import time
            import numpy as np

            # Create synthetic data
            X = np.random.randn(1000, 20)
            y = np.random.randint(0, 2, 1000)

            # Simulate training time based on complexity
            complexity = hp_config.get("n_estimators", 10) * hp_config.get("max_depth", 5)
            training_time = 0.005 * complexity

            # Simulate accuracy based on hyperparams - just for testing
            base_accuracy = 0.7
            depth_factor = min(hp_config.get("max_depth", 5) / 10, 0.15)  # More depth helps up to a point
            estimator_factor = min(hp_config.get("n_estimators", 10) / 100, 0.1)  # More estimators help up to a point
            random_factor = np.random.random() * 0.05  # Small random variation

            accuracy = base_accuracy + depth_factor + estimator_factor + random_factor

            # Simulate training
            time.sleep(training_time)

            return {
                "config": hp_config,
                "training_time": training_time,
                "accuracy": float(accuracy),
                "complexity": complexity
            }

        # Define hyperparameter search space
        param_grid = {
            "n_estimators": [10, 50, 100],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        }

        # Generate all combinations
        hp_configs = []
        for n_est in param_grid["n_estimators"]:
            for max_depth in param_grid["max_depth"]:
                for min_split in param_grid["min_samples_split"]:
                    hp_configs.append({
                        "n_estimators": n_est,
                        "max_depth": max_depth,
                        "min_samples_split": min_split
                    })

        logger.info(f"Testing {len(hp_configs)} hyperparameter configurations")

        # Run hyperparameter search
        with self.measure_time(f"Hyperparameter search with {len(hp_configs)} configs") as elapsed:
            results = ray.get([train_with_hyperparams.remote(config) for config in hp_configs])

        # Find best result
        best_result = max(results, key=lambda x: x["accuracy"])

        logger.info(f"Best hyperparameters: {best_result['config']} with accuracy: {best_result['accuracy']:.4f}")

        # Calculate throughput
        configs_per_second = len(hp_configs) / elapsed

        # Record performance metrics
        self.test_results["performance"]["hyperparameter_tuning"] = {
            "configs_tested": len(hp_configs),
            "configs_per_second": configs_per_second,
            "best_config": best_result["config"],
            "best_accuracy": best_result["accuracy"]
        }
