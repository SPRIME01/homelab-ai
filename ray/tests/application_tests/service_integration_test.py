"""
Test cases for Ray integration with external services.
"""

import time
import ray
import numpy as np
from typing import List, Dict, Any
import tempfile
import requests
import os
import subprocess
import socket
import json

from base_test import RayBaseTest, logger

class ServiceIntegrationTest(RayBaseTest):
    """Test suite for Ray integration with external services."""

    application_name = "service_integration"

    def setUp(self):
        """Set up for each test."""
        super().setUp()
        # Skip integration tests if explicitly disabled
        if os.environ.get("SKIP_INTEGRATION_TESTS", "").lower() == "true":
            self.skipTest("Integration tests disabled by environment variable")

    def test_triton_inference_server_integration(self):
        """Test integration with Triton Inference Server if available."""
        # Check if Triton server is available
        triton_url = os.environ.get("TRITON_URL", "localhost:8000")

        # Try to import Triton client
        try:
            import tritonclient.http
            triton_client = tritonclient.http.InferenceServerClient(
                url=triton_url, verbose=False
            )
            server_live = triton_client.is_server_live()
        except (ImportError, Exception) as e:
            logger.warning(f"Skipping Triton test: {e}")
            self.skipTest(f"Triton client not available or server not accessible: {e}")

        logger.info(f"Triton server is {'live' if server_live else 'not live'}")
        if not server_live:
            self.skipTest("Triton server is not live")

        # Get list of models
        try:
            models = triton_client.get_model_repository_index()
            logger.info(f"Found {len(models)} models in Triton: {[m['name'] for m in models]}")
        except Exception as e:
            logger.warning(f"Could not get model list: {e}")
            models = []

        if not models:
            self.skipTest("No models available on Triton server")

        # Create Ray tasks that call Triton
        @ray.remote
        def triton_inference_task(model_name, input_data):
            import tritonclient.http
            import numpy as np

            client = tritonclient.http.InferenceServerClient(url=triton_url, verbose=False)

            # Get model metadata
            metadata = client.get_model_metadata(model_name)

            # Create a simple input based on metadata
            input_tensor = tritonclient.http.InferInput(
                metadata["inputs"][0]["name"],
                input_data.shape,
                metadata["inputs"][0]["datatype"]
            )
            input_tensor.set_data_from_numpy(input_data)

            # Run inference
            start_time = time.time()
            response = client.infer(model_name, [input_tensor])
            elapsed = time.time() - start_time

            # Get output
            result = response.as_numpy(metadata["outputs"][0]["name"])

            return {
                "model": model_name,
                "output_shape": result.shape,
                "latency_ms": elapsed * 1000,
                "output_sample": result.flatten()[:5].tolist() if result.size > 0 else []
            }

        # Choose a model to test
        test_model = models[0]["name"]

        # Prepare some random input data (generic float32 tensor)
        input_data = np.random.randn(1, 16).astype(np.float32)

        # Run inference from multiple Ray workers
        num_workers = 5

        with self.measure_time(f"Triton inference across {num_workers} workers") as elapsed:
            results = ray.get([triton_inference_task.remote(test_model, input_data)
                              for _ in range(num_workers)])

        # Check that all workers got results
        self.assertEqual(len(results), num_workers, "All workers should get inference results")

        # Calculate average latency
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        logger.info(f"Average Triton inference latency: {avg_latency:.2f}ms")

        # Record performance
        self.test_results["performance"]["triton_integration"] = {
            "model": test_model,
            "workers": num_workers,
            "total_time_ms": elapsed * 1000,
            "avg_latency_ms": avg_latency,
            "throughput_infer_per_sec": num_workers / elapsed
        }

    def test_filesystem_integration(self):
        """Test integration with shared filesystem."""
        # Create temp directory for test
        temp_dir = tempfile.mkdtemp(prefix="ray_fs_test_")
        logger.info(f"Using temp directory: {temp_dir}")

        # Tasks to write files
        @ray.remote
        def write_file_task(path, data):
            with open(path, 'w') as f:
                f.write(data)
            return {"path": path, "size": len(data)}

        # Tasks to read files
        @ray.remote
        def read_file_task(path):
            if not os.path.exists(path):
                return {"path": path, "exists": False}

            with open(path, 'r') as f:
                content = f.read()
            return {"path": path, "exists": True, "size": len(content), "content": content}

        # Create some files with worker tasks
        num_files = 10
        files = []

        with self.measure_time(f"Writing {num_files} files"):
            for i in range(num_files):
                path = os.path.join(temp_dir, f"file_{i}.txt")
                content = f"File {i} content: {np.random.randn(10)}"
                files.append(path)
                ray.get(write_file_task.remote(path, content))

        # Try reading the files from different tasks
        with self.measure_time(f"Reading {num_files} files"):
            results = ray.get([read_file_task.remote(path) for path in files])

        # Verify all files were read correctly
        for result in results:
            self.assertTrue(result["exists"], f"File {result['path']} should exist")
            self.assertTrue(result["size"] > 0, f"File {result['path']} should have content")

        # Clean up
        for path in files:
            os.remove(path)
        os.rmdir(temp_dir)

    def test_remote_data_access(self):
        """Test accessing remote data sources."""
        # Use a public API for testing
        test_api_url = "https://jsonplaceholder.typicode.com/posts"

        # Define task that fetches remote data
        @ray.remote
        def fetch_data_task(url):
            import requests
            import time

            start_time = time.time()
            response = requests.get(url)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                return {
                    "url": url,
                    "status": response.status_code,
                    "data_size": len(response.content),
                    "latency_ms": elapsed * 1000,
                    "items": len(response.json())
                }
            else:
                return {
                    "url": url,
                    "status": response.status_code,
                    "error": response.text,
                    "latency_ms": elapsed * 1000
                }

        # Test multiple concurrent requests
        num_requests = 10

        with self.measure_time(f"Fetching remote data with {num_requests} concurrent requests"):
            results = ray.get([fetch_data_task.remote(test_api_url) for _ in range(num_requests)])

        # Verify results
        successful = [r for r in results if r["status"] == 200]
        self.assertEqual(len(successful), num_requests, "All requests should succeed")

        # Calculate average latency
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        logger.info(f"Average request latency: {avg_latency:.2f}ms")

        # Record performance metrics
        self.test_results["performance"]["remote_data_access"] = {
            "avg_latency_ms": avg_latency,
            "concurrent_requests": num_requests,
            "success_rate": len(successful) / num_requests
        }

    def test_database_integration(self):
        """Test integration with databases using SQLite as proxy."""
        import sqlite3

        # Create in-memory database for testing
        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        conn = sqlite3.connect(db_path)

        # Create test table
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value REAL
        )
        """)
        conn.commit()
        conn.close()

        # Ray task to insert data
        @ray.remote
        def db_insert_task(db_path, items):
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            start_time = time.time()
            for item in items:
                cursor.execute(
                    "INSERT INTO test_data (name, value) VALUES (?, ?)",
                    (item["name"], item["value"])
                )

            conn.commit()
            rows = cursor.rowcount
            elapsed = time.time() - start_time
            conn.close()

            return {
                "inserted": rows,
                "latency_ms": elapsed * 1000
            }

        # Ray task to query data
        @ray.remote
        def db_query_task(db_path, limit=None):
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM test_data"
            if limit:
                query += f" LIMIT {limit}"

            start_time = time.time()
            cursor.execute(query)
            rows = cursor.fetchall()
            elapsed = time.time() - start_time

            conn.close()

            return {
                "rows": len(rows),
                "latency_ms": elapsed * 1000,
                "sample": rows[:5] if rows else []
            }

        # Generate test data
        test_data = [{"name": f"item_{i}", "value": np.random.random()}
                     for i in range(100)]

        # Split data for concurrent inserts
        batch_size = 20
        batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]

        # Insert data using multiple tasks
        with self.measure_time("Concurrent database inserts"):
            insert_results = ray.get([db_insert_task.remote(db_path, batch) for batch in batches])

        # Verify inserts
        total_inserted = sum(r["inserted"] for r in insert_results)
        self.assertEqual(total_inserted, len(test_data), "All items should be inserted")

        # Run concurrent queries
        num_queries = 5
        with self.measure_time("Concurrent database queries"):
            query_results = ray.get([db_query_task.remote(db_path) for _ in range(num_queries)])

        # Verify queries
        for result in query_results:
            self.assertEqual(result["rows"], len(test_data), "Query should return all items")

        # Record performance metrics
        self.test_results["performance"]["database_integration"] = {
            "insert_tasks": len(batches),
            "total_records": len(test_data),
            "avg_insert_latency_ms": sum(r["latency_ms"] for r in insert_results) / len(insert_results),
            "avg_query_latency_ms": sum(r["latency_ms"] for r in query_results) / len(query_results)
        }

        # Clean up
        os.remove(db_path)
        os.rmdir(os.path.dirname(db_path))
