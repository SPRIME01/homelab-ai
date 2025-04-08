"""
Utility functions for Triton model testing.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import tritonclient.http
import tritonclient.grpc
from concurrent.futures import ThreadPoolExecutor

def generate_test_data(
    dtype: str,
    shape: List[int],
    distribution: str = "uniform"
) -> np.ndarray:
    """Generate random test data for model input.

    Args:
        dtype: Triton data type string
        shape: Shape of the array
        distribution: Distribution to use ('uniform', 'normal', or 'zeros')

    Returns:
        Numpy array with random data
    """
    # Convert Triton dtype to numpy dtype
    dtype_map = {
        "BOOL": np.bool_,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "BYTES": np.object_
    }

    np_dtype = dtype_map.get(dtype, np.float32)

    # Generate data based on distribution
    if dtype == "BYTES":
        if distribution == "zeros":
            return np.array([""] * np.prod(shape[:-1])).reshape(shape[:-1])
        else:
            # Generate random strings
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "

            def random_string(length=10):
                return ''.join(np.random.choice(list(alphabet), length))

            return np.array([random_string() for _ in range(np.prod(shape[:-1]))]).reshape(shape[:-1])

    elif np_dtype == np.bool_:
        return np.random.choice([True, False], size=shape).astype(np_dtype)

    else:
        if distribution == "uniform":
            if np.issubdtype(np_dtype, np.integer):
                low, high = -100, 100
                return np.random.randint(low, high, size=shape).astype(np_dtype)
            else:
                return np.random.uniform(-1.0, 1.0, size=shape).astype(np_dtype)

        elif distribution == "normal":
            return np.random.normal(0.0, 1.0, size=shape).astype(np_dtype)

        elif distribution == "zeros":
            return np.zeros(shape, dtype=np_dtype)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

def measure_throughput(
    client: Union[tritonclient.http.InferenceServerClient, tritonclient.grpc.InferenceServerClient],
    model_name: str,
    input_data: Dict[str, np.ndarray],
    batch_sizes: List[int] = [1, 2, 4, 8],
    concurrency: List[int] = [1, 2, 4, 8],
    iterations_per_concurrency: int = 10,
    model_version: str = ""
) -> Dict[str, Any]:
    """Measure throughput across different batch sizes and concurrency levels.

    Args:
        client: Triton client (HTTP or gRPC)
        model_name: Name of the model to test
        input_data: Dictionary of input name to numpy array (single sample)
        batch_sizes: List of batch sizes to test
        concurrency: List of concurrency levels to test
        iterations_per_concurrency: Number of iterations per concurrency level
        model_version: Model version (empty for latest)

    Returns:
        Dictionary with throughput results
    """
    results = {
        "batch_results": {},
        "concurrency_results": {}
    }

    # Determine client type
    is_grpc = isinstance(client, tritonclient.grpc.InferenceServerClient)

    # Helper function for single inference
    def run_inference(client, model_name, model_version, inputs):
        start_time = time.time()
        client.infer(model_name=model_name, model_version=model_version, inputs=inputs)
        return time.time() - start_time

    # Test different batch sizes
    for batch_size in batch_sizes:
        # Create batched inputs
        batched_inputs = []
        for input_name, array in input_data.items():
            # Determine batch dimension
            orig_shape = array.shape
            if len(orig_shape) > 0:
                batched_shape = (batch_size,) + orig_shape[1:]
                batched_array = np.repeat(array, batch_size, axis=0)
                batched_array = batched_array.reshape(batched_shape)
            else:
                batched_array = np.repeat(array, batch_size)

            # Create infer input object
            if is_grpc:
                inp = tritonclient.grpc.InferInput(input_name, batched_array.shape,
                                                  _get_dtype_string(batched_array.dtype))
                inp.set_data_from_numpy(batched_array)
            else:
                inp = tritonclient.http.InferInput(input_name, batched_array.shape,
                                                 _get_dtype_string(batched_array.dtype))
                inp.set_data_from_numpy(batched_array)

            batched_inputs.append(inp)

        try:
            # Run multiple iterations and compute average latency
            latencies = []
            for _ in range(5):  # Use fewer iterations for batch testing
                latency = run_inference(client, model_name, model_version, batched_inputs)
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies)
            throughput = batch_size / avg_latency

            results["batch_results"][str(batch_size)] = {
                "avg_latency_seconds": avg_latency,
                "throughput_samples_per_second": throughput
            }

        except Exception as e:
            results["batch_results"][str(batch_size)] = {
                "error": str(e)
            }

    # Test different concurrency levels
    for concurrency_level in concurrency:
        # Set up infer inputs (single sample for concurrency test)
        inputs = []
        for input_name, array in input_data.items():
            if is_grpc:
                inp = tritonclient.grpc.InferInput(input_name, array.shape,
                                                  _get_dtype_string(array.dtype))
                inp.set_data_from_numpy(array)
            else:
                inp = tritonclient.http.InferInput(input_name, array.shape,
                                                 _get_dtype_string(array.dtype))
                inp.set_data_from_numpy(array)

            inputs.append(inp)

        try:
            with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
                futures = []
                for _ in range(iterations_per_concurrency):
                    futures.append(executor.submit(
                        run_inference, client, model_name, model_version, inputs
                    ))

                # Get latencies from all futures
                latencies = [future.result() for future in futures]

            # Compute metrics
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = concurrency_level / avg_latency

            results["concurrency_results"][str(concurrency_level)] = {
                "avg_latency_seconds": avg_latency,
                "p95_latency_seconds": p95_latency,
                "p99_latency_seconds": p99_latency,
                "throughput_requests_per_second": throughput
            }

        except Exception as e:
            results["concurrency_results"][str(concurrency_level)] = {
                "error": str(e)
            }

    return results

def _get_dtype_string(dtype) -> str:
    """Convert numpy dtype to Triton dtype string."""
    dtype_map = {
        np.bool_: "BOOL",
        np.uint8: "UINT8",
        np.uint16: "UINT16",
        np.uint32: "UINT32",
        np.uint64: "UINT64",
        np.int8: "INT8",
        np.int16: "INT16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.float16: "FP16",
        np.float32: "FP32",
        np.float64: "FP64"
    }

    if dtype.type in dtype_map:
        return dtype_map[dtype.type]

    # Handle string types
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_):
        return "BYTES"

    # Default to FP32
    return "FP32"

def plot_performance_results(
    results: Dict[str, Any],
    output_dir: str,
    model_name: str
) -> None:
    """Plot performance results.

    Args:
        results: Performance results dictionary
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot batch size vs throughput
    if "batch_results" in results:
        batch_data = results["batch_results"]
        batch_sizes = []
        throughputs = []
        latencies = []

        for batch_size, metrics in batch_data.items():
            if "error" not in metrics:
                batch_sizes.append(int(batch_size))
                throughputs.append(metrics["throughput_samples_per_second"])
                latencies.append(metrics["avg_latency_seconds"] * 1000)  # Convert to ms

        if batch_sizes:
            # Throughput plot
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, throughputs, 'o-', linewidth=2)
            plt.title(f'Batch Size vs Throughput for {model_name}')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{model_name}_batch_throughput.png"))
            plt.close()

            # Latency plot
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, latencies, 'o-', linewidth=2)
            plt.title(f'Batch Size vs Latency for {model_name}')
            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{model_name}_batch_latency.png"))
            plt.close()

    # Plot concurrency vs throughput
    if "concurrency_results" in results:
        concurrency_data = results["concurrency_results"]
        concurrency_levels = []
        throughputs = []
        latencies = []
        p95_latencies = []

        for concurrency, metrics in concurrency_data.items():
            if "error" not in metrics:
                concurrency_levels.append(int(concurrency))
                throughputs.append(metrics["throughput_requests_per_second"])
                latencies.append(metrics["avg_latency_seconds"] * 1000)  # Convert to ms
                p95_latencies.append(metrics["p95_latency_seconds"] * 1000)  # Convert to ms

        if concurrency_levels:
            # Throughput plot
            plt.figure(figsize=(10, 6))
            plt.plot(concurrency_levels, throughputs, 'o-', linewidth=2)
            plt.title(f'Concurrency vs Throughput for {model_name}')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Throughput (requests/second)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{model_name}_concurrency_throughput.png"))
            plt.close()

            # Latency plot
            plt.figure(figsize=(10, 6))
            plt.plot(concurrency_levels, latencies, 'o-', label='Avg Latency', linewidth=2)
            plt.plot(concurrency_levels, p95_latencies, 'o--', label='P95 Latency', linewidth=2)
            plt.title(f'Concurrency vs Latency for {model_name}')
            plt.xlabel('Concurrency Level')
            plt.ylabel('Latency (ms)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{model_name}_concurrency_latency.png"))
            plt.close()
