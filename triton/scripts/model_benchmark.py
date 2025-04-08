import os
import sys
import logging
import numpy as np
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from config import config, logger
from utils import (
    verify_dependencies, get_file_hash, detect_model_type,
    get_model_metadata, create_model_info_file
)
from download_manager import DownloadManager

class ModelBenchmark:
    """
    Benchmark AI model performance on NVIDIA Jetson AGX Orin.
    Supports both direct model benchmarking and Triton Inference Server benchmarking.
    """

    def __init__(self):
        self.download_manager = DownloadManager()
        verify_dependencies()

        # Check for Triton client
        try:
            import tritonclient.grpc
            self.has_triton_client = True
        except ImportError:
            logger.warning("Triton client not found. Triton server benchmarking will not be available.")
            logger.info("Install with: pip install tritonclient[all]")
            self.has_triton_client = False

    def benchmark(
        self,
        model_source: str,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        batch_sizes: List[int] = [1, 2, 4, 8],
        duration_seconds: int = 10,
        concurrency: List[int] = [1, 2, 4, 8],
        triton_url: Optional[str] = None,
        warmup: int = 3,
        save_results: bool = True,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark a model's performance

        Args:
            model_source: Path or identifier for the model
            output_dir: Directory to save benchmark results and plots
            model_type: Type of model (language, vision, speech)
            batch_sizes: List of batch sizes to test
            duration_seconds: Duration of each benchmark test in seconds
            concurrency: List of concurrent request counts to test
            triton_url: URL of Triton server (if None, direct model inference is used)
            warmup: Number of warmup iterations before benchmarking
            save_results: Whether to save results to a file
            generate_plots: Whether to generate performance plots

        Returns:
            Dictionary of benchmark results
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(config.output_dir, "benchmarks")

        os.makedirs(output_dir, exist_ok=True)

        # Download or locate the model if not using Triton
        model_path = None
        if triton_url is None:
            model_path = self.download_manager.get_model(model_source)
        else:
            # Just use the model name for Triton benchmarks
            model_path = model_source if not os.path.exists(model_source) else os.path.basename(model_source)

            if not self.has_triton_client:
                logger.error("Triton client not available. Cannot benchmark against Triton server.")
                raise ImportError("Triton client not available. Install with: pip install tritonclient[all]")

        # Detect model type if not provided
        if model_type is None:
            model_type = detect_model_type(os.path.basename(model_path))

        # Create output directories
        benchmark_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(model_path))[0])
        os.makedirs(benchmark_dir, exist_ok=True)

        # Run benchmarks
        logger.info(f"Starting benchmark for {model_path}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"Concurrency: {concurrency}")
        logger.info(f"Duration: {duration_seconds} seconds per test")

        results = {}

        if triton_url is None:
            # Direct model inference benchmark
            results = self._benchmark_direct(
                model_path=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                warmup=warmup
            )
        else:
            # Triton server benchmark
            results = self._benchmark_triton(
                model_name=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                concurrency=concurrency,
                triton_url=triton_url,
                warmup=warmup
            )

        # Save results if requested
        if save_results:
            results_path = os.path.join(benchmark_dir, "benchmark_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved benchmark results to {results_path}")

        # Generate plots if requested
        if generate_plots:
            plots_dir = os.path.join(benchmark_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            self._generate_plots(results, plots_dir)
            logger.info(f"Saved benchmark plots to {plots_dir}")

        return results

    def _benchmark_direct(
        self,
        model_path: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Run direct model inference benchmark"""
        logger.info("Running direct model inference benchmark")

        # Determine model format
        model_format = os.path.splitext(model_path)[1].lower()

        results = {
            "model_path": model_path,
            "model_type": model_type,
            "model_format": model_format,
            "benchmark_type": "direct",
            "batch_sizes": batch_sizes,
            "duration_seconds": duration_seconds,
            "warmup": warmup,
            "metrics": {}
        }

        # Try to load the model based on format
        if model_format in ['.pt', '.pth']:
            # PyTorch model
            results["metrics"] = self._benchmark_pytorch(
                model_path=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                warmup=warmup
            )
        elif model_format == '.onnx':
            # ONNX model
            results["metrics"] = self._benchmark_onnx(
                model_path=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                warmup=warmup
            )
        elif model_format in ['.plan', '.trt']:
            # TensorRT model
            results["metrics"] = self._benchmark_tensorrt(
                model_path=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                warmup=warmup
            )
        elif model_format in ['.h5', '.keras']:
            # TensorFlow/Keras model
            results["metrics"] = self._benchmark_tensorflow(
                model_path=model_path,
                model_type=model_type,
                batch_sizes=batch_sizes,
                duration_seconds=duration_seconds,
                warmup=warmup
            )
        else:
            logger.error(f"Unsupported model format for direct benchmarking: {model_format}")
            raise ValueError(f"Unsupported model format for direct benchmarking: {model_format}")

        return results

    def _benchmark_pytorch(
        self,
        model_path: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark PyTorch model"""
        import torch

        logger.info(f"Benchmarking PyTorch model: {model_path}")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        model.eval()

        metrics = {}

        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Create dummy input
            dummy_input = self._create_dummy_input(model, model_type, batch_size)
            dummy_input = dummy_input.to(device) if isinstance(dummy_input, torch.Tensor) else dummy_input

            # Warmup
            logger.info(f"Warming up for {warmup} iterations")
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(dummy_input)

            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            iterations = 0
            latencies = []

            with torch.no_grad():
                while time.time() - start_time < duration_seconds:
                    iter_start = time.time()
                    _ = model(dummy_input)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    latencies.append((time.time() - iter_start) * 1000)  # ms
                    iterations += 1

            total_time = time.time() - start_time

            # Calculate metrics
            batch_metrics = {
                "iterations": iterations,
                "total_time_seconds": total_time,
                "throughput": iterations * batch_size / total_time,
                "latency_mean_ms": sum(latencies) / len(latencies),
                "latency_median_ms": sorted(latencies)[len(latencies) // 2],
                "latency_p90_ms": sorted(latencies)[int(len(latencies) * 0.9)],
                "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
                "samples_per_second": iterations * batch_size / total_time
            }

            metrics[f"batch_{batch_size}"] = batch_metrics

            logger.info(f"Batch size {batch_size}: {iterations} iterations in {total_time:.2f}s")
            logger.info(f"Throughput: {batch_metrics['throughput']:.2f} inferences/second")
            logger.info(f"Average latency: {batch_metrics['latency_mean_ms']:.2f} ms")

        return metrics

    def _benchmark_onnx(
        self,
        model_path: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark ONNX model"""
        import onnxruntime as ort
        import numpy as np

        logger.info(f"Benchmarking ONNX model: {model_path}")

        # Set up execution provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load model
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        # Get input name and shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_type = session.get_inputs()[0].type

        # Convert ONNX type to numpy type
        numpy_type = np.float32
        if 'float16' in input_type or 'float16' in input_type:
            numpy_type = np.float16
        elif 'int64' in input_type:
            numpy_type = np.int64
        elif 'int32' in input_type:
            numpy_type = np.int32

        metrics = {}

        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Create dummy input
            shape = list(input_shape)
            if len(shape) > 0 and shape[0] in [0, -1, None]:  # Dynamic batch dimension
                shape[0] = batch_size

            # Create random input based on input type
            if 'int' in input_type:
                dummy_input = np.random.randint(0, 100, size=shape).astype(numpy_type)
            else:
                dummy_input = np.random.randn(*shape).astype(numpy_type)

            input_dict = {input_name: dummy_input}

            # Warmup
            logger.info(f"Warming up for {warmup} iterations")
            for _ in range(warmup):
                _ = session.run(None, input_dict)

            # Benchmark
            start_time = time.time()
            iterations = 0
            latencies = []

            while time.time() - start_time < duration_seconds:
                iter_start = time.time()
                _ = session.run(None, input_dict)
                latencies.append((time.time() - iter_start) * 1000)  # ms
                iterations += 1

            total_time = time.time() - start_time

            # Calculate metrics
            batch_metrics = {
                "iterations": iterations,
                "total_time_seconds": total_time,
                "throughput": iterations * batch_size / total_time,
                "latency_mean_ms": sum(latencies) / len(latencies),
                "latency_median_ms": sorted(latencies)[len(latencies) // 2],
                "latency_p90_ms": sorted(latencies)[int(len(latencies) * 0.9)],
                "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
                "samples_per_second": iterations * batch_size / total_time
            }

            metrics[f"batch_{batch_size}"] = batch_metrics

            logger.info(f"Batch size {batch_size}: {iterations} iterations in {total_time:.2f}s")
            logger.info(f"Throughput: {batch_metrics['throughput']:.2f} inferences/second")
            logger.info(f"Average latency: {batch_metrics['latency_mean_ms']:.2f} ms")

        return metrics

    def _benchmark_tensorrt(
        self,
        model_path: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark TensorRT model"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            import numpy as np
        except ImportError:
            logger.error("tensorrt or pycuda not installed. Required for TensorRT benchmarking.")
            raise

        logger.info(f"Benchmarking TensorRT model: {model_path}")

        # Load TensorRT engine
        logger.info("Loading TensorRT engine")
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        metrics = {}

        # Create execution context
        context = engine.create_execution_context()

        # Get input and output names
        input_names = []
        output_names = []
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                input_names.append(engine.get_binding_name(i))
            else:
                output_names.append(engine.get_binding_name(i))

        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            # Skip batch sizes larger than max batch size
            if engine.max_batch_size > 0 and batch_size > engine.max_batch_size:
                logger.warning(f"Skipping batch size {batch_size} (exceeds max batch size {engine.max_batch_size})")
                continue

            logger.info(f"Testing batch size: {batch_size}")

            # Allocate memory for inputs and outputs
            bindings = []
            input_arrays = {}
            output_arrays = {}

            for binding in input_names + output_names:
                idx = engine.get_binding_index(binding)
                if engine.binding_is_input(idx):
                    # Input binding
                    shape = context.get_binding_shape(idx)
                    if shape[0] == -1:  # Dynamic batch dimension
                        shape[0] = batch_size
                        context.set_binding_shape(idx, shape)

                    size = trt.volume(shape) * engine.max_batch_size
                    dtype = trt.nptype(engine.get_binding_dtype(idx))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    input_arrays[binding] = {"host": host_mem, "device": device_mem, "shape": shape}
                else:
                    # Output binding
                    shape = context.get_binding_shape(idx)
                    size = trt.volume(shape) * engine.max_batch_size
                    dtype = trt.nptype(engine.get_binding_dtype(idx))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    output_arrays[binding] = {"host": host_mem, "device": device_mem, "shape": shape}

                bindings.append(int(device_mem))

            # Fill input arrays with random data
            for input_name, input_data in input_arrays.items():
                input_data["host"][:] = np.random.random(input_data["shape"]).astype(input_data["host"].dtype)
                cuda.memcpy_htod(input_data["device"], input_data["host"])

            # Warmup
            logger.info(f"Warming up for {warmup} iterations")
            for _ in range(warmup):
                context.execute_v2(bindings)

            # Benchmark
            start_time = time.time()
            iterations = 0
            latencies = []

            while time.time() - start_time < duration_seconds:
                # Copy inputs to device
                for input_name, input_data in input_arrays.items():
                    cuda.memcpy_htod(input_data["device"], input_data["host"])

                # Execute inference
                iter_start = time.time()
                context.execute_v2(bindings)
                cuda.Context.synchronize()
                latencies.append((time.time() - iter_start) * 1000)  # ms

                # Copy outputs from device
                for output_name, output_data in output_arrays.items():
                    cuda.memcpy_dtoh(output_data["host"], output_data["device"])

                iterations += 1

            total_time = time.time() - start_time

            # Calculate metrics
            batch_metrics = {
                "iterations": iterations,
                "total_time_seconds": total_time,
                "throughput": iterations * batch_size / total_time,
                "latency_mean_ms": sum(latencies) / len(latencies),
                "latency_median_ms": sorted(latencies)[len(latencies) // 2],
                "latency_p90_ms": sorted(latencies)[int(len(latencies) * 0.9)],
                "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
                "samples_per_second": iterations * batch_size / total_time
            }

            metrics[f"batch_{batch_size}"] = batch_metrics

            logger.info(f"Batch size {batch_size}: {iterations} iterations in {total_time:.2f}s")
            logger.info(f"Throughput: {batch_metrics['throughput']:.2f} inferences/second")
            logger.info(f"Average latency: {batch_metrics['latency_mean_ms']:.2f} ms")

        return metrics

    def _benchmark_tensorflow(
        self,
        model_path: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark TensorFlow model"""
        try:
            import tensorflow as tf
            import numpy as np
        except ImportError:
            logger.error("tensorflow not installed. Required for TensorFlow benchmarking.")
            raise

        logger.info(f"Benchmarking TensorFlow model: {model_path}")

        # Configure TensorFlow to use GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"Error configuring TensorFlow GPU: {e}")

        # Load model
        model = tf.keras.models.load_model(model_path)

        metrics = {}

        # Run benchmark for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")

            # Create dummy input based on model input shape
            input_shape = model.input_shape
            shape = list(input_shape)
            if shape[0] is None:  # Dynamic batch dimension
                shape[0] = batch_size

            dummy_input = np.random.random(shape).astype(np.float32)

            # Warmup
            logger.info(f"Warming up for {warmup} iterations")
            for _ in range(warmup):
                _ = model.predict(dummy_input)

            # Benchmark
            start_time = time.time()
            iterations = 0
            latencies = []

            while time.time() - start_time < duration_seconds:
                iter_start = time.time()
                _ = model.predict(dummy_input)
                latencies.append((time.time() - iter_start) * 1000)  # ms
                iterations += 1

            total_time = time.time() - start_time

            # Calculate metrics
            batch_metrics = {
                "iterations": iterations,
                "total_time_seconds": total_time,
                "throughput": iterations * batch_size / total_time,
                "latency_mean_ms": sum(latencies) / len(latencies),
                "latency_median_ms": sorted(latencies)[len(latencies) // 2],
                "latency_p90_ms": sorted(latencies)[int(len(latencies) * 0.9)],
                "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
                "samples_per_second": iterations * batch_size / total_time
            }

            metrics[f"batch_{batch_size}"] = batch_metrics

            logger.info(f"Batch size {batch_size}: {iterations} iterations in {total_time:.2f}s")
            logger.info(f"Throughput: {batch_metrics['throughput']:.2f} inferences/second")
            logger.info(f"Average latency: {batch_metrics['latency_mean_ms']:.2f} ms")

        return metrics

    def _benchmark_triton(
        self,
        model_name: str,
        model_type: str,
        batch_sizes: List[int],
        duration_seconds: int,
        concurrency: List[int],
        triton_url: str,
        warmup: int
    ) -> Dict[str, Any]:
        """Benchmark model on Triton Inference Server"""
        try:
            import tritonclient.grpc as grpcclient
            import tritonclient.grpc.model_config_pb2 as mc
            import tritonclient.http as httpclient
            import numpy as np
        except ImportError:
            logger.error("tritonclient not installed. Required for Triton benchmarking.")
            raise

        logger.info(f"Benchmarking model {model_name} on Triton server: {triton_url}")

        # Create client
        client = grpcclient.InferenceServerClient(url=triton_url)

        # Check if server is alive
        if not client.is_server_live():
            logger.error(f"Triton server at {triton_url} is not live")
            raise RuntimeError(f"Triton server at {triton_url} is not live")

        # Check if model is ready
        if not client.is_model_ready(model_name):
            logger.error(f"Model {model_name} is not ready on Triton server")
            raise RuntimeError(f"Model {model_name} is not ready on Triton server")

        # Get model metadata
        try:
            model_metadata = client.get_model_metadata(model_name)
            logger.info(f"Model metadata: {model_metadata}")
        except Exception as e:
            logger.warning(f"Failed to get model metadata: {e}")
            model_metadata = None

        # Get model configuration
        try:
            model_config = client.get_model_config(model_name)
            logger.info(f"Model config: {model_config}")
        except Exception as e:
            logger.warning(f"Failed to get model config: {e}")
            model_config = None

        metrics = {
            "model_name": model_name,
            "model_type": model_type,
            "triton_url": triton_url,
            "benchmark_type": "triton",
            "batch_sizes": batch_sizes,
            "concurrency": concurrency,
            "duration_seconds": duration_seconds,
            "warmup": warmup,
            "metrics": {}
        }

        # Create inputs based on model metadata
        if model_metadata is not None:
            inputs = []
            input_shapes = {}
            input_types = {}

            for input_meta in model_metadata.inputs:
                input_name = input_meta.name
                input_dtype = input_meta.datatype
                input_shape = input_meta.shape

                input_shapes[input_name] = input_shape
                input_types[input_name] = input_dtype

            # Run benchmark for each batch size and concurrency combination
            for batch_size in batch_sizes:
                for concurrent_clients in concurrency:
                    logger.info(f"Testing batch size: {batch_size}, concurrency: {concurrent_clients}")

                    # Create input data
                    inputs = []
                    for input_name, shape in input_shapes.items():
                        # Replace batch dimension
                        shape_list = list(shape)
                        if shape_list[0] == -1:
                            shape_list[0] = batch_size

                        # Create random data based on type
                        dtype = input_types[input_name]
                        if "INT" in dtype:
                            data = np.random.randint(0, 100, size=shape_list).astype(np.int32)
                        else:
                            data = np.random.random(shape_list).astype(np.float32)

                        inputs.append(grpcclient.InferInput(input_name, shape_list, dtype))
                        inputs[-1].set_data_from_numpy(data)

                    # Prepare outputs
                    outputs = []
                    for output_meta in model_metadata.outputs:
                        outputs.append(grpcclient.InferRequestedOutput(output_meta.name))

                    # Warmup
                    logger.info(f"Warming up for {warmup} iterations")
                    for _ in range(warmup):
                        client.infer(model_name, inputs, outputs=outputs)

                    # Setup concurrent requests
                    threads = []
                    thread_results = []
                    stop_event = threading.Event()

                    def run_inference(thread_id, results):
                        latencies = []
                        iterations = 0

                        while not stop_event.is_set():
                            try:
                                start_time = time.time()
                                response = client.infer(model_name, inputs, outputs=outputs)
                                latencies.append((time.time() - start_time) * 1000)  # ms
                                iterations += 1
                            except Exception as e:
                                logger.error(f"Thread {thread_id} inference error: {e}")

                        results.append({
                            "thread_id": thread_id,
                            "iterations": iterations,
                            "latencies": latencies
                        })

                    # Start threads
                    for i in range(concurrent_clients):
                        thread = threading.Thread(target=run_inference, args=(i, thread_results))
                        threads.append(thread)
                        thread.start()

                    # Run for specified duration
                    start_time = time.time()
                    time.sleep(duration_seconds)
                    stop_event.set()

                    # Wait for threads to finish
                    for thread in threads:
                        thread.join()

                    total_time = time.time() - start_time

                    # Aggregate results
                    total_iterations = sum(r["iterations"] for r in thread_results)
                    all_latencies = []
                    for r in thread_results:
                        all_latencies.extend(r["latencies"])

                    batch_concurrency_metrics = {
                        "iterations": total_iterations,
                        "total_time_seconds": total_time,
                        "throughput": total_iterations * batch_size / total_time,
                        "requests_per_second": total_iterations / total_time,
                        "latency_mean_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
                        "latency_median_ms": sorted(all_latencies)[len(all_latencies) // 2] if all_latencies else 0,
                        "latency_p90_ms": sorted(all_latencies)[int(len(all_latencies) * 0.9)] if all_latencies else 0,
                        "latency_p99_ms": sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0,
                        "samples_per_second": total_iterations * batch_size / total_time,
                        "concurrent_clients": concurrent_clients
                    }

                    metrics["metrics"][f"batch_{batch_size}_concurrency_{concurrent_clients}"] = batch_concurrency_metrics

                    logger.info(f"Batch {batch_size}, Concurrency {concurrent_clients}: {total_iterations} iterations in {total_time:.2f}s")
                    logger.info(f"Throughput: {batch_concurrency_metrics['throughput']:.2f} inferences/second")
                    logger.info(f"Average latency: {batch_concurrency_metrics['latency_mean_ms']:.2f} ms")
        else:
            logger.error("Could not get model metadata from Triton server")

        return metrics

    def _generate_plots(self, results: Dict[str, Any], output_dir: str) -> None:
        """Generate performance plots from benchmark results"""
        os.makedirs(output_dir, exist_ok=True)

        # Extract metrics for plotting
        if "metrics" not in results:
            logger.warning("No metrics found in results, skipping plot generation")
            return

        metrics = results["metrics"]

        # Check if this is a Triton benchmark or direct benchmark
        is_triton = "benchmark_type" in results and results["benchmark_type"] == "triton"

        # Plot throughput vs batch size
        if not is_triton:
            # For direct benchmark
            batch_sizes = []
            throughputs = []
            latencies = []

            for key, value in metrics.items():
                if key.startswith("batch_"):
                    batch_size = int(key.split("_")[1])
                    batch_sizes.append(batch_size)
                    throughputs.append(value["throughput"])
                    latencies.append(value["latency_mean_ms"])

            # Sort by batch size
            batch_sizes, throughputs, latencies = zip(*sorted(zip(batch_sizes, throughputs, latencies)))

            # Throughput plot
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (inferences/second)')
            plt.title('Throughput vs Batch Size')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'throughput_vs_batch.png'))
            plt.close()

            # Latency plot
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, latencies, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            plt.title('Latency vs Batch Size')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'latency_vs_batch.png'))
            plt.close()

            # Throughput and latency table
            df = pd.DataFrame({
                'Batch Size': batch_sizes,
                'Throughput (inferences/s)': throughputs,
                'Latency (ms)': latencies
            })

            # Save as CSV
            df.to_csv(os.path.join(output_dir, 'metrics_by_batch.csv'), index=False)

            # Save as styled HTML
            html = df.style.set_properties(**{'font-size': '10pt'}) \
                .format({'Throughput (inferences/s)': '{:.2f}', 'Latency (ms)': '{:.2f}'}) \
                .to_html()

            with open(os.path.join(output_dir, 'metrics_by_batch.html'), 'w') as f:
                f.write(html)

        else:
            # For Triton benchmark - handle concurrency as well
            data = []

            for key, value in metrics.items():
                if key.startswith("batch_"):
                    parts = key.split("_")
                    batch_size = int(parts[1])
                    concurrency = int(parts[3])

                    data.append({
                        'Batch Size': batch_size,
                        'Concurrency': concurrency,
                        'Throughput': value["throughput"],
                        'Latency': value["latency_mean_ms"]
                    })

            df = pd.DataFrame(data)

            # Plot heatmap for throughput
            batch_sizes = sorted(df['Batch Size'].unique())
            concurrencies = sorted(df['Concurrency'].unique())

            throughput_matrix = np.zeros((len(batch_sizes), len(concurrencies)))
            latency_matrix = np.zeros((len(batch_sizes), len(concurrencies)))

            for i, batch in enumerate(batch_sizes):
                for j, conc in enumerate(concurrencies):
                    filtered = df[(df['Batch Size'] == batch) & (df['Concurrency'] == conc)]
                    if not filtered.empty:
                        throughput_matrix[i, j] = filtered['Throughput'].values[0]
                        latency_matrix[i, j] = filtered['Latency'].values[0]

            # Throughput heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(throughput_matrix, cmap='viridis')
            plt.colorbar(label='Throughput (inferences/s)')
            plt.xticks(np.arange(len(concurrencies)), concurrencies)
            plt.yticks(np.arange(len(batch_sizes)), batch_sizes)
            plt.xlabel('Concurrency')
            plt.ylabel('Batch Size')
            plt.title('Throughput Heatmap')

            # Add text annotations
            for i in range(len(batch_sizes)):
                for j in range(len(concurrencies)):
                    plt.text(j, i, f'{throughput_matrix[i, j]:.1f}',
                            ha='center', va='center', color='white')

            plt.savefig(os.path.join(output_dir, 'throughput_heatmap.png'))
            plt.close()

            # Latency heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(latency_matrix, cmap='plasma')
            plt.colorbar(label='Latency (ms)')
            plt.xticks(np.arange(len(concurrencies)), concurrencies)
            plt.yticks(np.arange(len(batch_sizes)), batch_sizes)
            plt.xlabel('Concurrency')
            plt.ylabel('Batch Size')
            plt.title('Latency Heatmap')

            # Add text annotations
            for i in range(len(batch_sizes)):
                for j in range(len(concurrencies)):
                    plt.text(j, i, f'{latency_matrix[i, j]:.1f}',
                            ha='center', va='center', color='white')

            plt.savefig(os.path.join(output_dir, 'latency_heatmap.png'))
            plt.close()

            # Save full data as CSV
            df.to_csv(os.path.join(output_dir, 'metrics_by_batch_concurrency.csv'), index=False)

            # Save as styled HTML
            html = df.style.set_properties(**{'font-size': '10pt'}) \
                .format({'Throughput': '{:.2f}', 'Latency': '{:.2f}'}) \
                .to_html()

            with open(os.path.join(output_dir, 'metrics_by_batch_concurrency.html'), 'w') as f:
                f.write(html)

    def _create_dummy_input(self, model, model_type: str, batch_size: int = 1):
        """Create dummy input for a model based on its type"""
        import torch

        if model_type == "language":
            # Language model dummy input
            if hasattr(model, "config"):
                seq_length = getattr(model.config, "max_position_embeddings", 512)
                vocab_size = getattr(model.config, "vocab_size", 32000)
                return torch.randint(0, vocab_size, (batch_size, seq_length))
            return torch.randint(0, 32000, (batch_size, 512))

        elif model_type == "vision":
            # Vision model dummy input
            if hasattr(model, "config"):
                channels = 3
                img_size = getattr(model.config, "image_size", 224)
                if isinstance(img_size, list):
                    return torch.randn(batch_size, channels, img_size[0], img_size[1])
                return torch.randn(batch_size, channels, img_size, img_size)
            return torch.randn(batch_size, 3, 224, 224)

        elif model_type == "speech":
            # Speech model dummy input
            return torch.randn(batch_size, 80, 3000)  # Typical mel spectrogram

        else:
            # Default dummy input
            return torch.randn(batch_size, 3, 224, 224)


def main():
    """Command line interface for the model benchmark tool"""
    parser = argparse.ArgumentParser(description='Benchmark AI models on NVIDIA Jetson AGX Orin')
    parser.add_argument('model', type=str, help='Path or identifier for the model')
    parser.add_argument('--output', '-o', type=str, help='Output directory for benchmark results')
    parser.add_argument('--model-type', '-t', type=str, help='Model type (language, vision, speech)')
    parser.add_argument('--batch-sizes', '-b', type=str, default="1,2,4,8", help='Comma-separated batch sizes')
    parser.add_argument('--duration', '-d', type=int, default=10, help='Duration in seconds for each benchmark')
    parser.add_argument('--concurrency', '-c', type=str, default="1,2,4,8", help='Comma-separated concurrency values')
    parser.add_argument('--triton', '-u', type=str, help='Triton server URL (e.g., localhost:8001)')
    parser.add_argument('--warmup', '-w', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--no-save', action='store_true', help='Do not save benchmark results')
    parser.add_argument('--no-plots', action='store_true', help='Do not generate performance plots')

    args = parser.parse_args()

    # Parse batch sizes and concurrency
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    concurrency = [int(c) for c in args.concurrency.split(",")]

    # Run benchmark
    benchmark = ModelBenchmark()
    results = benchmark.benchmark(
        model_source=args.model,
        output_dir=args.output,
        model_type=args.model_type,
        batch_sizes=batch_sizes,
        duration_seconds=args.duration,
        concurrency=concurrency,
        triton_url=args.triton,
        warmup=args.warmup,
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )

    print(f"Benchmark completed successfully!")


if __name__ == "__main__":
    main()
