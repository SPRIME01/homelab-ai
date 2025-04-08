"""
Main entry point for the Ray Resource Management system.
Provides a unified interface for managing AI workloads in a homelab environment.
"""
import ray
import time
import logging
import argparse
import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Union
import numpy as np

# Import our modules
from config import load_config
import gpu_manager
import task_scheduler
import triton_integration
import resource_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ray_manager")

class RayManager:
    """
    Unified management interface for Ray-based AI workloads.
    Coordinates resource management, task scheduling, and inference.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the Ray Manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = load_config(config_path)
        self.initialized = False

        # Set log level from config
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level))

        logger.info("Ray Manager initialized with configuration")

    async def initialize(self):
        """Initialize the Ray environment and services."""
        if self.initialized:
            logger.warning("Ray Manager already initialized")
            return

        try:
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(
                    address=self.config.get("ray", {}).get("address", "auto"),
                    ignore_reinit_error=True
                )
                logger.info("Ray initialized")

            # Initialize GPU Manager
            self.gpu_manager = await gpu_manager.get_gpu_manager()
            logger.info("GPU Manager initialized")

            # Initialize Task Scheduler
            self.task_scheduler = await task_scheduler.get_task_scheduler()
            logger.info("Task Scheduler initialized")

            # Initialize Triton Service
            triton_config = self.config.get("triton", {})
            self.triton_service = triton_integration.TritonService.remote(self.config)
            connected = await ray.get(self.triton_service.connect.remote())
            if connected:
                logger.info("Connected to Triton Inference Server")
            else:
                logger.warning("Failed to connect to Triton Inference Server")

            # Initialize Resource Monitor
            self.resource_monitor = await resource_monitor.start_monitoring()
            logger.info("Resource Monitor started")

            self.initialized = True
            logger.info("Ray Manager initialization complete")

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    async def submit_inference_task(self, model_name: str, inputs: Dict[str, np.ndarray],
                                  model_type: str = "llm_small", priority: str = "batch_inference",
                                  wait: bool = True, timeout: float = None) -> Optional[Dict]:
        """
        Submit an inference task to the system.

        Args:
            model_name: Name of the model to run inference on
            inputs: Model input data as dictionary of numpy arrays
            model_type: Type/size of model for resource allocation
            priority: Priority level for the task
            wait: Whether to wait for the result
            timeout: Timeout in seconds for waiting

        Returns:
            Inference results if wait is True, otherwise task ID
        """
        if not self.initialized:
            await self.initialize()

        # Ensure inputs are numpy arrays
        for k, v in inputs.items():
            if not isinstance(v, np.ndarray):
                inputs[k] = np.array(v)

        # Define function to run inside Ray
        async def run_inference(task_id, gpu_index):
            try:
                start_time = time.time()
                result = await ray.get(self.triton_service.infer.remote(
                    model_name=model_name,
                    inputs=inputs,
                    request_id=task_id
                ))
                latency = time.time() - start_time

                # Record metrics
                self.resource_monitor.record_inference(
                    model_name=model_name,
                    model_type=model_type,
                    latency_seconds=latency
                )

                return result

            except Exception as e:
                logger.error(f"Error during inference: {e}")
                raise

        # Submit task
        task_id = await self.task_scheduler.submit_task(
            run_func=run_inference,
            model_type=model_type,
            priority=priority
        )

        logger.info(f"Submitted inference task {task_id} for model {model_name}")

        if wait:
            # Wait for and return result
            result = await self.task_scheduler.get_task_result(task_id, timeout=timeout)
            return result
        else:
            # Return the task ID for later retrieval
            return {"task_id": task_id}

    async def get_task_result(self, task_id: str, timeout: float = None) -> Dict:
        """
        Get the result of a previously submitted task.

        Args:
            task_id: ID of the task
            timeout: Timeout in seconds

        Returns:
            Task result
        """
        if not self.initialized:
            await self.initialize()

        return await self.task_scheduler.get_task_result(task_id, timeout=timeout)

    async def list_models(self) -> List[Dict]:
        """
        List available models on the Triton server.

        Returns:
            List of model information dictionaries
        """
        if not self.initialized:
            await self.initialize()

        return await ray.get(self.triton_service.get_model_list.remote())

    async def get_resource_status(self) -> Dict:
        """
        Get current resource usage status.

        Returns:
            Dictionary with resource usage information
        """
        if not self.initialized:
            await self.initialize()

        return await self.resource_monitor.get_resource_report()

    async def shutdown(self):
        """Shutdown the Ray Manager and release resources."""
        if not self.initialized:
            logger.warning("Ray Manager not initialized, nothing to shut down")
            return

        try:
            # Stop resource monitor
            self.resource_monitor.stop_monitoring()
            logger.info("Resource Monitor stopped")

            # Shutdown completed
            self.initialized = False
            logger.info("Ray Manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global singleton instance
_ray_manager = None

def get_ray_manager(config_path: str = None) -> RayManager:
    """
    Get the singleton instance of the Ray Manager.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        RayManager instance
    """
    global _ray_manager
    if _ray_manager is None:
        _ray_manager = RayManager(config_path)
    return _ray_manager

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray Resource Manager for AI Homelab")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--action", type=str, default="status",
                      choices=["status", "models", "test-inference"],
                      help="Action to perform")
    parser.add_argument("--model", type=str, help="Model name for inference")
    args = parser.parseArgs()

    async def main():
        manager = get_ray_manager(args.config)
        await manager.initialize()

        try:
            if args.action == "status":
                status = await manager.get_resource_status()
                print(json.dumps(status, indent=2))

            elif args.action == "models":
                models = await manager.list_models()
                print("Available models:")
                for model in models:
                    print(f"- {model['name']}")

            elif args.action == "test-inference" and args.model:
                # Simple test inference with dummy data
                print(f"Running test inference on model: {args.model}")
                # This is a dummy example - adjust for your actual model inputs
                inputs = {
                    "input": np.random.rand(1, 3, 224, 224).astype(np.float32)
                }
                result = await manager.submit_inference_task(
                    model_name=args.model,
                    inputs=inputs,
                    wait=True,
                    timeout=30.0
                )
                print("Inference result:")
                print(json.dumps(result, indent=2))

        finally:
            await manager.shutdown()

    asyncio.run(main())
