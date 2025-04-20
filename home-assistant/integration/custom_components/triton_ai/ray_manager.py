"""Task manager for Ray distributed computing framework."""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
import ray
import ray.dashboard

_LOGGER = logging.getLogger(__name__)

class RayTaskManager:
    """Manager for Ray distributed computing tasks."""

    def __init__(self, address: str):
        """Initialize the Ray task manager.

        Args:
            address: Ray cluster address (ray://ip:port)
        """
        self.address = address
        self.is_initialized = False
        self._tasks = {}

    async def initialize(self) -> bool:
        """Initialize connection to Ray cluster."""
        try:
            # Run ray.init in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            ray_init_result = await loop.run_in_executor(
                None,
                lambda: ray.init(
                    address=self.address,
                    namespace="home_assistant",
                    ignore_reinit_error=True,
                    logging_level=logging.INFO
                )
            )

            _LOGGER.info(
                "Connected to Ray cluster at %s with %s CPUs and %s GPUs",
                self.address,
                ray_init_result.get("num_cpus", "unknown"),
                ray_init_result.get("num_gpus", "unknown")
            )

            # Check cluster health
            nodes_info = await loop.run_in_executor(None, ray.nodes)
            if not nodes_info:
                _LOGGER.error("No active nodes found in Ray cluster")
                return False

            self.is_initialized = True
            return True

        except ConnectionError as err:
            _LOGGER.error("Error connecting to Ray cluster: %s", err)
            return False
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Unexpected error initializing Ray: %s", err)
            return False

    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
        memory: Optional[int] = None,
        resources: Optional[Dict[str, float]] = None,
        max_retries: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit a task to Ray for execution.

        Args:
            func: Function to execute (must be Ray remote compatible)
            *args: Arguments to pass to the function
            task_id: Optional identifier for the task
            num_cpus: Number of CPUs to allocate
            num_gpus: Number of GPUs to allocate
            memory: Memory to allocate in bytes
            resources: Additional resources to allocate
            max_retries: Maximum number of retries for the task
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary with task information
        """
        if not self.is_initialized:
            await self.initialize()
            if not self.is_initialized:
                return {
                    "status": "error",
                    "error": "Ray cluster not connected",
                    "task_id": task_id
                }

        try:
            # Create a unique task ID if none provided
            if task_id is None:
                task_id = f"hass_task_{len(self._tasks) + 1}"

            # Create remote options
            remote_options = {
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "max_retries": max_retries
            }

            if memory is not None:
                remote_options["memory"] = memory

            if resources is not None:
                remote_options.update({f"resources.{k}": v for k, v in resources.items()})

            # Submit task to Ray
            loop = asyncio.get_running_loop()

            # Create a remote function
            remote_func = ray.remote(**remote_options)(func)

            # Submit the task
            ref = await loop.run_in_executor(
                None,
                lambda: remote_func.remote(*args, **kwargs)
            )

            # Store task info
            self._tasks[task_id] = {
                "ref": ref,
                "status": "running",
                "options": remote_options
            }

            return {
                "status": "submitted",
                "task_id": task_id,
            }

        except ray.exceptions.RayTaskError as err:
            _LOGGER.error("Ray task submission failed: %s", err)
            return {
                "status": "error",
                "error": str(err),
                "task_id": task_id
            }
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error submitting Ray task: %s", err)
            return {
                "status": "error",
                "error": str(err),
                "task_id": task_id
            }

    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get the result of a submitted task.

        Args:
            task_id: ID of the task to get result for
            timeout: Timeout in seconds (None for no timeout)

        Returns:
            Dictionary with task result or status
        """
        if not self.is_initialized:
            return {
                "status": "error",
                "error": "Ray cluster not connected",
                "task_id": task_id
            }

        if task_id not in self._tasks:
            return {
                "status": "error",
                "error": f"Unknown task ID: {task_id}",
                "task_id": task_id
            }

        task = self._tasks[task_id]
        ref = task["ref"]

        try:
            # Check if result is ready
            loop = asyncio.get_running_loop()
            is_ready = await loop.run_in_executor(
                None,
                lambda: ray.wait([ref], timeout=0)[0]
            )

            if not is_ready:
                # Wait for result if timeout is specified
                if timeout is not None:
                    is_ready = await loop.run_in_executor(
                        None,
                        lambda: ray.wait([ref], timeout=timeout)[0]
                    )

                    if not is_ready:
                        return {
                            "status": "pending",
                            "task_id": task_id
                        }
                else:
                    return {
                        "status": "pending",
                        "task_id": task_id
                    }

            # Get the result
            result = await loop.run_in_executor(
                None,
                lambda: ray.get(ref)
            )

            # Update task status
            task["status"] = "completed"

            return {
                "status": "completed",
                "task_id": task_id,
                "result": result
            }

        except ray.exceptions.RayTaskError as err:
            _LOGGER.error("Ray task %s failed: %s", task_id, err)
            task["status"] = "failed"
            return {
                "status": "failed",
                "error": str(err),
                "task_id": task_id
            }
        except asyncio.TimeoutError:
            return {
                "status": "pending",
                "task_id": task_id
            }
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting Ray task result: %s", err)
            return {
                "status": "error",
                "error": str(err),
                "task_id": task_id
            }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a submitted task."""
        if not self.is_initialized or task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        ref = task["ref"]

        try:
            # Cancel the task
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: ray.cancel(ref)
            )

            # Update task status
            task["status"] = "cancelled"
            return True

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error cancelling Ray task: %s", err)
            return False

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status information about the Ray cluster."""
        if not self.is_initialized:
            await self.initialize()
            if not self.is_initialized:
                return {"status": "disconnected"}

        try:
            # Get cluster status
            loop = asyncio.get_running_loop()
            nodes = await loop.run_in_executor(None, ray.nodes)

            # Extract useful information
            status = {
                "status": "connected",
                "nodes": len(nodes),
                "total_cpus": sum(node.get("Resources", {}).get("CPU", 0) for node in nodes),
                "total_gpus": sum(node.get("Resources", {}).get("GPU", 0) for node in nodes),
                "available_cpus": sum(node.get("Resources", {}).get("CPU", 0) -
                                    node.get("ResourcesInUse", {}).get("CPU", 0) for node in nodes),
                "available_gpus": sum(node.get("Resources", {}).get("GPU", 0) -
                                    node.get("ResourcesInUse", {}).get("GPU", 0) for node in nodes),
                "active_tasks": len([t for t in self._tasks.values() if t["status"] == "running"])
            }

            return status

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting Ray cluster status: %s", err)
            return {"status": "error", "error": str(err)}

    async def close(self):
        """Close the Ray connection and clean up resources."""
        if self.is_initialized:
            try:
                # Cancel any running tasks
                for task_id, task in self._tasks.items():
                    if task["status"] == "running":
                        await self.cancel_task(task_id)

                # Disconnect from Ray
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, ray.shutdown)
                self.is_initialized = False

            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Error closing Ray connection: %s", err)
