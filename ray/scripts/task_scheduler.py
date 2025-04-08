"""
Priority-based task scheduler for Ray cluster.
"""
import ray
import time
import logging
import heapq
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
import threading
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("task_scheduler")

@dataclass(order=True)
class PrioritizedTask:
    """Task with priority for the scheduler queue."""
    priority: int
    submission_time: float
    task_id: str = field(compare=False)
    run_func: Callable = field(compare=False)
    args: tuple = field(default=(), compare=False)
    kwargs: Dict = field(default_factory=dict, compare=False)
    resource_request: Dict = field(default_factory=dict, compare=False)
    timeout: float = field(default=None, compare=False)
    result_callback: Callable = field(default=None, compare=False)
    error_callback: Callable = field(default=None, compare=False)

class TaskScheduler:
    def __init__(self, config: Dict, gpu_manager=None):
        """
        Initialize the task scheduler with configuration.

        Args:
            config: Configuration dictionary
            gpu_manager: Optional GPU manager instance
        """
        self.config = config
        self.priority_config = config["task_priority"]
        self.model_profiles = config["model_profiles"]
        self.gpu_manager = gpu_manager

        # Priority queue for tasks
        self.task_queue = []
        self.queue_lock = threading.Lock()

        # Track running tasks
        self.running_tasks = {}
        self.running_lock = threading.Lock()

        # Track task results
        self.task_results = {}
        self.results_lock = threading.Lock()

        # Background scheduler thread
        self.scheduler_running = False
        self.scheduler_thread = None

    def start(self):
        """Start the scheduler background thread."""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return

        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Task scheduler started")

    def stop(self):
        """Stop the scheduler background thread."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
            logger.info("Task scheduler stopped")

    def submit_task(self, run_func: Callable, task_type: str = "batch_inference",
                   model_profile: str = None, args: tuple = (), kwargs: Dict = None,
                   resource_request: Dict = None, timeout: float = None,
                   result_callback: Callable = None, error_callback: Callable = None) -> str:
        """
        Submit a task to the scheduler.

        Args:
            run_func: Function to run
            task_type: Type of task for priority determination
            model_profile: Model profile to use for resource allocation
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            resource_request: Explicit resource request (overrides model profile)
            timeout: Task timeout in seconds
            result_callback: Callback for task result
            error_callback: Callback for task error

        Returns:
            Task ID
        """
        if kwargs is None:
            kwargs = {}

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Determine task priority
        priority = self.priority_config.get(task_type, 100)  # Default to low priority

        # Determine resource request
        final_resource_request = {}
        if model_profile and model_profile in self.model_profiles:
            profile = self.model_profiles[model_profile]
            final_resource_request = {
                "gpu_memory_gb": profile["gpu_memory_gb"],
                "cpu_cores": profile["cpu_cores"],
                "max_batch_size": profile["max_batch_size"],
                "timeout_seconds": profile["timeout_seconds"]
            }

        # Override with explicit resource request if provided
        if resource_request:
            final_resource_request.update(resource_request)

        # Use explicit timeout if provided
        if timeout is not None:
            final_resource_request["timeout_seconds"] = timeout
        elif "timeout_seconds" in final_resource_request:
            timeout = final_resource_request["timeout_seconds"]

        # Create prioritized task
        task = PrioritizedTask(
            priority=priority,
            submission_time=time.time(),
            task_id=task_id,
            run_func=run_func,
            args=args,
            kwargs=kwargs,
            resource_request=final_resource_request,
            timeout=timeout,
            result_callback=result_callback,
            error_callback=error_callback
        )

        # Add to priority queue
        with self.queue_lock:
            heapq.heappush(self.task_queue, task)

        logger.info(f"Submitted task {task_id} with priority {priority}, model profile: {model_profile}")
        return task_id

    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """
        Get the result of a task.

        Args:
            task_id: Task ID
            timeout: How long to wait for result

        Returns:
            Task result or raises exception
        """
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.results_lock:
                if task_id in self.task_results:
                    result = self.task_results.pop(task_id)
                    if isinstance(result, Exception):
                        raise result
                    return result

            time.sleep(0.1)

        raise TimeoutError(f"Timeout waiting for task {task_id} result")

    def _scheduler_loop(self):
        """Main scheduler loop that processes tasks."""
        while self.scheduler_running:
            task_to_run = None

            # Get the next task from the priority queue
            with self.queue_lock:
                if self.task_queue:
                    task_to_run = heapq.heappop(self.task_queue)

            if task_to_run:
                self._process_task(task_to_run)
            else:
                # No tasks to run, sleep briefly
                time.sleep(0.1)

            # Check for completed tasks
            self._check_completed_tasks()

    def _process_task(self, task: PrioritizedTask):
        """
        Process a single task.

        Args:
            task: The task to process
        """
        # Check if GPU is required and available
        gpu_index = None
        if self.gpu_manager and "gpu_memory_gb" in task.resource_request:
            memory_bytes = int(task.resource_request["gpu_memory_gb"] * 1e9)
            gpu_index = self.gpu_manager.allocate_gpu_memory(task.task_id, memory_bytes)

            if gpu_index is None:
                # Put the task back in the queue if GPU not available
                with self.queue_lock:
                    heapq.heappush(self.task_queue, task)
                time.sleep(0.1)  # Avoid tight loop
                return

        # Prepare environment for the task
        task_env = {"gpu_index": gpu_index}
        if gpu_index is not None:
            task.kwargs["gpu_index"] = gpu_index

        # Submit to Ray
        try:
            # Use Ray remote function with resource constraints
            cpu_cores = task.resource_request.get("cpu_cores", 1)

            @ray.remote(num_cpus=cpu_cores)
            def run_task_wrapper(func, args, kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return e

            # Submit to Ray and track the reference
            ray_ref = run_task_wrapper.remote(task.run_func, task.args, task.kwargs)

            # Track the running task
            with self.running_lock:
                self.running_tasks[task.task_id] = {
                    "ray_ref": ray_ref,
                    "task": task,
                    "env": task_env,
                    "start_time": time.time()
                }

            logger.info(f"Started task {task.task_id} on Ray (GPU: {gpu_index}, CPU cores: {cpu_cores})")

        except Exception as e:
            logger.error(f"Error submitting task {task.task_id} to Ray: {e}")
            # Clean up GPU allocation if needed
            if self.gpu_manager and gpu_index is not None:
                self.gpu_manager.release_gpu_memory(task.task_id)

            # Store error result
            with self.results_lock:
                self.task_results[task.task_id] = e

            # Call error callback if provided
            if task.error_callback:
                try:
                    task.error_callback(task.task_id, e)
                except Exception as callback_err:
                    logger.error(f"Error in error callback for task {task.task_id}: {callback_err}")

    def _check_completed_tasks(self):
        """Check for completed tasks and process their results."""
        tasks_to_check = []

        # Get list of tasks to check (avoid holding lock during ray.get)
        with self.running_lock:
            for task_id, task_info in list(self.running_tasks.items()):
                # Check for timeout
                if (task_info["task"].timeout and
                    time.time() - task_info["start_time"] > task_info["task"].timeout):
                    logger.warning(f"Task {task_id} timed out after {task_info['task'].timeout} seconds")
                    # Will be handled as a result
                    tasks_to_check.append((task_id, task_info))
                    continue

                # Check if ready
                if ray.wait([task_info["ray_ref"]], timeout=0)[0]:
                    tasks_to_check.append((task_id, task_info))

        # Process completed tasks
        for task_id, task_info in tasks_to_check:
            try:
                # Get result (will be exception if task failed)
                result = ray.get(task_info["ray_ref"])

                # Store result
                with self.results_lock:
                    if isinstance(result, Exception):
                        logger.error(f"Task {task_id} failed: {result}")
                        self.task_results[task_id] = result

                        # Call error callback if provided
                        if task_info["task"].error_callback:
                            try:
                                task_info["task"].error_callback(task_id, result)
                            except Exception as callback_err:
                                logger.error(f"Error in error callback for task {task_id}: {callback_err}")
                    else:
                        self.task_results[task_id] = result

                        # Call result callback if provided
                        if task_info["task"].result_callback:
                            try:
                                task_info["task"].result_callback(task_id, result)
                            except Exception as callback_err:
                                logger.error(f"Error in result callback for task {task_id}: {callback_err}")

            except Exception as e:
                logger.error(f"Error getting result for task {task_id}: {e}")
                with self.results_lock:
                    self.task_results[task_id] = e

            finally:
                # Clean up regardless of success/failure
                with self.running_lock:
                    if task_id in self.running_tasks:
                        # Clean up GPU allocation if needed
                        if (self.gpu_manager and
                            task_info["env"].get("gpu_index") is not None):
                            self.gpu_manager.release_gpu_memory(task_id)

                        # Remove from running tasks
                        self.running_tasks.pop(task_id)

    def get_queue_status(self) -> Dict:
        """
        Get status of the task queue.

        Returns:
            Dictionary with queue statistics
        """
        with self.queue_lock, self.running_lock:
            queued_by_priority = {}
            for task in self.task_queue:
                priority = task.priority
                if priority not in queued_by_priority:
                    queued_by_priority[priority] = 0
                queued_by_priority[priority] += 1

            return {
                "queue_length": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "queued_by_priority": queued_by_priority,
                "oldest_task_age": min([time.time() - task.submission_time
                                      for task in self.task_queue]) if self.task_queue else 0
            }
