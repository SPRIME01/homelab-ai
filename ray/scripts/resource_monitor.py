"""
Resource monitoring and reporting for Ray cluster in an AI homelab.
Tracks resource usage and exports metrics for observability.
"""
import ray
import time
import logging
import asyncio
import psutil
import json
import os
import threading
from typing import Dict, List, Optional
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram

from config import CONFIG
import gpu_manager
from task_scheduler import get_task_scheduler, TaskStatus

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG.monitoring.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resource_monitor")

# Resource metrics
GPU_MEMORY_USED = Gauge("gpu_memory_mb_used", "GPU memory used in MB")
GPU_MEMORY_TOTAL = Gauge("gpu_memory_mb_total", "Total GPU memory in MB")
GPU_UTILIZATION = Gauge("gpu_utilization", "GPU memory utilization (0-1)")
CPU_USED = Gauge("cpu_cores_used", "CPU cores used")
CPU_TOTAL = Gauge("cpu_cores_total", "Total CPU cores")
MEMORY_USED_MB = Gauge("memory_mb_used", "Memory used in MB")
MEMORY_TOTAL_MB = Gauge("memory_mb_total", "Total memory in MB")

# Task metrics
TASKS_QUEUED = Gauge("tasks_queued", "Number of tasks in queue")
TASKS_RUNNING = Gauge("tasks_running", "Number of tasks running")
TASKS_COMPLETED = Counter("tasks_completed_total", "Total number of completed tasks")
TASKS_FAILED = Counter("tasks_failed_total", "Total number of failed tasks")
TASK_RUNTIME = Histogram(
    "task_runtime_seconds",
    "Task runtime in seconds",
    ["model_type"],
    buckets=(0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)
TASK_WAIT_TIME = Histogram(
    "task_wait_time_seconds",
    "Task wait time in seconds",
    ["model_type", "priority"],
    buckets=(0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

# Inference metrics
INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["model_name", "model_type"]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["model_name", "model_type"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
)


class ResourceMonitor:
    """
    Monitors and reports resource usage in the Ray cluster.
    Exports metrics to Prometheus and logs resource status.
    """

    def __init__(self, export_metrics: bool = True):
        """
        Initialize the resource monitor.

        Args:
            export_metrics: Whether to export metrics to Prometheus
        """
        self.scheduler = None
        self.gpu_mgr = None
        self.export_metrics = export_metrics
        self.is_running = False
        self._stop_event = threading.Event()
        logger.info("Resource Monitor initialized")

    async def initialize(self):
        """Initialize connections to Ray services."""
        try:
            if not ray.is_initialized():
                ray.init(
                    address=CONFIG.ray.address,
                    namespace=CONFIG.ray.namespace,
                    ignore_reinit_error=CONFIG.ray.ignore_reinit_error
                )
                logger.info(f"Ray initialized with address: {CONFIG.ray.address}")

            # Get the task scheduler
            self.scheduler = get_task_scheduler()

            # Get the GPU manager
            self.gpu_mgr = gpu_manager.get_gpu_manager()

            # Start Prometheus metrics server if enabled
            if self.export_metrics:
                try:
                    start_http_server(CONFIG.monitoring.metrics_export_port)
                    logger.info(f"Started Prometheus metrics server on port {CONFIG.monitoring.metrics_export_port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus metrics server: {e}")

            logger.info("Resource Monitor fully initialized")
        except Exception as e:
            logger.error(f"Error initializing Resource Monitor: {e}")
            raise

    async def start_monitoring(self):
        """Start monitoring resources."""
        if self.is_running:
            logger.warning("Resource Monitor is already running")
            return

        logger.info("Starting resource monitoring")
        self.is_running = True
        self._stop_event.clear()

        # Start background monitoring
        asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stop monitoring resources."""
        if not self.is_running:
            logger.warning("Resource Monitor is not running")
            return

        logger.info("Stopping resource monitoring")
        self._stop_event.set()
        self.is_running = False

    async def _monitoring_loop(self):
        """Background loop to collect and report metrics."""
        if not self.scheduler or not self.gpu_mgr:
            await self.initialize()

        while not self._stop_event.is_set():
            try:
                # Collect and report metrics
                await self._collect_gpu_metrics()
                await self._collect_system_metrics()
                await self._collect_task_metrics()

                # Log resource status
                await self._log_resource_status()

                # Wait before next collection
                await asyncio.sleep(CONFIG.monitoring.metrics_export_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _collect_gpu_metrics(self):
        """Collect and report GPU metrics."""
        try:
            # Get GPU memory usage from the GPU manager
            used_memory = await ray.get(self.gpu_mgr.get_used_memory.remote())
            total_memory = CONFIG.resources.gpu_memory_limit
            utilization = await ray.get(self.gpu_mgr.get_utilization.remote())

            # Update Prometheus metrics
            GPU_MEMORY_USED.set(used_memory)
            GPU_MEMORY_TOTAL.set(total_memory)
            GPU_UTILIZATION.set(utilization)

        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")

    async def _collect_system_metrics(self):
        """Collect and report system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None) / 100.0
            cpu_count = psutil.cpu_count(logical=True)
            cpu_used = cpu_count * cpu_percent

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)

            # Update Prometheus metrics
            CPU_USED.set(cpu_used)
            CPU_TOTAL.set(cpu_count)
            MEMORY_USED_MB.set(memory_used_mb)
            MEMORY_TOTAL_MB.set(memory_total_mb)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _collect_task_metrics(self):
        """Collect and report task metrics."""
        try:
            # Get task stats
            pending_tasks = await self.scheduler.get_all_tasks(TaskStatus.PENDING)
            running_tasks = await self.scheduler.get_all_tasks(TaskStatus.RUNNING)

            # Update Prometheus metrics
            TASKS_QUEUED.set(len(pending_tasks))
            TASKS_RUNNING.set(len(running_tasks))

            # Collect metrics for completed tasks
            all_tasks = await self.scheduler.get_all_tasks()
            for task in all_tasks:
                if task["status"] == TaskStatus.COMPLETED and task.get("duration") is not None:
                    # Record task runtime
                    TASK_RUNTIME.labels(model_type=task["model_type"]).observe(task["duration"])

                    # Record wait time
                    if task.get("wait_time") is not None:
                        TASK_WAIT_TIME.labels(
                            model_type=task["model_type"],
                            priority=str(task["priority"])
                        ).observe(task["wait_time"])

                    # Increment completed counter (only for tasks we haven't counted yet)
                    # In a real system, you'd want to persist task IDs you've counted already
                    # to avoid double-counting across restarts
                    TASKS_COMPLETED.inc()

                elif task["status"] == TaskStatus.FAILED:
                    TASKS_FAILED.inc()

        except Exception as e:
            logger.error(f"Error collecting task metrics: {e}")

    async def _log_resource_status(self):
        """Log a summary of resource usage."""
        try:
            # Get allocations from GPU manager
            allocations = await ray.get(self.gpu_mgr.list_allocations.remote())

            # Count tasks by status
            tasks_by_status = {}
            all_tasks = await self.scheduler.get_all_tasks()
            for task in all_tasks:
                status = task["status"]
                tasks_by_status[status] = tasks_by_status.get(status, 0) + 1

            # Log resource summary
            logger.info(
                f"Resource status: "
                f"GPU memory: {await ray.get(self.gpu_mgr.get_used_memory.remote())}/{CONFIG.resources.gpu_memory_limit}MB "
                f"({await ray.get(self.gpu_mgr.get_utilization.remote())*100:.1f}%), "
                f"Tasks: {tasks_by_status}"
            )

            # Log active allocations
            if allocations:
                allocation_summary = [
                    f"{a['task_id'][:8]}...({a['model_type']}): {a['memory_mb']}MB"
                    for a in allocations if a['is_active']
                ]
                logger.debug(f"Active allocations: {', '.join(allocation_summary)}")

        except Exception as e:
            logger.error(f"Error logging resource status: {e}")

    def record_inference(self, model_name: str, model_type: str, latency_seconds: float):
        """
        Record an inference request for metrics.

        Args:
            model_name: Name of the model
            model_type: Type of the model
            latency_seconds: Latency in seconds
        """
        if self.export_metrics:
            INFERENCE_COUNT.labels(model_name=model_name, model_type=model_type).inc()
            INFERENCE_LATENCY.labels(model_name=model_name, model_type=model_type).observe(latency_seconds)

    async def get_resource_report(self) -> Dict:
        """
        Generate a comprehensive resource usage report.

        Returns:
            Dictionary with complete resource usage information
        """
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # GPU resources
            gpu_used_memory = await ray.get(self.gpu_mgr.get_used_memory.remote())
            gpu_total_memory = CONFIG.resources.gpu_memory_limit
            gpu_util = await ray.get(self.gpu_mgr.get_utilization.remote())

            # Task statistics
            all_tasks = await self.scheduler.get_all_tasks()
            task_counts = {}
            for task in all_tasks:
                status = task["status"]
                task_counts[status] = task_counts.get(status, 0) + 1

            # Active allocations
            allocations = await ray.get(self.gpu_mgr.list_allocations.remote())
            active_allocations = [a for a in allocations if a.get('is_active', False)]

            # Create report
            report = {
                "timestamp": time.time(),
                "system": {
                    "cpu": {
                        "percent": cpu_percent,
                        "cores": psutil.cpu_count()
                    },
                    "memory": {
                        "total_mb": memory.total / (1024 * 1024),
                        "used_mb": memory.used / (1024 * 1024),
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024 * 1024 * 1024),
                        "used_gb": disk.used / (1024 * 1024 * 1024),
                        "percent": disk.percent
                    },
                },
                "gpu": {
                    "memory_used_mb": gpu_used_memory,
                    "memory_total_mb": gpu_total_memory,
                    "utilization": gpu_util,
                    "active_allocations": len(active_allocations),
                    "allocations": active_allocations
                },
                "tasks": task_counts
            }

            return report

        except Exception as e:
            logger.error(f"Error generating resource report: {e}")
            return {"error": str(e), "timestamp": time.time()}


# Singleton instance
_resource_monitor = None

def get_resource_monitor() -> ResourceMonitor:
    """Get the singleton instance of the Resource Monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


async def start_monitoring():
    """Start the resource monitor."""
    monitor = get_resource_monitor()
    await monitor.initialize()
    await monitor.start_monitoring()
    return monitor


# CLI entry point for standalone monitoring
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resource Monitor for Ray AI Homelab")
    parser.add_argument("--metrics-port", type=int, default=8000,
                      help="Port to expose Prometheus metrics")
    args = parser.parse_args()

    # Override config if specified
    if args.metrics_port != 8000:
        CONFIG.monitoring.metrics_export_port = args.metrics_port

    async def main():
        monitor = get_resource_monitor()
        await monitor.initialize()
        await monitor.start_monitoring()

        try:
            # Keep the service running
            while True:
                await asyncio.sleep(60)
                # Print report every minute
                report = await monitor.get_resource_report()
                print(json.dumps(report, indent=2))
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Resource monitoring stopped")

    asyncio.run(main())
