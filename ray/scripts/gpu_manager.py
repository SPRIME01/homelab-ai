"""
GPU memory management for Ray cluster.
"""
import ray
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pynvml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_manager")

class GPUManager:
    def __init__(self, config: Dict):
        """
        Initialize the GPU manager with configuration.

        Args:
            config: Configuration dictionary containing GPU settings
        """
        self.config = config["gpu"]
        self.monitoring_config = config.get("monitoring", {})
        self._initialize_pynvml()

        # Track memory allocations
        self.allocations = {}

    def _initialize_pynvml(self):
        """Initialize NVIDIA Management Library."""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Initialized NVML. Found {self.device_count} GPU devices.")
            self.devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.device_count = 0
            self.devices = []

    def get_available_memory(self) -> List[int]:
        """
        Get available memory for each GPU in bytes.

        Returns:
            List of available memory per GPU in bytes
        """
        available_memory = []

        for device in self.devices:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(device)
                # Account for reserved memory percentage
                reserve_percent = self.config["memory_reserve_percent"]
                available = info.free * (1 - reserve_percent / 100)
                available_memory.append(int(available))

            except pynvml.NVMLError as e:
                logger.error(f"Error getting memory info: {e}")
                available_memory.append(0)

        return available_memory

    def get_gpu_utilization(self) -> List[int]:
        """
        Get GPU utilization percentage for each GPU.

        Returns:
            List of GPU utilization percentages
        """
        utilization = []

        for device in self.devices:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(device)
                utilization.append(util.gpu)
            except pynvml.NVMLError as e:
                logger.error(f"Error getting GPU utilization: {e}")
                utilization.append(0)

        return utilization

    def allocate_gpu_memory(self, task_id: str, memory_required: int) -> Optional[int]:
        """
        Allocate GPU memory for a task.

        Args:
            task_id: Unique ID for the task
            memory_required: Memory required in bytes

        Returns:
            GPU index if allocation successful, None otherwise
        """
        if task_id in self.allocations:
            logger.warning(f"Task {task_id} already has GPU allocation")
            return self.allocations[task_id]["gpu_index"]

        available_memory = self.get_available_memory()

        # Find GPU with enough memory
        for gpu_index, mem in enumerate(available_memory):
            if mem >= memory_required:
                self.allocations[task_id] = {
                    "gpu_index": gpu_index,
                    "memory": memory_required,
                    "timestamp": time.time()
                }
                logger.info(f"Allocated {memory_required/1e9:.2f} GB on GPU {gpu_index} for task {task_id}")
                return gpu_index

        # If oversubscription is enabled, allow allocation even if it might exceed available memory
        if self.config["enable_fractional_allocation"]:
            oversubscription_factor = self.config["oversubscription_factor"]
            for gpu_index, mem in enumerate(available_memory):
                if mem * oversubscription_factor >= memory_required:
                    self.allocations[task_id] = {
                        "gpu_index": gpu_index,
                        "memory": memory_required,
                        "timestamp": time.time(),
                        "oversubscribed": True
                    }
                    logger.warning(f"Oversubscribed {memory_required/1e9:.2f} GB on GPU {gpu_index} for task {task_id}")
                    return gpu_index

        logger.error(f"Failed to allocate {memory_required/1e9:.2f} GB for task {task_id}")
        return None

    def release_gpu_memory(self, task_id: str) -> bool:
        """
        Release GPU memory allocated for a task.

        Args:
            task_id: Unique ID for the task

        Returns:
            True if release successful, False otherwise
        """
        if task_id not in self.allocations:
            logger.warning(f"Task {task_id} has no GPU allocation to release")
            return False

        allocation = self.allocations.pop(task_id)
        gpu_index = allocation["gpu_index"]
        memory = allocation["memory"]

        logger.info(f"Released {memory/1e9:.2f} GB from GPU {gpu_index} for task {task_id}")
        return True

    def get_optimal_gpu(self, memory_required: int) -> int:
        """
        Get the optimal GPU for a given memory requirement.

        Args:
            memory_required: Memory required in bytes

        Returns:
            Optimal GPU index, or -1 if none available
        """
        available_memory = self.get_available_memory()
        utilization = self.get_gpu_utilization()

        # Filter GPUs with enough memory
        candidate_gpus = [(i, mem, util) for i, (mem, util) in
                          enumerate(zip(available_memory, utilization))
                          if mem >= memory_required]

        if not candidate_gpus:
            return -1

        # Return GPU with lowest utilization among candidates
        return min(candidate_gpus, key=lambda x: x[2])[0]

    def get_memory_stats(self) -> Dict:
        """
        Get memory statistics for all GPUs.

        Returns:
            Dictionary with memory statistics
        """
        stats = {"gpus": []}

        for i, device in enumerate(self.devices):
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(device)
                util = pynvml.nvmlDeviceGetUtilizationRates(device)
                name = pynvml.nvmlDeviceGetName(device).decode('utf-8')

                stats["gpus"].append({
                    "index": i,
                    "name": name,
                    "total_memory": int(info.total),
                    "free_memory": int(info.free),
                    "used_memory": int(info.used),
                    "utilization": int(util.gpu),
                    "memory_utilization": int(util.memory)
                })
            except pynvml.NVMLError as e:
                logger.error(f"Error getting GPU stats for device {i}: {e}")

        return stats

    def cleanup(self):
        """Clean up NVML resources."""
        try:
            pynvml.nvmlShutdown()
            logger.info("NVML shutdown successfully")
        except pynvml.NVMLError as e:
            logger.error(f"Error during NVML shutdown: {e}")
