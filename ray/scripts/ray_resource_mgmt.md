# Ray Resource Management for AI Homelab

This documentation covers the Ray Resource Management system for AI workloads in a homelab environment. The system provides GPU memory management, priority-based task scheduling, resource allocation, and Triton Inference Server integration.

## Overview

The Ray Resource Management system includes several components:

1. **Configuration Management**: Centralized configuration for all components
2. **GPU Memory Management**: Efficient allocation and tracking of GPU memory
3. **Priority-based Task Scheduling**: Tasks run based on priority and resource availability
4. **Triton Inference Server Integration**: Seamless integration with Triton for model serving
5. **Resource Monitoring**: Tracks and reports resource usage metrics
6. **Main API**: Unified interface for the resource management system

![Ray Resource Management Architecture](../docs/images/ray_resource_mgmt_arch.png)

## Installation

### Prerequisites

- Python 3.8+
- Ray 2.0.0+
- NVIDIA GPU with CUDA support
- Triton Inference Server (optional)

### Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Configure the system by creating a custom config file:

```bash
cp example.config.yaml config.yaml
# Edit config.yaml with your specific settings
```

3. Start the resource management system:

```bash
python main.py
```

## Components

### Config Management (`config.py`)

Centralizes configuration for the entire system:

- Ray cluster connection settings
- Triton server endpoints
- Resource limits for GPU, CPU, and memory
- Task priorities for different workload types
- Model resource requirements
- Monitoring settings

```python
# Load custom configuration
from config import load_config
config = load_config("/path/to/config.yaml")
```

### GPU Memory Manager (`gpu_manager.py`)

Manages GPU memory allocation with support for:

- Priority-based memory allocation
- Resource tracking and utilization reporting
- Memory preemption for high-priority tasks
- Fractional GPU allocation

```python
# Get the GPU memory manager
from gpu_manager import get_gpu_manager
gpu_mgr = get_gpu_manager()

# Check if memory can be allocated
can_allocate, tasks_to_preempt = await ray.get(gpu_mgr.can_allocate.remote(2000, 100))

# Allocate memory
success = await ray.get(gpu_mgr.allocate.remote("task-123", "llm", 2000, 100))
```

### Task Scheduler (`task_scheduler.py`)

Schedules tasks based on priority and resource availability:

- Priority-based task execution
- Resource reservation
- Task preemption for higher-priority workloads
- Retry mechanism for failed tasks

```python
# Get the task scheduler
from task_scheduler import get_task_scheduler
scheduler = get_task_scheduler()

# Submit a task with priority
task_id = await scheduler.submit_task(
    my_function, args=[arg1, arg2],
    name="inference_task",
    priority=100,
    model_type="llm",
    gpu_mb=4000,
    cpu=2.0,
    memory_mb=8000
)

# Wait for result
result = await scheduler.get_result(task_id)
```

### Triton Integration (`triton_integration.py`)

Integrates with NVIDIA Triton Inference Server:

- Model discovery and metadata retrieval
- Optimized inference requests
- Priority-based scheduling
- Resource allocation based on model type

```python
# Get the Triton manager
from triton_integration import get_triton_manager
triton = get_triton_manager()

# Run inference with priority
result = await triton.infer(
    model_name="llama2-7b",
    inputs={"input_ids": input_tensor},
    priority=100,
    model_type="llm"
)
```

### Resource Monitoring (`resource_monitor.py`)

Monitors and reports resource usage:

- GPU, CPU, and memory utilization tracking
- Task statistics collection
- Prometheus metrics export
- Logging and alerting

```python
# Start the resource monitor
from resource_monitor import start_monitoring
monitor = await start_monitoring()

# Get current resource statistics
from main import HomelabRayManager
manager = HomelabRayManager()
await manager.initialize()
stats = await manager.get_resource_stats()
```

### Main API (`main.py`)

Provides a unified interface for the resource management system:

- Component initialization and management
- Task submission and monitoring
- Resource statistics reporting
- Model listing and management

```python
from main import HomelabRayManager

# Initialize the manager
manager = HomelabRayManager()
await manager.initialize()

# Submit a task
task_id = await manager.submit_task(
    my_function, arg1, arg2,
    task_args={
        "name": "important_task",
        "priority": 100,
        "model_type": "llm",
        "gpu_mb": 4000
    }
)
```

## Usage Examples

### Example 1: Basic Task Submission

```python
import asyncio
from main import HomelabRayManager

async def main():
    manager = HomelabRayManager()
    await manager.initialize()

    # Define a function to run
    def process_data(data):
        # Process the data
        return {"processed": data, "status": "success"}

    # Submit the task with priority
    task_id = await manager.submit_task(
        process_data,
        {"input": "example data"},
        task_args={
            "priority": 80,
            "model_type": "processing",
            "cpu": 1.0
        }
    )

    # Get the result
    result = await manager.scheduler.get_result(task_id)
    print(f"Task result: {result}")

    await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Using the Priority Task Decorator

```python
import asyncio
from task_scheduler import priority_task

@priority_task(
    priority=100,
    model_type="llm",
    gpu_mb=4000,
    cpu=2.0,
    memory_mb=8000
)
async def generate_text(prompt):
    # This function will run with the specified priority and resources
    # Complex processing here
    return {"text": f"Generated response for: {prompt}"}

async def main():
    # The decorator handles all the resource allocation
    result = await generate_text("Tell me about AI in homelabs")
    print(result["text"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 3: Triton Inference with the Decorator

```python
import asyncio
import numpy as np
from triton_integration import triton_inference

@triton_inference(
    model_name="stable-diffusion",
    model_type="vision",
    priority=90
)
async def generate_image(prompt, height=512, width=512):
    # Prepare inputs for the Triton model
    inputs = {
        "prompt": np.array([prompt.encode('utf-8')]),
        "height": np.array([height], dtype=np.int32),
        "width": np.array([width], dtype=np.int32)
    }

    # Define how to process the outputs
    def process_outputs(outputs):
        # Convert the output tensor to an image
        return {"image": outputs["image"][0]}

    return {
        "inputs": inputs,
        "process_outputs": process_outputs
    }

async def main():
    result = await generate_image("a beautiful mountain landscape")
    # result["image"] contains the generated image

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Resource Monitoring

```python
import asyncio
from main import HomelabRayManager
import json

async def monitor_resources():
    manager = HomelabRayManager()
    await manager.initialize()

    try:
        while True:
            # Get current resource statistics
            stats = await manager.get_resource_stats()

            # Print GPU utilization
            gpu_util = stats["gpu"]["utilization"] * 100
            print(f"GPU Memory: {stats['gpu']['used_memory']}/{stats['gpu']['total_memory']}MB ({gpu_util:.1f}%)")

            # Print task counts
            tasks = stats["tasks"]
            print(f"Tasks: {tasks['pending']} pending, {tasks['running']} running, "
                  f"{tasks['completed']} completed, {tasks['failed']} failed")

            # Wait before next update
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(monitor_resources())
```

## Best Practices

### Resource Allocation

1. **Set Appropriate Priorities**:
   - Critical tasks: 90-100
   - Interactive tasks: 70-89
   - Batch processing: 40-69
   - Background tasks: 10-39

2. **Estimate Resource Requirements**:
   - Overestimating leads to underutilization
   - Underestimating leads to performance issues
   - Benchmark your models to find optimal values

3. **Enable Fractional GPU**:
   - Useful for serving multiple smaller models
   - Disable for models that need dedicated GPU access

### Performance Optimization

1. **Use Dynamic Batching**:
   - Group similar requests together
   - Configure `triton_integration.py` with batch parameters

2. **Preemption Strategy**:
   - Configure critical tasks with high priority
   - Use `max_retries` to ensure task completion
   - Consider checkpointing for long-running tasks

3. **Monitor Resource Usage**:
   - Watch for GPU memory fragmentation
   - Adjust resource limits based on observations
   - Set up alerts for high utilization

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Task stays pending | Insufficient resources | Check GPU memory usage with `manager.get_resource_stats()` |
| Task fails with GPU memory error | Memory fragmentation | Restart the Ray cluster or reduce concurrent tasks |
| High latency for inference | Resource contention | Increase priority or adjust batch sizes |
| Ray connection errors | Cluster not available | Verify Ray cluster status with `ray status` |
| Triton server unreachable | Server not running | Check connectivity with `triton.is_healthy()` |

### Logging and Debugging

Enable detailed logging:

```bash
export RAY_LOG_LEVEL=DEBUG
python -m main --config=config.yaml
```

View task execution logs:

```bash
python -c "from main import HomelabRayManager; import asyncio; asyncio.run(HomelabRayManager().get_resource_stats())"
```

## Advanced Configuration

### Custom Resource Types

Define custom resources in `config.py`:

```yaml
resources:
  custom_types:
    my_special_resource:
      limit: 4
      priority_threshold: 80
```

### Integration with External Systems

Add custom services in `main.py`:

```python
class ExtendedRayManager(HomelabRayManager):
    async def initialize(self):
        await super().initialize()
        # Add custom initialization
        self.custom_service = MyCustomService()
        await self.custom_service.initialize()
```

## Security Considerations

1. **Network Security**:
   - Configure Ray to use TLS encryption
   - Use authentication for Ray and Triton endpoints

2. **Resource Limits**:
   - Set hard limits in `config.py` to prevent resource exhaustion
   - Configure timeouts for long-running tasks

3. **Model Access Control**:
   - Implement permission checks for model management
   - Audit logs for model deployments

## Contributing

To contribute to the Ray Resource Management system:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
