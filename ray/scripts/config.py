"""
Configuration settings for Ray resource management.
"""
import os
from typing import Dict, Any
import yaml

# Default configuration
DEFAULT_CONFIG = {
    # GPU memory settings
    "gpu": {
        "memory_reserve_percent": 10,  # Reserve 10% GPU memory for system
        "max_memory_per_task": 80,     # Max % of GPU memory per task
        "enable_fractional_allocation": True,
        "oversubscription_factor": 1.0  # No oversubscription by default
    },

    # Task priority settings (lower number = higher priority)
    "task_priority": {
        "interactive": 0,
        "batch_inference": 10,
        "training": 20,
        "background": 30
    },

    # Model type resource profiles
    "model_profiles": {
        "llm_small": {
            "gpu_memory_gb": 2,
            "cpu_cores": 2,
            "max_batch_size": 8,
            "timeout_seconds": 30
        },
        "llm_medium": {
            "gpu_memory_gb": 4,
            "cpu_cores": 4,
            "max_batch_size": 4,
            "timeout_seconds": 60
        },
        "llm_large": {
            "gpu_memory_gb": 8,
            "cpu_cores": 8,
            "max_batch_size": 2,
            "timeout_seconds": 120
        },
        "vision_model": {
            "gpu_memory_gb": 2,
            "cpu_cores": 2,
            "max_batch_size": 16,
            "timeout_seconds": 15
        },
        "speech_model": {
            "gpu_memory_gb": 3,
            "cpu_cores": 3,
            "max_batch_size": 4,
            "timeout_seconds": 30
        }
    },

    # Triton integration settings
    "triton": {
        "url": "localhost:8001",
        "verbose": False,
        "preferred_batch_sizes": [1, 2, 4, 8],
        "max_queue_delay_microseconds": 100,
        "status_check_interval_seconds": 30,
        "enable_metrics": True
    },

    # Monitoring settings
    "monitoring": {
        "interval_seconds": 10,
        "memory_threshold_percent": 90,
        "utilization_threshold_percent": 95,
        "alert_cooldown_seconds": 300,
        "metrics_history_size": 100
    }
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, with defaults for missing values.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Dict containing configuration settings
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        # Deep merge the user config into default config
        _deep_update(config, user_config)

    return config

def _deep_update(base_dict, update_dict):
    """
    Recursively update a nested dictionary with another nested dictionary.

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with new values
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
