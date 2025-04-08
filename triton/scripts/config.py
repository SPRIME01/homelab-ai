import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("model-tools")

# Default paths
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "homelab-ai" / "models"
DEFAULT_OUTPUT_DIR = Path.home() / "models" / "optimized"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "homelab-ai" / "model-tools.yaml"

# Jetson AGX Orin specific settings
JETSON_SPECS = {
    "gpu": "Jetson AGX Orin",
    "compute_capability": "7.2",  # Ampere architecture
    "max_gpu_memory": 32 * 1024 * 1024 * 1024,  # 32GB GPU memory
    "max_cpu_memory": 64 * 1024 * 1024 * 1024,  # 64GB CPU RAM
    "num_cores": 12,  # 12-core ARM CPU
    "tensorrt_version": "8.4",
    "cuda_version": "11.4",
}

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "cache_dir": str(DEFAULT_CACHE_DIR),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "triton_model_repo": "/models",
    },
    "hardware": JETSON_SPECS,
    "optimization": {
        "default_precision": "fp16",  # fp32, fp16, int8
        "enable_tensorrt": True,
        "enable_dynamic_shapes": True,
        "preferred_batch_size": [1, 2, 4, 8],
        "max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4GB workspace
        "trt_builder_flags": ["PrecisionMode.FP16", "MemoryPoolType.WORKSPACE"],
        "quantization": {
            "calibration_samples": 100,
            "int8_calibrator": "entropy",  # entropy, minmax, percentile
        },
        "pruning": {
            "target_sparsity": 0.5,  # 50% sparsity
            "pruning_method": "magnitude",  # magnitude, random, structured
        },
        "onnx": {
            "opset_version": 13,
            "optimize_level": 99,
        },
    },
    "model_types": {
        "language": {
            "default_seq_length": 512,
            "quantization": "int8",
            "dynamic_axes": {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
            },
        },
        "vision": {
            "default_image_size": [224, 224],
            "quantization": "fp16",
            "dynamic_axes": {
                "input": {0: "batch", 2: "height", 3: "width"},
            },
        },
        "speech": {
            "default_sample_rate": 16000,
            "quantization": "int8",
            "dynamic_axes": {
                "input": {0: "batch", 1: "time"},
            },
        },
    },
}


class Config:
    """Configuration manager for model optimization tools"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        self.config_path = config_path or os.environ.get("MODEL_TOOLS_CONFIG", DEFAULT_CONFIG_PATH)
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        """Load configuration from file if it exists"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        # Deep merge configs
                        self._update_dict(self.config, user_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _update_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
        return d

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {save_path}")

    @property
    def cache_dir(self) -> str:
        """Get model cache directory"""
        return self.config['paths']['cache_dir']

    @property
    def output_dir(self) -> str:
        """Get model output directory"""
        return self.config['paths']['output_dir']

    @property
    def triton_model_repo(self) -> str:
        """Get Triton model repository path"""
        return self.config['paths']['triton_model_repo']

    def get_model_type_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type"""
        if model_type in self.config['model_types']:
            return self.config['model_types'][model_type]
        return {}

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration"""
        return self.config['optimization']

    def get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications"""
        return self.config['hardware']


# Global configuration instance
config = Config()
