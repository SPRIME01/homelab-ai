import os
import sys
import logging
import hashlib
import requests
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import tempfile
import json
import time
from urllib.parse import urlparse
from tqdm import tqdm

from config import config, logger

def verify_dependencies() -> bool:
    """Verify all required dependencies are installed"""
    try:
        # Check for basic dependencies
        import torch
        import onnx
        import numpy as np

        # Try to import TensorRT if optimization is enabled
        if config.get_optimization_config()["enable_tensorrt"]:
            import tensorrt as trt
            logger.info(f"TensorRT version: {trt.__version__}")

        # Check for specific PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"ONNX version: {onnx.__version__}")

        # Check for CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.warning("CUDA is not available. Some optimizations will not work.")

        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url: str, output_path: str, force: bool = False) -> str:
    """Download a file from a URL with progress bar"""
    if os.path.exists(output_path) and not force:
        logger.info(f"File already exists at {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)

        progress_bar.close()
        logger.info(f"Downloaded {url} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def detect_model_type(model_name: str) -> str:
    """Detect the model type (language, vision, speech) based on model name"""
    # Language model patterns
    language_patterns = ["gpt", "llama", "bert", "t5", "roberta", "bloom", "phi", "gemma", "falcon"]
    # Vision model patterns
    vision_patterns = ["resnet", "yolo", "vit", "swin", "efficientnet", "densenet", "inception", "detr"]
    # Speech model patterns
    speech_patterns = ["wav2vec", "whisper", "hubert", "conformer", "encodec", "mms"]

    model_name_lower = model_name.lower()

    for pattern in language_patterns:
        if pattern in model_name_lower:
            return "language"

    for pattern in vision_patterns:
        if pattern in model_name_lower:
            return "vision"

    for pattern in speech_patterns:
        if pattern in model_name_lower:
            return "speech"

    # Default to language if we can't determine
    logger.warning(f"Could not determine model type for {model_name}, defaulting to language")
    return "language"

def get_model_path(model_name: str, framework: str = "pytorch", version: str = "latest", local_path: Optional[str] = None) -> str:
    """Get the path to a model, downloading it if necessary"""
    if local_path and os.path.exists(local_path):
        logger.info(f"Using local model at {local_path}")
        return local_path

    cache_dir = os.path.join(config.cache_dir, framework, model_name, version)
    os.makedirs(cache_dir, exist_ok=True)

    # Handle different model types and sources
    # In a real implementation, you would need to handle various model libraries and sources
    # For now, we'll just return the cache directory
    return cache_dir

def run_subprocess(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run a subprocess command and capture output"""
    logger.debug(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')

def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """Extract metadata from a model file"""
    metadata = {
        "path": model_path,
        "filename": os.path.basename(model_path),
        "size_bytes": os.path.getsize(model_path),
        "hash": get_file_hash(model_path),
        "format": detect_model_format(model_path),
    }
    return metadata

def detect_model_format(model_path: str) -> str:
    """Detect the format of a model file"""
    extension = os.path.splitext(model_path)[1].lower()

    if extension == '.pt' or extension == '.pth':
        return "pytorch"
    elif extension == '.h5' or extension == '.keras':
        return "tensorflow"
    elif extension == '.onnx':
        return "onnx"
    elif extension == '.trt':
        return "tensorrt"
    elif extension == '.bin':
        # Could be a quantized model or other format
        # We'd need more sophisticated detection here
        return "bin"
    else:
        return "unknown"

def create_model_info_file(model_path: str, info: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Create a JSON file with model information"""
    if output_dir is None:
        output_dir = os.path.dirname(model_path)

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(model_path).split('.')[0]
    info_path = os.path.join(output_dir, f"{basename}_info.json")

    # Add timestamp and other metadata
    info["timestamp"] = time.time()
    info["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    return info_path

def prepare_triton_model_repository(model_name: str, model_type: str, version: int = 1) -> str:
    """Prepare a directory structure for a Triton model"""
    triton_repo = config.triton_model_repo
    model_dir = os.path.join(triton_repo, model_name, str(version))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def is_url(path: str) -> bool:
    """Check if a path is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
