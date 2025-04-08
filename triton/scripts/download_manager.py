import os
import re
import json
import tempfile
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from tqdm import tqdm

from config import config, logger
from utils import download_file, is_url, get_file_hash

class DownloadManager:
    """
    Manager for downloading models from various sources (HuggingFace, GitHub, etc.)
    """

    def __init__(self):
        self.cache_dir = config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_model(self, model_source: str, output_dir: Optional[str] = None) -> str:
        """
        Download or locate a model from various sources

        Args:
            model_source: Path, URL, or identifier for the model
            output_dir: Directory to save the model to (default: cache_dir)

        Returns:
            Path to the downloaded model
        """
        if os.path.exists(model_source):
            # Local file or directory
            logger.info(f"Using local model at {model_source}")
            return model_source

        if is_url(model_source):
            # Direct URL download
            return self._download_from_url(model_source, output_dir)

        # Check if it's a known model source
        if model_source.startswith("hf://") or "/" in model_source and len(model_source.split("/")) == 2:
            # HuggingFace model
            if model_source.startswith("hf://"):
                model_id = model_source[5:]
            else:
                model_id = model_source
            return self._download_from_huggingface(model_id, output_dir)

        if model_source.startswith("github://"):
            # GitHub repository
            repo_info = model_source[9:]
            return self._download_from_github(repo_info, output_dir)

        if model_source.startswith("ollama://"):
            # Ollama model
            ollama_model = model_source[9:]
            return self._download_from_ollama(ollama_model, output_dir)

        # Try HuggingFace as a fallback
        logger.info(f"Trying to interpret {model_source} as a HuggingFace model ID")
        return self._download_from_huggingface(model_source, output_dir)

    def _download_from_url(self, url: str, output_dir: Optional[str] = None) -> str:
        """Download a model from a direct URL"""
        if output_dir is None:
            # Create a cache directory based on URL
            url_hash = get_file_hash(url.encode())[:10]
            output_dir = os.path.join(self.cache_dir, "url_downloads", url_hash)
            os.makedirs(output_dir, exist_ok=True)

        # Extract filename from URL
        filename = os.path.basename(url)
        if not filename:
            filename = "model.bin"

        output_path = os.path.join(output_dir, filename)
        download_file(url, output_path)

        # Handle archives
        if filename.endswith('.zip'):
            extract_dir = os.path.join(output_dir, "extracted")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            return self._find_model_file(extract_dir)

        if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            extract_dir = os.path.join(output_dir, "extracted")
            with tarfile.open(output_path) as tar:
                tar.extractall(path=extract_dir)
            return self._find_model_file(extract_dir)

        return output_path

    def _download_from_huggingface(self, model_id: str, output_dir: Optional[str] = None) -> str:
        """Download a model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download

            if output_dir is None:
                output_dir = os.path.join(self.cache_dir, "huggingface", model_id.replace("/", "--"))

            # Download the model files
            logger.info(f"Downloading model {model_id} from HuggingFace Hub")
            model_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=output_dir,
                local_dir=output_dir
            )

            # Find the main model file
            return self._find_model_file(model_dir)

        except ImportError:
            logger.error("huggingface_hub package is required to download from HuggingFace")
            logger.info("Install it with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            raise

    def _download_from_github(self, repo_info: str, output_dir: Optional[str] = None) -> str:
        """Download a model from a GitHub repository"""
        try:
            # Parse repo info (format: owner/repo[@branch_or_tag][/path/to/file])
            parts = repo_info.split("@")
            repo_path = parts[0]

            branch = "main"
            file_path = ""

            if len(parts) > 1:
                # Contains branch or tag
                branch_and_path = parts[1]
                if "/" in branch_and_path:
                    branch, file_path = branch_and_path.split("/", 1)
                else:
                    branch = branch_and_path

            owner, repo = repo_path.split("/", 1)

            if output_dir is None:
                output_dir = os.path.join(self.cache_dir, "github", f"{owner}--{repo}")
                os.makedirs(output_dir, exist_ok=True)

            # Create API URL for either a directory or a specific file
            if file_path:
                # Downloading a specific file
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
                output_file = os.path.join(output_dir, os.path.basename(file_path))
                return download_file(url, output_file)
            else:
                # Downloading a repository archive
                url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
                zip_path = os.path.join(output_dir, f"{repo}_{branch}.zip")
                download_file(url, zip_path)

                # Extract the ZIP file
                extract_dir = os.path.join(output_dir, f"{repo}_{branch}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Find the model file
                return self._find_model_file(extract_dir)

        except Exception as e:
            logger.error(f"Failed to download from GitHub: {e}")
            raise

    def _download_from_ollama(self, model_name: str, output_dir: Optional[str] = None) -> str:
        """
        Extract model files from Ollama local storage
        Note: This requires Ollama to be installed and the model to be pulled
        """
        try:
            # Determine Ollama storage location
            ollama_dir = None
            possible_dirs = [
                os.path.expanduser("~/.ollama/models"),  # Linux/macOS
                "C:\\Users\\Public\\Documents\\Ollama\\models",  # Windows
                "/var/lib/ollama/models",  # Docker/system installation
            ]

            for d in possible_dirs:
                if os.path.exists(d):
                    ollama_dir = d
                    break

            if ollama_dir is None:
                raise FileNotFoundError("Could not find Ollama models directory")

            # Create a target directory
            if output_dir is None:
                output_dir = os.path.join(self.cache_dir, "ollama", model_name)
                os.makedirs(output_dir, exist_ok=True)

            # Look for model files
            model_files = []
            for root, _, files in os.walk(ollama_dir):
                for file in files:
                    if model_name in file and (file.endswith('.bin') or file.endswith('.gguf')):
                        model_files.append(os.path.join(root, file))

            if not model_files:
                logger.error(f"No Ollama model files found for {model_name}")
                logger.info("Make sure you've pulled the model with 'ollama pull {model_name}'")
                raise FileNotFoundError(f"No model files found for {model_name}")

            # Copy the model file to output directory
            model_file = model_files[0]  # Use the first matching file
            target_file = os.path.join(output_dir, os.path.basename(model_file))
            shutil.copy2(model_file, target_file)
            logger.info(f"Copied Ollama model from {model_file} to {target_file}")

            return target_file

        except Exception as e:
            logger.error(f"Failed to extract from Ollama: {e}")
            raise

    def _find_model_file(self, directory: str) -> str:
        """
        Find a model file in a directory based on common extensions and names
        """
        # Priority order for model files
        extensions = [
            '.safetensors', '.bin', '.pt', '.pth', '.onnx', '.trt',
            '.gguf', '.ggml', '.pb', '.h5', '.keras'
        ]

        # Common model file names
        common_names = [
            'model', 'pytorch_model', 'weights', 'best', 'final',
            'encoder', 'decoder', 'generator', 'discriminator'
        ]

        # First, look for config files that might point to the model
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'config.json':
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            config_data = json.load(f)
                            # Extract model file information if available
                            if 'model_file' in config_data:
                                model_file = os.path.join(root, config_data['model_file'])
                                if os.path.exists(model_file):
                                    return model_file
                    except:
                        pass  # Ignore errors reading config files

        # Look for model files by extension and name
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append((file_path, os.path.getsize(file_path)))

        # Sort by size (largest first) - models are typically large files
        all_files.sort(key=lambda x: x[1], reverse=True)

        # First, try to find files with common extensions
        for file_path, _ in all_files:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in extensions:
                return file_path

        # If no matches by extension, look for common names
        for name in common_names:
            pattern = re.compile(rf'{name}.*', re.IGNORECASE)
            for file_path, _ in all_files:
                if pattern.match(os.path.basename(file_path)):
                    return file_path

        # If still no match, return the largest file (likely the model)
        if all_files:
            logger.warning(f"Could not find a specific model file in {directory}, using largest file")
            return all_files[0][0]

        raise FileNotFoundError(f"No model files found in {directory}")
