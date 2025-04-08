"""
Integration with Triton Inference Server for Ray.
"""
import ray
import time
import logging
import numpy as np
import requests
import json
from typing import Dict, List, Any, Optional, Union
import tritonclient.http
import tritonclient.grpc
from tritonclient.utils import InferenceServerException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("triton_integration")

class TritonClient:
    def __init__(self, config: Dict):
        """
        Initialize Triton client with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config["triton"]
        self.url = self.config["url"]
        self.verbose = self.config["verbose"]
        self.client = None
        self.is_connected = False
        self.protocol = self.config.get("protocol", "http")
        self.model_configs = {}
        self.model_metadata = {}

    def connect(self) -> bool:
        """
        Connect to Triton Inference Server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.protocol == "grpc":
                self.client = tritonclient.grpc.InferenceServerClient(
                    url=self.url, verbose=self.verbose
                )
            else:  # Default to HTTP
                self.client = tritonclient.http.InferenceServerClient(
                    url=self.url, verbose=self.verbose
                )

            if not self.client.is_server_live():
                logger.error(f"Triton server at {self.url} is not live")
                return False

            if not self.client.is_server_ready():
                logger.error(f"Triton server at {self.url} is not ready")
                return False

            self.is_connected = True
            logger.info(f"Connected to Triton server at {self.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            self.is_connected = False
            return False

    def load_model_metadata(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Load metadata for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Model metadata dictionary or None if failed
        """
        if not self.is_connected and not self.connect():
            return None

        model_key = f"{model_name}:{model_version}"
        if model_key in self.model_metadata:
            return self.model_metadata[model_key]

        try:
            metadata = self.client.get_model_metadata(model_name, model_version)
            self.model_metadata[model_key] = metadata
            logger.info(f"Loaded metadata for model {model_name} version {model_version or 'latest'}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to load metadata for model {model_name}: {e}")
            return None

    def load_model_config(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Load configuration for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Model configuration dictionary or None if failed
        """
        if not self.is_connected and not self.connect():
            return None

        model_key = f"{model_name}:{model_version}"
        if model_key in self.model_configs:
            return self.model_configs[model_key]

        try:
            config = self.client.get_model_config(model_name, model_version)
            self.model_configs[model_key] = config
            logger.info(f"Loaded config for model {model_name} version {model_version or 'latest'}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config for model {model_name}: {e}")
            return None

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray],
             model_version: str = "", request_id: str = None,
             output_names: List[str] = None) -> Optional[Dict]:
        """
        Run inference on a model.

        Args:
            model_name: Name of the model
            inputs: Dictionary of input name to numpy array
            model_version: Version of the model (empty for latest)
            request_id: Unique ID for the request
            output_names: List of output names to request

        Returns:
            Dictionary of outputs or None if failed
        """
        if not self.is_connected and not self.connect():
            return None

        # Get model metadata if we don't have it
        metadata = self.load_model_metadata(model_name, model_version)
        if metadata is None:
            logger.error(f"Unable to get metadata for model {model_name}")
            return None

        try:
            # Create input objects
            triton_inputs = []
            for input_name, input_array in inputs.items():
                if self.protocol == "grpc":
                    triton_input = tritonclient.grpc.InferInput(
                        input_name, input_array.shape, self._numpy_to_triton_dtype(input_array.dtype)
                    )
                    triton_input.set_data_from_numpy(input_array)
                else:
                    triton_input = tritonclient.http.InferInput(
                        input_name, input_array.shape, self._numpy_to_triton_dtype(input_array.dtype)
                    )
                    triton_input.set_data_from_numpy(input_array)

                triton_inputs.append(triton_input)

            # Create output objects if specified
            triton_outputs = []
            if output_names:
                for output_name in output_names:
                    if self.protocol == "grpc":
                        triton_outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))
                    else:
                        triton_outputs.append(tritonclient.http.InferRequestedOutput(output_name))

            # Run inference
            start_time = time.time()
            response = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs if output_names else None,
                model_version=model_version,
                request_id=request_id
            )
            inference_time = time.time() - start_time

            # Process results
            result = {
                "model_name": model_name,
                "model_version": response.get_model_version() if hasattr(response, "get_model_version") else model_version,
                "inference_time": inference_time,
                "outputs": {}
            }

            # Get all outputs
            for output_name in (output_names or self._get_output_names(metadata)):
                result["outputs"][output_name] = response.as_numpy(output_name)

            logger.info(f"Inference on model {model_name} completed in {inference_time:.4f}s")
            return result

        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            return None

    def _numpy_to_triton_dtype(self, dtype) -> str:
        """
        Convert numpy dtype to Triton dtype string.

        Args:
            dtype: Numpy dtype

        Returns:
            Triton dtype string
        """
        if np.issubdtype(dtype, np.bool_):
            return "BOOL"
        elif np.issubdtype(dtype, np.uint8):
            return "UINT8"
        elif np.issubdtype(dtype, np.uint16):
            return "UINT16"
        elif np.issubdtype(dtype, np.uint32):
            return "UINT32"
        elif np.issubdtype(dtype, np.uint64):
            return "UINT64"
        elif np.issubdtype(dtype, np.int8):
            return "INT8"
        elif np.issubdtype(dtype, np.int16):
            return "INT16"
        elif np.issubdtype(dtype, np.int32):
            return "INT32"
        elif np.issubdtype(dtype, np.int64):
            return "INT64"
        elif np.issubdtype(dtype, np.float16):
            return "FP16"
        elif np.issubdtype(dtype, np.float32):
            return "FP32"
        elif np.issubdtype(dtype, np.float64):
            return "FP64"
        elif np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_):
            return "BYTES"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _get_output_names(self, metadata) -> List[str]:
        """
        Get output names from model metadata.

        Args:
            metadata: Model metadata

        Returns:
            List of output names
        """
        if self.protocol == "grpc":
            return [output.name for output in metadata.outputs]
        else:
            return [output["name"] for output in metadata["outputs"]]

    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of models loaded on the server.

        Returns:
            List of model information dictionaries
        """
        if not self.is_connected and not self.connect():
            return []

        try:
            if self.protocol == "grpc":
                models = self.client.get_model_repository_index()
                return [{
                    "name": model.name,
                    "version": model.version,
                    "state": model.state,
                } for model in models]
            else:
                models = self.client.get_model_repository_index()
                return models

        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """
        Check if a model is ready.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            True if model is ready, False otherwise
        """
        if not self.is_connected and not self.connect():
            return False

        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception as e:
            logger.error(f"Failed to check if model {model_name} is ready: {e}")
            return False

    def load_model(self, model_name: str, model_version: str = None) -> bool:
        """
        Load a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (None for all versions)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected and not self.connect():
            return False

        try:
            self.client.load_model(model_name, config=None, files=None)
            logger.info(f"Loaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def unload_model(self, model_name: str, model_version: str = None) -> bool:
        """
        Unload a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (None for all versions)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected and not self.connect():
            return False

        try:
            self.client.unload_model(model_name, config=None, files=None)
            logger.info(f"Unloaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False

    def get_server_statistics(self) -> Optional[Dict]:
        """
        Get server statistics.

        Returns:
            Dictionary of server statistics or None if failed
        """
        if not self.is_connected and not self.connect():
            return None

        try:
            stats = self.client.get_inference_statistics()
            return stats
        except Exception as e:
            logger.error(f"Failed to get server statistics: {e}")
            return None

    def get_model_statistics(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Get statistics for a specific model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Dictionary of model statistics or None if failed
        """
        if not self.is_connected and not self.connect():
            return None

        try:
            stats = self.client.get_inference_statistics(model_name, model_version)
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics for model {model_name}: {e}")
            return None

@ray.remote
class TritonService:
    """Ray actor for interacting with Triton Inference Server."""

    def __init__(self, config: Dict):
        """
        Initialize the Triton service.

        Args:
            config: Configuration dictionary
        """
        self.triton_client = TritonClient(config)
        self.config = config
        self.connect()

    def connect(self) -> bool:
        """
        Connect to Triton server.

        Returns:
            True if connection successful, False otherwise
        """
        return self.triton_client.connect()

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray],
              model_version: str = "", request_id: str = None,
              output_names: List[str] = None) -> Optional[Dict]:
        """
        Run inference on a model.

        Args:
            model_name: Name of the model
            inputs: Dictionary of input name to numpy array
            model_version: Version of the model (empty for latest)
            request_id: Unique ID for the request
            output_names: List of output names to request

        Returns:
            Dictionary of outputs or None if failed
        """
        return self.triton_client.infer(
            model_name, inputs, model_version, request_id, output_names
        )

    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of models loaded on the server.

        Returns:
            List of model information dictionaries
        """
        return self.triton_client.get_model_list()

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """
        Check if a model is ready.

        Returns:
            True if model is ready, False otherwise
        """
        return self.triton_client.is_model_ready(model_name, model_version)

    def load_model(self, model_name: str) -> bool:
        """
        Load a model.

        Args:
            model_name: Name of the model

        Returns:
            True if successful, False otherwise
        """
        return self.triton_client.load_model(model_name)

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model.

        Args:
            model_name: Name of the model

        Returns:
            True if successful, False otherwise
        """
        return self.triton_client.unload_model(model_name)

    def get_server_statistics(self) -> Optional[Dict]:
        """
        Get server statistics.

        Returns:
            Dictionary of server statistics or None if failed
        """
        return self.triton_client.get_server_statistics()

    def get_model_statistics(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Get statistics for a specific model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Dictionary of model statistics or None if failed
        """
        return self.triton_client.get_model_statistics(model_name, model_version)

    def get_model_metadata(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Get metadata for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Model metadata or None if failed
        """
        return self.triton_client.load_model_metadata(model_name, model_version)

    def get_model_config(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        Get configuration for a model.

        Args:
            model_name: Name of the model
            model_version: Version of the model (empty for latest)

        Returns:
            Model configuration or None if failed
        """
        return self.triton_client.load_model_config(model_name, model_version)
