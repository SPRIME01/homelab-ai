"""Client for interacting with Triton Inference Server."""
import logging
import time
from typing import Dict, Any, Optional, List, Union
import numpy as np
import aiohttp
import json

_LOGGER = logging.getLogger(__name__)

class TritonClient:
    """Client for communicating with Triton Inference Server."""

    def __init__(self, url: str, timeout: float = 30.0):
        """Initialize the Triton client.

        Args:
            url: URL of the Triton Inference Server
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.server_metadata = None
        self.model_metadata = {}
        self.available_models = []

    async def initialize(self) -> bool:
        """Initialize the client and check server health."""
        try:
            self.session = aiohttp.ClientSession()

            # Check server health
            health_url = f"{self.url}/v2/health/ready"
            async with self.session.get(health_url, timeout=self.timeout) as response:
                if response.status != 200:
                    _LOGGER.error("Triton server is not ready: %s", response.status)
                    return False

            # Get server metadata
            metadata_url = f"{self.url}/v2"
            async with self.session.get(metadata_url, timeout=self.timeout) as response:
                if response.status == 200:
                    self.server_metadata = await response.json()
                else:
                    _LOGGER.warning("Failed to get server metadata: %s", response.status)

            # Get available models
            models_url = f"{self.url}/v2/models"
            async with self.session.get(models_url, timeout=self.timeout) as response:
                if response.status == 200:
                    models_data = await response.json()
                    self.available_models = [model["name"] for model in models_data["models"]]
                    _LOGGER.info("Available models: %s", self.available_models)
                else:
                    _LOGGER.warning("Failed to get available models: %s", response.status)

            return True

        except aiohttp.ClientError as err:
            _LOGGER.error("Error connecting to Triton server: %s", err)
            return False
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Unexpected error initializing Triton client: %s", err)
            return False

    async def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]

        try:
            url = f"{self.url}/v2/models/{model_name}"
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    metadata = await response.json()
                    self.model_metadata[model_name] = metadata
                    return metadata
                else:
                    _LOGGER.warning(
                        "Failed to get metadata for model %s: %s",
                        model_name, response.status
                    )
                    return None

        except aiohttp.ClientError as err:
            _LOGGER.error("Error getting model metadata: %s", err)
            return None
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Unexpected error getting model metadata: %s", err)
            return None

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run inference using the Triton server.

        Args:
            model_name: Name of the model to use
            inputs: Dictionary of input name to numpy array
            outputs: Optional list of output names to request
            parameters: Optional parameters for inference

        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()

        try:
            # Create inference request
            request = {
                "inputs": [],
                "outputs": []
            }

            # Add inputs
            for name, array in inputs.items():
                input_data = {
                    "name": name,
                    "shape": array.shape,
                    "datatype": self._numpy_to_triton_dtype(array.dtype),
                    "data": self._serialize_array(array)
                }
                request["inputs"].append(input_data)

            # Add requested outputs
            if outputs:
                for output_name in outputs:
                    request["outputs"].append({"name": output_name})

            # Add parameters if provided
            if parameters:
                request["parameters"] = parameters

            # Send inference request
            url = f"{self.url}/v2/models/{model_name}/infer"
            async with self.session.post(url, json=request, timeout=self.timeout) as response:
                if response.status == 200:
                    result = await response.json()

                    # Process outputs
                    outputs_dict = {}
                    for output in result.get("outputs", []):
                        name = output["name"]
                        shape = output["shape"]
                        dtype = self._triton_to_numpy_dtype(output["datatype"])
                        data = self._deserialize_array(output["data"], dtype, shape)
                        outputs_dict[name] = data

                    inference_time = time.time() - start_time

                    return {
                        "outputs": outputs_dict,
                        "model_name": model_name,
                        "inference_time": inference_time,
                        "status": "success"
                    }
                else:
                    error_text = await response.text()
                    _LOGGER.error(
                        "Inference request failed for model %s: %s - %s",
                        model_name, response.status, error_text
                    )
                    return {
                        "status": "error",
                        "error": f"HTTP error {response.status}: {error_text}",
                        "model_name": model_name,
                        "inference_time": time.time() - start_time
                    }

        except aiohttp.ClientError as err:
            _LOGGER.error("HTTP error during inference: %s", err)
            return {
                "status": "error",
                "error": f"Connection error: {err}",
                "model_name": model_name,
                "inference_time": time.time() - start_time
            }
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Unexpected error during inference: %s", err)
            return {
                "status": "error",
                "error": f"Unexpected error: {err}",
                "model_name": model_name,
                "inference_time": time.time() - start_time
            }

    def _numpy_to_triton_dtype(self, dtype) -> str:
        """Convert numpy dtype to Triton dtype string."""
        dtype_map = {
            np.bool_: "BOOL",
            np.int8: "INT8",
            np.int16: "INT16",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint8: "UINT8",
            np.uint16: "UINT16",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.float16: "FP16",
            np.float32: "FP32",
            np.float64: "FP64",
            np.object_: "BYTES",
            np.dtype('S'): "BYTES",
            np.dtype('U'): "BYTES",
        }

        if dtype in dtype_map:
            return dtype_map[dtype]

        # For strings and other object types
        if np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_) or dtype == np.dtype('O'):
            return "BYTES"

        _LOGGER.warning("Unknown dtype mapping for %s, using FP32", dtype)
        return "FP32"

    def _serialize_array(self, array: np.ndarray) -> Union[List, str]:
        """Serialize numpy array for Triton HTTP API."""
        if array.dtype == np.object_ or np.issubdtype(array.dtype, np.string_) or np.issubdtype(array.dtype, np.unicode_):
            # Handle string/object data
            if array.size == 1:
                # Single string
                return str(array.item())
            else:
                # Multiple strings
                return [str(item) for item in array.flatten()]
        else:
            # Numeric data
            return array.flatten().tolist()

    def _deserialize_array(self, data: Union[List, str], dtype, shape: List[int]) -> np.ndarray:
        """Deserialize array data from Triton response."""
        if dtype == np.object_ or np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_):
            # Handle string data
            if isinstance(data, str):
                return np.array([data], dtype=dtype)
            else:
                arr = np.array(data, dtype=dtype)
                if len(shape) > 1:
                    return arr.reshape(shape)
                return arr
        else:
            # Handle numeric data
            arr = np.array(data, dtype=dtype)
            if len(shape) > 1:
                return arr.reshape(shape)
            return arr

    def _triton_to_numpy_dtype(self, dtype_str: str):
        """Convert Triton dtype string to numpy dtype."""
        dtype_map = {
            "BOOL": np.bool_,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
            "UINT64": np.uint64,
            "FP16": np.float16,
            "FP32": np.float32,
            "FP64": np.float64,
            "BYTES": np.object_,
        }
        return dtype_map.get(dtype_str, np.float32)

    async def close(self):
        """Close the client and free resources."""
        if self.session:
            await self.session.close()
            self.session = None
