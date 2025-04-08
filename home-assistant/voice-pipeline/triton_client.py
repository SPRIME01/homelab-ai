import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
import aiohttp
import json

logger = logging.getLogger("triton_client")

class TritonClient:
    """Client for interacting with Triton Inference Server."""

    def __init__(self, url: str, timeout: float = 30.0):
        """
        Initialize the Triton client.

        Args:
            url: URL of the Triton Inference Server
            timeout: Timeout for requests in seconds
        """
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def infer(self, model_name: str,
                   inputs: Dict[str, np.ndarray],
                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform inference using Triton's HTTP API.

        Args:
            model_name: Name of the model to use
            inputs: Dictionary of input name to numpy array
            parameters: Optional parameters for the model

        Returns:
            Dictionary of output name to output data
        """
        try:
            session = await self._get_session()

            # Construct the request payload
            payload = {
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
                payload["inputs"].append(input_data)

            # Add parameters if specified
            if parameters:
                payload["parameters"] = parameters

            # Send the request
            url = f"{self.url}/v2/models/{model_name}/infer"
            async with session.post(
                url,
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_response(result)
                else:
                    error_text = await response.text()
                    logger.error(f"Inference request failed: {response.status}, {error_text}")
                    raise RuntimeError(f"Inference request failed: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during inference: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def _numpy_to_triton_dtype(self, dtype) -> str:
        """Convert numpy dtype to Triton dtype string."""
        mapping = {
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
            np.str_: "BYTES",
            np.bytes_: "BYTES",
        }

        if dtype in mapping:
            return mapping[dtype]

        # Handle string/object types
        if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_) or dtype == np.dtype('O'):
            return "BYTES"

        logger.warning(f"Unknown dtype mapping for {dtype}, using FP32")
        return "FP32"

    def _serialize_array(self, array: np.ndarray) -> Union[List, str]:
        """Serialize numpy array to format expected by Triton HTTP API."""
        if array.dtype == np.object_ or np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
            # Handle string arrays
            if array.size == 1:
                # Single string
                value = array.item()
                if isinstance(value, bytes):
                    return value.decode('utf-8')
                return str(value)
            else:
                # Multiple strings
                result = []
                for item in array.flatten():
                    if isinstance(item, bytes):
                        result.append(item.decode('utf-8'))
                    else:
                        result.append(str(item))
                return result
        else:
            # Numeric arrays
            return array.flatten().tolist()

    def _process_response(self, response: Dict) -> Dict[str, np.ndarray]:
        """Process response from Triton, converting outputs to numpy arrays."""
        result = {}

        if "outputs" in response:
            for output in response["outputs"]:
                name = output["name"]
                shape = output["shape"]
                dtype = self._triton_to_numpy_dtype(output["datatype"])

                # Convert data to numpy array
                if output["datatype"] == "BYTES":
                    # Handle string/bytes data
                    if "data" in output:
                        string_data = output["data"]
                        if isinstance(string_data, list):
                            # Multiple strings
                            result[name] = np.array(string_data, dtype=np.object_)
                        else:
                            # Single string
                            result[name] = np.array([string_data], dtype=np.object_)
                    else:
                        # Empty data
                        result[name] = np.array([], dtype=np.object_)
                else:
                    # Numeric data
                    data = np.array(output.get("data", []), dtype=dtype)
                    if len(shape) > 1:
                        data = data.reshape(shape)
                    result[name] = data

        return result

    def _triton_to_numpy_dtype(self, dtype_str: str):
        """Convert Triton dtype string to numpy dtype."""
        mapping = {
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

        return mapping.get(dtype_str, np.float32)

    async def cleanup(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
