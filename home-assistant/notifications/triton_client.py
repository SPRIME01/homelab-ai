import logging
import numpy as np
import json
from typing import Dict, Any, List, Union

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger("triton_client")

class TritonClient:
    def __init__(self, config: Dict[str, Any]):
        self.url = config.get("url", "localhost:8001")
        self.timeout = config.get("timeout_ms", 30000)
        self.client = None
        self.models_info = {}

    async def initialize(self):
        """Initialize connection to Triton server"""
        try:
            self.client = httpclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                concurrency=4
            )

            # Check server is alive
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {self.url} is not live")

            # Check server is ready
            if not self.client.is_server_ready():
                raise ConnectionError(f"Triton server at {self.url} is not ready")

            logger.info(f"Connected to Triton server at {self.url}")

            # Cache model information
            self._cache_models_info()

        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise

    def _cache_models_info(self):
        """Cache information about available models"""
        try:
            model_repository_index = self.client.get_model_repository_index()

            for model_info in model_repository_index:
                model_name = model_info["name"]
                self.models_info[model_name] = {
                    "name": model_name,
                    "versions": [str(v) for v in model_info.get("versions", [])],
                    "platform": model_info.get("platform", "")
                }

                # Get more detailed model metadata
                try:
                    metadata = self.client.get_model_metadata(model_name)
                    self.models_info[model_name]["inputs"] = [
                        {
                            "name": inp.name,
                            "datatype": inp.datatype,
                            "shape": inp.shape
                        }
                        for inp in metadata.inputs
                    ]
                    self.models_info[model_name]["outputs"] = [
                        {
                            "name": out.name,
                            "datatype": out.datatype,
                            "shape": out.shape
                        }
                        for out in metadata.outputs
                    ]
                except Exception as e:
                    logger.warning(f"Could not get metadata for model {model_name}: {e}")

            logger.info(f"Cached information for {len(self.models_info)} models")

        except Exception as e:
            logger.error(f"Failed to cache model information: {e}")

    async def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        version: str = "",
        output_names: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """Run inference on a model"""
        try:
            if not self.client:
                await self.initialize()

            # Prepare inference inputs
            triton_inputs = []
            for name, data in inputs.items():
                triton_inputs.append(httpclient.InferInput(name, data.shape, self._numpy_to_triton_dtype(data.dtype)))
                triton_inputs[-1].set_data_from_numpy(data)

            # Prepare output specification
            triton_outputs = []
            if output_names:
                for name in output_names:
                    triton_outputs.append(httpclient.InferRequestedOutput(name))

            # Run inference
            result = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
                model_version=version,
                client_timeout=self.timeout
            )

            # Process results
            output_dict = {}
            for output_name in (output_names or result.get_output_names()):
                output_dict[output_name] = result.as_numpy(output_name)

            return output_dict

        except InferenceServerException as e:
            logger.error(f"Inference error with model {model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def _numpy_to_triton_dtype(self, dtype):
        """Convert numpy dtype to Triton dtype string"""
        if np.issubdtype(dtype, np.float32):
            return "FP32"
        elif np.issubdtype(dtype, np.float64):
            return "FP64"
        elif np.issubdtype(dtype, np.float16):
            return "FP16"
        elif np.issubdtype(dtype, np.int64):
            return "INT64"
        elif np.issubdtype(dtype, np.int32):
            return "INT32"
        elif np.issubdtype(dtype, np.int16):
            return "INT16"
        elif np.issubdtype(dtype, np.int8):
            return "INT8"
        elif np.issubdtype(dtype, np.uint8):
            return "UINT8"
        elif np.issubdtype(dtype, np.bool_):
            return "BOOL"
        elif np.issubdtype(dtype, np.str_):
            return "STRING"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    async def close(self):
        """Close connection to Triton server"""
        if self.client:
            self.client.close()
            logger.info("Closed connection to Triton server")
