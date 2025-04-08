import os
import sys
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import tempfile
import time
import shutil

from config import config, logger
from utils import (
    verify_dependencies, get_file_hash, detect_model_type,
    get_model_metadata, create_model_info_file
)
from download_manager import DownloadManager

class ModelConverter:
    """
    Convert models between different formats with focus on ONNX as an intermediate format
    """

    def __init__(self):
        self.download_manager = DownloadManager()
        verify_dependencies()

    def convert(
        self,
        model_source: str,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 13,
        simplify: bool = True,
        target_format: str = "onnx",
    ) -> str:
        """
        Convert a model to ONNX format

        Args:
            model_source: Path or identifier for the model
            output_dir: Directory to save the output model
            model_type: Type of model (language, vision, speech)
            framework: Source framework (pytorch, tensorflow, auto)
            input_shapes: Dictionary of input names to shapes
            dynamic_axes: Dictionary of input/output names to dynamic axes
            opset_version: ONNX opset version
            simplify: Whether to simplify the ONNX model
            target_format: Target format (currently only 'onnx' supported)

        Returns:
            Path to the converted model
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(config.output_dir, "onnx")

        os.makedirs(output_dir, exist_ok=True)

        # Download or locate the model
        model_path = self.download_manager.get_model(model_source)

        # Detect model properties if not provided
        if framework is None:
            framework = self._detect_framework(model_path)

        if model_type is None:
            model_type = detect_model_type(os.path.basename(model_path))

        # Get default configurations for model type
        model_config = config.get_model_type_config(model_type)
        if dynamic_axes is None and "dynamic_axes" in model_config:
            dynamic_axes = model_config["dynamic_axes"]

        # Create a meaningful output filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(output_dir, f"{model_name}.onnx")

        # Convert based on source framework
        if framework.lower() == "pytorch":
            return self._convert_pytorch_to_onnx(
                model_path, output_path, model_type, input_shapes, dynamic_axes,
                opset_version, simplify
            )
        elif framework.lower() == "tensorflow":
            return self._convert_tensorflow_to_onnx(
                model_path, output_path, model_type, input_shapes, dynamic_axes,
                opset_version, simplify
            )
        elif framework.lower() in ["gguf", "ggml"]:
            return self._convert_gguf_to_onnx(
                model_path, output_path, model_type, input_shapes
            )
        elif framework.lower() == "safetensors":
            return self._convert_safetensors_to_onnx(
                model_path, output_path, model_type, input_shapes, dynamic_axes,
                opset_version, simplify
            )
        else:
            logger.error(f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")

    def _detect_framework(self, model_path: str) -> str:
        """Detect the framework of a model based on file extension and contents"""
        extension = os.path.splitext(model_path)[1].lower()

        if extension in ['.pt', '.pth']:
            return "pytorch"
        elif extension in ['.h5', '.keras', '.pb', '.saved_model']:
            return "tensorflow"
        elif extension == '.onnx':
            return "onnx"
        elif extension == '.safetensors':
            return "safetensors"
        elif extension in ['.gguf', '.ggml']:
            return "gguf"
        else:
            # Try to load with PyTorch
            try:
                import torch
                model = torch.load(model_path, map_location="cpu")
                return "pytorch"
            except:
                pass

            # Try to load with TensorFlow
            try:
                import tensorflow as tf
                model = tf.saved_model.load(model_path)
                return "tensorflow"
            except:
                pass

        logger.warning(f"Could not detect framework for {model_path}, defaulting to PyTorch")
        return "pytorch"

    def _convert_pytorch_to_onnx(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 13,
        simplify: bool = True
    ) -> str:
        """Convert a PyTorch model to ONNX"""
        import torch
        from torch.autograd import Variable

        logger.info(f"Converting PyTorch model {model_path} to ONNX")

        # Create default input shapes if not provided
        if input_shapes is None:
            input_shapes = self._get_default_input_shapes(model_type)

        # Load the model
        try:
            # First try directly loading the state dict
            state_dict = torch.load(model_path, map_location="cpu")

            # If it's just a state dict and not the full model
            if isinstance(state_dict, dict) and all(isinstance(x, torch.Tensor) for x in state_dict.values()):
                # We need the model class to load the state dict
                # For transformers models, we can try to load from HuggingFace
                try:
                    from transformers import AutoModel, AutoConfig

                    # Try to find a config file in the same directory
                    config_path = os.path.join(os.path.dirname(model_path), "config.json")
                    if os.path.exists(config_path):
                        config = AutoConfig.from_pretrained(config_path)
                        model = AutoModel.from_config(config)
                        model.load_state_dict(state_dict)
                    else:
                        logger.error("Could not find model config to load state dict")
                        raise ValueError("Model config not found for state dict")
                except ImportError:
                    logger.error("transformers library required to load this model")
                    raise
            else:
                # It's a full model or a checkpoint with model included
                model = state_dict

            # Make sure model is in eval mode
            model.eval()

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise

        # Create dummy inputs based on input shapes
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape)

        # If no input names were provided, use a default input
        if not dummy_inputs:
            dummy_inputs = torch.randn(1, 3, 224, 224)

        # Export to ONNX
        try:
            # If dynamic_axes is not provided, create a default one based on model type
            if dynamic_axes is None:
                dynamic_axes = self._get_default_dynamic_axes(model_type)

            input_names = list(dummy_inputs.keys()) if isinstance(dummy_inputs, dict) else ["input"]
            output_names = ["output"]

            # If dummy_inputs is a dictionary, unpack it
            if isinstance(dummy_inputs, dict):
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()),
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            else:
                torch.onnx.export(
                    model,
                    dummy_inputs,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )

            logger.info(f"Exported ONNX model to {output_path}")

            # Simplify the model if requested
            if simplify:
                output_path = self._simplify_onnx(output_path)

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "original_format": "pytorch",
                "model_type": model_type,
                "opset_version": opset_version,
                "input_shapes": input_shapes,
                "dynamic_axes": dynamic_axes,
                "simplify": simplify
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to export PyTorch model to ONNX: {e}")
            raise

    def _convert_tensorflow_to_onnx(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 13,
        simplify: bool = True
    ) -> str:
        """Convert a TensorFlow model to ONNX"""
        try:
            import tensorflow as tf
            import tf2onnx
        except ImportError:
            logger.error("tf2onnx required for TensorFlow to ONNX conversion")
            logger.info("Install with: pip install tf2onnx")
            raise

        logger.info(f"Converting TensorFlow model {model_path} to ONNX")

        # Create default input shapes if not provided
        if input_shapes is None:
            input_shapes = self._get_default_input_shapes(model_type)

        # Load the model
        try:
            # Try different ways to load the model
            if os.path.isdir(model_path):
                # SavedModel format
                model = tf.saved_model.load(model_path)
            elif model_path.endswith(".h5") or model_path.endswith(".keras"):
                # Keras H5 format
                model = tf.keras.models.load_model(model_path)
            else:
                logger.error(f"Unsupported TensorFlow model format: {model_path}")
                raise ValueError(f"Unsupported TensorFlow model format: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise

        # Convert to ONNX
        try:
            if hasattr(model, "signatures"):
                # SavedModel with signatures
                model_proto, _ = tf2onnx.convert.from_saved_model(
                    model_path,
                    opset=opset_version,
                    output_path=output_path
                )
            else:
                # Keras model
                input_spec = []
                for name, shape in input_shapes.items():
                    input_spec.append(tf.TensorSpec(shape, tf.float32, name=name))

                model_proto, _ = tf2onnx.convert.from_keras(
                    model,
                    input_signature=input_spec,
                    opset=opset_version,
                    output_path=output_path
                )

            logger.info(f"Exported ONNX model to {output_path}")

            # Simplify the model if requested
            if simplify:
                output_path = self._simplify_onnx(output_path)

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "original_format": "tensorflow",
                "model_type": model_type,
                "opset_version": opset_version,
                "input_shapes": input_shapes,
                "dynamic_axes": dynamic_axes,
                "simplify": simplify
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to export TensorFlow model to ONNX: {e}")
            raise

    def _convert_gguf_to_onnx(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> str:
        """Convert a GGUF model to ONNX format"""
        try:
            # Check if the required libraries are installed
            from ctransformers import AutoModelForCausalLM
            import torch
        except ImportError:
            logger.error("ctransformers and torch required for GGUF to ONNX conversion")
            logger.info("Install with: pip install ctransformers torch")
            raise

        logger.info(f"Converting GGUF model {model_path} to ONNX")

        try:
            # Load the model with ctransformers
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama"  # We assume GGUF models are LLaMA-based
            )

            # Create a PyTorch wrapper for the model
            class LLMWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids):
                    return torch.tensor(self.model.generate(input_ids.tolist()[0]))

            torch_model = LLMWrapper(model)
            torch_model.eval()

            # Create dummy input
            if input_shapes is None:
                input_shapes = {"input_ids": [1, 10]}  # Default sequence length of 10

            dummy_input = torch.randint(0, 32000, input_shapes["input_ids"])

            # Export to ONNX
            dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}}

            torch.onnx.export(
                torch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["output"],
                dynamic_axes=dynamic_axes
            )

            logger.info(f"Exported ONNX model to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "original_format": "gguf",
                "model_type": model_type,
                "input_shapes": input_shapes,
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to convert GGUF model to ONNX: {e}")
            raise

    def _convert_safetensors_to_onnx(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 13,
        simplify: bool = True
    ) -> str:
        """Convert a safetensors model to ONNX"""
        try:
            from safetensors import safe_open
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError:
            logger.error("safetensors, torch, and transformers required for safetensors conversion")
            logger.info("Install with: pip install safetensors torch transformers")
            raise

        logger.info(f"Converting safetensors model {model_path} to ONNX")

        try:
            # Try to find the model configuration
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
            if not os.path.exists(config_path):
                logger.error("Could not find config.json for safetensors model")
                raise ValueError("config.json not found for safetensors model")

            # Load the model configuration and create the model
            config = AutoConfig.from_pretrained(config_path)
            model = AutoModel.from_config(config)

            # Load weights from safetensors
            with safe_open(model_path, framework="pt", device="cpu") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}

            # Load state dict into the model
            model.load_state_dict(state_dict)
            model.eval()

            # Create default input shapes if not provided
            if input_shapes is None:
                input_shapes = self._get_default_input_shapes(model_type)

            # Create dummy inputs based on input shapes
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randint(0, 100, shape) if "ids" in name else torch.randn(*shape)

            # If dynamic_axes is not provided, create a default one
            if dynamic_axes is None:
                dynamic_axes = self._get_default_dynamic_axes(model_type)

            # Export to ONNX
            input_names = list(dummy_inputs.keys())
            output_names = ["output"]

            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            logger.info(f"Exported ONNX model to {output_path}")

            # Simplify the model if requested
            if simplify:
                output_path = self._simplify_onnx(output_path)

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "original_format": "safetensors",
                "model_type": model_type,
                "opset_version": opset_version,
                "input_shapes": input_shapes,
                "dynamic_axes": dynamic_axes,
                "simplify": simplify
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to convert safetensors model to ONNX: {e}")
            raise

    def _simplify_onnx(self, model_path: str) -> str:
        """Simplify an ONNX model with onnxsim"""
        try:
            import onnx
            from onnxsim import simplify
        except ImportError:
            logger.warning("onnxsim not found, skipping model simplification")
            logger.info("Install with: pip install onnxsim")
            return model_path

        logger.info(f"Simplifying ONNX model: {model_path}")

        try:
            # Load the model
            model = onnx.load(model_path)

            # Simplify the model
            model_simplified, check = simplify(model)

            if not check:
                logger.warning("Simplified ONNX model could not be validated")
                return model_path

            # Save the simplified model (overwrite the original)
            onnx.save(model_simplified, model_path)
            logger.info(f"Simplified model saved to {model_path}")

            return model_path

        except Exception as e:
            logger.error(f"Failed to simplify ONNX model: {e}")
            return model_path

    def _get_default_input_shapes(self, model_type: str) -> Dict[str, List[int]]:
        """Get default input shapes based on model type"""
        model_config = config.get_model_type_config(model_type)

        if model_type == "language":
            seq_length = model_config.get("default_seq_length", 512)
            return {
                "input_ids": [1, seq_length],
                "attention_mask": [1, seq_length],
            }

        elif model_type == "vision":
            image_size = model_config.get("default_image_size", [224, 224])
            return {
                "input": [1, 3, image_size[0], image_size[1]],
            }

        elif model_type == "speech":
            return {
                "input": [1, 80, 3000],  # [batch, features, time]
            }

        else:
            return {"input": [1, 3, 224, 224]}  # Default to vision

    def _get_default_dynamic_axes(self, model_type: str) -> Dict[str, Dict[int, str]]:
        """Get default dynamic axes based on model type"""
        model_config = config.get_model_type_config(model_type)

        if "dynamic_axes" in model_config:
            return model_config["dynamic_axes"]

        if model_type == "language":
            return {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "output": {0: "batch", 1: "sequence"}
            }

        elif model_type == "vision":
            return {
                "input": {0: "batch"},
                "output": {0: "batch"}
            }

        elif model_type == "speech":
            return {
                "input": {0: "batch", 2: "time"},
                "output": {0: "batch", 1: "time"}
            }

        else:
            return {
                "input": {0: "batch"},
                "output": {0: "batch"}
            }


def main():
    """Command line interface for the model converter"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert models to ONNX format')
    parser.add_argument('model', type=str, help='Path or identifier for the input model')
    parser.add_argument('--output', '-o', type=str, help='Output directory or file path')
    parser.add_argument('--framework', '-f', type=str, help='Source framework (pytorch, tensorflow, auto)')
    parser.add_argument('--model-type', '-t', type=str, help='Model type (language, vision, speech)')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true', help='Disable model simplification')

    args = parser.parse_args()

    converter = ModelConverter()
    output_path = converter.convert(
        model_source=args.model,
        output_dir=args.output,
        model_type=args.model_type,
        framework=args.framework,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )

    print(f"Model converted successfully: {output_path}")


if __name__ == "__main__":
    main()
