import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import tempfile
import json
import time
import shutil
import torch

from config import config, logger
from utils import (
    verify_dependencies, get_file_hash, detect_model_type,
    get_model_metadata, create_model_info_file
)
from download_manager import DownloadManager

class ModelQuantizer:
    """
    Quantize models to lower precision formats (FP16, INT8, 4-bit)
    for efficient inference on NVIDIA Jetson AGX Orin
    """

    def __init__(self):
        self.download_manager = DownloadManager()
        verify_dependencies()

        # Add LLM-specific settings
        self.llm_config = {
            "max_sequence_length": 2048,
            "load_in_8bit": True,  # Use 8-bit loading for initial load
            "trust_remote_code": True,
            "use_bettertransformer": True,  # Use BetterTransformer for optimized inference
            "device_map": "auto",  # Automatic device mapping for large models
            "max_memory": {0: "28GiB"},  # Reserve memory for Jetson AGX Orin
            "offload_folder": "offload_folder",  # Folder for weight offloading
        }

    def quantize(
        self,
        model_source: str,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        precision: str = "fp16",
        calibration_data: Optional[str] = None,
        calibration_samples: int = 100,
        quantization_method: str = "static",
        save_onnx: bool = True,
    ) -> str:
        """
        Quantize a model to lower precision

        Args:
            model_source: Path or identifier for the model
            output_dir: Directory to save the quantized model
            model_type: Type of model (language, vision, speech)
            precision: Target precision (fp16, int8, 4bit)
            calibration_data: Path to calibration data for post-training quantization
            calibration_samples: Number of calibration samples to use
            quantization_method: Quantization method (static, dynamic, aware)
            save_onnx: Whether to save as ONNX format (otherwise save in native format)

        Returns:
            Path to the quantized model
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(config.output_dir, "quantized", precision)

        os.makedirs(output_dir, exist_ok=True)

        # Download or locate the model
        model_path = self.download_manager.get_model(model_source)

        # Detect model properties if not provided
        if model_type is None:
            model_type = detect_model_type(os.path.basename(model_path))

        # Create a meaningful output filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(output_dir, f"{model_name}_{precision}")
        if save_onnx:
            output_path += ".onnx"

        # Quantize based on target precision
        if precision == "fp16":
            return self._quantize_fp16(model_path, output_path, model_type, save_onnx)
        elif precision == "int8":
            return self._quantize_int8(model_path, output_path, model_type,
                                     calibration_data, calibration_samples, save_onnx)
        elif precision.lower() in ["4bit", "4-bit", "int4"]:
            return self._quantize_4bit(model_path, output_path, model_type,
                                     quantization_method, save_onnx)
        else:
            logger.error(f"Unsupported precision mode: {precision}")
            raise ValueError(f"Unsupported precision mode: {precision}")

    def quantize_llm(
        self,
        model_source: str,
        output_dir: Optional[str] = None,
        quantization_method: str = "gptq",
        bits: int = 4,
        group_size: int = 128,
        use_triton: bool = True,
        calibration_dataset: Optional[str] = None,
        validation_dataset: Optional[str] = None,
        perplexity_threshold: float = 1.1,  # Max allowed perplexity increase
        memory_efficient: bool = True
    ) -> str:
        """
        Quantize a large language model with advanced settings

        Args:
            model_source: HuggingFace model ID or local path
            output_dir: Output directory
            quantization_method: Quantization method (gptq, awq, gguf)
            bits: Number of bits for quantization (default: 4)
            group_size: Size of quantization groups
            use_triton: Generate Triton configuration
            calibration_dataset: Path to calibration dataset
            validation_dataset: Path to validation dataset
            perplexity_threshold: Maximum allowed perplexity degradation
            memory_efficient: Use memory efficient techniques
        """
        try:
            # Setup memory efficient loading
            if memory_efficient:
                self._setup_memory_efficient_loading()

            # Load and prepare model
            model_path = self._prepare_llm(model_source)

            # Quantize based on method
            if quantization_method == "gptq":
                output_path = self._quantize_gptq_enhanced(
                    model_path, output_dir, bits, group_size,
                    calibration_dataset, validation_dataset,
                    perplexity_threshold
                )
            elif quantization_method == "awq":
                output_path = self._quantize_awq_enhanced(
                    model_path, output_dir, bits, group_size,
                    calibration_dataset, validation_dataset,
                    perplexity_threshold
                )
            elif quantization_method == "gguf":
                output_path = self._quantize_gguf(
                    model_path, output_dir, bits,
                    validation_dataset, perplexity_threshold
                )
            else:
                raise ValueError(f"Unsupported quantization method: {quantization_method}")

            # Generate Triton configuration if requested
            if use_triton:
                self._generate_triton_config(
                    output_path,
                    quantization_method,
                    bits
                )

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize LLM: {e}")
            raise

    def _setup_memory_efficient_loading(self):
        """Configure memory efficient model loading"""
        import torch
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)

        # Configure disk offload
        os.makedirs(self.llm_config["offload_folder"], exist_ok=True)

        # Enable attention memory optimizations
        if self.llm_config["use_bettertransformer"]:
            from optimum.bettertransformer import BetterTransformer
            self.better_transformer = BetterTransformer

    def _prepare_llm(self, model_source: str) -> str:
        """Prepare LLM for quantization with memory efficient loading"""
        try:
            import accelerate
            from transformers import AutoConfig

            # Get model config first
            config = AutoConfig.from_pretrained(
                model_source,
                trust_remote_code=self.llm_config["trust_remote_code"]
            )

            # Calculate memory requirements
            model_size_gb = self._estimate_model_size(config)
            logger.info(f"Estimated model size: {model_size_gb:.2f}GB")

            # Adjust loading strategy based on size
            if model_size_gb > 20:  # For very large models
                return self._load_large_model(model_source, config)
            else:
                return self._load_normal_model(model_source)

        except Exception as e:
            logger.error(f"Failed to prepare LLM: {e}")
            raise

    def _estimate_model_size(self, config) -> float:
        """Estimate model size in GB based on config"""
        # Basic estimation based on parameters
        num_params = config.num_parameters if hasattr(config, 'num_parameters') else \
                    config.num_hidden_layers * config.hidden_size * config.hidden_size * 4

        # Account for different precisions
        bytes_per_param = 2  # FP16
        estimated_size = (num_params * bytes_per_param) / (1024 * 1024 * 1024)  # Convert to GB
        return estimated_size

    def _validate_quantization(
        self,
        original_model,
        quantized_model,
        validation_dataset: str,
        perplexity_threshold: float
    ) -> bool:
        """Validate quantized model performance"""
        try:
            from datasets import load_dataset
            import evaluate

            # Load evaluation metric
            perplexity = evaluate.load("perplexity", module_type="metric")

            # Load validation dataset
            if validation_dataset.startswith(("wikitext", "c4", "pile")):
                dataset = load_dataset(validation_dataset, split="validation")
            else:
                dataset = load_dataset(validation_dataset)

            # Calculate perplexity for both models
            original_ppl = self._calculate_perplexity(original_model, dataset)
            quantized_ppl = self._calculate_perplexity(quantized_model, dataset)

            # Compare results
            degradation = quantized_ppl / original_ppl
            logger.info(f"Perplexity - Original: {original_ppl:.2f}, Quantized: {quantized_ppl:.2f}")
            logger.info(f"Degradation factor: {degradation:.2f}x")

            return degradation <= perplexity_threshold

        except Exception as e:
            logger.error(f"Failed to validate quantization: {e}")
            raise

    def _generate_triton_config(
        self,
        model_path: str,
        quantization_method: str,
        bits: int
    ):
        """Generate Triton Inference Server configuration"""
        config_dir = os.path.join(os.path.dirname(model_path), "triton_config")
        os.makedirs(config_dir, exist_ok=True)

        # Basic model configuration
        model_config = {
            "backend": "pytorch",
            "max_batch_size": 8,
            "input": [
                {
                    "name": "input_ids",
                    "data_type": "TYPE_INT64",
                    "dims": [-1]
                },
                {
                    "name": "attention_mask",
                    "data_type": "TYPE_INT64",
                    "dims": [-1]
                }
            ],
            "output": [
                {
                    "name": "logits",
                    "data_type": "TYPE_FP32",
                    "dims": [-1, -1]
                }
            ],
            "instance_group": [
                {
                    "count": 1,
                    "kind": "KIND_GPU"
                }
            ],
            "optimization": {
                "priority": "PRIORITY_MAX",
                "cuda": {
                    "graphs": True,
                    "batched_input": True,
                    "output_copy_stream": True
                }
            },
            "parameters": {
                "quantization_method": quantization_method,
                "bits": bits
            }
        }

        # Save configuration
        with open(os.path.join(config_dir, "config.pbtxt"), 'w') as f:
            import json
            json.dump(model_config, f, indent=2)

    def _quantize_fp16(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        save_onnx: bool = True
    ) -> str:
        """Quantize a model to FP16 precision"""
        try:
            import torch
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            logger.info(f"Quantizing model {model_path} to FP16")

            # Determine model format
            model_format = os.path.splitext(model_path)[1].lower()

            # For PyTorch models
            if model_format in ['.pt', '.pth']:
                model = torch.load(model_path, map_location="cpu")
                if hasattr(model, 'half'):
                    # Convert model to half precision
                    fp16_model = model.half()

                    if save_onnx:
                        # Export to ONNX
                        dummy_input = self._create_dummy_input(model, model_type)
                        torch.onnx.export(
                            fp16_model,
                            dummy_input,
                            output_path,
                            export_params=True,
                            opset_version=13,
                            do_constant_folding=True
                        )
                    else:
                        # Save as PyTorch model
                        if not output_path.endswith('.pt'):
                            output_path += '.pt'
                        torch.save(fp16_model, output_path)
                else:
                    logger.warning("Model doesn't support .half() method, trying alternative approach")
                    # For state_dict models
                    if isinstance(model, dict):
                        for k in model.keys():
                            if isinstance(model[k], torch.Tensor):
                                model[k] = model[k].half()
                        torch.save(model, output_path)

            # For ONNX models
            elif model_format == '.onnx':
                # Load ONNX model
                onnx_model = onnx.load(model_path)

                # Convert all float tensors to float16
                for tensor in onnx_model.graph.initializer:
                    if tensor.data_type == 1:  # FLOAT
                        # Convert to float16
                        tensor.data_type = 10  # FLOAT16

                # Save the model
                onnx.save(onnx_model, output_path)

            # For other formats, try using ONNX conversion first
            else:
                logger.warning(f"Unsupported model format for direct FP16 conversion: {model_format}")
                logger.info("Trying to convert to ONNX first")

                from model_converter import ModelConverter
                converter = ModelConverter()
                onnx_path = converter.convert(
                    model_source=model_path,
                    model_type=model_type
                )

                # Then quantize the ONNX model
                return self._quantize_fp16(onnx_path, output_path, model_type, save_onnx)

            logger.info(f"Model quantized to FP16 and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "precision": "fp16",
                "model_type": model_type,
                "quantization_method": "float16_conversion"
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize model to FP16: {e}")
            raise

    def _quantize_int8(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        calibration_data: Optional[str] = None,
        calibration_samples: int = 100,
        save_onnx: bool = True
    ) -> str:
        """Quantize a model to INT8 precision"""
        try:
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
            from onnxruntime.quantization.calibrate import CalibrationDataReader
            import onnx

            logger.info(f"Quantizing model {model_path} to INT8")

            # Convert to ONNX first if not already ONNX
            if not model_path.endswith(".onnx"):
                from model_converter import ModelConverter
                converter = ModelConverter()
                model_path = converter.convert(
                    model_source=model_path,
                    model_type=model_type
                )

            # If we have calibration data, use static quantization (more accurate)
            if calibration_data:
                # Implement calibration data reader
                class CalibrationReader(CalibrationDataReader):
                    def __init__(self, calibration_data_path: str, input_names: List[str]):
                        self.data_path = calibration_data_path
                        self.input_names = input_names
                        self.files = os.listdir(calibration_data_path)
                        self.index = 0
                        self.max_samples = min(len(self.files), calibration_samples)

                    def get_next(self) -> Dict[str, np.ndarray]:
                        if self.index >= self.max_samples:
                            return None

                        file_path = os.path.join(self.data_path, self.files[self.index])
                        self.index += 1

                        # Load sample data
                        try:
                            if file_path.endswith('.npz'):
                                data = np.load(file_path)
                                return {name: data[name] for name in self.input_names if name in data}
                            elif file_path.endswith('.npy'):
                                data = np.load(file_path)
                                # Assume single input for .npy files
                                return {self.input_names[0]: data}
                            else:
                                logger.warning(f"Unsupported calibration data format: {file_path}")
                                return self.get_next()  # Skip and get next
                        except Exception as e:
                            logger.warning(f"Error loading calibration data: {e}")
                            return self.get_next()  # Skip and get next

                # Get model input names
                onnx_model = onnx.load(model_path)
                input_names = [node.name for node in onnx_model.graph.input]

                # Create calibration data reader
                calibration_reader = CalibrationReader(calibration_data, input_names)

                # Perform static quantization
                quantize_static(
                    model_input=model_path,
                    model_output=output_path,
                    calibration_data_reader=calibration_reader,
                    quant_format=QuantType.QInt8,
                    optimize_model=True,
                    per_channel=False,
                    reduce_range=False,
                    weight_type=QuantType.QInt8
                )
            else:
                # Use dynamic quantization (no calibration data needed)
                quantize_dynamic(
                    model_input=model_path,
                    model_output=output_path,
                    weight_type=QuantType.QInt8,
                    optimize_model=True
                )

            logger.info(f"Model quantized to INT8 and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "precision": "int8",
                "model_type": model_type,
                "quantization_method": "static" if calibration_data else "dynamic",
                "calibration_samples": calibration_samples if calibration_data else 0
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize model to INT8: {e}")
            raise

    def _quantize_4bit(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        quantization_method: str = "gptq",
        save_onnx: bool = True
    ) -> str:
        """Quantize a model to 4-bit precision (for LLMs)"""
        try:
            logger.info(f"Quantizing model {model_path} to 4-bit using {quantization_method}")

            # Check if model is a language model
            if model_type != "language":
                logger.warning(f"4-bit quantization is typically for language models, but model type is {model_type}")

            # GPTQ quantization
            if quantization_method.lower() == "gptq":
                return self._quantize_gptq(model_path, output_path, model_type, save_onnx)

            # AWQ quantization
            elif quantization_method.lower() == "awq":
                return self._quantize_awq(model_path, output_path, model_type, save_onnx)

            # EETQ quantization (Efficient & Easy Transformer Quantization)
            elif quantization_method.lower() == "eetq":
                return self._quantize_eetq(model_path, output_path, model_type, save_onnx)

            else:
                logger.error(f"Unsupported 4-bit quantization method: {quantization_method}")
                raise ValueError(f"Unsupported 4-bit quantization method: {quantization_method}")

        except Exception as e:
            logger.error(f"Failed to quantize model to 4-bit: {e}")
            raise

    def _quantize_gptq(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        save_onnx: bool = True
    ) -> str:
        """Quantize a model using GPTQ (4-bit, typically for LLMs)"""
        try:
            # Check for optimum and transformers
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from optimum.gptq import GPTQQuantizer, load_quantized_model
                import torch
            except ImportError:
                logger.error("Missing dependencies for GPTQ quantization")
                logger.info("Install with: pip install transformers optimum[gptq]")
                raise

            # Check if it's a local directory with model files or single file
            if os.path.isdir(model_path):
                # Load model from directory
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            else:
                # Assume it's a state dict file
                logger.error("GPTQ requires a complete transformers model. Use a directory path containing config.json")
                raise ValueError("Unsupported model format for GPTQ")

            # Get calibration dataset (use WikiText-2 for simplicity)
            from datasets import load_dataset
            calibration_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            calibration_dataset = calibration_dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)

            # Create quantizer
            quantizer = GPTQQuantizer(
                bits=4,
                dataset=calibration_dataset,
                tokenizer=tokenizer,
                block_name_to_quantize="model.layers"
            )

            # Quantize model
            quantized_model = quantizer.quantize_model(model, calibration_dataset)

            # Create output directory if output_path doesn't have extension
            if not os.path.splitext(output_path)[1]:
                os.makedirs(output_path, exist_ok=True)
            else:
                # Make sure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Remove the extension and create directory
                dir_path = os.path.splitext(output_path)[0]
                os.makedirs(dir_path, exist_ok=True)
                output_path = dir_path

            # Save model
            tokenizer.save_pretrained(output_path)
            quantized_model.save_pretrained(output_path)

            # Save to ONNX if requested
            if save_onnx:
                onnx_path = os.path.join(output_path, "model.onnx")
                from transformers.onnx import export
                export(
                    preprocessor=tokenizer,
                    model=quantized_model,
                    opset=13,
                    output=Path(onnx_path)
                )

            logger.info(f"Model quantized to 4-bit with GPTQ and saved to {output_path}")

            # Create model info file
            metadata = {
                "original_model": model_path,
                "precision": "4bit",
                "model_type": model_type,
                "quantization_method": "gptq",
                "bits": 4,
                "tokenizer": tokenizer.__class__.__name__,
                "model_type": model.__class__.__name__,
            }
            create_model_info_file(os.path.join(output_path, "model.bin"), metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize model with GPTQ: {e}")
            raise

    def _quantize_awq(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        save_onnx: bool = True
    ) -> str:
        """Quantize a model using AWQ (Activation-aware Weight Quantization)"""
        try:
            # Check for awq
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
                from awq import AutoAWQForCausalLM
            except ImportError:
                logger.error("Missing dependencies for AWQ quantization")
                logger.info("Install with: pip install transformers git+https://github.com/casper-hansen/AutoAWQ.git")
                raise

            # Check if it's a local directory with model files or single file
            if os.path.isdir(model_path):
                # Initialize model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # Assume it's a state dict file
                logger.error("AWQ requires a complete transformers model. Use a directory path containing config.json")
                raise ValueError("Unsupported model format for AWQ")

            # Create quantizer and load model
            model = AutoAWQForCausalLM.from_pretrained(model_path)

            # Create output directory if output_path doesn't have extension
            if not os.path.splitext(output_path)[1]:
                os.makedirs(output_path, exist_ok=True)
            else:
                # Make sure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Remove the extension and create directory
                dir_path = os.path.splitext(output_path)[0]
                os.makedirs(dir_path, exist_ok=True)
                output_path = dir_path

            # Get calibration dataset
            from datasets import load_dataset
            calibration_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text_samples = calibration_dataset["text"][:100]  # Use 100 samples

            # Quantize model with AWQ
            model.quantize(
                tokenizer=tokenizer,
                quant_config={
                    "zero_point": True,
                    "q_group_size": 128,
                    "w_bit": 4,
                    "version": "GEMM"
                },
                calib_data=text_samples
            )

            # Save model
            tokenizer.save_pretrained(output_path)
            model.save_pretrained(output_path)

            # Save to ONNX if requested
            if save_onnx:
                onnx_path = os.path.join(output_path, "model.onnx")
                from transformers.onnx import export
                export(
                    preprocessor=tokenizer,
                    model=model,
                    opset=13,
                    output=Path(onnx_path)
                )

            logger.info(f"Model quantized to 4-bit with AWQ and saved to {output_path}")

            # Create model info file
            metadata = {
                "original_model": model_path,
                "precision": "4bit",
                "model_type": model_type,
                "quantization_method": "awq",
                "bits": 4,
                "group_size": 128,
                "tokenizer": tokenizer.__class__.__name__,
            }
            create_model_info_file(os.path.join(output_path, "model.bin"), metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize model with AWQ: {e}")
            raise

    def _quantize_eetq(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        save_onnx: bool = True
    ) -> str:
        """Quantize a model using EETQ (Efficient & Easy Transformer Quantization)"""
        try:
            # Check for eetq
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import eetq
            except ImportError:
                logger.error("Missing dependencies for EETQ quantization")
                logger.info("Install with: pip install transformers eetq")
                raise

            # Check if it's a local directory with model files or single file
            if os.path.isdir(model_path):
                # Load model with EETQ quantization
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=eetq.EETQConfig(
                        bits=4,
                        group_size=128,
                    )
                )
            else:
                # Assume it's a state dict file
                logger.error("EETQ requires a complete transformers model. Use a directory path containing config.json")
                raise ValueError("Unsupported model format for EETQ")

            # Create output directory if output_path doesn't have extension
            if not os.path.splitext(output_path)[1]:
                os.makedirs(output_path, exist_ok=True)
            else:
                # Make sure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Remove the extension and create directory
                dir_path = os.path.splitext(output_path)[0]
                os.makedirs(dir_path, exist_ok=True)
                output_path = dir_path

            # Save model
            tokenizer.save_pretrained(output_path)
            model.save_pretrained(output_path)

            # Save to ONNX if requested
            if save_onnx:
                onnx_path = os.path.join(output_path, "model.onnx")
                from transformers.onnx import export
                export(
                    preprocessor=tokenizer,
                    model=model,
                    opset=13,
                    output=Path(onnx_path)
                )

            logger.info(f"Model quantized to 4-bit with EETQ and saved to {output_path}")

            # Create model info file
            metadata = {
                "original_model": model_path,
                "precision": "4bit",
                "model_type": model_type,
                "quantization_method": "eetq",
                "bits": 4,
                "group_size": 128,
                "tokenizer": tokenizer.__class__.__name__,
            }
            create_model_info_file(os.path.join(output_path, "model.bin"), metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to quantize model with EETQ: {e}")
            raise

    def _create_dummy_input(self, model, model_type: str):
        """Create dummy input for a model based on its type"""
        if model_type == "language":
            # Language model dummy input (batch_size=1, seq_len=512)
            return torch.randint(0, 1000, (1, 512))
        elif model_type == "vision":
            # Vision model dummy input (batch_size=1, channels=3, height=224, width=224)
            return torch.randn(1, 3, 224, 224)
        elif model_type == "speech":
            # Speech model dummy input (batch_size=1, channels=1, time_steps=16000)
            return torch.randn(1, 1, 16000)
        else:
            # Default dummy input
            return torch.randn(1, 3, 224, 224)


def main():
    """Command line interface for the model quantizer"""
    import argparse

    parser = argparse.ArgumentParser(description='Quantize models to lower precision')
    parser.add_argument('model', type=str, help='Path or identifier for the input model')
    parser.add_argument('--output', '-o', type=str, help='Output directory or file path')
    parser.add_argument('--precision', '-p', choices=['fp16', 'int8', '4bit'], default='fp16',
                        help='Target precision (default: fp16)')
    parser.add_argument('--model-type', '-t', type=str, help='Model type (language, vision, speech)')
    parser.add_argument('--calibration', '-c', type=str, help='Path to calibration data for INT8 quantization')
    parser.add_argument('--samples', '-s', type=int, default=100, help='Number of calibration samples')
    parser.add_argument('--method', '-m', type=str, default='',
                        help='Quantization method (for 4-bit: gptq, awq, eetq)')
    parser.add_argument('--no-onnx', action='store_true', help='Do not save as ONNX format')

    args = parser.parse_args()

    # Determine quantization method based on precision if not specified
    method = args.method
    if args.precision == '4bit' and not method:
        method = 'gptq'  # Default for 4-bit

    quantizer = ModelQuantizer()
    output_path = quantizer.quantize(
        model_source=args.model,
        output_dir=args.output,
        model_type=args.model_type,
        precision=args.precision,
        calibration_data=args.calibration,
        calibration_samples=args.samples,
        quantization_method=method,
        save_onnx=not args.no_onnx
    )

    print(f"Model quantized successfully: {output_path}")


if __name__ == "__main__":
    main()
