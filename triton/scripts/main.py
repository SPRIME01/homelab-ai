import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

from config import config, logger
from utils import verify_dependencies
from download_manager import DownloadManager
from model_converter import ModelConverter
from tensorrt_optimizer import TensorRTOptimizer
from model_quantizer import ModelQuantizer
from model_pruner import ModelPruner
from model_benchmark import ModelBenchmark

def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Model Optimization Toolkit for NVIDIA Jetson AGX Orin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a PyTorch model to ONNX
  python main.py convert pytorch_model.pt --output models/onnx/

  # Optimize an ONNX model with TensorRT
  python main.py optimize model.onnx --precision fp16

  # Quantize a model to INT8
  python main.py quantize model.onnx --precision int8

  # Prune a model to reduce size
  python main.py prune model.pt --sparsity 0.5

  # Benchmark a model
  python main.py benchmark model.trt --batch-sizes 1,2,4,8 --duration 10

  # Download a model from HuggingFace
  python main.py download facebook/opt-350m --output models/language/
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--output", "-o", type=str, help="Output directory")
    common_parser.add_argument("--model-type", "-t", type=str, help="Model type (language, vision, speech)")
    common_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # Convert command
    convert_parser = subparsers.add_parser("convert", parents=[common_parser], help="Convert models to ONNX format")
    convert_parser.add_argument("model", type=str, help="Path or identifier for the input model")
    convert_parser.add_argument("--framework", "-f", type=str, help="Source framework (pytorch, tensorflow, auto)")
    convert_parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    convert_parser.add_argument("--no-simplify", action="store_true", help="Disable model simplification")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", parents=[common_parser], help="Optimize models with TensorRT")
    optimize_parser.add_argument("model", type=str, help="Path or identifier for the input model")
    optimize_parser.add_argument("--precision", "-p", choices=["fp32", "fp16", "int8", "auto"], default="fp16",
                               help="Precision mode (default: fp16)")
    optimize_parser.add_argument("--workspace", "-w", type=int, default=4, help="TensorRT workspace size in GB")
    optimize_parser.add_argument("--batch-size", "-b", type=int, default=8, help="Maximum batch size")
    optimize_parser.add_argument("--calibration", "-c", type=str, help="Path to calibration data for INT8")

    # Quantize command
    quantize_parser = subparsers.add_parser("quantize", parents=[common_parser], help="Quantize models to lower precision")
    quantize_parser.add_argument("model", type=str, help="Path or identifier for the input model")
    quantize_parser.add_argument("--precision", "-p", choices=["fp16", "int8", "4bit"], default="fp16",
                               help="Target precision (default: fp16)")
    quantize_parser.add_argument("--method", "-m", type=str, help="Quantization method (for 4-bit: gptq, awq, eetq)")
    quantize_parser.add_argument("--calibration", "-c", type=str, help="Path to calibration data")
    quantize_parser.add_argument("--samples", "-s", type=int, default=100, help="Number of calibration samples")
    quantize_parser.add_argument("--no-onnx", action="store_true", help="Do not save as ONNX format")

    # Prune command
    prune_parser = subparsers.add_parser("prune", parents=[common_parser], help="Prune models to reduce size")
    prune_parser.add_argument("model", type=str, help="Path or identifier for the input model")
    prune_parser.add_argument("--method", "-m", choices=["magnitude", "structured", "uniform", "distillation"],
                            default="magnitude", help="Pruning method (default: magnitude)")
    prune_parser.add_argument("--sparsity", "-s", type=float, default=0.5,
                            help="Target sparsity level (0-1, default: 0.5)")
    prune_parser.add_argument("--no-onnx", action="store_true", help="Do not save as ONNX format")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", parents=[common_parser], help="Benchmark model performance")
    benchmark_parser.add_argument("model", type=str, help="Path or identifier for the input model")
    benchmark_parser.add_argument("--batch-sizes", "-b", type=str, default="1,2,4,8", help="Comma-separated batch sizes")
    benchmark_parser.add_argument("--duration", "-d", type=int, default=10, help="Duration in seconds for each benchmark")
    benchmark_parser.add_argument("--concurrency", "-c", type=str, default="1,2,4,8", help="Comma-separated concurrency values")
    benchmark_parser.add_argument("--triton", "-u", type=str, help="Triton server URL (e.g., localhost:8001)")
    benchmark_parser.add_argument("--warmup", "-w", type=int, default=3, help="Number of warmup iterations")
    benchmark_parser.add_argument("--no-save", action="store_true", help="Do not save benchmark results")
    benchmark_parser.add_argument("--no-plots", action="store_true", help="Do not generate performance plots")

    # Download command
    download_parser = subparsers.add_parser("download", parents=[common_parser], help="Download models from various sources")
    download_parser.add_argument("model", type=str, help="Model identifier (e.g., 'hf://facebook/opt-350m', GitHub URL, direct URL)")
    download_parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if model exists")

    # Config command
    config_parser = subparsers.add_parser("config", help="View or modify configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--reset", action="store_true", help="Reset to default configuration")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set configuration value (e.g., --set paths.output_dir /new/path)")

    return parser

def main():
    """Main entry point"""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Set log level
    if args.command and hasattr(args, "verbose") and args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Verify environment
    verify_dependencies()

    # Process commands
    if args.command == "convert":
        converter = ModelConverter()
        output_path = converter.convert(
            model_source=args.model,
            output_dir=args.output,
            model_type=args.model_type,
            framework=args.framework,
            opset_version=args.opset,
            simplify=not args.no_simplify
        )
        print(f"Model converted successfully: {output_path}")

    elif args.command == "optimize":
        optimizer = TensorRTOptimizer()
        output_path = optimizer.optimize(
            model_source=args.model,
            output_dir=args.output,
            model_type=args.model_type,
            precision=args.precision,
            workspace_size=args.workspace * 1024 * 1024 * 1024,
            max_batch_size=args.batch_size,
            calibration_data=args.calibration
        )
        print(f"Model optimized successfully: {output_path}")

    elif args.command == "quantize":
        quantizer = ModelQuantizer()
        output_path = quantizer.quantize(
            model_source=args.model,
            output_dir=args.output,
            model_type=args.model_type,
            precision=args.precision,
            calibration_data=args.calibration,
            calibration_samples=args.samples,
            quantization_method=args.method,
            save_onnx=not args.no_onnx
        )
        print(f"Model quantized successfully: {output_path}")

    elif args.command == "prune":
        pruner = ModelPruner()
        output_path = pruner.prune(
            model_source=args.model,
            output_dir=args.output,
            model_type=args.model_type,
            pruning_method=args.method,
            target_sparsity=args.sparsity,
            save_onnx=not args.no_onnx
        )
        print(f"Model pruned successfully: {output_path}")

    elif args.command == "benchmark":
        # Parse batch sizes and concurrency
        batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
        concurrency = [int(c) for c in args.concurrency.split(",")]

        benchmark = ModelBenchmark()
        results = benchmark.benchmark(
            model_source=args.model,
            output_dir=args.output,
            model_type=args.model_type,
            batch_sizes=batch_sizes,
            duration_seconds=args.duration,
            concurrency=concurrency,
            triton_url=args.triton,
            warmup=args.warmup,
            save_results=not args.no_save,
            generate_plots=not args.no_plots
        )
        print(f"Benchmark completed successfully!")

    elif args.command == "download":
        downloader = DownloadManager()
        model_path = downloader.get_model(
            model_source=args.model,
            output_dir=args.output
        )
        print(f"Model downloaded successfully: {model_path}")

    elif args.command == "config":
        if args.show:
            print(json.dumps(config.config, indent=2))
        elif args.reset:
            config = config.__class__()
            config.save_config()
            print("Configuration reset to defaults")
        elif args.set:
            key_path, value = args.set
            keys = key_path.split('.')

            # Navigate to the correct nested dictionary
            current = config.config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value with appropriate type conversion
            try:
                # Try to interpret as int, float, or boolean
                if value.lower() == "true":
                    current[keys[-1]] = True
                elif value.lower() == "false":
                    current[keys[-1]] = False
                elif value.isdigit():
                    current[keys[-1]] = int(value)
                elif is_float(value):
                    current[keys[-1]] = float(value)
                else:
                    current[keys[-1]] = value
            except:
                # Default to string if conversion fails
                current[keys[-1]] = value

            config.save_config()
            print(f"Set {key_path} to {value}")
        else:
            parser.parse_args(["config", "--help"])
    else:
        parser.print_help()

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    main()
