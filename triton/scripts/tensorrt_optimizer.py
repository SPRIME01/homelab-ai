import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class TensorRTOptimizer:
    def __init__(self):
        self.trt_version = self._get_trt_version()

    def _get_trt_version(self) -> str:
        import tensorrt as trt
        return trt.__version__

    def optimize(
        self,
        model_source: str,
        output_dir: str,
        model_type: str,
        precision: str = "fp16",
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        max_batch_size: int = 8,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        calibration_data: Optional[str] = None,
    ) -> str:
        """Optimize a model with TensorRT"""
        if precision == "fp32":
            return self._optimize_fp32(model_source, output_dir, input_shapes, workspace_size, max_batch_size)
        elif precision == "fp16":
            return self._optimize_fp16(model_source, output_dir, input_shapes, workspace_size, max_batch_size)
        elif precision == "int8":
            return self._optimize_int8(model_source, output_dir, input_shapes, workspace_size, max_batch_size, calibration_data)
        else:
            logger.warning(f"Unknown precision: {precision}. Defaulting to FP16.")
            return self._optimize_fp16(model_source, output_dir, input_shapes, workspace_size, max_batch_size)

    def _optimize_fp32(
        self,
        model_path: str,
        output_path: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        max_batch_size: int = 8
    ) -> str:
        """Optimize an ONNX model with TensorRT FP32 precision"""
        import tensorrt as trt
        from polygraphy.backend.trt import CreateConfig
        from polygraphy.backend.onnx import ConvertToTensorRTEngine

        logger.info(f"Optimizing ONNX model {model_path} with TensorRT (FP32)")

        try:
            # Create TensorRT config for FP32
            config = CreateConfig(
                max_workspace_size=workspace_size,
                tf32=False,
                fp16=False,
                int8=False,
                profiles=[profile_from_shapes(input_shapes)] if input_shapes else None,
                max_batch_size=max_batch_size,
                strict_types=False
            )

            # Convert ONNX to TensorRT engine
            engine = ConvertToTensorRTEngine(
                model_path, config=config, save_timing_cache=True
            )

            # Save the engine
            with open(output_path, "wb") as f:
                f.write(engine.serialize())
            logger.info(f"Saved TensorRT engine to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "precision": "fp32",
                "tensorrt_version": self.trt_version,
                "max_batch_size": max_batch_size,
                "workspace_size": workspace_size,
                "input_shapes": input_shapes
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to optimize model with TensorRT (FP32): {e}")
            raise

    def _optimize_fp16(
        self,
        model_path: str,
        output_path: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        max_batch_size: int = 8
    ) -> str:
        """Optimize an ONNX model with TensorRT FP16 precision"""
        import tensorrt as trt
        from polygraphy.backend.trt import CreateConfig
        from polygraphy.backend.onnx import ConvertToTensorRTEngine

        logger.info(f"Optimizing ONNX model {model_path} with TensorRT (FP16)")

        try:
            # Create TensorRT config for FP16
            config = CreateConfig(
                max_workspace_size=workspace_size,
                tf32=False,
                fp16=True,
                int8=False,
                profiles=[profile_from_shapes(input_shapes)] if input_shapes else None,
                max_batch_size=max_batch_size,
                strict_types=False
            )

            # Convert ONNX to TensorRT engine
            engine = ConvertToTensorRTEngine(
                model_path, config=config, save_timing_cache=True
            )

            # Save the engine
            with open(output_path, "wb") as f:
                f.write(engine.serialize())
            logger.info(f"Saved TensorRT engine to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "precision": "fp16",
                "tensorrt_version": self.trt_version,
                "max_batch_size": max_batch_size,
                "workspace_size": workspace_size,
                "input_shapes": input_shapes
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to optimize model with TensorRT (FP16): {e}")
            raise

    def _optimize_int8(
        self,
        model_path: str,
        output_path: str,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        max_batch_size: int = 8,
        calibration_data: Optional[str] = None,
        calibrator_algo: str = "entropy",
        save_engine: bool = True
    ) -> str:
        """Optimize an ONNX model with TensorRT INT8 precision"""
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda
        import numpy as np
        from polygraphy.backend.trt import CreateConfig
        from polygraphy.backend.onnx import ConvertToTensorRTEngine
        from polygraphy.backend.trt import TrtRunner

        logger.info(f"Optimizing ONNX model {model_path} with TensorRT (INT8)")

        # INT8 calibration requires calibration data
        if not calibration_data:
            logger.warning("No calibration data provided for INT8 quantization. Using synthetic data.")
            calibration_data = self._generate_synthetic_calibration_data(model_path, input_shapes)

        try:
            # Create a calibrator based on algorithm
            if calibrator_algo == "entropy":
                calibrator = EntropyCalibrator(calibration_data, input_shapes)
            elif calibrator_algo == "minmax":
                calibrator = MinMaxCalibrator(calibration_data, input_shapes)
            elif calibrator_algo == "percentile":
                calibrator = PercentileCalibrator(calibration_data, input_shapes)
            else:
                logger.warning(f"Unknown calibrator algorithm: {calibrator_algo}. Using entropy calibrator.")
                calibrator = EntropyCalibrator(calibration_data, input_shapes)

            # Create TensorRT config for INT8
            config = CreateConfig(
                max_workspace_size=workspace_size,
                tf32=False,
                fp16=True,  # Enable FP16 for better performance
                int8=True,
                profiles=[profile_from_shapes(input_shapes)] if input_shapes else None,
                max_batch_size=max_batch_size,
                strict_types=False,
                int8_calib_dataset=calibration_data,
                int8_calibrator=calibrator
            )

            # Convert ONNX to TensorRT engine
            engine = ConvertToTensorRTEngine(
                model_path, config=config, save_timing_cache=True
            )

            # Save the engine if requested
            if save_engine:
                with open(output_path, "wb") as f:
                    f.write(engine.serialize())
                logger.info(f"Saved TensorRT engine to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path if save_engine else {})
            metadata.update({
                "original_model": model_path,
                "precision": "int8",
                "tensorrt_version": self.trt_version,
                "max_batch_size": max_batch_size,
                "workspace_size": workspace_size,
                "calibrator": calibrator_algo,
                "input_shapes": input_shapes
            })
            create_model_info_file(output_path if save_engine else model_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to optimize model with TensorRT (INT8): {e}")
            raise

    def _generate_synthetic_calibration_data(self, model_path: str, input_shapes: Optional[Dict[str, List[int]]] = None) -> str:
        """Generate synthetic calibration data for INT8 quantization"""
        import onnx
        import numpy as np
        import os
        import tempfile

        logger.info("Generating synthetic calibration data for INT8 quantization")

        # Create temp directory for calibration data
        temp_dir = os.path.join(tempfile.gettempdir(), "trt_calibration")
        os.makedirs(temp_dir, exist_ok=True)

        # Load the ONNX model to get input shapes if not provided
        if not input_shapes:
            model = onnx.load(model_path)
            input_shapes = {}
            for input_tensor in model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        shape.append(1)  # Replace dynamic dimensions with 1
                    else:
                        shape.append(dim.dim_value)
                input_shapes[input_tensor.name] = shape

        # Generate random data for each input
        for i in range(100):  # Generate 100 calibration samples
            sample = {}
            for name, shape in input_shapes.items():
                # Generate random data of appropriate shape and type
                # For uint8 data (images), we use uint8 random data
                if "image" in name.lower():
                    data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
                # For embeddings and other float data, we use float32
                else:
                    data = np.random.randn(*shape).astype(np.float32)
                sample[name] = data

            # Save sample to disk
            sample_path = os.path.join(temp_dir, f"sample_{i}")
            np.savez(sample_path, **sample)

        logger.info(f"Generated synthetic calibration data at {temp_dir}")
        return temp_dir

# Calibrator classes for INT8 quantization
class EntropyCalibrator:
    """Entropy calibrator for INT8 quantization"""

    def __init__(self, calibration_data_dir: str, input_shapes: Dict[str, List[int]]):
        self.calibration_data_dir = calibration_data_dir
        self.input_shapes = input_shapes
        self.calibration_files = [
            os.path.join(calibration_data_dir, f)
            for f in os.listdir(calibration_data_dir)
            if f.endswith(".npz")
        ]
        self.current_file_idx = 0
        self.batch_size = 1

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, name):
        if self.current_file_idx >= len(self.calibration_files):
            return None

        calibration_file = self.calibration_files[self.current_file_idx]
        self.current_file_idx += 1

        data = np.load(calibration_file)
        return data[name]

    def read_calibration_cache(self):
        cache_file = os.path.join(self.calibration_data_dir, "calibration.cache")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        cache_file = os.path.join(self.calibration_data_dir, "calibration.cache")
        with open(cache_file, "wb") as f:
            f.write(cache)


class MinMaxCalibrator(EntropyCalibrator):
    """MinMax calibrator for INT8 quantization"""
    pass  # Same implementation as EntropyCalibrator, just using different internal algorithm


class PercentileCalibrator(EntropyCalibrator):
    """Percentile calibrator for INT8 quantization"""
    pass  # Same implementation as EntropyCalibrator, just using different internal algorithm


def profile_from_shapes(input_shapes: Dict[str, List[int]]) -> Dict:
    """Create an optimization profile from input shapes"""
    if not input_shapes:
        return {}

    profile = {}
    for name, shape in input_shapes.items():
        # Set min/opt/max dimensions for dynamic shapes
        min_shape = []
        opt_shape = []
        max_shape = []

        for dim in shape:
            if dim == -1 or dim == 0:  # Dynamic dimension
                min_shape.append(1)  # Minimum size 1
                opt_shape.append(16)  # Optimal size 16 (typical batch size)
                max_shape.append(64)  # Maximum size 64 (adjust as needed)
            else:
                min_shape.append(dim)
                opt_shape.append(dim)
                max_shape.append(dim)

        profile[name] = (min_shape, opt_shape, max_shape)

    return profile


def main():
    """Command line interface for the TensorRT optimizer"""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize models with TensorRT')
    parser.add_argument('model', type=str, help='Path or identifier for the input model')
    parser.add_argument('--output', '-o', type=str, help='Output directory or file path')
    parser.add_argument('--precision', '-p', choices=['fp32', 'fp16', 'int8', 'auto'], default='fp16',
                    help='Precision mode for optimization (default: fp16)')
    parser.add_argument('--calibration', '-c', type=str, help='Path to calibration data for INT8 quantization')
    parser.add_argument('--workspace', '-w', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Maximum batch size')
    parser.add_argument('--model-type', '-t', type=str, help='Model type (language, vision, speech)')

    args = parser.parse_args()

    optimizer = TensorRTOptimizer()
    output_path = optimizer.optimize(
        model_source=args.model,
        output_dir=args.output,
        model_type=args.model_type,
        precision=args.precision,
        workspace_size=args.workspace * 1024 * 1024 * 1024,
        max_batch_size=args.batch_size,
        calibration_data=args.calibration,
    )

    print(f"Model optimized successfully: {output_path}")


if __name__ == "__main__":
    main()
