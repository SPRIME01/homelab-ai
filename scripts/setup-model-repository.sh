#!/bin/bash
set -e

# Setup script for Triton Inference Server model repository
# Optimized for NVIDIA Jetson AGX Orin

# Configuration variables
MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-/models}"
S3_ENDPOINT="${S3_ENDPOINT:-minio.ai.svc.cluster.local:9000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_USE_HTTPS="${S3_USE_HTTPS:-false}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-models}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Triton Model Repository Setup ===${NC}"
echo -e "Setting up model repository at: ${MODEL_REPO_ROOT}"

# Create the main directory structure
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p "${MODEL_REPO_ROOT}"

# Create model type directories
MODEL_TYPES=("language" "vision" "speech" "multimodal")
for model_type in "${MODEL_TYPES[@]}"; do
    mkdir -p "${MODEL_REPO_ROOT}/${model_type}"
    echo -e "Created ${model_type} directory"
done

# Create version control directory
mkdir -p "${MODEL_REPO_ROOT}/version_control"
echo -e "Created version control directory"

# Create utility scripts directory
mkdir -p "${MODEL_REPO_ROOT}/utils"

# Create example model directories for each type
# Language Models
LLM_MODELS=("llama2-7b" "llama2-13b" "phi2" "mistral-7b")
for model in "${LLM_MODELS[@]}"; do
    # Create model directory with version
    mkdir -p "${MODEL_REPO_ROOT}/language/${model}/1"
    mkdir -p "${MODEL_REPO_ROOT}/language/${model}/config"
done

# Vision Models
VISION_MODELS=("yolov8" "resnet50" "mobilenet" "vitdet")
for model in "${VISION_MODELS[@]}"; do
    # Create model directory with version
    mkdir -p "${MODEL_REPO_ROOT}/vision/${model}/1"
    mkdir -p "${MODEL_REPO_ROOT}/vision/${model}/config"
done

# Speech Models
SPEECH_MODELS=("whisper" "fastconformer" "wav2vec" "espnet")
for model in "${SPEECH_MODELS[@]}"; do
    # Create model directory with version
    mkdir -p "${MODEL_REPO_ROOT}/speech/${model}/1"
    mkdir -p "${MODEL_REPO_ROOT}/speech/${model}/config"
done

# Multimodal Models
MM_MODELS=("clip" "llava" "imagebind")
for model in "${MM_MODELS[@]}"; do
    # Create model directory with version
    mkdir -p "${MODEL_REPO_ROOT}/multimodal/${model}/1"
    mkdir -p "${MODEL_REPO_ROOT}/multimodal/${model}/config"
done

# Create utility scripts
echo -e "\n${GREEN}Creating utility scripts...${NC}"

# Create model conversion utilities
cat > "${MODEL_REPO_ROOT}/utils/pytorch_to_onnx.py" << 'EOF'
#!/usr/bin/env python3
# Convert PyTorch model to ONNX format

import argparse
import torch
import os

def convert_model(model_path, output_path, input_shape, dynamic_axes=None):
    """Convert PyTorch model to ONNX format."""
    try:
        # Load the model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

        print(f"Model successfully converted to ONNX: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument('--model', required=True, help='Path to PyTorch model')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--shape', required=True, help='Input shape (comma-separated list)')
    parser.add_argument('--dynamic_batch', action='store_true', help='Use dynamic batch size')

    args = parser.parse_args()

    # Parse input shape
    shape = [int(dim) for dim in args.shape.split(',')]

    # Set up dynamic axes if needed
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    # Convert model
    convert_model(args.model, args.output, shape, dynamic_axes)
EOF

cat > "${MODEL_REPO_ROOT}/utils/optimize_onnx.py" << 'EOF'
#!/usr/bin/env python3
# Optimize ONNX models for NVIDIA Jetson AGX Orin

import argparse
import onnx
import onnxruntime as ort
import onnxoptimizer
import numpy as np
import os

def optimize_model(model_path, output_path, optimization_level=99):
    """Optimize ONNX model for Jetson AGX Orin."""
    try:
        # Load model
        model = onnx.load(model_path)

        # Check model
        onnx.checker.check_model(model)

        # Apply optimizations
        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized_model = onnxoptimizer.optimize(model, passes)

        # Save optimized model
        onnx.save(optimized_model, output_path)

        print(f"Model successfully optimized: {output_path}")
        return True
    except Exception as e:
        print(f"Error optimizing model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ONNX model for Jetson AGX Orin")
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--output', required=True, help='Output optimized ONNX file path')
    parser.add_argument('--level', type=int, default=99, help='Optimization level')

    args = parser.parse_args()
    optimize_model(args.model, args.output, args.level)
EOF

cat > "${MODEL_REPO_ROOT}/utils/convert_to_tensorrt.py" << 'EOF'
#!/usr/bin/env python3
# Convert ONNX model to TensorRT for NVIDIA Jetson AGX Orin

import argparse
import os
import sys
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_to_tensorrt(onnx_model_path, output_path, precision="fp16", workspace_size=4,
                        max_batch_size=8, calibrator=None):
    """Convert ONNX model to TensorRT engine."""
    try:
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # Set max workspace size (in GB)
        config.max_workspace_size = workspace_size * (1 << 30)

        # Set precision mode
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if calibrator:
                config.int8_calibrator = calibrator
            print("Using INT8 precision")
        else:
            print("Using default precision (FP32)")

        # Parse ONNX model
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parser Error: {parser.get_error(error)}")
                raise ValueError("Failed to parse ONNX model")

        # Set optimization profiles for dynamic shapes if needed
        # This part would need customization based on your model

        # Build engine
        print("Building TensorRT engine. This might take a while...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise ValueError("Failed to build TensorRT engine")

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())

        print(f"TensorRT engine saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting to TensorRT: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT")
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--output', required=True, help='Output TensorRT engine path')
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16', 'int8'],
                        help='Precision mode')
    parser.add_argument('--workspace', type=int, default=4, help='Max workspace size in GB')
    parser.add_argument('--batch_size', type=int, default=8, help='Max batch size')

    args = parser.parse_args()

    convert_to_tensorrt(args.model, args.output, args.precision,
                        args.workspace, args.batch_size)
EOF

cat > "${MODEL_REPO_ROOT}/utils/sync_models.sh" << 'EOF'
#!/bin/bash
# Sync models with S3-compatible storage (MinIO)

# Read environment variables
S3_ENDPOINT="${S3_ENDPOINT:-minio.ai.svc.cluster.local:9000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_USE_HTTPS="${S3_USE_HTTPS:-false}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-models}"
MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-/models}"

# Set MC configuration
export MC_HOST_minio=http://${S3_ACCESS_KEY}:${S3_SECRET_KEY}@${S3_ENDPOINT}

if [ "$S3_USE_HTTPS" = "true" ]; then
    export MC_HOST_minio=https://${S3_ACCESS_KEY}:${S3_SECRET_KEY}@${S3_ENDPOINT}
fi

# Check if mc is installed
if ! command -v mc &> /dev/null; then
    echo "MinIO client (mc) not found. Installing..."
    wget https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc
    chmod +x /usr/local/bin/mc
fi

# Function to sync from S3 to local
sync_from_s3() {
    echo "Syncing models from S3 to local storage..."
    mc mirror --json minio/${S3_BUCKET_NAME} ${MODEL_REPO_ROOT}
}

# Function to sync from local to S3
sync_to_s3() {
    echo "Syncing models from local storage to S3..."
    mc mirror --json ${MODEL_REPO_ROOT} minio/${S3_BUCKET_NAME}
}

# Main
case "$1" in
    "pull")
        sync_from_s3
        ;;
    "push")
        sync_to_s3
        ;;
    *)
        echo "Usage: $0 [pull|push]"
        echo "  pull: Download models from S3 to local storage"
        echo "  push: Upload local models to S3 storage"
        exit 1
        ;;
esac

echo "Sync completed!"
EOF

# Make utility scripts executable
chmod +x "${MODEL_REPO_ROOT}/utils/pytorch_to_onnx.py"
chmod +x "${MODEL_REPO_ROOT}/utils/optimize_onnx.py"
chmod +x "${MODEL_REPO_ROOT}/utils/convert_to_tensorrt.py"
chmod +x "${MODEL_REPO_ROOT}/utils/sync_models.sh"

# Create example model configuration files
echo -e "\n${GREEN}Creating example model configuration files...${NC}"

# Example TensorRT config for YOLOv8
cat > "${MODEL_REPO_ROOT}/vision/yolov8/config/config.pbtxt" << 'EOF'
name: "yolov8"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 5000
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      {
        name : "tensorrt"
        parameters {
          key: "precision_mode"
          value: "FP16"
        }
        parameters {
          key: "max_workspace_size_bytes"
          value: "1073741824"
        }
      }
    ]
  }
}
EOF

# Example ONNX config for Whisper
cat > "${MODEL_REPO_ROOT}/speech/whisper/config/config.pbtxt" << 'EOF'
name: "whisper"
platform: "onnxruntime_onnx"
max_batch_size: 1
input [
  {
    name: "input_features"
    data_type: TYPE_FP32
    dims: [ 80, -1 ]
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
dynamic_batching { }
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "onnxruntime"
        parameters {
          key: "gpu_mem_limit"
          value: "2147483648"
        }
        parameters {
          key: "execution_mode"
          value: "0"
        }
        parameters {
          key: "provider"
          value: "CUDAExecutionProvider"
        }
      }
    ]
  }
  graph {
    level: 99
  }
}
EOF

# Example PyTorch config for LLaMa2
cat > "${MODEL_REPO_ROOT}/language/llama2-7b/config/config.pbtxt" << 'EOF'
name: "llama2-7b"
backend: "python"
max_batch_size: 4
input [
  {
    name: "INPUT_0"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "INPUT_1"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }
]
output [
  {
    name: "OUTPUT_0"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 50000
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
parameters [
  {
    key: "model_type"
    value: { string_value: "llama" }
  },
  {
    key: "model_path"
    value: { string_value: "/models/language/llama2-7b/1/model" }
  },
  {
    key: "quantization"
    value: { string_value: "awq" }
  },
  {
    key: "max_tokens"
    value: { string_value: "2048" }
  },
  {
    key: "tokenizer_path"
    value: { string_value: "/models/language/llama2-7b/1/tokenizer" }
  }
]
EOF

# Example TensorFlow config for MobileNet
cat > "${MODEL_REPO_ROOT}/vision/mobilenet/config/config.pbtxt" << 'EOF'
name: "mobilenet"
platform: "tensorflow_savedmodel"
max_batch_size: 64
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
  }
]
output [
  {
    name: "MobilenetV2/Predictions/Reshape_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    label_filename: "labels.txt"
  }
]
dynamic_batching {
  preferred_batch_size: [ 4, 16, 32, 64 ]
  max_queue_delay_microseconds: 1000
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "tensorflow"
        parameters {
          key: "precision_mode"
          value: "FP16"
        }
      }
    ]
  }
}
EOF

# Create a README file in the repository root
cat > "${MODEL_REPO_ROOT}/README.md" << 'EOF'
# Triton Model Repository

This repository contains models for NVIDIA Triton Inference Server optimized for Jetson AGX Orin.

## Directory Structure

```
/models
├── language/           # Language models (LLMs, embeddings)
├── vision/             # Computer vision models
├── speech/             # Speech recognition and synthesis models
├── multimodal/         # Multi-modal models
├── version_control/    # Version history and metadata
└── utils/              # Utility scripts for model management
```

## Model Categories

### Language Models
- LLaMa 2 (7B & 13B parameters)
- Phi-2
- Mistral 7B

### Vision Models
- YOLOv8 (Object Detection)
- ResNet50 (Classification)
- MobileNet (Classification)
- ViTDet (Detection)

### Speech Models
- Whisper (Speech Recognition)
- FastConformer (Speech Recognition)
- Wav2Vec (Speech Recognition)
- ESPnet (Text-to-Speech)

### Multi-modal Models
- CLIP (Image-Text)
- LLaVA (Vision-Language)
- ImageBind (Multi-modal)

## Model Management

### Adding a New Model

1. Create a directory for your model:
   ```
   mkdir -p /models/<category>/<model_name>/1
   ```

2. Create a config file:
   ```
   mkdir -p /models/<category>/<model_name>/config
   ```

3. Convert your model to an optimized format (ONNX, TensorRT)
   ```
   /models/utils/pytorch_to_onnx.py --model source_model.pt --output /models/<category>/<model_name>/1/model.onnx --shape <input_shape>
   /models/utils/optimize_onnx.py --model /models/<category>/<model_name>/1/model.onnx --output /models/<category>/<model_name>/1/model.opt.onnx
   /models/utils/convert_to_tensorrt.py --model /models/<category>/<model_name>/1/model.opt.onnx --output /models/<category>/<model_name>/1/model.plan --precision fp16
   ```

4. Create a model configuration file in the config directory

### Syncing with S3/MinIO

To push models to S3:
```
/models/utils/sync_models.sh push
```

To pull models from S3:
```
/models/utils/sync_models.sh pull
```

## Optimization for Jetson AGX Orin

Models are optimized for the Jetson AGX Orin hardware:
- FP16 precision for most models
- INT8 quantization for selected models
- AWQ/GPTQ 4-bit quantization for large language models
- Optimized GPU memory usage
- Dynamic batching where appropriate
EOF

# Create a configuration file for the model repository
cat > "${MODEL_REPO_ROOT}/repository_config.json" << 'EOF'
{
  "repository_name": "triton_model_repo",
  "repository_version": "1.0",
  "default_batch_size": 1,
  "default_precision": "fp16",
  "repository_agent": {
    "polling_interval_seconds": 30,
    "auto_update": true,
    "notifications_enabled": true
  },
  "optimization_defaults": {
    "gpu_memory_limit_mb": 8192,
    "dynamic_batching_enabled": true,
    "batch_sizes": [1, 2, 4, 8],
    "max_queue_delay_microseconds": 5000
  },
  "jetson_agx_orin_config": {
    "compute_capability": "7.2",
    "max_gpu_memory_mb": 32768,
    "recommended_model_size_limit_mb": 16000,
    "tensorrt_optimization_level": 5
  }
}
EOF

echo -e "\n${GREEN}Model repository structure has been set up successfully!${NC}"
echo -e "Model repository location: ${MODEL_REPO_ROOT}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Deploy the Triton Inference Server pointing to this repository"
echo -e "2. Convert and optimize your models using the utility scripts"
echo -e "3. Use the example configuration files as templates for your models"
echo -e "4. Set up model synchronization with MinIO/S3 storage"
echo -e "\n${BLUE}For more information, see the README.md in the repository root.${NC}"

exit 0
