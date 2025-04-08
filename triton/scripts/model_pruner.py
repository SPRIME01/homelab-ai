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

class ModelPruner:
    """
    Prune AI models to reduce size while maintaining accuracy
    for efficient inference on NVIDIA Jetson AGX Orin
    """

    def __init__(self):
        self.download_manager = DownloadManager()
        verify_dependencies()

    def prune(
        self,
        model_source: str,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        pruning_method: str = "magnitude",
        target_sparsity: float = 0.5,
        save_onnx: bool = True
    ) -> str:
        """
        Prune a model to reduce size

        Args:
            model_source: Path or identifier for the model
            output_dir: Directory to save the pruned model
            model_type: Type of model (language, vision, speech)
            pruning_method: Pruning method (magnitude, structured, uniform, distillation)
            target_sparsity: Target sparsity level (0-1, where 1 is fully sparse)
            save_onnx: Whether to save as ONNX format

        Returns:
            Path to the pruned model
        """
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(config.output_dir, "pruned")

        os.makedirs(output_dir, exist_ok=True)

        # Download or locate the model
        model_path = self.download_manager.get_model(model_source)

        # Detect model properties if not provided
        if model_type is None:
            model_type = detect_model_type(os.path.basename(model_path))

        # Create a meaningful output filename
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        sparsity_str = str(int(target_sparsity * 100))
        output_path = os.path.join(output_dir, f"{model_name}_pruned_{sparsity_str}pct")
        if save_onnx:
            output_path += ".onnx"

        # Prune based on method
        if pruning_method == "magnitude":
            return self._prune_magnitude(model_path, output_path, model_type, target_sparsity, save_onnx)
        elif pruning_method == "structured":
            return self._prune_structured(model_path, output_path, model_type, target_sparsity, save_onnx)
        elif pruning_method == "uniform":
            return self._prune_uniform(model_path, output_path, model_type, target_sparsity, save_onnx)
        elif pruning_method == "distillation":
            return self._prune_distillation(model_path, output_path, model_type, target_sparsity, save_onnx)
        else:
            logger.error(f"Unsupported pruning method: {pruning_method}")
            raise ValueError(f"Unsupported pruning method: {pruning_method}")

    def _prune_magnitude(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        target_sparsity: float = 0.5,
        save_onnx: bool = True
    ) -> str:
        """Prune a model using magnitude-based weight pruning"""
        try:
            import torch
            from torch.nn.utils import prune

            logger.info(f"Pruning model {model_path} with magnitude pruning (target sparsity: {target_sparsity:.2f})")

            # Check if it's a PyTorch model
            if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
                logger.error("Magnitude pruning currently only supports PyTorch models")
                raise ValueError("Unsupported model format for magnitude pruning")

            # Load the model
            model = torch.load(model_path, map_location="cpu")
            if isinstance(model, dict):  # state dict
                logger.error("Please provide a full model, not just state_dict for pruning")
                raise ValueError("State dict pruning not supported")

            model.eval()  # Set model to evaluation mode

            # Apply pruning to all parameters
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

            # Global pruning (across all parameters)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=target_sparsity,
            )

            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)

            # Save pruned model
            if not save_onnx:
                # Save as PyTorch model
                if not output_path.endswith('.pt') and not output_path.endswith('.pth'):
                    output_path += '.pt'
                torch.save(model, output_path)
            else:
                # Save as ONNX model
                input_tensor = self._create_dummy_input(model, model_type)
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True
                )

            logger.info(f"Model pruned with magnitude-based pruning and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "pruning_method": "magnitude",
                "target_sparsity": target_sparsity,
                "model_type": model_type
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to prune model with magnitude pruning: {e}")
            raise

    def _prune_structured(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        target_sparsity: float = 0.5,
        save_onnx: bool = True
    ) -> str:
        """Prune a model using structured pruning"""
        try:
            import torch
            from torch.nn.utils import prune

            logger.info(f"Pruning model {model_path} with structured pruning (target sparsity: {target_sparsity:.2f})")

            # Check if it's a PyTorch model
            if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
                logger.error("Structured pruning currently only supports PyTorch models")
                raise ValueError("Unsupported model format for structured pruning")

            # Load the model
            model = torch.load(model_path, map_location="cpu")
            if isinstance(model, dict):  # state dict
                logger.error("Please provide a full model, not just state_dict for pruning")
                raise ValueError("State dict pruning not supported")

            model.eval()  # Set model to evaluation mode

            # Apply structured pruning to convolutional layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=target_sparsity,
                        n=2,  # L2 norm for structured pruning
                        dim=0  # Prune output channels
                    )
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=target_sparsity,
                        n=2,  # L2 norm for structured pruning
                        dim=0  # Prune output features
                    )

            # Make pruning permanent
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    try:
                        prune.remove(module, 'weight')
                    except:
                        pass  # Skip if weight wasn't pruned

            # Save pruned model
            if not save_onnx:
                # Save as PyTorch model
                if not output_path.endswith('.pt') and not output_path.endswith('.pth'):
                    output_path += '.pt'
                torch.save(model, output_path)
            else:
                # Save as ONNX model
                input_tensor = self._create_dummy_input(model, model_type)
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True
                )

            logger.info(f"Model pruned with structured pruning and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "pruning_method": "structured",
                "target_sparsity": target_sparsity,
                "model_type": model_type
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to prune model with structured pruning: {e}")
            raise

    def _prune_uniform(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        target_sparsity: float = 0.5,
        save_onnx: bool = True
    ) -> str:
        """Prune a model using uniform pruning"""
        try:
            import torch
            from torch.nn.utils import prune

            logger.info(f"Pruning model {model_path} with uniform pruning (target sparsity: {target_sparsity:.2f})")

            # Check if it's a PyTorch model
            if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
                logger.error("Uniform pruning currently only supports PyTorch models")
                raise ValueError("Unsupported model format for uniform pruning")

            # Load the model
            model = torch.load(model_path, map_location="cpu")
            if isinstance(model, dict):  # state dict
                logger.error("Please provide a full model, not just state_dict for pruning")
                raise ValueError("State dict pruning not supported")

            model.eval()  # Set model to evaluation mode

            # Apply uniform pruning to all parameters
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    prune.random_unstructured(module, name='weight', amount=target_sparsity)

            # Make pruning permanent
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    try:
                        prune.remove(module, 'weight')
                    except:
                        pass  # Skip if weight wasn't pruned

            # Save pruned model
            if not save_onnx:
                # Save as PyTorch model
                if not output_path.endswith('.pt') and not output_path.endswith('.pth'):
                    output_path += '.pt'
                torch.save(model, output_path)
            else:
                # Save as ONNX model
                input_tensor = self._create_dummy_input(model, model_type)
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True
                )

            logger.info(f"Model pruned with uniform pruning and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "pruning_method": "uniform",
                "target_sparsity": target_sparsity,
                "model_type": model_type
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to prune model with uniform pruning: {e}")
            raise

    def _prune_distillation(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        target_sparsity: float = 0.5,
        save_onnx: bool = True
    ) -> str:
        """Prune a model using knowledge distillation (creating a smaller model)"""
        try:
            import torch

            logger.info(f"Pruning model {model_path} with knowledge distillation")
            logger.warning("Knowledge distillation pruning requires training data and is resource-intensive")
            logger.warning("This is a simplified implementation and might not work optimally for all models")

            # Check if it's a PyTorch model
            if not model_path.endswith('.pt') and not model_path.endswith('.pth'):
                logger.error("Knowledge distillation currently only supports PyTorch models")
                raise ValueError("Unsupported model format for knowledge distillation")

            # Load the model
            teacher_model = torch.load(model_path, map_location="cpu")
            if isinstance(teacher_model, dict):  # state dict
                logger.error("Please provide a full model, not just state_dict for pruning")
                raise ValueError("State dict pruning not supported")

            teacher_model.eval()  # Set model to evaluation mode

            # Create a smaller student model
            # Note: This is a simplified approach and in practice, would need to be
            # tailored to the specific model architecture
            if model_type == "vision":
                student_model = self._create_student_model_vision(teacher_model, target_sparsity)
            elif model_type == "language":
                student_model = self._create_student_model_language(teacher_model, target_sparsity)
            else:
                logger.error(f"Knowledge distillation not implemented for model type: {model_type}")
                raise ValueError(f"Unsupported model type for knowledge distillation: {model_type}")

            # In a real implementation, we would perform knowledge distillation training here
            # This would involve:
            # 1. Loading training data
            # 2. Forward pass through both teacher and student models
            # 3. Computing distillation loss
            # 4. Updating student model weights

            logger.warning("Skipping actual distillation training - in a real scenario, this would require data and computation")

            # Save the student model
            if not save_onnx:
                # Save as PyTorch model
                if not output_path.endswith('.pt') and not output_path.endswith('.pth'):
                    output_path += '.pt'
                torch.save(student_model, output_path)
            else:
                # Save as ONNX model
                input_tensor = self._create_dummy_input(student_model, model_type)
                torch.onnx.export(
                    student_model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True
                )

            logger.info(f"Model pruned with knowledge distillation and saved to {output_path}")

            # Create model info file
            metadata = get_model_metadata(output_path)
            metadata.update({
                "original_model": model_path,
                "pruning_method": "distillation",
                "target_sparsity": target_sparsity,
                "model_type": model_type,
                "teacher_model": os.path.basename(model_path)
            })
            create_model_info_file(output_path, metadata)

            return output_path

        except Exception as e:
            logger.error(f"Failed to prune model with knowledge distillation: {e}")
            raise

    def _create_student_model_vision(self, teacher_model, target_sparsity):
        """Create a smaller student model for vision tasks"""
        # This is a simplified approach - in a real implementation, you would
        # create a proper student architecture based on the teacher
        import torch.nn as nn

        # Basic implementation - reducing number of filters in convolutional layers
        class SmallerModel(nn.Module):
            def __init__(self, teacher_model):
                super().__init__()
                # Extract input and output dimensions from teacher
                self.input_size = self._find_input_size(teacher_model)
                self.output_size = self._find_output_size(teacher_model)

                # Create a simplified model
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(128, self.output_size)

            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

            def _find_input_size(self, model):
                # Try to determine input size from model
                return 3  # Default to 3 channels

            def _find_output_size(self, model):
                # Try to determine output size from model
                if hasattr(model, 'fc'):
                    return model.fc.out_features
                elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
                    return model.classifier.out_features
                else:
                    return 1000  # Default to 1000 classes

        return SmallerModel(teacher_model)

    def _create_student_model_language(self, teacher_model, target_sparsity):
        """Create a smaller student model for language tasks"""
        # This is a simplified approach - in a real implementation, you would
        # create a proper student architecture based on the teacher
        import torch.nn as nn

        # Basic implementation - reducing model dimensions
        class SmallerLanguageModel(nn.Module):
            def __init__(self, teacher_model):
                super().__init__()
                # Extract dimensions from teacher
                self.vocab_size = self._find_vocab_size(teacher_model)
                self.hidden_size = self._find_hidden_size(teacher_model)
                reduced_hidden = max(64, int(self.hidden_size * (1 - target_sparsity)))

                # Create a simplified model
                self.embeddings = nn.Embedding(self.vocab_size, reduced_hidden)
                self.transformer = nn.TransformerEncoderLayer(
                    d_model=reduced_hidden,
                    nhead=4,
                    dim_feedforward=reduced_hidden * 2,
                    batch_first=True
                )
                self.output = nn.Linear(reduced_hidden, self.vocab_size)

            def forward(self, x):
                x = self.embeddings(x)
                x = self.transformer(x)
                x = self.output(x)
                return x

            def _find_vocab_size(self, model):
                # Try to determine vocabulary size from model
                if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
                    return model.embeddings.word_embeddings.num_embeddings
                else:
                    return 30000  # Default

            def _find_hidden_size(self, model):
                # Try to determine hidden size from model
                if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                    return model.config.hidden_size
                elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
                    return model.embeddings.word_embeddings.embedding_dim
                else:
                    return 768  # Default

        return SmallerLanguageModel(teacher_model)

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
    """Command line interface for the model pruner"""
    import argparse

    parser = argparse.ArgumentParser(description='Prune models to reduce size')
    parser.add_argument('model', type=str, help='Path or identifier for the input model')
    parser.add_argument('--output', '-o', type=str, help='Output directory or file path')
    parser.add_argument('--model-type', '-t', type=str, help='Model type (language, vision, speech)')
    parser.add_argument('--method', '-m', choices=['magnitude', 'structured', 'uniform', 'distillation'],
                      default='magnitude', help='Pruning method (default: magnitude)')
    parser.add_argument('--sparsity', '-s', type=float, default=0.5,
                      help='Target sparsity level (0-1, default: 0.5)')
    parser.add_argument('--no-onnx', action='store_true', help='Do not save as ONNX format')

    args = parser.parse_args()

    # Validate sparsity level
    if args.sparsity < 0 or args.sparsity > 0.95:
        parser.error("Sparsity level must be between 0 and 0.95")

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


if __name__ == "__main__":
    main()
