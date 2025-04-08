import os
import time
import logging
import numpy as np
from typing import Dict, List, Optional
import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm

from model_quantizer import ModelQuantizer

logger = logging.getLogger(__name__)

class LLMBenchmark:
    """Benchmark quantized LLMs for accuracy and performance"""

    def __init__(self):
        self.perplexity = evaluate.load("perplexity")
        self.quantizer = ModelQuantizer()

    def run_benchmark(
        self,
        model_path: str,
        quantization_methods: List[str] = ["gptq", "awq", "gguf"],
        bits: List[int] = [4, 8],
        benchmark_dataset: str = "wikitext-2-raw-v1",
        num_samples: int = 100,
        max_length: int = 512
    ) -> Dict:
        """Run comprehensive benchmarks on quantized models"""
        results = {}

        # Load original model for baseline
        original_model = self._load_model(model_path)
        baseline_metrics = self._evaluate_model(original_model, benchmark_dataset, num_samples, max_length)
        results["baseline"] = baseline_metrics

        # Test each quantization method and bit depth
        for method in quantization_methods:
            for bit in bits:
                try:
                    # Quantize model
                    quantized_path = self.quantizer.quantize_llm(
                        model_path,
                        quantization_method=method,
                        bits=bit,
                        validation_dataset=benchmark_dataset
                    )

                    # Load quantized model
                    quantized_model = self._load_model(quantized_path)

                    # Evaluate
                    metrics = self._evaluate_model(quantized_model, benchmark_dataset, num_samples, max_length)

                    # Calculate relative metrics
                    relative_metrics = {
                        "perplexity_increase": metrics["perplexity"] / baseline_metrics["perplexity"],
                        "latency_reduction": baseline_metrics["latency"] / metrics["latency"],
                        "memory_reduction": baseline_metrics["memory_used"] / metrics["memory_used"]
                    }

                    results[f"{method}_{bit}bit"] = {
                        **metrics,
                        "relative": relative_metrics
                    }

                except Exception as e:
                    logger.error(f"Failed to benchmark {method} {bit}-bit: {e}")
                    results[f"{method}_{bit}bit"] = {"error": str(e)}

        return results

    def _evaluate_model(
        self,
        model,
        dataset_name: str,
        num_samples: int,
        max_length: int
    ) -> Dict:
        """Evaluate model performance metrics"""
        # Load dataset
        dataset = load_dataset(dataset_name, split="test")
        eval_samples = dataset.select(range(num_samples))

        # Track metrics
        latencies = []
        memory_usage = []
        total_tokens = 0

        # Evaluate perplexity and performance
        model.eval()
        with torch.no_grad():
            for sample in tqdm(eval_samples, desc="Evaluating"):
                # Tokenize
                inputs = model.tokenizer(
                    sample["text"],
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)

                # Measure inference
                start_time = time.time()
                outputs = model(**inputs)
                latencies.append(time.time() - start_time)

                # Track memory
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                total_tokens += inputs.input_ids.size(1)

        # Calculate perplexity
        perplexity = self.perplexity.compute(
            predictions=outputs.logits.cpu(),
            model_id=model.config._name_or_path
        )["perplexity"]

        # Calculate throughput
        avg_latency = np.mean(latencies)
        throughput = total_tokens / sum(latencies)

        return {
            "perplexity": perplexity,
            "latency": avg_latency,
            "latency_p90": np.percentile(latencies, 90),
            "latency_p99": np.percentile(latencies, 99),
            "throughput": throughput,
            "memory_used": np.mean(memory_usage),
            "memory_peak": np.max(memory_usage)
        }

def main():
    """Run benchmarks from command line"""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark quantized LLMs")
    parser.add_argument("model", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--methods", type=str, default="gptq,awq,gguf",
                      help="Comma-separated list of quantization methods")
    parser.add_argument("--bits", type=str, default="4,8",
                      help="Comma-separated list of bit depths")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1",
                      help="Dataset for evaluation")
    parser.add_argument("--samples", type=int, default=100,
                      help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Run benchmarks
    benchmark = LLMBenchmark()
    results = benchmark.run_benchmark(
        args.model,
        quantization_methods=args.methods.split(","),
        bits=[int(b) for b in args.bits.split(",")],
        benchmark_dataset=args.dataset,
        num_samples=args.samples
    )

    # Save or display results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
