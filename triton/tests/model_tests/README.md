# Triton Inference Server Model Testing Framework

This package provides a Test-Driven Development (TDD) framework for testing models deployed on NVIDIA Triton Inference Server. It supports comprehensive testing throughout the model development lifecycle and integration into CI/CD pipelines.

## Features

- **Model Loading Tests**: Validate that models load correctly with proper metadata and configuration
- **Input Validation**: Test model behavior with valid and invalid inputs
- **Inference Correctness**: Verify model outputs against reference data
- **Performance Requirements**: Test throughput, latency, and resource usage
- **Error Handling**: Validate model behavior under error conditions
- **Comprehensive Reporting**: Generate detailed HTML and optional JUnit XML reports

## Available Test Classes

- `BaseModelTest`: Base class for all model tests
- `LanguageModelTest`: Specialized test class for language models
- `VisionModelTest`: Specialized test class for vision models

## Usage

### Basic Usage

```bash
# Run tests for all models
python -m homelab-ai.triton.tests.model_tests

# Run tests for specific models
python -m homelab-ai.triton.tests.model_tests --models llama2-7b-q4 yolov8n

# Connect to a specific Triton server
python -m homelab-ai.triton.tests.model_tests --url triton-server:8000
```

### Advanced Options

```bash
# Use configuration file
python -m homelab-ai.triton.tests.model_tests --config-file my_config.yaml

# Generate JUnit XML reports for CI integration
python -m homelab-ai.triton.tests.model_tests --junit

# Test Triton server deployed in Kubernetes
python -m homelab-ai.triton.tests.model_tests --kubernetes --namespace ai --service triton-inference
```

## Creating Tests for a New Model

1. Create a new Python file for your model tests:

```python
from model_tests import BaseModelTest
# or use a specialized test class:
# from model_tests import LanguageModelTest, VisionModelTest

class MyModelTest(BaseModelTest):
    # Specify the model to test
    model_name = "my_model"
    model_version = "1"

    @classmethod
    def setup_test_data(cls):
        # Setup test data for your model
        pass

    def test_basic_inference(self):
        # Test basic inference functionality
        pass

    def test_performance_requirements(self):
        # Test that model meets performance requirements
        pass
```

2. Add the model test to your test directory
3. Run the tests using the command-line interface

## Configuration

You can configure the tests using:

1. A YAML configuration file
2. Command-line arguments
3. Environment variables

Example configuration file:

```yaml
url: localhost:8000
protocol: http
kubernetes:
  enabled: false
  namespace: ai
  service: triton-inference-server
performance:
  max_latency_ms: 100
  min_throughput: 10
  batch_sizes: [1, 2, 4, 8]
```

## CI/CD Integration

This framework is designed for easy integration with CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Model Tests

on:
  push:
    paths:
      - 'models/**'
      - 'triton/**'

jobs:
  test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v2
      - name: Run model tests
        run: |
          python -m homelab-ai.triton.tests.model_tests \
            --kubernetes \
            --namespace ai \
            --junit
```

## HTML Reports

The framework generates detailed HTML reports with:

- Test results for each model
- Performance metrics with visualizations
- Input/output validation results
- Summary report across all models

Reports are saved to the `test_results` directory by default.
