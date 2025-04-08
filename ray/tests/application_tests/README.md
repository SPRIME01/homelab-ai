# Ray Application Testing Framework

A comprehensive testing framework for Ray applications following Test-Driven Development (TDD) principles. This framework is designed to validate distributed AI workloads and ensure reliable operation of Ray-based applications.

## Features

- Task distribution testing
- Resource allocation validation
- Fault tolerance verification
- Service integration testing
- AI workload performance testing
- Comprehensive reporting
- CI/CD pipeline integration

## Test Categories

1. **Task Distribution Tests**
   - Basic task distribution
   - Task locality
   - Task dependencies
   - Concurrent task scaling

2. **Resource Allocation Tests**
   - CPU allocation
   - Memory allocation
   - GPU allocation
   - Custom resource handling

3. **Fault Tolerance Tests**
   - Task failure recovery
   - Actor failure recovery
   - Distributed state recovery
   - Load balancing with failures

4. **Service Integration Tests**
   - Triton Inference Server integration
   - Remote data access
   - Database integration

5. **AI Workload Tests**
   - Distributed data processing
   - Distributed training
   - Inference scaling
   - Hyperparameter tuning

## Usage

### Basic Usage

```bash
# Run all tests
python -m ray.tests.application_tests.run_tests

# Run specific test categories
python -m ray.tests.application_tests.run_tests --tests task resource fault

# Connect to specific Ray cluster
python -m ray.tests.application_tests.run_tests --ray-address="ray://localhost:10001"
```

### Advanced Options

```bash
# Generate HTML report
python -m ray.tests.application_tests.run_tests --html-report

# Generate JUnit XML report for CI/CD
python -m ray.tests.application_tests.run_tests --junit-xml=results.xml

# Use custom configuration
python -m ray.tests.application_tests.run_tests --config-file=my_config.yaml
```

## Writing Tests

### Creating a New Test Class

```python
from ray.tests.application_tests import RayBaseTest

class MyCustomTest(RayBaseTest):
    """Custom test class for specific functionality."""

    application_name = "my_app"

    def test_my_feature(self):
        """Test specific feature."""
        # Your test code here
        pass
```

### Test Configuration

Create a YAML configuration file:

```yaml
ray:
  address: "auto"
  namespace: "test"
  runtime_env:
    pip: ["numpy", "pandas"]
resources:
  cpu_tests: [1, 2, 4]
  gpu_tests: [0.25, 0.5, 1.0]
  memory_tests: ["500MB", "1GB"]
```

## Test Reports

The framework generates:

1. Console output with detailed logs
2. JSON summary of test results
3. HTML report with visualizations (optional)
4. JUnit XML report for CI/CD integration (optional)

## Directory Structure

