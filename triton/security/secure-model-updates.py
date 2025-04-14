import os
import sys
import json
import shutil
import hashlib
import logging
import requests
import subprocess
import time
import yaml
import threading
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/triton/model_updates.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("secure-model-updates")

class ModelVerifier:
    """Handles verification of model files using checksums or signatures."""

    def __init__(self, public_key_path: Optional[str] = None):
        self.public_key_path = public_key_path
        self.public_key = None
        if public_key_path:
            try:
                with open(public_key_path, 'rb') as key_file:
                    self.public_key = load_pem_public_key(key_file.read())
            except Exception as e:
                logger.error(f"Failed to load public key: {str(e)}")

    def verify_checksum(self, file_path: str, expected_checksum: str,
                        algorithm: str = 'sha256') -> bool:
        """Verify file integrity using checksum."""
        try:
            if algorithm == 'sha256':
                hasher = hashlib.sha256()
            elif algorithm == 'md5':
                hasher = hashlib.md5()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")

            with open(file_path, 'rb') as f:
                buffer = f.read(65536)  # Read in 64k chunks
                while len(buffer) > 0:
                    hasher.update(buffer)
                    buffer = f.read(65536)

            computed_hash = hasher.hexdigest()
            result = computed_hash == expected_checksum

            if result:
                logger.info(f"Checksum verification successful for {file_path}")
            else:
                logger.error(f"Checksum verification failed for {file_path}. "
                            f"Expected: {expected_checksum}, Got: {computed_hash}")

            return result
        except Exception as e:
            logger.error(f"Error during checksum verification: {str(e)}")
            return False

    def verify_signature(self, file_path: str, signature_path: str) -> bool:
        """Verify file integrity using digital signature."""
        if not self.public_key:
            logger.error("No public key loaded for signature verification")
            return False

        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()

            with open(signature_path, 'rb') as f:
                signature = f.read()

            try:
                self.public_key.verify(
                    signature,
                    file_data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                logger.info(f"Signature verification successful for {file_path}")
                return True
            except Exception:
                logger.error(f"Signature verification failed for {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error during signature verification: {str(e)}")
            return False


class ModelValidator:
    """Validates model metadata and configuration."""

    def __init__(self, schema_dir: Optional[str] = None):
        self.schema_dir = schema_dir or "/home/sprime01/homelab/homelab-ai/triton/schemas"

    def validate_config(self, model_config_path: str) -> Tuple[bool, List[str]]:
        """Validate model configuration against schema."""
        try:
            with open(model_config_path, 'r') as f:
                config = yaml.safe_load(f)

            errors = []

            # Basic validation
            required_fields = ['name', 'platform', 'max_batch_size', 'input', 'output']
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field: {field}")

            # Validate inputs and outputs
            if 'input' in config and isinstance(config['input'], list):
                for idx, inp in enumerate(config['input']):
                    if 'name' not in inp or 'data_type' not in inp or 'dims' not in inp:
                        errors.append(f"Input {idx} missing required properties")

            if 'output' in config and isinstance(config['output'], list):
                for idx, out in enumerate(config['output']):
                    if 'name' not in out or 'data_type' not in out or 'dims' not in out:
                        errors.append(f"Output {idx} missing required properties")

            is_valid = len(errors) == 0
            if is_valid:
                logger.info(f"Model config validation successful for {model_config_path}")
            else:
                logger.error(f"Model config validation failed for {model_config_path}: {errors}")

            return is_valid, errors
        except Exception as e:
            logger.error(f"Error validating model config: {str(e)}")
            return False, [str(e)]

    def validate_model_structure(self, model_dir: str) -> Tuple[bool, List[str]]:
        """Validate the structure of the model directory."""
        try:
            errors = []
            required_files = ['config.pbtxt']

            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    errors.append(f"Missing required file: {file}")

            is_valid = len(errors) == 0
            if is_valid:
                logger.info(f"Model structure validation successful for {model_dir}")
            else:
                logger.error(f"Model structure validation failed for {model_dir}: {errors}")

            return is_valid, errors
        except Exception as e:
            logger.error(f"Error validating model structure: {str(e)}")
            return False, [str(e)]


class CanaryDeployer:
    """Handles canary deployment of models."""

    def __init__(self, triton_server_url: str, model_repository: str):
        self.triton_url = triton_server_url
        self.model_repo = model_repository

    def deploy_canary(self, model_name: str, version: int,
                       traffic_percentage: int = 10) -> bool:
        """Deploy a model as canary with specified traffic percentage."""
        try:
            # Create a canary-specific configuration
            canary_config_path = os.path.join(self.model_repo, model_name,
                                             f"config.pbtxt.canary")

            # Copy original config
            original_config = os.path.join(self.model_repo, model_name, "config.pbtxt")
            shutil.copy(original_config, canary_config_path)

            # Modify the copy with canary settings
            with open(canary_config_path, "a") as f:
                f.write(f"\n# Canary deployment configuration\n")
                f.write(f"version_policy {{\n")
                f.write(f"  specific {{\n")
                f.write(f"    versions: [{version}]\n")
                f.write(f"    traffic_percentage: {traffic_percentage}\n")
                f.write(f"  }}\n")
                f.write(f"}}\n")

            # Apply the canary configuration
            shutil.move(canary_config_path, original_config)

            # Reload the model via Triton API
            response = requests.post(
                f"{self.triton_url}/v2/repository/models/{model_name}/load"
            )

            if response.status_code == 200:
                logger.info(f"Canary deployment successful for {model_name} v{version} "
                           f"with {traffic_percentage}% traffic")
                return True
            else:
                logger.error(f"Canary deployment failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error during canary deployment: {str(e)}")
            return False

    def promote_canary(self, model_name: str, version: int) -> bool:
        """Promote canary to full production traffic."""
        try:
            # Update config to direct 100% traffic to the canary version
            config_path = os.path.join(self.model_repo, model_name, "config.pbtxt")

            with open(config_path, "r") as f:
                config_content = f.read()

            # Replace traffic percentage with 100%
            if "traffic_percentage:" in config_content:
                config_content = config_content.replace(
                    "traffic_percentage:", "traffic_percentage: 100 #"
                )

            with open(config_path, "w") as f:
                f.write(config_content)

            # Reload the model
            response = requests.post(
                f"{self.triton_url}/v2/repository/models/{model_name}/load"
            )

            if response.status_code == 200:
                logger.info(f"Canary promotion successful for {model_name} v{version}")
                return True
            else:
                logger.error(f"Canary promotion failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error during canary promotion: {str(e)}")
            return False


class ModelTester:
    """Handles automated testing of models before deployment."""

    def __init__(self, triton_server_url: str, test_data_dir: str):
        self.triton_url = triton_server_url
        self.test_data_dir = test_data_dir

    def load_test_data(self, model_name: str) -> List[Dict]:
        """Load test data for a specific model."""
        try:
            test_file = os.path.join(self.test_data_dir, f"{model_name}_test_data.json")
            if not os.path.exists(test_file):
                logger.warning(f"No test data found for {model_name}")
                return []

            with open(test_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return []

    def run_performance_test(self, model_name: str,
                             batch_sizes: List[int] = [1, 4, 8],
                             concurrency: int = 4,
                             duration_ms: int = 5000) -> Tuple[bool, Dict]:
        """Run performance testing on the model using perf_analyzer."""
        try:
            results = {}
            success = True

            for batch_size in batch_sizes:
                cmd = [
                    "perf_analyzer",
                    "-m", model_name,
                    "-b", str(batch_size),
                    "-p", str(duration_ms),
                    "-c", str(concurrency),
                    "--json-output"
                ]

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )

                if process.returncode != 0:
                    logger.error(f"Performance test failed for {model_name}: {process.stderr}")
                    success = False
                    continue

                try:
                    perf_results = json.loads(process.stdout)
                    results[f"batch_{batch_size}"] = {
                        "throughput": perf_results.get("throughput", 0),
                        "latency_avg": perf_results.get("latency_avg", 0),
                        "latency_p90": perf_results.get("latency_p90", 0),
                        "latency_p95": perf_results.get("latency_p95", 0),
                        "latency_p99": perf_results.get("latency_p99", 0),
                    }
                    logger.info(f"Performance test completed for {model_name} with batch size {batch_size}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse performance results for {model_name}")
                    success = False

            return success, results

        except Exception as e:
            logger.error(f"Error during performance testing: {str(e)}")
            return False, {}

    def run_inference_tests(self, model_name: str) -> Tuple[bool, Dict]:
        """Run functional inference tests on the model."""
        try:
            test_data = self.load_test_data(model_name)
            if not test_data:
                return False, {"error": "No test data available"}

            results = {"passed": 0, "failed": 0, "details": []}

            for i, test_case in enumerate(test_data):
                inputs = test_case.get("inputs", [])
                expected_outputs = test_case.get("expected_outputs", [])

                # Prepare the inference request
                request_data = {
                    "inputs": inputs,
                    "outputs": []
                }

                # Send inference request to Triton
                response = requests.post(
                    f"{self.triton_url}/v2/models/{model_name}/infer",
                    json=request_data
                )

                if response.status_code != 200:
                    results["failed"] += 1
                    results["details"].append({
                        "test_case": i,
                        "status": "failed",
                        "error": f"HTTP Error: {response.status_code}",
                        "message": response.text
                    })
                    continue

                # Process the response
                response_data = response.json()
                actual_outputs = response_data.get("outputs", [])

                # Compare with expected outputs (simplified)
                test_passed = len(actual_outputs) == len(expected_outputs)

                if test_passed:
                    results["passed"] += 1
                    results["details"].append({
                        "test_case": i,
                        "status": "passed"
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "test_case": i,
                        "status": "failed",
                        "expected": expected_outputs,
                        "actual": actual_outputs
                    })

            success = results["failed"] == 0 and results["passed"] > 0
            if success:
                logger.info(f"All {results['passed']} inference tests passed for {model_name}")
            else:
                logger.error(f"Inference tests: {results['passed']} passed, {results['failed']} failed for {model_name}")

            return success, results

        except Exception as e:
            logger.error(f"Error during inference testing: {str(e)}")
            return False, {"error": str(e)}


class RollbackManager:
    """Handles rollback of failed model updates."""

    def __init__(self, model_repository: str, backup_dir: str):
        self.model_repo = model_repository
        self.backup_dir = backup_dir
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

    def backup_model(self, model_name: str, version: int) -> str:
        """Create a backup of the current model before updating."""
        try:
            # Define paths
            model_path = os.path.join(self.model_repo, model_name, str(version))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"{model_name}_{version}_{timestamp}")

            # Create backup
            if os.path.exists(model_path):
                shutil.copytree(model_path, backup_path)
                logger.info(f"Backup created for {model_name} v{version} at {backup_path}")

                # Also backup the config
                config_path = os.path.join(self.model_repo, model_name, "config.pbtxt")
                if os.path.exists(config_path):
                    shutil.copy2(
                        config_path,
                        os.path.join(self.backup_dir, f"{model_name}_config_{timestamp}.pbtxt")
                    )

                return backup_path
            else:
                logger.warning(f"Model {model_name} v{version} not found, skipping backup")
                return ""
        except Exception as e:
            logger.error(f"Error during model backup: {str(e)}")
            return ""

    def rollback(self, model_name: str, version: int, backup_path: str) -> bool:
        """Rollback to a previous version of the model."""
        try:
            if not backup_path or not os.path.exists(backup_path):
                logger.error(f"Backup path {backup_path} does not exist for rollback")
                return False

            # Define paths
            model_path = os.path.join(self.model_repo, model_name, str(version))

            # Remove current version
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            # Restore from backup
            shutil.copytree(backup_path, model_path)

            # Find and restore config if needed
            backup_dir = os.path.dirname(backup_path)
            timestamp = os.path.basename(backup_path).split('_')[-1]
            config_backup = os.path.join(backup_dir, f"{model_name}_config_{timestamp}.pbtxt")

            if os.path.exists(config_backup):
                shutil.copy2(
                    config_backup,
                    os.path.join(self.model_repo, model_name, "config.pbtxt")
                )

            logger.info(f"Rollback successful for {model_name} v{version} from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
            return False


class ModelUpdateManager:
    """Main class to orchestrate the secure model update process."""

    def __init__(self,
                 triton_url: str = "http://localhost:8000",
                 model_repository: str = "/home/sprime01/homelab/models",
                 backup_dir: str = "/home/sprime01/homelab/model_backups",
                 test_data_dir: str = "/home/sprime01/homelab/test_data",
                 public_key_path: Optional[str] = None):

        self.triton_url = triton_url
        self.model_repo = model_repository

        # Initialize components
        self.verifier = ModelVerifier(public_key_path)
        self.validator = ModelValidator()
        self.canary = CanaryDeployer(triton_url, model_repository)
        self.tester = ModelTester(triton_url, test_data_dir)
        self.rollback = RollbackManager(model_repository, backup_dir)

        # Thresholds for automated decisions
        self.perf_degradation_threshold = 0.2  # 20% degradation allowed
        self.canary_traffic_percentage = 10    # 10% traffic to canary
        self.canary_monitoring_duration = 600  # 10 minutes

    def update_model(self,
                     model_name: str,
                     version: int,
                     model_files: Dict[str, str],
                     checksums: Dict[str, str],
                     skip_tests: bool = False,
                     skip_canary: bool = False,
                     force: bool = False) -> bool:
        """
        Execute the complete model update process.

        Args:
            model_name: Name of the model to update
            version: Version number for the update
            model_files: Dict of {file_path: source_path} to copy
            checksums: Dict of {file_path: expected_checksum} for verification
            skip_tests: Skip testing phase
            skip_canary: Skip canary deployment
            force: Force update even if some checks fail

        Returns:
            bool: True if update was successful
        """
        try:
            logger.info(f"Starting update process for {model_name} v{version}")

            # Step 1: Create backup
            backup_path = self.rollback.backup_model(model_name, version)
            if not backup_path and not force:
                logger.error("Failed to create backup, aborting update")
                return False

            # Step 2: Copy and verify model files
            model_dir = os.path.join(self.model_repo, model_name, str(version))
            os.makedirs(model_dir, exist_ok=True)

            verification_failed = False
            for dest_rel_path, source_path in model_files.items():
                dest_path = os.path.join(model_dir, dest_rel_path)

                # Create directories if needed
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copy the file
                shutil.copy2(source_path, dest_path)

                # Verify checksum if provided
                if dest_rel_path in checksums:
                    if not self.verifier.verify_checksum(dest_path, checksums[dest_rel_path]):
                        verification_failed = True
                        if not force:
                            logger.error(f"Checksum verification failed for {dest_path}, aborting update")
                            self._cleanup_and_rollback(model_name, version, backup_path)
                            return False

            # Step 3: Validate model configuration
            config_path = os.path.join(self.model_repo, model_name, "config.pbtxt")
            is_valid, errors = self.validator.validate_config(config_path)
            if not is_valid and not force:
                logger.error(f"Model config validation failed: {errors}")
                self._cleanup_and_rollback(model_name, version, backup_path)
                return False

            # Also validate model structure
            is_valid, errors = self.validator.validate_model_structure(os.path.join(self.model_repo, model_name))
            if not is_valid and not force:
                logger.error(f"Model structure validation failed: {errors}")
                self._cleanup_and_rollback(model_name, version, backup_path)
                return False

            # Step 4: Run automated tests if not skipped
            if not skip_tests:
                # Run inference tests
                test_success, test_results = self.tester.run_inference_tests(model_name)
                if not test_success and not force:
                    logger.error(f"Inference tests failed: {test_results}")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False

                # Run performance tests
                perf_success, perf_results = self.tester.run_performance_test(model_name)
                if not perf_success and not force:
                    logger.error(f"Performance tests failed: {perf_results}")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False

            # Step 5: Deploy with canary if not skipped
            if not skip_canary:
                # Deploy as canary with limited traffic
                if not self.canary.deploy_canary(model_name, version, self.canary_traffic_percentage):
                    logger.error("Canary deployment failed")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False

                # Monitor canary deployment
                canary_success = self._monitor_canary(model_name, version)
                if not canary_success and not force:
                    logger.error("Canary monitoring detected issues, rolling back")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False

                # Promote canary to full traffic
                if not self.canary.promote_canary(model_name, version):
                    logger.error("Canary promotion failed")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False
            else:
                # Direct deployment without canary
                logger.info(f"Skipping canary, deploying {model_name} v{version} directly")

                # Load the model via Triton API
                response = requests.post(
                    f"{self.triton_url}/v2/repository/models/{model_name}/load"
                )

                if response.status_code != 200:
                    logger.error(f"Model loading failed: {response.text}")
                    self._cleanup_and_rollback(model_name, version, backup_path)
                    return False

            logger.info(f"Model update successful for {model_name} v{version}")
            return True

        except Exception as e:
            logger.error(f"Error during model update: {str(e)}")
            if backup_path:
                self._cleanup_and_rollback(model_name, version, backup_path)
            return False

    def _cleanup_and_rollback(self, model_name: str, version: int, backup_path: str) -> None:
        """Clean up and roll back in case of failure."""
        logger.info(f"Initiating rollback for {model_name} v{version}")
        self.rollback.rollback(model_name, version, backup_path)

    def _monitor_canary(self, model_name: str, version: int) -> bool:
        """Monitor canary deployment for issues."""
        logger.info(f"Monitoring canary deployment for {model_name} v{version} for "
                   f"{self.canary_monitoring_duration} seconds")

        # Simple implementation - wait and check health
        start_time = time.time()
        end_time = start_time + self.canary_monitoring_duration

        # Check model health periodically
        while time.time() < end_time:
            try:
                # Check model health via Triton API
                response = requests.get(
                    f"{self.triton_url}/v2/health/ready"
                )

                if response.status_code != 200:
                    logger.error(f"Triton server health check failed during canary monitoring")
                    return False

                # Check model metrics if available
                # This is a simplified placeholder - in a real implementation,
                # you would check error rates, latency, etc.

                # Sleep before next check
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error during canary monitoring: {str(e)}")
                return False

        logger.info(f"Canary monitoring completed successfully for {model_name} v{version}")
        return True


# Example CI/CD integration function
def ci_cd_integration(
    model_name: str,
    version: int,
    model_files_dir: str,
    checksums_file: str
) -> bool:
    """Integration point for CI/CD pipelines."""
    try:
        # Load checksums
        with open(checksums_file, 'r') as f:
            checksums = json.load(f)

        # Map model files
        model_files = {}
        for root, _, files in os.walk(model_files_dir):
            for file in files:
                source_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_path, model_files_dir)
                model_files[rel_path] = source_path

        # Initialize update manager
        manager = ModelUpdateManager(
            triton_url="http://localhost:8000",
            model_repository="/home/sprime01/homelab/models",
            backup_dir="/home/sprime01/homelab/model_backups",
            test_data_dir="/home/sprime01/homelab/test_data"
        )

        # Execute update
        return manager.update_model(
            model_name=model_name,
            version=version,
            model_files=model_files,
            checksums=checksums,
            skip_canary=False  # Use canary deployment
        )
    except Exception as e:
        logger.error(f"Error in CI/CD integration: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        print("Running example model update...")

        # Mock parameters
        model_name = "example_model"
        version = 1
        model_files_dir = "/tmp/model_files"
        checksums_file = "/tmp/checksums.json"

        # Create temporary directories and files for the example
        os.makedirs(model_files_dir, exist_ok=True)
        with open(os.path.join(model_files_dir, "model.bin"), "w") as f:
            f.write("mock model content")

        with open(checksums_file, "w") as f:
            json.dump({"model.bin": hashlib.sha256(b"mock model content").hexdigest()}, f)

        # Run example update
        success = ci_cd_integration(model_name, version, model_files_dir, checksums_file)
        print(f"Example update {'succeeded' if success else 'failed'}")
    else:
        print("Usage:")
        print("  python secure-model-updates.py --example   # Run an example update")
        print("Import this module in your CI/CD scripts to use the functionality")
