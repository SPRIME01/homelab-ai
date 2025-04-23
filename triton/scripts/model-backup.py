#!/usr/bin/env python3

import os
import logging
import datetime
import tarfile
import json
import shutil
import sys
from minio import Minio
from minio.error import S3Error

# --- Configuration (Prefer environment variables) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT") # e.g., "minio.homelab:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ai-models-backup")
MINIO_SECURE = os.getenv("MINIO_SECURE", "true").lower() == "true"

# Path to the Triton model repository root directory
MODEL_REPO_PATH = os.getenv("MODEL_REPO_PATH") # e.g., "/models" or "/path/to/triton/models"
# Specific model name to back up (optional, backs up entire repo if not set)
MODEL_NAME = os.getenv("MODEL_NAME")
# Local directory for staging backups before upload
STAGING_DIR = os.getenv("STAGING_DIR", "/tmp/model_backups")

# Version tag for the backup (e.g., git commit hash, training run ID, or timestamp)
# If not provided, a timestamp will be used.
BACKUP_VERSION_TAG = os.getenv("BACKUP_VERSION_TAG")

# Placeholder for lineage info - should be passed or read from a file
# Example: GIT_COMMIT, TRAINING_DATA_VERSION, SOURCE_NOTEBOOK
LINEAGE_METADATA = json.loads(os.getenv("LINEAGE_METADATA", "{}"))

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Helper Functions ---
def get_minio_client():
    """Initializes and returns a Minio client."""
    if not all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY]):
        logging.error("MinIO connection details (endpoint, access key, secret key) not fully configured.")
        return None
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        # Ping the server to check connection
        client.list_buckets()
        logging.info(f"Successfully connected to MinIO at {MINIO_ENDPOINT}")
        return client
    except S3Error as e:
        logging.error(f"MinIO S3 Error: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to connect to MinIO at {MINIO_ENDPOINT}: {e}")
        return None

def create_local_archive(model_repo_root, staging_dir, model_name=None):
    """
    Creates a compressed tar archive of the model repository or a specific model.
    Includes model files and config.pbtxt.
    """
    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    archive_basename = f"{model_name or 'full_repo'}_{timestamp}.tar.gz"
    archive_path = os.path.join(staging_dir, archive_basename)

    source_path = model_repo_root
    arcname_prefix = "" # Store relative paths from repo root

    if model_name:
        source_path = os.path.join(model_repo_root, model_name)
        arcname_prefix = model_name # Store files under model_name dir in archive
        if not os.path.isdir(source_path):
            logging.error(f"Model directory not found: {source_path}")
            return None
        logging.info(f"Archiving specific model '{model_name}' from {source_path}...")
    else:
        logging.info(f"Archiving entire model repository from {model_repo_root}...")

    if not os.path.exists(staging_dir):
        logging.info(f"Creating staging directory: {staging_dir}")
        os.makedirs(staging_dir, exist_ok=True)

    try:
        logging.info(f"Creating archive: {archive_path}")
        with tarfile.open(archive_path, "w:gz") as tar:
            # Add the directory contents. arcname controls the path inside the tar file.
            tar.add(source_path, arcname=arcname_prefix)
        logging.info(f"Successfully created local archive: {archive_path}")
        return archive_path
    except Exception as e:
        logging.error(f"Failed to create archive for {source_path}: {e}")
        if os.path.exists(archive_path):
            os.remove(archive_path) # Clean up partial archive
        return None

def upload_to_minio(client, bucket_name, model_name, version_tag, local_archive_path, metadata):
    """Uploads the local archive to MinIO with versioning and metadata."""
    if not client or not local_archive_path:
        logging.error("Upload skipped: MinIO client not initialized or local archive missing.")
        return False

    object_name_base = os.path.basename(local_archive_path)
    # Structure: model_name/version_tag/archive_file.tar.gz
    # If backing up full repo, use a placeholder like '_full_repo_'
    object_name = f"{model_name or '_full_repo_'}/{version_tag}/{object_name_base}"

    try:
        # Ensure bucket exists
        found = client.bucket_exists(bucket_name)
        if not found:
            logging.info(f"Bucket '{bucket_name}' not found. Creating...")
            client.make_bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            logging.info(f"Using existing bucket: '{bucket_name}'")

        logging.info(f"Uploading {local_archive_path} to MinIO as {bucket_name}/{object_name}")

        # MinIO Python SDK uses 'metadata' which maps to x-amz-meta-* headers
        # Ensure keys are simple strings, values are strings.
        s3_metadata = {f"x-amz-meta-{k.lower()}": str(v) for k, v in metadata.items()}

        result = client.fput_object(
            bucket_name,
            object_name,
            local_archive_path,
            metadata=s3_metadata # Pass user metadata here
        )
        logging.info(
            f"Successfully uploaded {object_name} (etag: {result.etag}, version_id: {result.version_id})"
        )
        return True
    except S3Error as e:
        logging.error(f"MinIO S3 Error during upload: {e}")
        return False
    except Exception as e:
        logging.error(f"Failed to upload {local_archive_path} to MinIO: {e}")
        return False

# --- Main Execution ---
def main():
    logging.info("Starting AI Model Backup Process...")
    start_time = datetime.datetime.now()
    backup_success = False

    # --- Basic Configuration Checks ---
    if not MODEL_REPO_PATH or not os.path.isdir(MODEL_REPO_PATH):
        logging.error(f"MODEL_REPO_PATH ('{MODEL_REPO_PATH}') is not set or not a valid directory.")
        sys.exit(1)

    minio_client = get_minio_client()
    if not minio_client:
        logging.error("Failed to initialize MinIO client. Exiting.")
        sys.exit(1)

    # --- Determine Version Tag ---
    version_tag = BACKUP_VERSION_TAG or datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    logging.info(f"Using backup version tag: {version_tag}")

    # --- Create Local Archive ---
    local_archive = create_local_archive(MODEL_REPO_PATH, STAGING_DIR, MODEL_NAME)

    if local_archive:
        # --- Prepare Metadata ---
        # Note: Lineage tracking is complex. This metadata should ideally be
        # generated by the preceding step (e.g., training/deployment pipeline)
        # and passed via LINEAGE_METADATA env var.
        metadata = {
            "backup_timestamp": start_time.isoformat(),
            "source_path": MODEL_REPO_PATH,
            "version_tag": version_tag,
            "backed_up_model": MODEL_NAME or "full_repository",
            **LINEAGE_METADATA # Merge lineage info if provided
        }
        logging.info(f"Prepared metadata: {json.dumps(metadata)}")

        # --- Upload to MinIO ---
        # Use the specific model name for grouping, or a general name if backing up the whole repo
        upload_model_name_key = MODEL_NAME or "_full_repo_"
        if upload_to_minio(minio_client, MINIO_BUCKET, upload_model_name_key, version_tag, local_archive, metadata):
            backup_success = True
        else:
            logging.error("Failed to upload backup to MinIO.")

        # --- Clean up local archive ---
        try:
            logging.info(f"Cleaning up local archive: {local_archive}")
            os.remove(local_archive)
        except Exception as e:
            logging.warning(f"Failed to clean up local archive {local_archive}: {e}")
    else:
        logging.error("Failed to create local backup archive.")

    # --- Reporting ---
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Model backup process finished in {duration}.")

    if backup_success:
        logging.info("Model backup completed successfully.")
        sys.exit(0)
    else:
        logging.error("Model backup process failed.")
        sys.exit(1)

if __name__ == "__main__":
    # --- Prerequisite Check ---
    try:
        import minio
    except ImportError:
        logging.error("The 'minio' Python package is not installed.")
        logging.error("Please install it: pip install minio")
        sys.exit(2)

    main()

# --- Integration Notes ---
#
# This script is designed to be part of a larger workflow.
#
# 1.  **Triggering:**
#     - Run manually after significant model changes.
#     - Integrate into a CI/CD pipeline for models (e.g., after successful training and validation).
#     - Run as a scheduled job (e.g., Kubernetes CronJob) for regular snapshots, though pipeline integration is often preferred.
#
# 2.  **Environment Variables:**
#     - Ensure all required environment variables (MINIO_*, MODEL_REPO_PATH) are set in the execution environment (e.g., K8s pod spec, CI/CD variables).
#     - Pass `MODEL_NAME` if backing up a single model.
#     - Pass `BACKUP_VERSION_TAG` (e.g., Git commit hash, training run ID) for meaningful versioning.
#     - Pass `LINEAGE_METADATA` as a JSON string to capture relevant context.
#
# 3.  **Containerization:**
#     - Package this script, Python, the `minio` library, and `tar` into a Docker container.
#     - The container needs access to the model repository (e.g., via volume mount).
#
# 4.  **Space Efficiency:**
#     - The script uses `.tar.gz` compression.
#     - MinIO's built-in versioning can help manage multiple copies efficiently if enabled on the bucket.
#     - For very large models, consider tools/strategies for deduplication or differential backups if MinIO's features aren't sufficient, though this adds complexity.
#
# 5.  **Recovery:**
#     - To restore, list objects in MinIO for the desired model/version.
#     - Download the `.tar.gz` archive using the MinIO client or UI.
#     - Extract the archive to the target Triton model repository location.
#     - Check associated metadata in MinIO for context.
