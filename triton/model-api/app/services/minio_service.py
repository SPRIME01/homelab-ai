import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error

from app.models.model_schemas import ModelMetadata, ModelVersion

logger = logging.getLogger("minio-service")

class MinioService:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        """Initialize MinIO client for model storage."""
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.models_bucket = "models"
        self.metadata_bucket = "model-metadata"
        self.ensure_buckets()

    def ensure_buckets(self) -> None:
        """Ensure the required MinIO buckets exist."""
        try:
            for bucket in [self.models_bucket, self.metadata_bucket]:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
        except S3Error as e:
            logger.error(f"Error ensuring buckets exist: {str(e)}")
            raise

    async def check_health(self) -> Dict[str, Any]:
        """Check MinIO service health."""
        try:
            # List buckets as a simple check
            buckets = self.client.list_buckets()
            bucket_count = len(buckets)
            return {
                "status": "healthy",
                "bucket_count": bucket_count,
                "buckets_exist": {
                    "models": self.client.bucket_exists(self.models_bucket),
                    "metadata": self.client.bucket_exists(self.metadata_bucket)
                }
            }
        except Exception as e:
            logger.error(f"MinIO health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def upload_model(self, model_file: UploadFile, model_metadata: ModelMetadata) -> str:
        """Upload model file to MinIO and store metadata."""
        try:
            # Generate object name
            model_id = f"{model_metadata.name}/{model_metadata.version}/{model_file.filename}"

            # Upload file
            file_data = await model_file.read()
            file_size = len(file_data)
            model_metadata.size = file_size

            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.models_bucket,
                object_name=model_id,
                data=file_data,
                length=file_size,
                content_type=model_file.content_type
            )

            # Store metadata
            metadata_id = f"{model_metadata.name}/{model_metadata.version}/metadata.json"
            metadata_bytes = json.dumps(model_metadata.dict()).encode("utf-8")
            self.client.put_object(
                bucket_name=self.metadata_bucket,
                object_name=metadata_id,
                data=bytes(metadata_bytes),
                length=len(metadata_bytes),
                content_type="application/json"
            )

            # Return the model ID
            return model_id
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise

    async def get_model_metadata(self, model_name: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model metadata from MinIO."""
        try:
            if version:
                # Get specific version
                metadata_id = f"{model_name}/{version}/metadata.json"
                metadata_obj = self.client.get_object(
                    bucket_name=self.metadata_bucket,
                    object_name=metadata_id
                )
                metadata = json.loads(metadata_obj.read().decode("utf-8"))
                return ModelMetadata(**metadata)
            else:
                # Get latest version
                versions = await self.list_model_versions(model_name)
                if not versions:
                    return None
                latest = versions[0]  # Assuming versions are sorted by date
                return await self.get_model_metadata(model_name, latest.version)
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return None
            logger.error(f"Error getting model metadata: {str(e)}")
            raise

    async def update_model_metadata(self, model_metadata: ModelMetadata) -> None:
        """Update model metadata in MinIO."""
        try:
            metadata_id = f"{model_metadata.name}/{model_metadata.version}/metadata.json"
            metadata_bytes = json.dumps(model_metadata.dict()).encode("utf-8")
            self.client.put_object(
                bucket_name=self.metadata_bucket,
                object_name=metadata_id,
                data=bytes(metadata_bytes),
                length=len(metadata_bytes),
                content_type="application/json"
            )
        except Exception as e:
            logger.error(f"Error updating model metadata: {str(e)}")
            raise

    async def list_models(self) -> List[ModelMetadata]:
        """List all models in the repository."""
        try:
            models = {}
            objects = self.client.list_objects(self.metadata_bucket, recursive=True)

            for obj in objects:
                if obj.object_name.endswith("metadata.json"):
                    try:
                        # Parse path to get model name and version
                        parts = obj.object_name.split("/")
                        if len(parts) >= 3:
                            model_name = parts[0]
                            version = parts[1]

                            # Get metadata
                            metadata = await self.get_model_metadata(model_name, version)
                            if metadata:
                                # Keep only the latest version in the list
                                if model_name not in models or version > models[model_name].version:
                                    models[model_name] = metadata
                    except Exception as e:
                        logger.warning(f"Error processing metadata for {obj.object_name}: {str(e)}")

            return list(models.values())
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise

    async def list_model_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        try:
            versions = []
            prefix = f"{model_name}/"
            objects = self.client.list_objects(self.metadata_bucket, prefix=prefix, recursive=True)

            for obj in objects:
                if obj.object_name.endswith("metadata.json"):
                    try:
                        # Parse path to get version
                        parts = obj.object_name.split("/")
                        if len(parts) >= 3:
                            version = parts[1]

                            # Get metadata for this version
                            metadata = await self.get_model_metadata(model_name, version)
                            if metadata:
                                # Create ModelVersion object
                                model_version = ModelVersion(
                                    version=metadata.version,
                                    upload_date=metadata.upload_date,
                                    uploaded_by=metadata.uploaded_by,
                                    size=metadata.size,
                                    status=metadata.status,
                                    is_optimized=metadata.optimization is not None,
                                    is_deployed=metadata.deployment is not None,
                                    optimization_params=metadata.optimization,
                                    deployment_params=metadata.deployment
                                )
                                versions.append(model_version)
                    except Exception as e:
                        logger.warning(f"Error processing version for {obj.object_name}: {str(e)}")

            # Sort versions by upload date (newest first)
            versions.sort(key=lambda x: x.upload_date, reverse=True)
            return versions
        except Exception as e:
            logger.error(f"Error listing model versions: {str(e)}")
            raise

    async def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific version of a model."""
        try:
            # Delete model files
            objects = self.client.list_objects(
                self.models_bucket,
                prefix=f"{model_name}/{version}/",
                recursive=True
            )
            for obj in objects:
                self.client.remove_object(self.models_bucket, obj.object_name)

            # Delete metadata
            metadata_id = f"{model_name}/{version}/metadata.json"
            self.client.remove_object(self.metadata_bucket, metadata_id)
        except Exception as e:
            logger.error(f"Error deleting model version: {str(e)}")
            raise

    async def delete_model(self, model_name: str) -> None:
        """Delete all versions of a model."""
        try:
            # Delete all model files
            objects = self.client.list_objects(
                self.models_bucket,
                prefix=f"{model_name}/",
                recursive=True
            )
            for obj in objects:
                self.client.remove_object(self.models_bucket, obj.object_name)

            # Delete all metadata
            metadata_objects = self.client.list_objects(
                self.metadata_bucket,
                prefix=f"{model_name}/",
                recursive=True
            )
            for obj in metadata_objects:
                self.client.remove_object(self.metadata_bucket, obj.object_name)
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            raise
