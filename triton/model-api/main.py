from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import uuid
import os
import json
import logging
from datetime import datetime, timedelta

# Local modules
from app.services.minio_service import MinioService
from app.services.kubernetes_service import KubernetesService
from app.services.optimization_service import OptimizationService
from app.services.monitoring_service import MonitoringService
from app.auth.auth_handler import create_access_token, verify_token
from app.models.model_schemas import (
    ModelMetadata,
    OptimizationRequest,
    DeploymentRequest,
    ModelVersion,
    ModelPerformance
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model-api")

# Create FastAPI app
app = FastAPI(
    title="Model Management API",
    description="API service for managing AI models in a homelab environment",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# Initialize services
minio_service = MinioService(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio.local:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)

kubernetes_service = KubernetesService(
    namespace=os.getenv("KUBERNETES_NAMESPACE", "triton-inference")
)

optimization_service = OptimizationService()
monitoring_service = MonitoringService(
    prometheus_url=os.getenv("PROMETHEUS_URL", "http://prometheus.monitoring:9090")
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dependency for authentication
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Routes for UI
@app.get("/")
async def redirect_to_ui():
    return {"message": "API is running. Visit /ui for the web interface, or /docs for API documentation"}

# Model Management Endpoints
@app.get("/api/models", response_model=List[ModelMetadata], tags=["Models"])
async def list_models(current_user: dict = Depends(get_current_user)):
    """List all available models"""
    try:
        models = await minio_service.list_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/api/models/{model_name}", response_model=ModelMetadata, tags=["Models"])
async def get_model(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get model details"""
    try:
        model = await minio_service.get_model_metadata(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")

@app.post("/api/models/upload", response_model=ModelMetadata, tags=["Models"])
async def upload_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form(...),
    version: str = Form("1.0.0"),
    description: str = Form(None),
    metadata: str = Form("{}"),
    current_user: dict = Depends(get_current_user)
):
    """Upload a new model or a new version of an existing model"""
    try:
        # Parse metadata from JSON string
        metadata_dict = json.loads(metadata)

        # Create model metadata
        model_metadata = ModelMetadata(
            name=model_name,
            type=model_type,
            framework=framework,
            version=version,
            description=description or f"{model_name} model",
            upload_date=datetime.now().isoformat(),
            uploaded_by=current_user.get("username", "unknown"),
            size=0,  # Will be updated during upload
            status="uploading",
            metadata=metadata_dict
        )

        # Upload model to MinIO
        model_id = await minio_service.upload_model(
            model_file=model_file,
            model_metadata=model_metadata
        )

        # Update metadata after successful upload
        model_metadata.status = "uploaded"
        await minio_service.update_model_metadata(model_metadata)

        # Log the upload
        logger.info(f"Model {model_name} v{version} uploaded successfully")

        return model_metadata
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@app.post("/api/models/{model_name}/optimize", response_model=ModelMetadata, tags=["Optimization"])
async def optimize_model(
    model_name: str,
    optimization_request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger model optimization"""
    try:
        # Get model metadata
        model = await minio_service.get_model_metadata(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Start optimization in the background
        background_tasks.add_task(
            optimization_service.optimize_model,
            model=model,
            optimization_params=optimization_request,
            minio_service=minio_service
        )

        # Update model status
        model.status = "optimizing"
        await minio_service.update_model_metadata(model)

        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize model: {str(e)}")

@app.post("/api/models/{model_name}/deploy", response_model=Dict[str, Any], tags=["Deployment"])
async def deploy_model(
    model_name: str,
    deployment_request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Deploy model to Triton Inference Server"""
    try:
        # Get model metadata
        model = await minio_service.get_model_metadata(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Deploy model to Kubernetes
        deployment_result = await kubernetes_service.deploy_model(
            model=model,
            deployment_params=deployment_request
        )

        # Update model status
        model.status = "deployed"
        model.deployment = deployment_result
        await minio_service.update_model_metadata(model)

        return {
            "status": "success",
            "message": f"Model {model_name} deployment initiated",
            "deployment": deployment_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {str(e)}")

@app.get("/api/models/{model_name}/versions", response_model=List[ModelVersion], tags=["Versions"])
async def list_model_versions(
    model_name: str,
    current_user: dict = Depends(get_current_user)
):
    """List all versions of a model"""
    try:
        versions = await minio_service.list_model_versions(model_name)
        return versions
    except Exception as e:
        logger.error(f"Error listing versions for model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list model versions: {str(e)}")

@app.get("/api/models/{model_name}/performance", response_model=ModelPerformance, tags=["Monitoring"])
async def get_model_performance(
    model_name: str,
    version: Optional[str] = None,
    timeframe: str = Query("1h", description="Timeframe for metrics (e.g., 1h, 24h, 7d)"),
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics for a deployed model"""
    try:
        performance = await monitoring_service.get_model_performance(
            model_name=model_name,
            version=version,
            timeframe=timeframe
        )
        return performance
    except Exception as e:
        logger.error(f"Error getting performance for model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@app.delete("/api/models/{model_name}", response_model=Dict[str, str], tags=["Models"])
async def delete_model(
    model_name: str,
    version: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Delete a model or a specific version of a model"""
    try:
        if version:
            # Delete specific version
            await minio_service.delete_model_version(model_name, version)
            return {"message": f"Model {model_name} version {version} deleted successfully"}
        else:
            # Delete all versions of the model
            await minio_service.delete_model(model_name)
            return {"message": f"Model {model_name} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# Authentication endpoints
@app.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get access token for API authentication"""
    # This is a simplified example - in production use a proper auth system
    if form_data.username == os.getenv("API_USERNAME", "admin") and \
       form_data.password == os.getenv("API_PASSWORD", "password"):
        access_token = create_access_token(
            data={"sub": form_data.username},
            expires_delta=timedelta(hours=24)
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=401,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for the API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "minio": await minio_service.check_health(),
            "kubernetes": await kubernetes_service.check_health(),
            "monitoring": await monitoring_service.check_health()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
