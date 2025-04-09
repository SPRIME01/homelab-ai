#!/usr/bin/env python3

import os
import json
import time
import logging
import datetime
import hashlib
import re
import uuid
import threading
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
import queue

# Third-party imports
import jwt
import requests
import redis
from fastapi import FastAPI, HTTPException, Header, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/triton/access-control.log')
    ]
)
logger = logging.getLogger("model-access-control")

# Define constants
DEFAULT_CONFIG_PATH = "/etc/triton/model-access-config.json"
TOKEN_EXPIRY_SECONDS = 3600  # 1 hour
RATE_LIMIT_WINDOW = 60  # 1 minute
AUDIT_LOG_BATCH_SIZE = 50
AUDIT_LOG_FLUSH_INTERVAL = 10  # seconds

class AccessLevel(str, Enum):
    """Access levels for model endpoints"""
    READ_ONLY = "read_only"
    INFERENCE = "inference"
    FULL_ACCESS = "full_access"
    ADMIN = "admin"

class AuditAction(str, Enum):
    """Audit action types"""
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    INFERENCE_REQUEST = "inference_request"
    ACCESS_DENIED = "access_denied"
    CONFIG_CHANGE = "config_change"
    SYSTEM = "system"

@dataclass
class AuditEvent:
    """Structured audit event for logging"""
    timestamp: str
    action: AuditAction
    user_id: str
    client_ip: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    request_id: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        return result

class AuthConfig(BaseModel):
    """Authentication configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_issuer: Optional[str] = None
    token_expiry_seconds: int = TOKEN_EXPIRY_SECONDS
    admin_tokens: List[str] = []

class ModelAccessConfig(BaseModel):
    """Model access configuration"""
    model_name: str
    versions: Optional[List[str]] = None  # None means all versions
    allowed_roles: List[str] = []
    allowed_users: List[str] = []
    denied_users: List[str] = []
    require_auth: bool = True
    rate_limit: Optional[int] = None  # Requests per minute

class RoleConfig(BaseModel):
    """Role configuration"""
    name: str
    allowed_models: List[str] = []
    access_level: AccessLevel = AccessLevel.INFERENCE
    rate_limit: Optional[int] = None  # Requests per minute

class QuotaConfig(BaseModel):
    """Quota configuration"""
    user_id: str
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None

class AccessControlConfig(BaseModel):
    """Main configuration model"""
    auth: AuthConfig
    models: List[ModelAccessConfig]
    roles: List[RoleConfig]
    quotas: List[QuotaConfig] = []
    default_rate_limit: int = 100  # Default rate limit per minute
    redis_url: Optional[str] = None
    enable_audit_logging: bool = True
    audit_log_path: str = "/var/log/triton/audit.log"

class ModelAccessRequest(BaseModel):
    """Model access request details"""
    model_name: str
    model_version: str = "1"
    access_type: AccessLevel = AccessLevel.INFERENCE

class TokenRequest(BaseModel):
    """Token request model"""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class ModelAccessControl:
    """Main access control class for Triton models"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize with configuration from file"""
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize Redis if configured
        self.redis_client = None
        if self.config.redis_url:
            self.redis_client = redis.Redis.from_url(self.config.redis_url)
            logger.info(f"Connected to Redis at {self.config.redis_url}")

        # Setup audit logging
        self.enable_audit_logging = self.config.enable_audit_logging
        self.audit_log_path = self.config.audit_log_path
        self.audit_queue = queue.Queue()
        self.audit_batch = []

        if self.enable_audit_logging:
            # Start audit logging thread
            self.audit_thread = threading.Thread(target=self._audit_logger_thread, daemon=True)
            self.audit_thread.start()

            # Log startup event
            self.log_audit_event(
                AuditEvent(
                    timestamp=self._get_timestamp(),
                    action=AuditAction.SYSTEM,
                    user_id="system",
                    client_ip="127.0.0.1",
                    details={"message": "Access control system started"}
                )
            )

    def _load_config(self) -> AccessControlConfig:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            return AccessControlConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Create a minimal default configuration
            return AccessControlConfig(
                auth=AuthConfig(jwt_secret=os.environ.get("JWT_SECRET", self._generate_secret())),
                models=[],
                roles=[]
            )

    def _generate_secret(self) -> str:
        """Generate a random secret key"""
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.datetime.utcnow().isoformat() + "Z"

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return token if successful"""
        # In a real implementation, this would verify against a user database
        # For demo purposes, we'll use a simplistic approach
        # NOTE: Replace with actual authentication logic

        # Hash the password (in a real system, you'd verify against stored hash)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check against an external authentication system
        # For demo, we'll assume success if username isn't empty
        if username:
            # Generate JWT token
            payload = {
                "sub": username,
                "iat": datetime.datetime.utcnow(),
                "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=self.config.auth.token_expiry_seconds),
                "roles": self._get_user_roles(username)
            }

            if self.config.auth.jwt_issuer:
                payload["iss"] = self.config.auth.jwt_issuer

            token = jwt.encode(
                payload,
                self.config.auth.jwt_secret,
                algorithm=self.config.auth.jwt_algorithm
            )

            return token

        return None

    def _get_user_roles(self, username: str) -> List[str]:
        """Get roles for a user - replace with actual role lookup logic"""
        # In a real implementation, this would query a user/role database
        if username == "admin":
            return ["admin"]
        elif username.startswith("model_"):
            return ["model_developer"]
        else:
            return ["user"]

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token and return the payload"""
        try:
            # Check if it's an admin token
            if token in self.config.auth.admin_tokens:
                return {
                    "sub": "admin",
                    "roles": ["admin"],
                    "is_admin_token": True
                }

            # Otherwise verify JWT
            payload = jwt.decode(
                token,
                self.config.auth.jwt_secret,
                algorithms=[self.config.auth.jwt_algorithm],
                options={"verify_signature": True}
            )

            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )

    def check_model_access(
        self,
        model_name: str,
        model_version: str,
        user_id: str,
        user_roles: List[str],
        access_type: AccessLevel = AccessLevel.INFERENCE,
        client_ip: str = "0.0.0.0",
        request_id: Optional[str] = None
    ) -> bool:
        """Check if a user has access to a specific model version"""
        # Find the model configuration
        model_config = None
        for model in self.config.models:
            if model.model_name == model_name:
                model_config = model
                break

        # If model isn't in config, default to requiring auth
        if not model_config:
            # Log attempt to access undefined model
            self.log_audit_event(
                AuditEvent(
                    timestamp=self._get_timestamp(),
                    action=AuditAction.ACCESS_DENIED,
                    user_id=user_id,
                    client_ip=client_ip,
                    model_name=model_name,
                    model_version=model_version,
                    request_id=request_id,
                    details={"reason": "Model not defined in access control config"}
                )
            )
            return False

        # Check if model requires authentication
        if model_config.require_auth:
            # Check denied users list
            if user_id in model_config.denied_users:
                self.log_audit_event(
                    AuditEvent(
                        timestamp=self._get_timestamp(),
                        action=AuditAction.ACCESS_DENIED,
                        user_id=user_id,
                        client_ip=client_ip,
                        model_name=model_name,
                        model_version=model_version,
                        request_id=request_id,
                        details={"reason": "User explicitly denied"}
                    )
                )
                return False

            # Check if user is in allowed users
            user_explicitly_allowed = user_id in model_config.allowed_users

            # Check if user has an allowed role
            role_allowed = False
            highest_access_level = None

            for role_name in user_roles:
                if role_name in model_config.allowed_roles:
                    role_allowed = True

                    # Find the role's access level
                    for role_config in self.config.roles:
                        if role_config.name == role_name:
                            # Check if this role allows the specific model
                            if model_name in role_config.allowed_models or "*" in role_config.allowed_models:
                                if highest_access_level is None or role_config.access_level.value > highest_access_level.value:
                                    highest_access_level = role_config.access_level

            # Check if user has sufficient access level
            if highest_access_level is not None:
                access_level_sufficient = highest_access_level.value >= access_type.value
            else:
                access_level_sufficient = False

            # User must be explicitly allowed or have an allowed role with sufficient access level
            if not (user_explicitly_allowed or (role_allowed and access_level_sufficient)):
                self.log_audit_event(
                    AuditEvent(
                        timestamp=self._get_timestamp(),
                        action=AuditAction.ACCESS_DENIED,
                        user_id=user_id,
                        client_ip=client_ip,
                        model_name=model_name,
                        model_version=model_version,
                        request_id=request_id,
                        details={
                            "reason": "Insufficient access privileges",
                            "required_access": access_type.value,
                            "highest_access": highest_access_level.value if highest_access_level else None
                        }
                    )
                )
                return False

            # Check model version restrictions if specified
            if model_config.versions and model_version not in model_config.versions:
                self.log_audit_event(
                    AuditEvent(
                        timestamp=self._get_timestamp(),
                        action=AuditAction.ACCESS_DENIED,
                        user_id=user_id,
                        client_ip=client_ip,
                        model_name=model_name,
                        model_version=model_version,
                        request_id=request_id,
                        details={"reason": "Requested version not allowed"}
                    )
                )
                return False

        # Check rate limits
        if not self.check_rate_limit(user_id, model_name, model_config.rate_limit):
            self.log_audit_event(
                AuditEvent(
                    timestamp=self._get_timestamp(),
                    action=AuditAction.ACCESS_DENIED,
                    user_id=user_id,
                    client_ip=client_ip,
                    model_name=model_name,
                    model_version=model_version,
                    request_id=request_id,
                    details={"reason": "Rate limit exceeded"}
                )
            )
            return False

        # Check quotas
        if not self.check_quota(user_id):
            self.log_audit_event(
                AuditEvent(
                    timestamp=self._get_timestamp(),
                    action=AuditAction.ACCESS_DENIED,
                    user_id=user_id,
                    client_ip=client_ip,
                    model_name=model_name,
                    model_version=model_version,
                    request_id=request_id,
                    details={"reason": "Quota exceeded"}
                )
            )
            return False

        # All checks passed
        self.log_audit_event(
            AuditEvent(
                timestamp=self._get_timestamp(),
                action=AuditAction.INFERENCE_REQUEST,
                user_id=user_id,
                client_ip=client_ip,
                model_name=model_name,
                model_version=model_version,
                request_id=request_id,
                status_code=200,
                details={"access_granted": True}
            )
        )
        return True

    def check_rate_limit(self, user_id: str, model_name: str, model_rate_limit: Optional[int] = None) -> bool:
        """Check if a user has exceeded rate limits"""
        if not self.redis_client:
            # If Redis is not configured, skip rate limiting
            return True

        # Get user-specific rate limit
        user_rate_limit = None
        for quota in self.config.quotas:
            if quota.user_id == user_id:
                for role_config in self.config.roles:
                    if role_config.name in self._get_user_roles(user_id):
                        user_rate_limit = role_config.rate_limit
                        break
                break

        # Determine which rate limit to apply (most restrictive)
        rate_limit = self.config.default_rate_limit  # Default
        if user_rate_limit is not None:
            rate_limit = min(rate_limit, user_rate_limit)
        if model_rate_limit is not None:
            rate_limit = min(rate_limit, model_rate_limit)

        # Use Redis to track request counts
        key = f"rate_limit:{user_id}:{model_name}:{int(time.time() / RATE_LIMIT_WINDOW)}"
        current = self.redis_client.incr(key)

        # Set expiry on the key if this is the first request in the window
        if current == 1:
            self.redis_client.expire(key, RATE_LIMIT_WINDOW * 2)  # Double the window for safety

        # Check if rate limit is exceeded
        return current <= rate_limit

    def check_quota(self, user_id: str) -> bool:
        """Check if a user has exceeded their quota"""
        if not self.redis_client:
            # If Redis is not configured, skip quota checks
            return True

        # Find user's quota configuration
        user_quota = None
        for quota in self.config.quotas:
            if quota.user_id == user_id:
                user_quota = quota
                break

        if not user_quota or (not user_quota.daily_limit and not user_quota.monthly_limit):
            # No quotas defined for this user
            return True

        # Get current date components
        now = datetime.datetime.utcnow()
        day_key = f"quota:daily:{user_id}:{now.strftime('%Y-%m-%d')}"
        month_key = f"quota:monthly:{user_id}:{now.strftime('%Y-%m')}"

        # Check daily quota
        if user_quota.daily_limit:
            daily_usage = int(self.redis_client.get(day_key) or 0)
            if daily_usage >= user_quota.daily_limit:
                return False

            # Increment and set expiry
            self.redis_client.incr(day_key)
            self.redis_client.expire(day_key, 86400 * 2)  # 2 days

        # Check monthly quota
        if user_quota.monthly_limit:
            monthly_usage = int(self.redis_client.get(month_key) or 0)
            if monthly_usage >= user_quota.monthly_limit:
                return False

            # Increment and set expiry
            self.redis_client.incr(month_key)
            self.redis_client.expire(month_key, 86400 * 32)  # ~1 month

        return True

    def log_audit_event(self, event: AuditEvent) -> None:
        """Add an event to the audit log queue"""
        if self.enable_audit_logging:
            self.audit_queue.put(event)

    def _audit_logger_thread(self) -> None:
        """Background thread that processes and writes audit events"""
        last_flush_time = time.time()

        while True:
            try:
                # Try to get an event with timeout
                try:
                    event = self.audit_queue.get(timeout=1.0)
                    self.audit_batch.append(event)
                    self.audit_queue.task_done()
                except queue.Empty:
                    # No new events
                    pass

                # Check if it's time to flush the batch
                current_time = time.time()
                should_flush = (
                    len(self.audit_batch) >= AUDIT_LOG_BATCH_SIZE or
                    (len(self.audit_batch) > 0 and
                     current_time - last_flush_time >= AUDIT_LOG_FLUSH_INTERVAL)
                )

                if should_flush:
                    self._flush_audit_logs()
                    last_flush_time = current_time

            except Exception as e:
                logger.error(f"Error in audit logger thread: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors

    def _flush_audit_logs(self) -> None:
        """Write accumulated audit logs to file"""
        if not self.audit_batch:
            return

        try:
            with open(self.audit_log_path, 'a') as f:
                for event in self.audit_batch:
                    f.write(json.dumps(event.to_dict()) + "\n")

            self.audit_batch = []
        except Exception as e:
            logger.error(f"Failed to write audit logs: {e}")


# FastAPI application for serving the access control API
app = FastAPI(title="Triton Model Access Control API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize controller
access_control = ModelAccessControl()

@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(request: Request, token_request: TokenRequest):
    """Authenticate and get access token"""
    token = access_control.authenticate(token_request.username, token_request.password)
    if not token:
        # Log failed authentication
        access_control.log_audit_event(
            AuditEvent(
                timestamp=access_control._get_timestamp(),
                action=AuditAction.ACCESS_DENIED,
                user_id=token_request.username,
                client_ip=request.client.host,
                details={"reason": "Authentication failed"}
            )
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Log successful authentication
    access_control.log_audit_event(
        AuditEvent(
            timestamp=access_control._get_timestamp(),
            action=AuditAction.SYSTEM,
            user_id=token_request.username,
            client_ip=request.client.host,
            details={"event": "Authentication successful"}
        )
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": access_control.config.auth.token_expiry_seconds
    }

@app.post("/check_access")
async def check_model_access(
    request: Request,
    access_request: ModelAccessRequest,
    authorization: str = Header(...),
    x_request_id: Optional[str] = Header(None)
):
    """Check if the authenticated user has access to a model"""
    try:
        request_id = x_request_id or str(uuid.uuid4())

        # Extract token
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization.replace("Bearer ", "")

        # Verify token
        payload = access_control.verify_token(token)
        user_id = payload.get("sub", "anonymous")
        user_roles = payload.get("roles", [])

        # Check access
        has_access = access_control.check_model_access(
            model_name=access_request.model_name,
            model_version=access_request.model_version,
            user_id=user_id,
            user_roles=user_roles,
            access_type=access_request.access_type,
            client_ip=request.client.host,
            request_id=request_id
        )

        return {"access_granted": has_access}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking access: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking access: {str(e)}"
        )

@app.get("/models/allowed")
async def get_allowed_models(
    request: Request,
    authorization: str = Header(...),
):
    """Get list of models the authenticated user has access to"""
    try:
        # Extract token
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization.replace("Bearer ", "")

        # Verify token
        payload = access_control.verify_token(token)
        user_id = payload.get("sub", "anonymous")
        user_roles = payload.get("roles", [])

        # Determine which models the user can access
        allowed_models = []
        for model_config in access_control.config.models:
            # Skip if user is denied
            if user_id in model_config.denied_users:
                continue

            # Include if user is explicitly allowed
            if user_id in model_config.allowed_users:
                allowed_models.append({
                    "model_name": model_config.model_name,
                    "versions": model_config.versions or ["all"],
                })
                continue

            # Check role-based access
            for role_name in user_roles:
                if role_name in model_config.allowed_roles:
                    allowed_models.append({
                        "model_name": model_config.model_name,
                        "versions": model_config.versions or ["all"],
                    })
                    break

        return {"allowed_models": allowed_models}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving allowed models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving allowed models: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()
