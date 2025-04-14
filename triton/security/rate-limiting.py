"""
Rate Limiting Module for AI Services

This module provides rate limiting functionality for AI services in a homelab environment.
It can be deployed as a sidecar or API gateway component to protect backend services from
excessive usage.
"""

import time
import threading
import logging
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
import redis  # You might need to install this: pip install redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai-rate-limiter')

# Rate Limiting Types
class LimitType(Enum):
    USER = "user"
    IP = "ip"
    APPLICATION = "application"
    MODEL = "model"

@dataclass
class RateLimitRule:
    """Configuration for a rate limit rule"""
    limit_type: LimitType
    requests_per_period: int
    period_seconds: int
    burst_size: Optional[int] = None

    def __post_init__(self):
        # If burst size is not explicitly set, use the requests_per_period
        if self.burst_size is None:
            self.burst_size = self.requests_per_period

class TokenBucket:
    """
    Implementation of the Token Bucket algorithm for rate limiting.
    A token bucket has a capacity and fills at a constant rate. Each request
    consumes one or more tokens. If the bucket has enough tokens, the request
    is allowed; otherwise, it's denied.
    """

    def __init__(self, capacity: int, fill_rate: float):
        """
        Initialize a token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold
            fill_rate: Rate at which tokens are added to the bucket (tokens per second)
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.RLock()

    def refill(self) -> None:
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        with self.lock:
            elapsed = now - self.last_refill
            new_tokens = elapsed * self.fill_rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        self.refill()
        with self.lock:
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False

class ResourceQuota:
    """
    Manages resource quotas for compute-intensive models.
    """

    def __init__(self, resource_type: str, max_usage: float, current_usage: float = 0):
        """
        Initialize a resource quota.

        Args:
            resource_type: Type of resource (e.g., 'gpu-memory', 'cpu-time')
            max_usage: Maximum allowed usage
            current_usage: Current usage level
        """
        self.resource_type = resource_type
        self.max_usage = max_usage
        self.current_usage = current_usage
        self.lock = threading.RLock()

    def can_allocate(self, amount: float) -> bool:
        """Check if the requested amount can be allocated."""
        with self.lock:
            return self.current_usage + amount <= self.max_usage

    def allocate(self, amount: float) -> bool:
        """Try to allocate the requested amount of resource."""
        with self.lock:
            if self.can_allocate(amount):
                self.current_usage += amount
                return True
            return False

    def release(self, amount: float) -> None:
        """Release the allocated amount of resource."""
        with self.lock:
            self.current_usage = max(0, self.current_usage - amount)


class RateLimitExceededError(Exception):
    """Exception raised when a rate limit is exceeded."""

    def __init__(self, limit_type: LimitType, key: str, retry_after: float):
        self.limit_type = limit_type
        self.key = key
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {limit_type.value} '{key}'. Retry after {retry_after:.2f} seconds.")


class AbusePrevention:
    """
    Detects and prevents abuse patterns in API usage.
    """

    def __init__(self, alert_threshold: int = 5, window_seconds: int = 300):
        """
        Initialize abuse prevention.

        Args:
            alert_threshold: Number of rate limit violations before triggering an alert
            window_seconds: Time window for tracking violations
        """
        self.violations = defaultdict(list)
        self.alert_threshold = alert_threshold
        self.window_seconds = window_seconds
        self.alert_callbacks = []

    def register_violation(self, limit_type: LimitType, key: str) -> None:
        """Register a rate limit violation."""
        now = time.time()
        violation_key = f"{limit_type.value}:{key}"

        # Remove old violations outside the time window
        self.violations[violation_key] = [t for t in self.violations[violation_key]
                                         if now - t < self.window_seconds]

        # Add the new violation
        self.violations[violation_key].append(now)

        # Check if we need to trigger an alert
        if len(self.violations[violation_key]) >= self.alert_threshold:
            self._trigger_alert(limit_type, key, len(self.violations[violation_key]))

    def add_alert_callback(self, callback: Callable[[LimitType, str, int], None]) -> None:
        """
        Add a callback function to be called when an alert is triggered.

        Args:
            callback: Function that takes (limit_type, key, violation_count) as arguments
        """
        self.alert_callbacks.append(callback)

    def _trigger_alert(self, limit_type: LimitType, key: str, violation_count: int) -> None:
        """Trigger alerts through all registered callbacks."""
        logger.warning(f"Abuse detected for {limit_type.value} '{key}' with {violation_count} violations")
        for callback in self.alert_callbacks:
            try:
                callback(limit_type, key, violation_count)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


class RateLimiter:
    """
    Main rate limiter class that manages rate limits for different entities.
    """

    def __init__(self, rules: List[RateLimitRule] = None, use_redis: bool = False, redis_url: str = None):
        """
        Initialize the rate limiter.

        Args:
            rules: List of rate limit rules to enforce
            use_redis: Whether to use Redis for distributed rate limiting
            redis_url: Redis connection URL (if use_redis is True)
        """
        self.rules = rules or []
        self.buckets: Dict[str, Dict[str, TokenBucket]] = {
            LimitType.USER.value: {},
            LimitType.IP.value: {},
            LimitType.APPLICATION.value: {},
            LimitType.MODEL.value: {}
        }
        self.resource_quotas: Dict[str, ResourceQuota] = {}
        self.abuse_prevention = AbusePrevention()

        # Set up Redis if requested
        self.use_redis = use_redis
        self.redis_client = None
        if use_redis:
            if not redis_url:
                redis_url = "redis://localhost:6379/0"
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.use_redis = False

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a new rate limit rule."""
        self.rules.append(rule)

    def add_resource_quota(self, model_id: str, resource_type: str, max_usage: float) -> None:
        """Add a resource quota for a specific model."""
        key = f"{model_id}:{resource_type}"
        self.resource_quotas[key] = ResourceQuota(resource_type, max_usage)

    def _get_bucket_key(self, limit_type: LimitType, key: str, rule: RateLimitRule) -> str:
        """Generate a unique key for a token bucket."""
        return f"{limit_type.value}:{key}:{rule.requests_per_period}:{rule.period_seconds}"

    def _get_or_create_bucket(self, limit_type: LimitType, key: str, rule: RateLimitRule) -> TokenBucket:
        """Get an existing token bucket or create a new one."""
        bucket_key = self._get_bucket_key(limit_type, key, rule)
        buckets = self.buckets[limit_type.value]

        if bucket_key not in buckets:
            # Calculate fill rate in tokens per second
            fill_rate = rule.requests_per_period / rule.period_seconds
            buckets[bucket_key] = TokenBucket(rule.burst_size, fill_rate)

        return buckets[bucket_key]

    def check_rate_limit(self,
                         user_id: Optional[str] = None,
                         ip_address: Optional[str] = None,
                         app_id: Optional[str] = None,
                         model_id: Optional[str] = None,
                         tokens: int = 1) -> bool:
        """
        Check if a request is allowed under the rate limits.

        Args:
            user_id: User identifier
            ip_address: Client IP address
            app_id: Application identifier
            model_id: Model identifier
            tokens: Number of tokens to consume (for weighted requests)

        Returns:
            True if the request is allowed, False otherwise

        Raises:
            RateLimitExceededError: If a rate limit is exceeded
        """
        # Check each applicable rule
        for rule in self.rules:
            key = None

            if rule.limit_type == LimitType.USER and user_id:
                key = user_id
            elif rule.limit_type == LimitType.IP and ip_address:
                key = ip_address
            elif rule.limit_type == LimitType.APPLICATION and app_id:
                key = app_id
            elif rule.limit_type == LimitType.MODEL and model_id:
                key = model_id

            if key:
                # Get or create the appropriate token bucket
                bucket = self._get_or_create_bucket(rule.limit_type, key, rule)

                # Try to consume tokens
                if not bucket.consume(tokens):
                    # Calculate time until next token is available
                    seconds_per_token = 1.0 / (rule.requests_per_period / rule.period_seconds)
                    retry_after = seconds_per_token * tokens

                    # Register the violation for abuse detection
                    self.abuse_prevention.register_violation(rule.limit_type, key)

                    # Raise exception with retry information
                    raise RateLimitExceededError(rule.limit_type, key, retry_after)

        return True

    def allocate_resources(self, model_id: str, resource_type: str, amount: float) -> bool:
        """
        Try to allocate resources for a compute-intensive operation.

        Args:
            model_id: Model identifier
            resource_type: Type of resource to allocate
            amount: Amount of resource to allocate

        Returns:
            True if resources were successfully allocated, False otherwise
        """
        key = f"{model_id}:{resource_type}"
        if key in self.resource_quotas:
            return self.resource_quotas[key].allocate(amount)
        return True  # No quota defined, allow by default

    def release_resources(self, model_id: str, resource_type: str, amount: float) -> None:
        """Release previously allocated resources."""
        key = f"{model_id}:{resource_type}"
        if key in self.resource_quotas:
            self.resource_quotas[key].release(amount)

    def add_abuse_alert_callback(self, callback: Callable[[LimitType, str, int], None]) -> None:
        """Add a callback for abuse alerts."""
        self.abuse_prevention.add_alert_callback(callback)


class RateLimitMiddleware:
    """
    Middleware for integrating rate limiting with web frameworks.
    This is a generic implementation that can be adapted to specific frameworks
    like Flask, FastAPI, Django, etc.
    """

    def __init__(self, rate_limiter: RateLimiter,
                 get_user_id=None,
                 get_app_id=None,
                 get_model_id=None):
        """
        Initialize the middleware.

        Args:
            rate_limiter: The RateLimiter instance to use
            get_user_id: Function to extract user ID from the request
            get_app_id: Function to extract application ID from the request
            get_model_id: Function to extract model ID from the request
        """
        self.rate_limiter = rate_limiter
        self.get_user_id = get_user_id
        self.get_app_id = get_app_id
        self.get_model_id = get_model_id

    def handle_request(self, request: Any, get_response: Callable) -> Any:
        """
        Process a request through the rate limiter.

        Args:
            request: The web request (framework-specific)
            get_response: Function to get the response if rate limit is not exceeded

        Returns:
            Response object (framework-specific)
        """
        try:
            # Extract identifiers from the request
            user_id = self.get_user_id(request) if self.get_user_id else None
            app_id = self.get_app_id(request) if self.get_app_id else None
            model_id = self.get_model_id(request) if self.get_model_id else None
            ip_address = self._get_client_ip(request)

            # Check rate limits
            self.rate_limiter.check_rate_limit(
                user_id=user_id,
                ip_address=ip_address,
                app_id=app_id,
                model_id=model_id
            )

            # If no rate limit is exceeded, process the request
            return get_response(request)

        except RateLimitExceededError as e:
            # Handle rate limit exceeded
            return self._create_rate_limit_response(e)

    def _get_client_ip(self, request: Any) -> str:
        """Extract client IP address from the request (framework-specific)."""
        # This is a generic implementation that should be overridden
        # for specific web frameworks
        return getattr(request, 'remote_addr', '0.0.0.0')

    def _create_rate_limit_response(self, error: RateLimitExceededError) -> Any:
        """Create a response for rate limit exceeded (framework-specific)."""
        # This is a generic implementation that should be overridden
        # for specific web frameworks
        response_data = {
            'error': 'Rate limit exceeded',
            'retry_after': error.retry_after
        }
        return response_data


# Example usage and default configuration

def log_abuse_alert(limit_type: LimitType, key: str, violation_count: int) -> None:
    """Example callback for abuse alerts that logs to a file."""
    with open("abuse_alerts.log", "a") as f:
        f.write(f"{time.time()},{limit_type.value},{key},{violation_count}\n")

def create_default_rate_limiter() -> RateLimiter:
    """Create a rate limiter with sensible defaults for a homelab environment."""
    rules = [
        # Limit by user: 100 requests per minute with burst of 120
        RateLimitRule(LimitType.USER, 100, 60, 120),

        # Limit by IP: 60 requests per minute with burst of 80
        RateLimitRule(LimitType.IP, 60, 60, 80),

        # Limit by application: 1000 requests per minute with burst of 1200
        RateLimitRule(LimitType.APPLICATION, 1000, 60, 1200),

        # Model-specific limits for GPU-intensive models
        RateLimitRule(LimitType.MODEL, 20, 60, 25),
    ]

    # Create the rate limiter
    limiter = RateLimiter(rules)

    # Add resource quotas
    limiter.add_resource_quota("gpt-4", "gpu-memory", 16.0)  # 16GB VRAM limit
    limiter.add_resource_quota("stable-diffusion", "gpu-memory", 12.0)  # 12GB VRAM limit

    # Add abuse alert callback
    limiter.add_abuse_alert_callback(log_abuse_alert)

    return limiter


# Example middleware adapters for popular frameworks
# These would need to be extended and completed for actual use

class FlaskRateLimitMiddleware(RateLimitMiddleware):
    """Flask-specific rate limit middleware."""

    def _get_client_ip(self, request):
        """Get client IP from Flask request."""
        return request.remote_addr

    def _create_rate_limit_response(self, error):
        """Create Flask response for rate limit error."""
        from flask import jsonify
        response = jsonify({
            'error': str(error),
            'retry_after': error.retry_after
        })
        response.status_code = 429
        response.headers['Retry-After'] = str(int(error.retry_after))
        return response


class FastAPIRateLimitMiddleware(RateLimitMiddleware):
    """FastAPI-specific rate limit middleware."""

    async def __call__(self, request, call_next):
        """Process FastAPI request."""
        try:
            # Extract identifiers
            user_id = self.get_user_id(request) if self.get_user_id else None
            app_id = self.get_app_id(request) if self.get_app_id else None
            model_id = self.get_model_id(request) if self.get_model_id else None
            ip_address = self._get_client_ip(request)

            # Check rate limits
            self.rate_limiter.check_rate_limit(user_id, ip_address, app_id, model_id)

            # Process the request
            return await call_next(request)

        except RateLimitExceededError as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": str(e), "retry_after": e.retry_after},
                headers={"Retry-After": str(int(e.retry_after))}
            )

    def _get_client_ip(self, request):
        """Get client IP from FastAPI request."""
        return request.client.host


if __name__ == "__main__":
    # Example of using the rate limiter in a standalone script
    limiter = create_default_rate_limiter()

    # Simulate requests
    try:
        for i in range(150):
            try:
                limiter.check_rate_limit(user_id="user1", ip_address="127.0.0.1")
                print(f"Request {i+1} allowed")
            except RateLimitExceededError as e:
                print(f"Request {i+1} denied: {e}")
            time.sleep(0.1)  # Short delay between requests
    except KeyboardInterrupt:
        print("Simulation stopped by user")
