"""
Rate limiting middleware with token bucket algorithm.
In-memory implementation for MVP with per-API-key limits.
"""
import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if rate limited
        """
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until enough tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available
        """
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """In-memory rate limiter with token bucket algorithm."""
    
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        
        # Default rate limits (can be configured per API key)
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "uploads_per_hour": 100
        }
    
    async def is_allowed(self, api_key: str, endpoint_type: str = "general") -> Tuple[bool, Optional[float]]:
        """
        Check if request is allowed for the given API key.
        
        Args:
            api_key: API key identifier
            endpoint_type: Type of endpoint (general, upload, etc.)
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        async with self._lock:
            bucket_key = f"{api_key}:{endpoint_type}"
            
            # Get or create bucket
            if bucket_key not in self._buckets:
                self._buckets[bucket_key] = self._create_bucket(endpoint_type)
            
            bucket = self._buckets[bucket_key]
            
            # Try to consume token
            if bucket.consume():
                return True, None
            else:
                retry_after = bucket.time_until_available()
                return False, retry_after
    
    def _create_bucket(self, endpoint_type: str) -> TokenBucket:
        """Create a new token bucket for the given endpoint type."""
        now = time.time()
        
        if endpoint_type == "upload":
            # More restrictive limits for upload endpoints
            capacity = 10
            refill_rate = 100 / 3600  # 100 per hour
        elif endpoint_type == "status":
            # More lenient for status checks
            capacity = 100
            refill_rate = 1000 / 3600  # 1000 per hour
        else:
            # General endpoints
            capacity = 60
            refill_rate = 60 / 60  # 60 per minute
        
        return TokenBucket(
            capacity=capacity,
            tokens=capacity,  # Start with full bucket
            refill_rate=refill_rate,
            last_refill=now
        )
    
    async def cleanup_old_buckets(self, max_age_seconds: int = 3600):
        """Clean up old unused buckets to prevent memory leaks."""
        async with self._lock:
            now = time.time()
            to_remove = []
            
            for key, bucket in self._buckets.items():
                if now - bucket.last_refill > max_age_seconds:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._buckets[key]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old rate limit buckets")


# Global rate limiter instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware for FastAPI.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint
        
    Returns:
        Response or rate limit error
    """
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Get API key from request context (set by auth middleware)
    api_key = getattr(request.state, "api_key", None)
    
    if not api_key:
        # No API key means auth middleware will handle it
        return await call_next(request)
    
    # Determine endpoint type
    endpoint_type = "general"
    if "/upload" in request.url.path or "/submit" in request.url.path:
        endpoint_type = "upload"
    elif "/status" in request.url.path:
        endpoint_type = "status"
    
    # Check rate limit
    is_allowed, retry_after = await rate_limiter.is_allowed(api_key, endpoint_type)
    
    if not is_allowed:
        logger.warning(
            f"Rate limit exceeded for API key {api_key[:8]}... on {endpoint_type} endpoint",
            extra={
                "api_key_prefix": api_key[:8],
                "endpoint_type": endpoint_type,
                "retry_after": retry_after
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded for {endpoint_type} endpoints",
                "retry_after": int(retry_after) if retry_after else 60
            },
            headers={
                "Retry-After": str(int(retry_after) if retry_after else 60)
            }
        )
    
    # Process request
    response = await call_next(request)
    
    return response


class StandardErrorHandler:
    """Standardized error handling with PRD error codes."""
    
    ERROR_CODES = {
        # Authentication errors
        "invalid_api_key": {
            "status_code": 401,
            "message": "Invalid or missing API key"
        },
        "api_key_inactive": {
            "status_code": 401,
            "message": "API key is inactive"
        },
        
        # Validation errors
        "invalid_processing_mode": {
            "status_code": 400,
            "message": "Invalid processing mode. Must be one of: STTN, LAMA, PROPAINTER"
        },
        "invalid_subtitle_area": {
            "status_code": 400,
            "message": "Invalid subtitle area coordinates"
        },
        "invalid_video_url": {
            "status_code": 400,
            "message": "Invalid video URL. Must be HTTPS and point to a video file"
        },
        "invalid_video_file": {
            "status_code": 400,
            "message": "Invalid video file format or size"
        },
        
        # Resource errors
        "job_not_found": {
            "status_code": 404,
            "message": "Job not found"
        },
        "video_not_found": {
            "status_code": 404,
            "message": "Video file not found"
        },
        
        # Rate limiting
        "rate_limit_exceeded": {
            "status_code": 429,
            "message": "Rate limit exceeded"
        },
        
        # File size errors
        "file_too_large": {
            "status_code": 413,
            "message": "File size exceeds maximum allowed limit"
        },
        
        # Processing errors
        "processing_failed": {
            "status_code": 500,
            "message": "Video processing failed"
        },
        "insufficient_gpu_memory": {
            "status_code": 503,
            "message": "Insufficient GPU memory for processing"
        },
        
        # Generic errors
        "internal_error": {
            "status_code": 500,
            "message": "Internal server error"
        }
    }
    
    @classmethod
    def create_error_response(cls, error_code: str, details: Optional[str] = None) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            error_code: Error code from ERROR_CODES
            details: Optional additional details
            
        Returns:
            JSONResponse with standardized error format
        """
        if error_code not in cls.ERROR_CODES:
            error_code = "internal_error"
        
        error_info = cls.ERROR_CODES[error_code]
        
        content = {
            "error": error_code,
            "message": error_info["message"]
        }
        
        if details:
            content["details"] = details
        
        return JSONResponse(
            status_code=error_info["status_code"],
            content=content
        )
    
    @classmethod
    def handle_validation_error(cls, exc: Exception) -> JSONResponse:
        """Handle Pydantic validation errors."""
        error_details = str(exc)
        
        # Map common validation errors to specific codes
        if "processing_mode" in error_details.lower():
            return cls.create_error_response("invalid_processing_mode", error_details)
        elif "subtitle_area" in error_details.lower():
            return cls.create_error_response("invalid_subtitle_area", error_details)
        elif "video_url" in error_details.lower():
            return cls.create_error_response("invalid_video_url", error_details)
        else:
            return cls.create_error_response("internal_error", error_details)


# Background task to clean up old rate limit buckets
async def cleanup_rate_limiter():
    """Background task to clean up old rate limit buckets."""
    while True:
        try:
            await rate_limiter.cleanup_old_buckets()
            await asyncio.sleep(300)  # Clean up every 5 minutes
        except Exception as e:
            logger.error(f"Error cleaning up rate limiter: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error
