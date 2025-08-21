"""API Key authentication middleware for VSR API."""

import hashlib
import hmac
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from vsr_shared.logging import get_logger
from vsr_shared.models import ApiKey
from vsr_shared.db.client import get_mongodb_client
from vsr_shared.db.dal import ApiKeysDAL

logger = get_logger(__name__)

# Security scheme for OpenAPI documentation
api_key_scheme = HTTPBearer(
    scheme_name="API Key",
    description="API key authentication using Bearer token in Authorization header"
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and rate limiting."""

    def __init__(self, app, excluded_paths: Optional[list] = None):
        """
        Initialize API key authentication middleware.

        Args:
            app: FastAPI application instance
            excluded_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/health",
            "/healthz",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request with API key authentication."""
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        try:
            # Extract and validate API key
            api_key_doc = await self._authenticate_request(request)
            
            # Add authenticated API key to request state
            request.state.api_key = api_key_doc
            request.state.api_key_id = api_key_doc.id
            
            # Log successful authentication
            logger.info(
                "API key authenticated",
                api_key_id=str(api_key_doc.id),
                path=request.url.path,
                method=request.method,
            )
            
            return await call_next(request)
            
        except HTTPException:
            # Re-raise HTTP exceptions (authentication failures)
            raise
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal authentication error"
            )

    async def _authenticate_request(self, request: Request) -> ApiKey:
        """
        Authenticate request using API key.

        Args:
            request: FastAPI request object

        Returns:
            ApiKey: Authenticated API key document

        Raises:
            HTTPException: If authentication fails
        """
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse Bearer token
        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid scheme")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Use 'Bearer <token>'",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key
        api_key_doc = await self._validate_api_key(token)
        if not api_key_doc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if API key is active
        if not api_key_doc.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is disabled",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return api_key_doc

    async def _validate_api_key(self, token: str) -> Optional[ApiKey]:
        """
        Validate API key token against database.

        Args:
            token: API key token to validate

        Returns:
            ApiKey document if valid, None otherwise
        """
        try:
            # Get MongoDB client and DAL
            mongo_client = get_mongodb_client()
            await mongo_client.connect()
            db = mongo_client.get_database()
            api_keys_dal = ApiKeysDAL(db)
            
            # Hash the provided token for comparison
            token_hash = self._hash_api_key(token)
            
            # Find API key by hash
            api_key_doc = await api_keys_dal.get_by_key_hash(token_hash)
            
            return api_key_doc
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    def _hash_api_key(self, api_key: str) -> str:
        """
        Hash API key using secure algorithm.

        Args:
            api_key: Raw API key string

        Returns:
            Hashed API key string
        """
        # Use SHA-256 for API key hashing
        return hashlib.sha256(api_key.encode()).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        True if strings are equal, False otherwise
    """
    return hmac.compare_digest(a, b)


async def get_current_api_key(request: Request) -> ApiKey:
    """
    Dependency to get current authenticated API key.

    Args:
        request: FastAPI request object

    Returns:
        ApiKey: Current authenticated API key

    Raises:
        HTTPException: If no API key is authenticated
    """
    api_key = getattr(request.state, "api_key", None)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authenticated API key found"
        )
    return api_key


async def get_current_api_key_id(request: Request) -> str:
    """
    Dependency to get current authenticated API key ID.

    Args:
        request: FastAPI request object

    Returns:
        str: Current authenticated API key ID

    Raises:
        HTTPException: If no API key is authenticated
    """
    api_key_id = getattr(request.state, "api_key_id", None)
    if not api_key_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authenticated API key found"
        )
    return str(api_key_id)


class RateLimiter:
    """In-memory rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.buckets = {}  # api_key_id -> (tokens, last_refill)

    async def is_allowed(self, api_key_id: str) -> Tuple[bool, dict]:
        """
        Check if request is allowed for API key.

        Args:
            api_key_id: API key identifier

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        
        # Get or create bucket for API key
        if api_key_id not in self.buckets:
            self.buckets[api_key_id] = (self.max_requests, current_time)
        
        tokens, last_refill = self.buckets[api_key_id]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = int(time_elapsed * (self.max_requests / self.window_seconds))
        
        # Refill bucket (capped at max_requests)
        tokens = min(self.max_requests, tokens + tokens_to_add)
        last_refill = current_time
        
        # Check if request is allowed
        if tokens > 0:
            tokens -= 1
            self.buckets[api_key_id] = (tokens, last_refill)
            allowed = True
        else:
            allowed = False
        
        # Rate limit info for headers
        rate_limit_info = {
            "limit": self.max_requests,
            "remaining": max(0, tokens),
            "reset": int(last_refill + self.window_seconds),
            "window": self.window_seconds,
        }
        
        return allowed, rate_limit_info


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 requests per hour


async def check_rate_limit(request: Request) -> dict:
    """
    Check rate limit for current API key.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit information

    Raises:
        HTTPException: If rate limit is exceeded
    """
    api_key_id = await get_current_api_key_id(request)
    
    allowed, rate_limit_info = await rate_limiter.is_allowed(api_key_id)
    
    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            api_key_id=api_key_id,
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
                "X-RateLimit-Reset": str(rate_limit_info["reset"]),
                "Retry-After": str(rate_limit_info["window"]),
            },
        )
    
    return rate_limit_info
