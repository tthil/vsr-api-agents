"""
Security hardening middleware for VSR API.

Implements HTTPS enforcement, CORS configuration, request size limits,
and audit logging for production security.
"""

import time
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog
import os

logger = structlog.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security hardening middleware for production deployment.
    
    Enforces HTTPS, adds security headers, and implements request size limits.
    """
    
    def __init__(self, app, enforce_https: bool = None):
        super().__init__(app)
        self.enforce_https = enforce_https if enforce_https is not None else os.getenv("ENFORCE_HTTPS", "false").lower() == "true"
        self.max_request_size = int(os.getenv("MAX_REQUEST_SIZE", str(100 * 1024 * 1024)))  # 100MB default
        
    async def dispatch(self, request: Request, call_next):
        # HTTPS enforcement
        if self.enforce_https and request.url.scheme != "https":
            # Allow health checks and metrics on HTTP for internal monitoring
            if request.url.path not in ["/health/healthz", "/health/readyz", "/metrics"]:
                return JSONResponse(
                    status_code=426,
                    content={
                        "error": "https_required",
                        "message": "HTTPS is required for this endpoint",
                        "upgrade_required": "https"
                    }
                )
        
        # Request size limit
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request size exceeds limit of {self.max_request_size} bytes",
                    "max_size": self.max_request_size
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        # HSTS (only if HTTPS)
        if self.enforce_https:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


class AuditLogger:
    """
    Audit logging for security events.
    
    Logs authentication failures, suspicious activities, and security violations.
    """
    
    def __init__(self):
        self.failed_auth_attempts = {}
        self.suspicious_ips = set()
        
    def log_auth_failure(self, request: Request, reason: str, api_key: str = None):
        """Log authentication failure."""
        client_ip = self._get_client_ip(request)
        
        # Track failed attempts per IP
        if client_ip not in self.failed_auth_attempts:
            self.failed_auth_attempts[client_ip] = []
        
        self.failed_auth_attempts[client_ip].append({
            "timestamp": time.time(),
            "reason": reason,
            "api_key": api_key[:8] + "..." if api_key else None,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "path": str(request.url.path)
        })
        
        # Mark IP as suspicious after multiple failures
        recent_failures = [
            attempt for attempt in self.failed_auth_attempts[client_ip]
            if time.time() - attempt["timestamp"] < 3600  # Last hour
        ]
        
        if len(recent_failures) >= 5:
            self.suspicious_ips.add(client_ip)
            logger.warning(
                "Suspicious IP detected",
                ip=client_ip,
                failures_last_hour=len(recent_failures),
                reason=reason
            )
        
        logger.warning(
            "Authentication failure",
            ip=client_ip,
            reason=reason,
            api_key=api_key[:8] + "..." if api_key else None,
            user_agent=request.headers.get("user-agent", "unknown"),
            path=str(request.url.path),
            failures_from_ip=len(recent_failures)
        )
    
    def log_security_violation(self, request: Request, violation_type: str, details: Dict[str, Any]):
        """Log security violation."""
        client_ip = self._get_client_ip(request)
        
        logger.error(
            "Security violation",
            ip=client_ip,
            violation_type=violation_type,
            details=details,
            user_agent=request.headers.get("user-agent", "unknown"),
            path=str(request.url.path)
        )
    
    def is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is marked as suspicious."""
        return ip in self.suspicious_ips
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for security endpoints.
    
    Implements rate limiting to prevent abuse of authentication and sensitive endpoints.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = {}
        self.window_size = 300  # 5 minutes
        self.max_requests = 100  # Max requests per window
        
        # Stricter limits for auth endpoints
        self.auth_endpoints = ["/api/auth", "/api/login"]
        self.auth_max_requests = 10
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip, request.url.path, current_time):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": self.window_size
                },
                headers={"Retry-After": str(self.window_size)}
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def _is_rate_limited(self, ip: str, path: str, current_time: float) -> bool:
        """Check if IP is rate limited."""
        if ip not in self.request_counts:
            return False
        
        # Count recent requests
        recent_requests = [
            timestamp for timestamp in self.request_counts[ip]
            if current_time - timestamp < self.window_size
        ]
        
        # Determine limit based on endpoint
        limit = self.auth_max_requests if any(path.startswith(ep) for ep in self.auth_endpoints) else self.max_requests
        
        return len(recent_requests) >= limit
    
    def _record_request(self, ip: str, timestamp: float):
        """Record request timestamp."""
        if ip not in self.request_counts:
            self.request_counts[ip] = []
        
        self.request_counts[ip].append(timestamp)
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove old request records."""
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                timestamp for timestamp in self.request_counts[ip]
                if current_time - timestamp < self.window_size
            ]
            
            # Remove empty entries
            if not self.request_counts[ip]:
                del self.request_counts[ip]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"


def get_cors_middleware():
    """
    Get configured CORS middleware for production.
    
    Returns:
        Configured CORSMiddleware instance
    """
    # Production CORS settings
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
    if not allowed_origins or allowed_origins == [""]:
        # Default to no CORS in production for security
        allowed_origins = []
    
    return CORSMiddleware(
        allow_origins=allowed_origins,
        allow_credentials=False,  # Disable credentials for security
        allow_methods=["GET", "POST"],  # Only allow necessary methods
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=["X-Response-Time", "X-Daily-Quota-Used", "X-Daily-Quota-Remaining"]
    )


# Global audit logger instance
audit_logger = AuditLogger()


def log_auth_failure(request: Request, reason: str, api_key: str = None):
    """Log authentication failure."""
    audit_logger.log_auth_failure(request, reason, api_key)


def log_security_violation(request: Request, violation_type: str, details: Dict[str, Any]):
    """Log security violation."""
    audit_logger.log_security_violation(request, violation_type, details)


def is_suspicious_ip(request: Request) -> bool:
    """Check if request IP is suspicious."""
    client_ip = request.client.host if request.client else "unknown"
    return audit_logger.is_suspicious_ip(client_ip)
