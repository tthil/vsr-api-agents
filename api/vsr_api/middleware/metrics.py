"""
Metrics collection middleware for VSR API.

Tracks request counts, latency, error rates, and other observability metrics.
"""

import time
from typing import Dict, Any
from collections import defaultdict, Counter
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """
    In-memory metrics collector for API observability.
    
    Collects and stores metrics about API requests, response times,
    error rates, and other operational metrics.
    """
    
    def __init__(self):
        self.request_count = Counter()
        self.request_duration = defaultdict(list)
        self.error_count = Counter()
        self.status_codes = Counter()
        self.active_requests = 0
        self.start_time = time.time()
        
    def record_request(self, method: str, path: str, status_code: int, duration: float):
        """Record a completed request."""
        # Request counting
        self.request_count[f"{method}:{path}"] += 1
        self.request_count["total"] += 1
        
        # Duration tracking
        self.request_duration[f"{method}:{path}"].append(duration)
        self.request_duration["all"].append(duration)
        
        # Status code tracking
        self.status_codes[status_code] += 1
        
        # Error tracking
        if status_code >= 400:
            self.error_count[f"{method}:{path}"] += 1
            self.error_count["total"] += 1
            
            if 400 <= status_code < 500:
                self.error_count["4xx"] += 1
            elif status_code >= 500:
                self.error_count["5xx"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate average response times
        avg_durations = {}
        for endpoint, durations in self.request_duration.items():
            if durations:
                avg_durations[endpoint] = {
                    "avg_ms": round(sum(durations) / len(durations) * 1000, 2),
                    "min_ms": round(min(durations) * 1000, 2),
                    "max_ms": round(max(durations) * 1000, 2),
                    "count": len(durations)
                }
        
        # Calculate error rates
        error_rates = {}
        for endpoint, count in self.request_count.items():
            if endpoint != "total" and count > 0:
                errors = self.error_count.get(endpoint, 0)
                error_rates[endpoint] = round((errors / count) * 100, 2)
        
        return {
            "timestamp": current_time,
            "uptime_seconds": round(uptime, 2),
            "requests": {
                "total": self.request_count["total"],
                "by_endpoint": dict(self.request_count),
                "active": self.active_requests
            },
            "response_times": avg_durations,
            "status_codes": dict(self.status_codes),
            "errors": {
                "total": self.error_count["total"],
                "4xx": self.error_count["4xx"],
                "5xx": self.error_count["5xx"],
                "by_endpoint": dict(self.error_count),
                "error_rates_percent": error_rates
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.request_count.clear()
        self.request_duration.clear()
        self.error_count.clear()
        self.status_codes.clear()
        self.active_requests = 0
        self.start_time = time.time()


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for collecting request metrics.
    
    Automatically tracks all HTTP requests passing through the API,
    recording timing, status codes, and error rates.
    """
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip metrics collection for certain paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Record request start
        start_time = time.time()
        metrics_collector.active_requests += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            metrics_collector.record_request(
                method=request.method,
                path=self._normalize_path(request.url.path),
                status_code=response.status_code,
                duration=duration
            )
            
            # Add metrics headers
            response.headers["X-Response-Time"] = f"{duration*1000:.2f}ms"
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics_collector.record_request(
                method=request.method,
                path=self._normalize_path(request.url.path),
                status_code=500,
                duration=duration
            )
            
            logger.error(
                "Request processing error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration * 1000,
                error=str(e)
            )
            
            raise
        
        finally:
            metrics_collector.active_requests -= 1
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize URL paths to avoid high cardinality metrics.
        
        Replaces dynamic path segments (UUIDs, IDs) with placeholders.
        """
        import re
        
        # Replace UUIDs with placeholder
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace other numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path


def get_metrics() -> Dict[str, Any]:
    """Get current metrics from the global collector."""
    return metrics_collector.get_metrics()


def reset_metrics():
    """Reset all metrics (useful for testing)."""
    metrics_collector.reset_metrics()
