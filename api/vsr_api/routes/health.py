"""
Health check and readiness endpoints for VSR API.

Provides observability endpoints for monitoring system health and readiness.
"""

import asyncio
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from vsr_shared.db import get_db
from vsr_shared.queue import RabbitMQClient
from vsr_shared.spaces import SpacesClient


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str = "1.0.0"
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    timestamp: float
    services: Dict[str, Dict[str, Any]]
    overall_ready: bool


class ServiceStatus(BaseModel):
    """Individual service status model."""
    status: str
    response_time_ms: float
    error: str = None


router = APIRouter(prefix="/health", tags=["health"])

# Track application start time for uptime calculation
_start_time = time.time()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns basic application health status without checking dependencies.
    This endpoint should always return 200 OK if the application is running.
    """
    current_time = time.time()
    uptime = current_time - _start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        uptime_seconds=uptime
    )


@router.get("/readyz", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check endpoint.
    
    Checks if the application is ready to serve traffic by verifying
    connectivity to all required services (MongoDB, RabbitMQ, Spaces).
    """
    current_time = time.time()
    services = {}
    overall_ready = True
    
    # Check MongoDB connectivity
    try:
        start = time.time()
        db = await get_db()
        # Simple ping to verify connection
        await db.command("ping")
        response_time = (time.time() - start) * 1000
        
        services["mongodb"] = {
            "status": "healthy",
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        services["mongodb"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "error": str(e)
        }
        overall_ready = False
    
    # Check RabbitMQ connectivity
    try:
        start = time.time()
        rabbitmq_client = RabbitMQClient()
        # Test connection by checking if we can connect
        connection = await rabbitmq_client.get_connection()
        if connection and not connection.is_closed:
            response_time = (time.time() - start) * 1000
            services["rabbitmq"] = {
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            }
        else:
            raise Exception("Connection is closed or None")
    except Exception as e:
        services["rabbitmq"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "error": str(e)
        }
        overall_ready = False
    
    # Check Spaces/MinIO connectivity
    try:
        start = time.time()
        spaces_client = SpacesClient()
        # Test by listing buckets (lightweight operation)
        await spaces_client.list_buckets()
        response_time = (time.time() - start) * 1000
        
        services["spaces"] = {
            "status": "healthy",
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        services["spaces"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "error": str(e)
        }
        overall_ready = False
    
    status = "ready" if overall_ready else "not_ready"
    
    response = ReadinessResponse(
        status=status,
        timestamp=current_time,
        services=services,
        overall_ready=overall_ready
    )
    
    # Return 503 Service Unavailable if not ready
    if not overall_ready:
        raise HTTPException(status_code=503, detail=response.dict())
    
    return response


@router.get("/metrics/basic")
async def basic_metrics():
    """
    Basic metrics endpoint for simple monitoring.
    
    Provides basic application metrics without full Prometheus format.
    """
    current_time = time.time()
    uptime = current_time - _start_time
    
    # Get queue depth from RabbitMQ
    queue_depth = 0
    try:
        rabbitmq_client = RabbitMQClient()
        # This would need to be implemented in RabbitMQClient
        # queue_depth = await rabbitmq_client.get_queue_depth("vsr.process.q")
    except Exception:
        pass
    
    return {
        "timestamp": current_time,
        "uptime_seconds": uptime,
        "queue_depth": queue_depth,
        "status": "healthy"
    }
