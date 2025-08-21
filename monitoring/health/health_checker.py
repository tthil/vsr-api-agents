"""
Comprehensive health check system for VSR API services.
Monitors all components and provides detailed health status.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import time

import httpx
import pymongo
import aio_pika
import redis.asyncio as redis
import boto3
from botocore.exceptions import ClientError

from vsr_shared.db.client import get_database
from vsr_shared.queue.config import RABBITMQ_CONFIG
from vsr_shared.spaces import get_spaces_client


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class ComponentHealthChecker:
    """Base class for component health checkers."""
    
    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
    
    async def check_health(self) -> HealthCheck:
        """Perform health check for this component."""
        start_time = time.time()
        try:
            details = await self._perform_check()
            response_time = time.time() - start_time
            
            return HealthCheck(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Component is healthy",
                response_time=response_time,
                timestamp=datetime.utcnow(),
                details=details
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time=response_time,
                timestamp=datetime.utcnow()
            )
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError


class APIHealthChecker(ComponentHealthChecker):
    """Health checker for VSR API service."""
    
    def __init__(self, api_url: str):
        super().__init__("vsr-api")
        self.api_url = api_url.rstrip('/')
    
    async def _perform_check(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Check health endpoint
            health_response = await client.get(f"{self.api_url}/healthz")
            health_response.raise_for_status()
            
            # Check metrics endpoint
            metrics_response = await client.get(f"{self.api_url}/metrics")
            metrics_response.raise_for_status()
            
            return {
                "health_status": health_response.json(),
                "metrics_available": True,
                "response_code": health_response.status_code
            }


class DatabaseHealthChecker(ComponentHealthChecker):
    """Health checker for MongoDB database."""
    
    def __init__(self):
        super().__init__("mongodb")
    
    async def _perform_check(self) -> Dict[str, Any]:
        db = await get_database()
        
        # Test basic connectivity
        await db.admin.command("ping")
        
        # Check server status
        server_status = await db.admin.command("serverStatus")
        
        # Check replica set status if applicable
        replica_status = None
        try:
            replica_status = await db.admin.command("replSetGetStatus")
        except Exception:
            pass  # Not a replica set
        
        # Check collection stats
        jobs_stats = await db.command("collStats", "jobs")
        
        return {
            "server_version": server_status.get("version"),
            "uptime": server_status.get("uptime"),
            "connections": server_status.get("connections"),
            "replica_set": replica_status is not None,
            "jobs_collection_count": jobs_stats.get("count", 0),
            "jobs_collection_size": jobs_stats.get("size", 0)
        }


class RabbitMQHealthChecker(ComponentHealthChecker):
    """Health checker for RabbitMQ message broker."""
    
    def __init__(self):
        super().__init__("rabbitmq")
    
    async def _perform_check(self) -> Dict[str, Any]:
        connection = await aio_pika.connect_robust(
            host=RABBITMQ_CONFIG["host"],
            port=RABBITMQ_CONFIG["port"],
            login=RABBITMQ_CONFIG["username"],
            password=RABBITMQ_CONFIG["password"],
            virtualhost=RABBITMQ_CONFIG["vhost"]
        )
        
        try:
            channel = await connection.channel()
            
            # Check queue status
            processing_queue = await channel.get_queue(
                RABBITMQ_CONFIG["queues"]["processing"],
                ensure=False
            )
            
            queue_info = await processing_queue.get_queue_info()
            
            return {
                "connection_state": "open",
                "processing_queue_messages": queue_info.message_count,
                "processing_queue_consumers": queue_info.consumer_count,
                "channel_open": True
            }
        finally:
            await connection.close()


class RedisHealthChecker(ComponentHealthChecker):
    """Health checker for Redis cache."""
    
    def __init__(self, redis_url: str):
        super().__init__("redis")
        self.redis_url = redis_url
    
    async def _perform_check(self) -> Dict[str, Any]:
        client = redis.from_url(self.redis_url)
        
        try:
            # Test basic connectivity
            await client.ping()
            
            # Get server info
            info = await client.info()
            
            # Test set/get operation
            test_key = "health_check_test"
            await client.set(test_key, "test_value", ex=60)
            test_value = await client.get(test_key)
            await client.delete(test_key)
            
            return {
                "redis_version": info.get("redis_version"),
                "uptime": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "test_operation": test_value == b"test_value"
            }
        finally:
            await client.close()


class SpacesHealthChecker(ComponentHealthChecker):
    """Health checker for DigitalOcean Spaces."""
    
    def __init__(self, bucket_name: str):
        super().__init__("spaces")
        self.bucket_name = bucket_name
    
    async def _perform_check(self) -> Dict[str, Any]:
        client = get_spaces_client()
        
        # Test bucket access
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.head_bucket(Bucket=self.bucket_name)
        )
        
        # Test object operations
        test_key = f"health-check/{datetime.utcnow().isoformat()}.txt"
        test_content = b"Health check test content"
        
        # Upload test object
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.put_object(
                Bucket=self.bucket_name,
                Key=test_key,
                Body=test_content
            )
        )
        
        # Download test object
        downloaded = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.get_object(Bucket=self.bucket_name, Key=test_key)
        )
        
        # Clean up test object
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.delete_object(Bucket=self.bucket_name, Key=test_key)
        )
        
        return {
            "bucket_accessible": True,
            "upload_test": True,
            "download_test": downloaded['Body'].read() == test_content,
            "cleanup_test": True
        }


class GPUHealthChecker(ComponentHealthChecker):
    """Health checker for GPU resources."""
    
    def __init__(self):
        super().__init__("gpu")
    
    async def _perform_check(self) -> Dict[str, Any]:
        try:
            import torch
            
            if not torch.cuda.is_available():
                raise Exception("CUDA is not available")
            
            device_count = torch.cuda.device_count()
            if device_count == 0:
                raise Exception("No CUDA devices found")
            
            # Get GPU information
            gpu_info = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "memory_allocated": memory_allocated,
                    "memory_reserved": memory_reserved,
                    "memory_free": props.total_memory - memory_reserved,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            
            # Test basic GPU operation
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            test_successful = result.shape == (100, 100)
            
            return {
                "cuda_available": True,
                "device_count": device_count,
                "gpu_info": gpu_info,
                "test_operation": test_successful
            }
            
        except ImportError:
            raise Exception("PyTorch not available")
        except Exception as e:
            raise Exception(f"GPU check failed: {e}")


class SystemHealthChecker:
    """Main system health checker that orchestrates all component checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkers: List[ComponentHealthChecker] = []
        self._initialize_checkers()
    
    def _initialize_checkers(self):
        """Initialize all component health checkers."""
        # API Health Checker
        if "api_url" in self.config:
            self.checkers.append(APIHealthChecker(self.config["api_url"]))
        
        # Database Health Checker
        self.checkers.append(DatabaseHealthChecker())
        
        # RabbitMQ Health Checker
        self.checkers.append(RabbitMQHealthChecker())
        
        # Redis Health Checker
        if "redis_url" in self.config:
            self.checkers.append(RedisHealthChecker(self.config["redis_url"]))
        
        # Spaces Health Checker
        if "spaces_bucket" in self.config:
            self.checkers.append(SpacesHealthChecker(self.config["spaces_bucket"]))
        
        # GPU Health Checker
        if self.config.get("check_gpu", False):
            self.checkers.append(GPUHealthChecker())
    
    async def check_all_components(self) -> Dict[str, Any]:
        """Perform health checks on all components."""
        start_time = time.time()
        
        # Run all health checks concurrently
        tasks = [checker.check_health() for checker in self.checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        health_checks = []
        overall_status = HealthStatus.HEALTHY
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions
                health_check = HealthCheck(
                    name=self.checkers[i].name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                    response_time=0.0,
                    timestamp=datetime.utcnow()
                )
            else:
                health_check = result
            
            health_checks.append(health_check)
            
            # Determine overall status
            if health_check.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif health_check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        total_time = time.time() - start_time
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "total_check_time": total_time,
            "components": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                for check in health_checks
            ],
            "summary": {
                "total_components": len(health_checks),
                "healthy": len([c for c in health_checks if c.status == HealthStatus.HEALTHY]),
                "degraded": len([c for c in health_checks if c.status == HealthStatus.DEGRADED]),
                "unhealthy": len([c for c in health_checks if c.status == HealthStatus.UNHEALTHY])
            }
        }
    
    async def get_component_health(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a specific component."""
        checker = next((c for c in self.checkers if c.name == component_name), None)
        if not checker:
            return None
        
        health_check = await checker.check_health()
        return {
            "name": health_check.name,
            "status": health_check.status.value,
            "message": health_check.message,
            "response_time": health_check.response_time,
            "timestamp": health_check.timestamp.isoformat(),
            "details": health_check.details
        }


# Health check endpoint for FastAPI
async def create_health_endpoint(config: Dict[str, Any]):
    """Create health check endpoint function."""
    health_checker = SystemHealthChecker(config)
    
    async def health_check_endpoint():
        """Health check endpoint that returns system status."""
        try:
            health_status = await health_checker.check_all_components()
            
            # Determine HTTP status code based on overall health
            if health_status["overall_status"] == "healthy":
                status_code = 200
            elif health_status["overall_status"] == "degraded":
                status_code = 200  # Still operational
            else:
                status_code = 503  # Service unavailable
            
            return {
                "status_code": status_code,
                "content": health_status
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status_code": 503,
                "content": {
                    "overall_status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            }
    
    return health_check_endpoint


# Standalone health check script
async def main():
    """Main health check function for standalone execution."""
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        "api_url": os.getenv("VSR_API_URL", "http://localhost:8000"),
        "redis_url": os.getenv("VSR_REDIS_URL", "redis://localhost:6379"),
        "spaces_bucket": os.getenv("VSR_SPACES_BUCKET"),
        "check_gpu": os.getenv("VSR_CHECK_GPU", "false").lower() == "true"
    }
    
    # Initialize health checker
    health_checker = SystemHealthChecker(config)
    
    try:
        logger.info("Starting comprehensive health check...")
        health_status = await health_checker.check_all_components()
        
        # Print results
        print(json.dumps(health_status, indent=2))
        
        # Exit with appropriate code
        if health_status["overall_status"] == "unhealthy":
            exit(1)
        else:
            exit(0)
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
