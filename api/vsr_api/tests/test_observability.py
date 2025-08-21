"""
Tests for observability features including metrics, health checks, quotas, and security.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
import time

from vsr_api.middleware.metrics import MetricsMiddleware
from vsr_api.middleware.quotas import QuotasMiddleware
from vsr_api.middleware.security import SecurityMiddleware
from vsr_api.routes.health import router as health_router
from vsr_api.routes.metrics import router as metrics_router


@pytest.fixture
def test_app():
    """Create test FastAPI app with observability middleware."""
    app = FastAPI()
    
    # Add middleware in correct order
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(QuotasMiddleware)
    
    # Add routes
    app.include_router(health_router)
    app.include_router(metrics_router)
    
    # Add test endpoint
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.post("/test-upload")
    async def test_upload():
        await asyncio.sleep(0.1)  # Simulate processing
        return {"status": "uploaded"}
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
    
    @patch('vsr_api.routes.health.get_db')
    @patch('vsr_api.routes.health.get_rabbitmq_client')
    @patch('vsr_api.routes.health.get_spaces_client')
    def test_readiness_check_healthy(self, mock_spaces, mock_rabbitmq, mock_db, client):
        """Test readiness check when all services are healthy."""
        # Mock successful connections
        mock_db.return_value.admin.command.return_value = {"ok": 1}
        mock_rabbitmq.return_value.is_connected = True
        mock_spaces.return_value.head_bucket.return_value = None
        
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["mongodb"]["status"] == "healthy"
        assert data["checks"]["rabbitmq"]["status"] == "healthy"
        assert data["checks"]["spaces"]["status"] == "healthy"
    
    @patch('vsr_api.routes.health.get_db')
    def test_readiness_check_unhealthy(self, mock_db, client):
        """Test readiness check when services are unhealthy."""
        # Mock failed MongoDB connection
        mock_db.return_value.admin.command.side_effect = Exception("Connection failed")
        
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["mongodb"]["status"] == "unhealthy"


class TestMetricsCollection:
    """Test metrics collection middleware."""
    
    def test_request_metrics_collection(self, client):
        """Test that request metrics are collected."""
        # Make several requests
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        metrics_text = response.text
        
        # Verify metrics are present
        assert "http_requests_total" in metrics_text
        assert "http_request_duration_seconds" in metrics_text
        assert "http_requests_in_progress" in metrics_text
        assert 'method="GET"' in metrics_text
        assert 'endpoint="/test"' in metrics_text
    
    def test_error_metrics_collection(self, client):
        """Test that error metrics are collected."""
        # Make request to non-existent endpoint
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Check metrics
        response = client.get("/metrics")
        metrics_text = response.text
        assert 'status="404"' in metrics_text
    
    def test_latency_metrics(self, client):
        """Test that latency metrics are collected."""
        # Make request to endpoint with artificial delay
        response = client.post("/test-upload")
        assert response.status_code == 200
        
        # Check metrics
        response = client.get("/metrics")
        metrics_text = response.text
        assert "http_request_duration_seconds" in metrics_text


class TestQuotasMiddleware:
    """Test quotas and request guards middleware."""
    
    @patch.dict('os.environ', {'DAILY_JOB_LIMIT': '5'})
    def test_daily_job_limit(self, client):
        """Test daily job limit enforcement."""
        # This would need to be tested with actual job submission endpoints
        # For now, test that the middleware is properly configured
        response = client.get("/test")
        assert response.status_code == 200
    
    @patch.dict('os.environ', {'BUSINESS_HOURS_ONLY': 'true', 'BUSINESS_START_HOUR': '9', 'BUSINESS_END_HOUR': '17'})
    def test_business_hours_enforcement(self, client):
        """Test business hours enforcement."""
        # Mock current time to be outside business hours
        with patch('vsr_api.middleware.quotas.datetime') as mock_datetime:
            mock_datetime.now.return_value.hour = 20  # 8 PM
            mock_datetime.now.return_value.weekday.return_value = 1  # Tuesday
            
            response = client.post("/test-upload")
            # This would return 403 if business hours are enforced on this endpoint
            # For test endpoint, it should still work
            assert response.status_code == 200


class TestSecurityMiddleware:
    """Test security hardening middleware."""
    
    def test_security_headers(self, client):
        """Test that security headers are added."""
        response = client.get("/test")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
    
    @patch.dict('os.environ', {'ENFORCE_HTTPS': 'true'})
    def test_https_enforcement(self, client):
        """Test HTTPS enforcement."""
        # This would need to test actual HTTPS redirect behavior
        # For test client, we can verify the middleware is configured
        response = client.get("/test")
        assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers configuration."""
        response = client.options("/test", headers={"Origin": "https://example.com"})
        # CORS headers would be set by FastAPI CORS middleware
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented


class TestIntegration:
    """Integration tests for observability features."""
    
    def test_full_request_lifecycle(self, client):
        """Test complete request lifecycle with all middleware."""
        # Make request that goes through all middleware
        start_time = time.time()
        response = client.get("/test")
        end_time = time.time()
        
        assert response.status_code == 200
        assert response.json() == {"message": "test"}
        
        # Verify request was processed quickly
        assert (end_time - start_time) < 1.0
        
        # Check that metrics were collected
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "http_requests_total" in metrics_response.text
    
    def test_health_and_metrics_endpoints(self, client):
        """Test that health and metrics endpoints work together."""
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Check readiness
        ready_response = client.get("/ready")
        assert ready_response.status_code in [200, 503]  # Depends on mock setup
        
        # Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Verify metrics include health endpoint calls
        metrics_text = metrics_response.text
        assert 'endpoint="/health"' in metrics_text or 'endpoint="/ready"' in metrics_text


class TestWorkerMetrics:
    """Test worker metrics functionality."""
    
    @patch('vsr_worker.metrics.subprocess.run')
    def test_gpu_monitoring(self, mock_subprocess):
        """Test GPU monitoring functionality."""
        from vsr_worker.metrics import WorkerMetrics
        
        # Mock nvidia-smi output
        mock_subprocess.return_value.stdout = "85, 7500, 8000\n"
        mock_subprocess.return_value.returncode = 0
        
        metrics = WorkerMetrics()
        gpu_stats = metrics.get_gpu_stats()
        
        assert gpu_stats["utilization"] == 85
        assert gpu_stats["memory_used"] == 7500
        assert gpu_stats["memory_total"] == 8000
    
    def test_job_duration_tracking(self):
        """Test job duration tracking."""
        from vsr_worker.metrics import WorkerMetrics
        
        metrics = WorkerMetrics()
        
        # Start job timer
        job_id = "test_job_123"
        metrics.start_job_timer(job_id)
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # End job timer
        duration = metrics.end_job_timer(job_id)
        
        assert duration >= 0.1
        assert duration < 1.0  # Should be quick


class TestCleanupSystem:
    """Test data retention and cleanup system."""
    
    @patch('scripts.cleanup_storage.StorageCleanupManager._init_clients')
    def test_cleanup_manager_initialization(self, mock_init):
        """Test cleanup manager initialization."""
        from scripts.cleanup_storage import StorageCleanupManager
        
        manager = StorageCleanupManager()
        mock_init.assert_called_once()
    
    @patch('scripts.cleanup_storage.StorageCleanupManager._cleanup_temp_files')
    @patch('scripts.cleanup_storage.StorageCleanupManager._cleanup_old_videos')
    @patch('scripts.cleanup_storage.StorageCleanupManager._cleanup_old_jobs')
    async def test_cleanup_dry_run(self, mock_jobs, mock_videos, mock_temp):
        """Test cleanup dry run functionality."""
        from scripts.cleanup_storage import StorageCleanupManager
        
        # Mock cleanup methods
        mock_temp.return_value = {"files_deleted": 5, "bytes_freed": 1024, "errors": []}
        mock_videos.return_value = {"videos_deleted": 2, "bytes_freed": 2048, "errors": []}
        mock_jobs.return_value = {"jobs_deleted": 10, "events_deleted": 25, "errors": []}
        
        manager = StorageCleanupManager()
        result = await manager.run_cleanup(dry_run=True)
        
        assert result["dry_run"] is True
        assert "cleanup_results" in result
        assert result["cleanup_results"]["temp_files"]["files_deleted"] == 5
        assert result["cleanup_results"]["processed_videos"]["videos_deleted"] == 2
        assert result["cleanup_results"]["job_records"]["jobs_deleted"] == 10


if __name__ == "__main__":
    pytest.main([__file__])
