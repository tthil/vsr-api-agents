"""
Comprehensive test suite for VSR API endpoints.
Tests all endpoints with happy paths and error cases.
"""
import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json
from datetime import datetime

from vsr_api.main import app
from vsr_shared.models import Job, JobStatus, ProcessingMode, SubtitleArea


# Test client setup
client = TestClient(app)

# Test data
VALID_API_KEY = "test-api-key-12345"
INVALID_API_KEY = "invalid-key"
TEST_JOB_ID = str(uuid.uuid4())
TEST_VIDEO_URL = "https://example.com/test-video.mp4"
TEST_CALLBACK_URL = "https://example.com/webhook"


class TestAuthentication:
    """Test authentication middleware."""
    
    def test_missing_authorization_header(self):
        """Test request without Authorization header."""
        response = client.post("/api/upload-and-submit")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["error"] == "invalid_api_key"
    
    def test_invalid_authorization_format(self):
        """Test invalid Authorization header format."""
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": "InvalidFormat"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["error"] == "invalid_api_key"
    
    def test_invalid_api_key(self):
        """Test with invalid API key."""
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {INVALID_API_KEY}"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["error"] == "invalid_api_key"
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    def test_valid_api_key(self, mock_verify):
        """Test with valid API key."""
        mock_verify.return_value = "test-user-id"
        
        response = client.get(
            "/healthz",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"}
        )
        assert response.status_code == status.HTTP_200_OK


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_rate_limit_exceeded(self, mock_rate_limit, mock_verify):
        """Test rate limit exceeded response."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (False, 60)
        
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"}
        )
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert response.json()["error"] == "rate_limit_exceeded"
        assert "Retry-After" in response.headers
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_rate_limit_allowed(self, mock_rate_limit, mock_verify):
        """Test request allowed by rate limiter."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        response = client.get(
            "/healthz",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"}
        )
        assert response.status_code == status.HTTP_200_OK


class TestUploadAndSubmit:
    """Test upload and submit endpoint."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    @patch('vsr_shared.db.client.get_database')
    @patch('vsr_shared.spaces.get_spaces_client')
    @patch('vsr_shared.queue.integration.publish_job')
    def test_upload_and_submit_success(self, mock_publish, mock_spaces, mock_db, mock_rate_limit, mock_verify):
        """Test successful video upload and submission."""
        # Setup mocks
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        mock_db_instance = AsyncMock()
        mock_db_instance.jobs.count_documents.return_value = 2
        mock_db_instance.jobs.insert_one.return_value = None
        mock_db.return_value = mock_db_instance
        
        mock_spaces_instance = AsyncMock()
        mock_spaces_instance.upload_fileobj.return_value = None
        mock_spaces.return_value = mock_spaces_instance
        
        mock_publish.return_value = None
        
        # Create test file
        test_file = ("video.mp4", b"fake video content", "video/mp4")
        
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={
                "mode": "LAMA",
                "subtitle_area": "[100,400,800,500]",
                "callback_url": TEST_CALLBACK_URL
            },
            files={"video_file": test_file}
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["mode"] == "LAMA"
        assert data["subtitle_area"] == {"x1": 100, "y1": 400, "x2": 800, "y2": 500}
        assert data["eta_seconds"] == 180  # 120 + 2*30
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_upload_invalid_file_type(self, mock_rate_limit, mock_verify):
        """Test upload with invalid file type."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        test_file = ("document.txt", b"not a video", "text/plain")
        
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={"mode": "LAMA"},
            files={"video_file": test_file}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["error"] == "invalid_video_file"
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_upload_invalid_processing_mode(self, mock_rate_limit, mock_verify):
        """Test upload with invalid processing mode."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        test_file = ("video.mp4", b"fake video content", "video/mp4")
        
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={"mode": "INVALID_MODE"},
            files={"video_file": test_file}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["error"] == "invalid_processing_mode"


class TestSubmitVideoURL:
    """Test submit video URL endpoint."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    @patch('vsr_shared.db.client.get_database')
    @patch('vsr_shared.spaces.get_spaces_client')
    @patch('vsr_shared.queue.integration.publish_job')
    @patch('httpx.AsyncClient')
    def test_submit_video_url_success(self, mock_httpx, mock_publish, mock_spaces, mock_db, mock_rate_limit, mock_verify):
        """Test successful video URL submission."""
        # Setup mocks
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        mock_db_instance = AsyncMock()
        mock_db_instance.jobs.count_documents.return_value = 1
        mock_db_instance.jobs.insert_one.return_value = None
        mock_db.return_value = mock_db_instance
        
        mock_spaces_instance = AsyncMock()
        mock_spaces_instance.upload_fileobj.return_value = None
        mock_spaces.return_value = mock_spaces_instance
        
        mock_publish.return_value = None
        
        # Mock HTTP client
        mock_head_response = MagicMock()
        mock_head_response.headers = {"content-type": "video/mp4", "content-length": "1000000"}
        mock_head_response.raise_for_status.return_value = None
        
        mock_stream_response = MagicMock()
        mock_stream_response.headers = {"content-type": "video/mp4"}
        mock_stream_response.raise_for_status.return_value = None
        mock_stream_response.aiter_bytes.return_value = [b"fake video chunk"] * 100
        
        mock_client_instance = AsyncMock()
        mock_client_instance.head.return_value = mock_head_response
        mock_client_instance.stream.return_value.__aenter__.return_value = mock_stream_response
        
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.post(
            "/api/submit-video",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={
                "mode": "STTN",
                "video_url": TEST_VIDEO_URL,
                "subtitle_area": "[50,350,900,450]"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["mode"] == "STTN"
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_submit_invalid_video_url(self, mock_rate_limit, mock_verify):
        """Test submit with invalid video URL."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        response = client.post(
            "/api/submit-video",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={
                "mode": "LAMA",
                "video_url": "http://invalid-url.com/video.mp4"  # HTTP not HTTPS
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["error"] == "invalid_video_url"


class TestJobStatus:
    """Test job status endpoint."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    @patch('vsr_shared.db.client.get_database')
    @patch('vsr_shared.spaces.get_spaces_client')
    def test_get_job_status_success(self, mock_spaces, mock_db, mock_rate_limit, mock_verify):
        """Test successful job status retrieval."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        # Mock job data
        job_data = {
            "_id": uuid.UUID(TEST_JOB_ID),
            "status": JobStatus.COMPLETED.value,
            "mode": ProcessingMode.LAMA.value,
            "subtitle_area": {"x1": 100, "y1": 400, "x2": 800, "y2": 500},
            "progress": 100,
            "video_key": "uploads/test/video.mp4",
            "processed_video_key": "processed/test/video.mp4",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "started_at": datetime.now(),
            "completed_at": datetime.now()
        }
        
        mock_db_instance = AsyncMock()
        mock_db_instance.jobs.find_one.return_value = job_data
        mock_db.return_value = mock_db_instance
        
        # Mock presigned URL generation
        mock_spaces_instance = AsyncMock()
        mock_spaces.return_value = mock_spaces_instance
        
        with patch('vsr_shared.presigned.generate_presigned_get_url') as mock_presigned:
            mock_presigned.return_value = "https://signed-url.com/video.mp4"
            
            response = client.get(
                f"/api/job-status/{TEST_JOB_ID}",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == TEST_JOB_ID
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert "processed_video_url" in data
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    @patch('vsr_shared.db.client.get_database')
    def test_get_job_status_not_found(self, mock_db, mock_rate_limit, mock_verify):
        """Test job status for non-existent job."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        mock_db_instance = AsyncMock()
        mock_db_instance.jobs.find_one.return_value = None
        mock_db.return_value = mock_db_instance
        
        response = client.get(
            f"/api/job-status/{TEST_JOB_ID}",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["error"] == "job_not_found"


class TestGenerateUploadURL:
    """Test generate upload URL endpoint."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    @patch('vsr_shared.presigned.generate_upload_url')
    def test_generate_upload_url_success(self, mock_generate_url, mock_rate_limit, mock_verify):
        """Test successful upload URL generation."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        mock_generate_url.return_value = {
            "upload_url": "https://presigned-url.com/upload",
            "key": "uploads/test/video.mp4",
            "expires_in": 3600
        }
        
        response = client.post(
            "/api/generate-upload-url",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            json={"content_type": "video/mp4"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "upload_url" in data
        assert "key" in data
        assert data["expires_in"] == 3600
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_generate_upload_url_invalid_content_type(self, mock_rate_limit, mock_verify):
        """Test upload URL generation with invalid content type."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        response = client.post(
            "/api/generate-upload-url",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            json={"content_type": "text/plain"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "ok"


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    @patch('vsr_api.middleware.auth.verify_api_key')
    @patch('vsr_api.middleware.rate_limit.rate_limiter.is_allowed')
    def test_internal_server_error(self, mock_rate_limit, mock_verify):
        """Test internal server error handling."""
        mock_verify.return_value = "test-user-id"
        mock_rate_limit.return_value = (True, None)
        
        with patch('vsr_shared.db.client.get_database') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get(
                f"/api/job-status/{TEST_JOB_ID}",
                headers={"Authorization": f"Bearer {VALID_API_KEY}"}
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert response.json()["error"] == "internal_error"
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        response = client.post(
            "/api/upload-and-submit",
            headers={"Authorization": f"Bearer {VALID_API_KEY}"},
            data={"mode": ""}  # Empty mode should fail validation
        )
        
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]


# Integration test fixtures
@pytest.fixture
def test_video_file():
    """Create a test video file for upload tests."""
    return ("test_video.mp4", b"fake video content" * 1000, "video/mp4")


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for integration tests."""
    with patch.multiple(
        'vsr_api.middleware.auth',
        verify_api_key=AsyncMock(return_value="test-user-id")
    ), patch.multiple(
        'vsr_api.middleware.rate_limit',
        rate_limiter=MagicMock()
    ), patch.multiple(
        'vsr_shared.db.client',
        get_database=AsyncMock()
    ), patch.multiple(
        'vsr_shared.spaces',
        get_spaces_client=AsyncMock()
    ), patch.multiple(
        'vsr_shared.queue.integration',
        publish_job=AsyncMock()
    ):
        yield


class TestIntegration:
    """Integration tests with all components."""
    
    def test_full_workflow_upload_and_submit(self, mock_dependencies, test_video_file):
        """Test complete workflow from upload to job creation."""
        # This would be a comprehensive integration test
        # Testing the full flow with all components working together
        pass
    
    def test_full_workflow_url_submit(self, mock_dependencies):
        """Test complete workflow from URL submission to job creation."""
        # This would test the URL submission flow end-to-end
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
