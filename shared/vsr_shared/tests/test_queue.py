"""Tests for RabbitMQ queue functionality."""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from vsr_shared.models import JobMessage, JobMessageResponse, ProcessingMode, JobStatus
from vsr_shared.queue.client import RabbitMQClient
from vsr_shared.queue.publisher import JobPublisher
from vsr_shared.queue.consumer import JobConsumer
from vsr_shared.queue.topology import setup_queue_topology
from vsr_shared.queue.integration import publish_job_for_processing


class TestRabbitMQClient:
    """Test RabbitMQ client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test RabbitMQ client initialization."""
        client = RabbitMQClient(url="amqp://guest:guest@localhost:5672/")
        assert client.url == "amqp://guest:guest@localhost:5672/"
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.connection is None
        assert client.channel is None

    @pytest.mark.asyncio
    async def test_url_masking(self):
        """Test URL credential masking for logging."""
        client = RabbitMQClient(url="amqp://user:password@localhost:5672/")
        masked_url = client._mask_url(client.url)
        assert "password" not in masked_url
        assert "****" in masked_url

    @pytest.mark.asyncio
    async def test_health_check_no_connection(self):
        """Test health check with no connection."""
        client = RabbitMQClient()
        # Mock the connection to avoid actual RabbitMQ dependency
        client.connection = AsyncMock()
        client.connection.is_closed = False
        
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue
        client.channel = mock_channel
        
        result = await client.health_check()
        assert result is True


class TestJobMessage:
    """Test JobMessage model."""

    def test_job_message_creation(self):
        """Test JobMessage creation with required fields."""
        job_id = uuid.uuid4()
        message = JobMessage(
            job_id=job_id,
            mode=ProcessingMode.STTN,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        assert message.job_id == job_id
        assert message.mode == ProcessingMode.STTN
        assert message.video_key_in == "uploads/20240101/test-uuid/video.mp4"
        assert message.video_url is None
        assert message.subtitle_area is None
        assert message.callback_url is None
        assert message.trace_id is None
        assert isinstance(message.requested_at, datetime)

    def test_job_message_serialization(self):
        """Test JobMessage JSON serialization."""
        job_id = uuid.uuid4()
        message = JobMessage(
            job_id=job_id,
            mode=ProcessingMode.LAMA,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
            trace_id="test-trace-123",
        )
        
        json_data = message.model_dump_json()
        assert str(job_id) in json_data
        assert "lama" in json_data
        assert "test-trace-123" in json_data


class TestJobPublisher:
    """Test job publisher functionality."""

    @pytest.mark.asyncio
    async def test_publisher_setup(self):
        """Test publisher setup."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_client.get_channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = AsyncMock()
        
        publisher = JobPublisher(mock_client)
        await publisher.setup()
        
        assert publisher.exchange is not None
        mock_channel.confirm_delivery.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_job_message(self):
        """Test publishing a job message."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        mock_exchange = AsyncMock()
        
        publisher = JobPublisher(mock_client)
        publisher.exchange = mock_exchange
        
        job_message = JobMessage(
            job_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        result = await publisher.publish_job(job_message)
        
        assert result is True
        mock_exchange.publish.assert_called_once()


class TestJobConsumer:
    """Test job consumer functionality."""

    @pytest.mark.asyncio
    async def test_consumer_setup(self):
        """Test consumer setup."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        
        mock_client.get_channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue
        
        def mock_handler(job_message: JobMessage) -> JobMessageResponse:
            return JobMessageResponse(
                job_id=job_message.job_id,
                status=JobStatus.COMPLETED,
            )
        
        consumer = JobConsumer(mock_client, mock_handler)
        await consumer.setup()
        
        assert consumer.queue is not None

    @pytest.mark.asyncio
    async def test_message_handler_success(self):
        """Test successful message handling."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        
        def mock_handler(job_message: JobMessage) -> JobMessageResponse:
            return JobMessageResponse(
                job_id=job_message.job_id,
                status=JobStatus.COMPLETED,
                processed_video_key="processed/20240101/test-uuid/video.mp4",
            )
        
        consumer = JobConsumer(mock_client, mock_handler)
        
        job_message = JobMessage(
            job_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        response = await consumer._call_handler_safely(job_message)
        
        assert response.job_id == job_message.job_id
        assert response.status == JobStatus.COMPLETED
        assert response.processed_video_key == "processed/20240101/test-uuid/video.mp4"

    @pytest.mark.asyncio
    async def test_message_handler_error(self):
        """Test message handler error handling."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        
        def failing_handler(job_message: JobMessage) -> JobMessageResponse:
            raise Exception("Processing failed")
        
        consumer = JobConsumer(mock_client, failing_handler)
        
        job_message = JobMessage(
            job_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        response = await consumer._call_handler_safely(job_message)
        
        assert response.job_id == job_message.job_id
        assert response.status == "failed"
        assert "Processing failed" in response.error_message


class TestIntegration:
    """Test RabbitMQ integration functionality."""

    @pytest.mark.asyncio
    async def test_publish_job_for_processing(self):
        """Test publishing job for processing."""
        from vsr_shared.models import Job
        
        # Create mock publisher
        mock_publisher = AsyncMock(spec=JobPublisher)
        mock_publisher.publish_job.return_value = True
        
        # Create test job
        job = Job(
            id=uuid.uuid4(),
            video_key="uploads/20240101/test-uuid/video.mp4",
            api_key_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
        )
        
        # Test publishing
        result = await publish_job_for_processing(
            job=job,
            trace_id="test-trace-123",
            publisher=mock_publisher,
        )
        
        assert result is True
        mock_publisher.publish_job.assert_called_once()
        
        # Verify the job message was created correctly
        call_args = mock_publisher.publish_job.call_args
        job_message = call_args[1]["job_message"]
        assert job_message.job_id == job.id
        assert job_message.mode == job.mode
        assert job_message.video_key_in == job.video_key


# Integration test markers
pytestmark = pytest.mark.asyncio
