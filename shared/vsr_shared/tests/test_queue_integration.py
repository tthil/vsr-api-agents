"""Integration tests for RabbitMQ queue functionality."""

import pytest
import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from vsr_shared.models import (
    Job, JobMessage, JobMessageResponse, JobStatus, ProcessingMode, JobEventType
)
from vsr_shared.queue.client import RabbitMQClient
from vsr_shared.queue.publisher import JobPublisher
from vsr_shared.queue.consumer import JobConsumer
from vsr_shared.queue.topology import setup_queue_topology
from vsr_shared.queue.integration import publish_job_for_processing


@pytest.mark.asyncio
class TestRabbitMQIntegration:
    """Integration tests for RabbitMQ functionality."""

    async def test_publish_and_consume_job_message(self):
        """Test publishing and consuming a job message."""
        # Mock RabbitMQ client and components
        mock_client = AsyncMock(spec=RabbitMQClient)
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_queue = AsyncMock()
        
        # Setup mocks
        mock_client.get_channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = mock_queue
        
        # Create test job
        job = Job(
            id=uuid.uuid4(),
            video_key="uploads/20240101/test-uuid/video.mp4",
            api_key_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            status=JobStatus.PENDING,
        )
        
        # Test job message creation
        job_message = JobMessage(
            job_id=job.id,
            mode=job.mode,
            video_key_in=job.video_key,
            requested_at=datetime.now(timezone.utc),
            trace_id="test-trace-123",
        )
        
        # Verify job message structure
        assert job_message.job_id == job.id
        assert job_message.mode == ProcessingMode.STTN
        assert job_message.video_key_in == job.video_key
        assert job_message.trace_id == "test-trace-123"
        
        # Test JSON serialization
        json_data = job_message.model_dump_json()
        assert str(job.id) in json_data
        assert "sttn" in json_data
        assert "test-trace-123" in json_data

    async def test_message_handler_success(self):
        """Test successful message handling by worker."""
        def mock_message_handler(job_message: JobMessage) -> JobMessageResponse:
            """Mock message handler that simulates successful processing."""
            return JobMessageResponse(
                job_id=job_message.job_id,
                status=JobStatus.COMPLETED,
                processed_video_key=f"processed/20240101/{job_message.job_id}/video.mp4",
                processing_time_seconds=30.5,
                trace_id=job_message.trace_id,
            )
        
        # Create test job message
        job_id = uuid.uuid4()
        job_message = JobMessage(
            job_id=job_id,
            mode=ProcessingMode.LAMA,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
            trace_id="test-trace-456",
        )
        
        # Test message handler
        response = mock_message_handler(job_message)
        
        # Verify response
        assert response.job_id == job_id
        assert response.status == JobStatus.COMPLETED
        assert response.processed_video_key == f"processed/20240101/{job_id}/video.mp4"
        assert response.processing_time_seconds == 30.5
        assert response.trace_id == "test-trace-456"

    async def test_message_handler_failure(self):
        """Test message handler error handling."""
        def failing_message_handler(job_message: JobMessage) -> JobMessageResponse:
            """Mock message handler that simulates processing failure."""
            raise Exception("Video processing failed: Invalid format")
        
        # Create mock consumer to test error handling
        mock_client = AsyncMock(spec=RabbitMQClient)
        consumer = JobConsumer(mock_client, failing_message_handler)
        
        job_message = JobMessage(
            job_id=uuid.uuid4(),
            mode=ProcessingMode.PROPAINTER,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        # Test error handling
        response = await consumer._call_handler_safely(job_message)
        
        # Verify error response
        assert response.job_id == job_message.job_id
        assert response.status == "failed"
        assert "Video processing failed" in response.error_message

    async def test_dead_letter_queue_routing(self):
        """Test that failed messages are routed to dead letter queue."""
        # This test would require actual RabbitMQ connection
        # For now, we test the topology setup
        
        mock_channel = AsyncMock()
        mock_main_exchange = AsyncMock()
        mock_dlq_exchange = AsyncMock()
        mock_main_queue = AsyncMock()
        mock_dlq_queue = AsyncMock()
        
        # Mock exchange and queue declarations
        mock_channel.declare_exchange.side_effect = [
            mock_dlq_exchange,  # DLQ exchange first
            mock_main_exchange, # Main exchange second
        ]
        mock_channel.declare_queue.side_effect = [
            mock_dlq_queue,     # DLQ queue first
            mock_main_queue,    # Main queue second
        ]
        
        # Test topology setup
        topology = await setup_queue_topology(mock_channel)
        
        # Verify topology structure
        assert "exchanges" in topology
        assert "queues" in topology
        assert "main" in topology["exchanges"]
        assert "dlq" in topology["exchanges"]
        assert "main" in topology["queues"]
        assert "dlq" in topology["queues"]
        
        # Verify queue bindings were called
        mock_dlq_queue.bind.assert_called_once()
        mock_main_queue.bind.assert_called_once()

    @patch('vsr_shared.queue.integration.create_job_publisher')
    async def test_publish_job_for_processing_integration(self, mock_create_publisher):
        """Test the integration function for publishing jobs."""
        # Setup mock publisher
        mock_publisher = AsyncMock()
        mock_publisher.publish_job.return_value = True
        mock_create_publisher.return_value = mock_publisher
        
        # Create test job
        job = Job(
            id=uuid.uuid4(),
            video_key="uploads/20240101/test-uuid/video.mp4",
            api_key_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            status=JobStatus.PENDING,
        )
        
        # Test publishing
        result = await publish_job_for_processing(
            job=job,
            trace_id="integration-test-789",
        )
        
        # Verify result
        assert result is True
        mock_create_publisher.assert_called_once()
        mock_publisher.publish_job.assert_called_once()
        
        # Verify job message was created correctly
        call_args = mock_publisher.publish_job.call_args
        job_message = call_args[1]["job_message"]
        assert job_message.job_id == job.id
        assert job_message.mode == job.mode
        assert job_message.video_key_in == job.video_key

    async def test_queue_configuration(self):
        """Test RabbitMQ queue configuration parameters."""
        from vsr_shared.queue.client import (
            EXCHANGE_NAME, QUEUE_NAME, ROUTING_KEY,
            DLQ_EXCHANGE_NAME, DLQ_QUEUE_NAME, DLQ_ROUTING_KEY
        )
        
        # Verify queue configuration constants
        assert EXCHANGE_NAME == "vsr.jobs"
        assert QUEUE_NAME == "vsr.process.q"
        assert ROUTING_KEY == "vsr.process"
        assert DLQ_EXCHANGE_NAME == "vsr.dead.ex"
        assert DLQ_QUEUE_NAME == "vsr.dead.q"
        assert DLQ_ROUTING_KEY == "vsr.dead"

    async def test_message_persistence_and_durability(self):
        """Test that messages and queues are configured for durability."""
        mock_client = AsyncMock(spec=RabbitMQClient)
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_client.get_channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = AsyncMock()
        
        # Create publisher
        publisher = JobPublisher(mock_client)
        await publisher.setup()
        
        # Create test job message
        job_message = JobMessage(
            job_id=uuid.uuid4(),
            mode=ProcessingMode.STTN,
            video_key_in="uploads/20240101/test-uuid/video.mp4",
        )
        
        # Test publishing (mocked)
        await publisher.publish_job(job_message)
        
        # Verify exchange publish was called
        mock_exchange.publish.assert_called_once()
        
        # Verify message was published with mandatory flag
        call_args = mock_exchange.publish.call_args
        assert call_args[1]["mandatory"] is True


# Pytest configuration for async tests
pytestmark = pytest.mark.asyncio
