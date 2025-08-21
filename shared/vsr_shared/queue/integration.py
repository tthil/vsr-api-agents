"""RabbitMQ integration helpers for VSR API."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from vsr_shared.logging import get_logger
from vsr_shared.models import Job, JobMessage, ProcessingMode, SubtitleArea
from .publisher import JobPublisher, create_job_publisher
from .client import get_rabbitmq_client

logger = get_logger(__name__)


async def publish_job_for_processing(
    job: Job,
    trace_id: Optional[str] = None,
    publisher: Optional[JobPublisher] = None,
) -> bool:
    """
    Publish a job for processing via RabbitMQ.

    Args:
        job: Job instance to process
        trace_id: Optional trace ID for request tracking
        publisher: Optional publisher instance (will create if None)

    Returns:
        bool: True if job was published successfully

    Raises:
        Exception: If publishing fails
    """
    try:
        # Create publisher if not provided
        if publisher is None:
            publisher = await create_job_publisher()

        # Create job message from job instance
        job_message = JobMessage(
            job_id=job.id,
            mode=job.mode,
            video_key_in=job.video_key,
            subtitle_area=job.subtitle_area,
            callback_url=job.callback_url,
            requested_at=job.created_at,
            trace_id=trace_id,
        )

        # Publish the message
        success = await publisher.publish_job(
            job_message=job_message,
            correlation_id=trace_id,
        )

        if success:
            logger.info(f"Successfully published job {job.id} for processing")
        else:
            logger.error(f"Failed to publish job {job.id} for processing")

        return success

    except Exception as e:
        logger.error(f"Error publishing job {job.id} for processing: {e}")
        raise


async def create_and_publish_job(
    video_key: str,
    api_key_id: uuid.UUID,
    mode: ProcessingMode = ProcessingMode.STTN,
    subtitle_area: Optional[SubtitleArea] = None,
    callback_url: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Job:
    """
    Create a new job and publish it for processing.

    Args:
        video_key: S3 key for input video
        api_key_id: API key ID for the job
        mode: Processing mode
        subtitle_area: Optional subtitle area coordinates
        callback_url: Optional webhook callback URL
        trace_id: Optional trace ID for request tracking

    Returns:
        Job: Created job instance

    Raises:
        Exception: If job creation or publishing fails
    """
    # Create job instance
    job = Job(
        video_key=video_key,
        api_key_id=api_key_id,
        mode=mode,
        subtitle_area=subtitle_area,
        callback_url=callback_url,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Publish job for processing
    await publish_job_for_processing(job, trace_id=trace_id)

    return job


class RabbitMQLifespan:
    """RabbitMQ lifespan manager for FastAPI applications."""

    def __init__(self, rabbitmq_url: Optional[str] = None):
        """
        Initialize RabbitMQ lifespan manager.

        Args:
            rabbitmq_url: RabbitMQ connection URL
        """
        self.rabbitmq_url = rabbitmq_url
        self.client = None

    async def startup(self) -> None:
        """Initialize RabbitMQ connection and topology on startup."""
        try:
            logger.info("Initializing RabbitMQ connection")
            
            # Get RabbitMQ client
            self.client = await get_rabbitmq_client(url=self.rabbitmq_url)
            
            # Set up topology
            from .topology import setup_topology_with_client
            await setup_topology_with_client(self.client)
            
            # Test connection
            health_ok = await self.client.health_check()
            if not health_ok:
                raise Exception("RabbitMQ health check failed")
                
            logger.info("RabbitMQ initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up RabbitMQ resources on shutdown."""
        try:
            logger.info("Shutting down RabbitMQ connection")
            
            if self.client:
                await self.client.disconnect()
                
            # Close global client
            from .client import close_rabbitmq_client
            await close_rabbitmq_client()
            
            logger.info("RabbitMQ shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during RabbitMQ shutdown: {e}")


def create_rabbitmq_lifespan(rabbitmq_url: Optional[str] = None) -> RabbitMQLifespan:
    """
    Create RabbitMQ lifespan manager.

    Args:
        rabbitmq_url: RabbitMQ connection URL

    Returns:
        RabbitMQLifespan instance
    """
    return RabbitMQLifespan(rabbitmq_url=rabbitmq_url)
