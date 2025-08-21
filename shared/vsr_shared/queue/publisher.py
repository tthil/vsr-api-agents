"""RabbitMQ job publisher for VSR API."""

import json
import uuid
from typing import Optional

from aio_pika import Message, DeliveryMode
from aio_pika.abc import AbstractChannel, AbstractExchange

from vsr_shared.logging import get_logger
from vsr_shared.models import JobMessage
from .client import EXCHANGE_NAME, ROUTING_KEY, RabbitMQClient
from .topology import setup_queue_topology

logger = get_logger(__name__)


class JobPublisher:
    """Publisher for job processing messages."""

    def __init__(self, client: RabbitMQClient):
        """
        Initialize job publisher.

        Args:
            client: RabbitMQ client instance
        """
        self.client = client
        self.exchange: Optional[AbstractExchange] = None

    async def setup(self) -> None:
        """Set up publisher (declare topology and get exchange)."""
        channel = await self.client.get_channel()
        
        # Set up topology if not already done
        topology = await setup_queue_topology(channel)
        self.exchange = topology["exchanges"]["main"]
        
        # Publisher confirms disabled for local development
        logger.info("Job publisher setup completed")

    async def publish_job(
        self,
        job_message: JobMessage,
        priority: int = 0,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish a job message to the processing queue.

        Args:
            job_message: Job message to publish
            priority: Message priority (0-255)
            correlation_id: Correlation ID for tracking

        Returns:
            bool: True if message was published successfully

        Raises:
            Exception: If publishing fails
        """
        if not self.exchange:
            await self.setup()

        try:
            # Serialize job message to JSON
            message_body = job_message.model_dump_json()
            
            # Create AMQP message
            message = Message(
                body=message_body.encode('utf-8'),
                content_type="application/json",
                delivery_mode=DeliveryMode.PERSISTENT,  # Make message persistent
                priority=priority,
                correlation_id=correlation_id or str(uuid.uuid4()),
                message_id=str(job_message.job_id),
                timestamp=job_message.requested_at,
                headers={
                    "job_id": str(job_message.job_id),
                    "mode": job_message.mode.value,
                    "trace_id": job_message.trace_id,
                },
            )

            # Publish with mandatory flag to ensure queue exists
            await self.exchange.publish(
                message,
                routing_key=ROUTING_KEY,
                mandatory=True,  # Ensure message is routed to a queue
            )

            logger.info(
                f"Published job message for job {job_message.job_id} "
                f"with mode {job_message.mode.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish job message for job {job_message.job_id}: {e}")
            raise

    async def publish_job_dict(
        self,
        job_data: dict,
        priority: int = 0,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish a job message from dictionary data.

        Args:
            job_data: Job data dictionary
            priority: Message priority (0-255)
            correlation_id: Correlation ID for tracking

        Returns:
            bool: True if message was published successfully
        """
        # Validate and create JobMessage from dict
        job_message = JobMessage(**job_data)
        return await self.publish_job(job_message, priority, correlation_id)

    async def close(self) -> None:
        """Close publisher resources."""
        # Publisher doesn't hold additional resources beyond the client
        logger.info("Job publisher closed")


async def create_job_publisher(client: Optional[RabbitMQClient] = None) -> JobPublisher:
    """
    Create and set up a job publisher.

    Args:
        client: RabbitMQ client (will create new one if None)

    Returns:
        Configured JobPublisher instance
    """
    if client is None:
        from .client import get_rabbitmq_client
        client = await get_rabbitmq_client()

    publisher = JobPublisher(client)
    await publisher.setup()
    return publisher
