"""RabbitMQ job consumer for VSR API."""

import json
import asyncio
from typing import Callable, Optional, Any, Dict
from datetime import datetime

from aio_pika import IncomingMessage
from aio_pika.abc import AbstractQueue, AbstractIncomingMessage

from vsr_shared.logging import get_logger
from vsr_shared.models import JobMessage, JobMessageResponse
from .client import RabbitMQClient
from .topology import setup_queue_topology

logger = get_logger(__name__)


class JobConsumer:
    """Consumer for job processing messages."""

    def __init__(
        self,
        client: RabbitMQClient,
        message_handler: Callable[[JobMessage], JobMessageResponse],
    ):
        """
        Initialize job consumer.

        Args:
            client: RabbitMQ client instance
            message_handler: Function to handle job messages
        """
        self.client = client
        self.message_handler = message_handler
        self.queue: Optional[AbstractQueue] = None
        self.consumer_tag: Optional[str] = None
        self._consuming = False

    async def setup(self) -> None:
        """Set up consumer (declare topology and get queue)."""
        channel = await self.client.get_channel()
        
        # Set up topology if not already done
        topology = await setup_queue_topology(channel)
        self.queue = topology["queues"]["main"]
        
        logger.info("Job consumer setup completed")

    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        if not self.queue:
            await self.setup()

        if self._consuming:
            logger.warning("Consumer is already consuming messages")
            return

        try:
            # Start consuming with manual acknowledgment
            self.consumer_tag = await self.queue.consume(
                callback=self._handle_message,
                no_ack=False,  # Manual acknowledgment for reliability
            )
            
            self._consuming = True
            logger.info(f"Started consuming messages from queue {self.queue.name}")
            
        except Exception as e:
            logger.error(f"Failed to start consuming messages: {e}")
            raise

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        if not self._consuming:
            return

        try:
            if self.consumer_tag and self.queue:
                await self.queue.cancel(self.consumer_tag)
                self.consumer_tag = None
                
            self._consuming = False
            logger.info("Stopped consuming messages")
            
        except Exception as e:
            logger.error(f"Error stopping consumer: {e}")

    async def _handle_message(self, message: AbstractIncomingMessage) -> None:
        """
        Handle incoming job message.

        Args:
            message: Incoming RabbitMQ message
        """
        start_time = datetime.utcnow()
        job_id = None
        
        try:
            # Parse message body
            message_body = message.body.decode('utf-8')
            message_data = json.loads(message_body)
            
            # Create JobMessage from parsed data
            job_message = JobMessage(**message_data)
            job_id = job_message.job_id
            
            logger.info(f"Processing job message for job {job_id}")
            
            # Call the message handler
            response = await self._call_handler_safely(job_message)
            
            # Log processing result
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Completed processing job {job_id} in {processing_time:.2f}s "
                f"with status {response.status.value}"
            )
            
            # Acknowledge message on successful processing
            await message.ack()
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            # Reject message and send to DLQ
            await message.reject(requeue=False)
            
        except Exception as e:
            logger.error(f"Error processing job message {job_id}: {e}")
            # Reject message and send to DLQ
            await message.reject(requeue=False)

    async def _call_handler_safely(self, job_message: JobMessage) -> JobMessageResponse:
        """
        Call message handler with error handling.

        Args:
            job_message: Job message to process

        Returns:
            JobMessageResponse: Response from handler
        """
        try:
            # Check if handler is async or sync
            if asyncio.iscoroutinefunction(self.message_handler):
                response = await self.message_handler(job_message)
            else:
                response = self.message_handler(job_message)
                
            return response
            
        except Exception as e:
            logger.error(f"Message handler failed for job {job_message.job_id}: {e}")
            
            # Return error response
            return JobMessageResponse(
                job_id=job_message.job_id,
                status="failed",
                error_message=str(e),
                trace_id=job_message.trace_id,
            )

    async def consume_single_message(self, timeout: float = 30.0) -> Optional[JobMessage]:
        """
        Consume a single message (useful for testing).

        Args:
            timeout: Timeout in seconds

        Returns:
            JobMessage if received, None if timeout
        """
        if not self.queue:
            await self.setup()

        try:
            message = await asyncio.wait_for(
                self.queue.get(no_ack=False),
                timeout=timeout
            )
            
            if message:
                try:
                    message_body = message.body.decode('utf-8')
                    message_data = json.loads(message_body)
                    job_message = JobMessage(**message_data)
                    
                    # Acknowledge message
                    await message.ack()
                    
                    return job_message
                    
                except Exception as e:
                    logger.error(f"Error processing single message: {e}")
                    await message.reject(requeue=False)
                    return None
                    
        except asyncio.TimeoutError:
            logger.debug(f"No message received within {timeout}s timeout")
            return None

    async def close(self) -> None:
        """Close consumer resources."""
        await self.stop_consuming()
        logger.info("Job consumer closed")

    @property
    def is_consuming(self) -> bool:
        """Check if consumer is currently consuming messages."""
        return self._consuming


async def create_job_consumer(
    message_handler: Callable[[JobMessage], JobMessageResponse],
    client: Optional[RabbitMQClient] = None,
) -> JobConsumer:
    """
    Create and set up a job consumer.

    Args:
        message_handler: Function to handle job messages
        client: RabbitMQ client (will create new one if None)

    Returns:
        Configured JobConsumer instance
    """
    if client is None:
        from .client import get_rabbitmq_client
        client = await get_rabbitmq_client()

    consumer = JobConsumer(client, message_handler)
    await consumer.setup()
    return consumer
