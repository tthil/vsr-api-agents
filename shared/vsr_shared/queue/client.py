"""RabbitMQ client for VSR API."""

import os
import asyncio
from typing import Optional
from urllib.parse import urlparse

import aio_pika
from aio_pika import Connection, Channel, ExchangeType
from aio_pika.abc import AbstractConnection, AbstractChannel

from vsr_shared.logging import get_logger

logger = get_logger(__name__)

# Queue and exchange configuration
EXCHANGE_NAME = "vsr.jobs"
QUEUE_NAME = "vsr.process.q"
ROUTING_KEY = "vsr.process"
DLQ_EXCHANGE_NAME = "vsr.dead.ex"
DLQ_QUEUE_NAME = "vsr.dead.q"
DLQ_ROUTING_KEY = "vsr.dead"


class RabbitMQClient:
    """RabbitMQ client for VSR API."""

    def __init__(
        self,
        url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize RabbitMQ client.

        Args:
            url: RabbitMQ connection URL
            max_retries: Maximum connection retries
            retry_delay: Delay between retries in seconds
        """
        self.url = url or os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self._closed = False

    async def connect(self) -> None:
        """Connect to RabbitMQ with retries."""
        if self.connection and not self.connection.is_closed:
            return

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to RabbitMQ at {self._mask_url(self.url)} (attempt {attempt + 1})")
                self.connection = await aio_pika.connect_robust(
                    self.url,
                    client_properties={"connection_name": "vsr-api"},
                )
                self.channel = await self.connection.channel()
                
                # Set QoS to prefetch=1 to avoid concurrency issues
                await self.channel.set_qos(prefetch_count=1)
                
                logger.info("Successfully connected to RabbitMQ")
                return
                
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to RabbitMQ after {self.max_retries} attempts")

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self._closed:
            return
            
        self._closed = True
        
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
                logger.info("Closed RabbitMQ channel")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ channel: {e}")
            
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logger.info("Closed RabbitMQ connection")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")

    async def get_channel(self) -> AbstractChannel:
        """Get RabbitMQ channel, connecting if necessary."""
        if not self.connection or self.connection.is_closed:
            await self.connect()
        
        if not self.channel or self.channel.is_closed:
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
            
        return self.channel

    async def health_check(self) -> bool:
        """Check RabbitMQ connection health."""
        try:
            channel = await self.get_channel()
            # Try to declare a temporary queue to test connection
            temp_queue = await channel.declare_queue(
                name="",  # Auto-generated name
                exclusive=True,
                auto_delete=True,
            )
            await temp_queue.delete()
            return True
        except Exception as e:
            logger.error(f"RabbitMQ health check failed: {e}")
            return False

    def _mask_url(self, url: str) -> str:
        """Mask credentials in URL for logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                masked = url.replace(parsed.password, "****")
                return masked
            return url
        except Exception:
            return "****"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Global client instance
_rabbitmq_client: Optional[RabbitMQClient] = None


async def get_rabbitmq_client(
    url: Optional[str] = None,
    force_new: bool = False,
) -> RabbitMQClient:
    """
    Get RabbitMQ client instance.

    Args:
        url: RabbitMQ connection URL
        force_new: Force creation of new client

    Returns:
        RabbitMQClient instance
    """
    global _rabbitmq_client

    if _rabbitmq_client is None or force_new:
        _rabbitmq_client = RabbitMQClient(url=url)

    return _rabbitmq_client


async def close_rabbitmq_client() -> None:
    """Close global RabbitMQ client."""
    global _rabbitmq_client
    
    if _rabbitmq_client:
        await _rabbitmq_client.disconnect()
        _rabbitmq_client = None
