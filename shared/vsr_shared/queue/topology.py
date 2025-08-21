"""RabbitMQ topology setup for VSR API."""

from typing import Dict, Any
from aio_pika import ExchangeType
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractQueue

from vsr_shared.logging import get_logger
from .client import (
    EXCHANGE_NAME,
    QUEUE_NAME,
    ROUTING_KEY,
    DLQ_EXCHANGE_NAME,
    DLQ_QUEUE_NAME,
    DLQ_ROUTING_KEY,
)

logger = get_logger(__name__)


async def setup_queue_topology(channel: AbstractChannel) -> Dict[str, Any]:
    """
    Set up RabbitMQ topology (exchanges, queues, bindings).
    
    This function is idempotent and can be called multiple times.
    
    Args:
        channel: RabbitMQ channel
        
    Returns:
        Dict containing created exchanges and queues
    """
    logger.info("Setting up RabbitMQ topology")
    
    # Declare dead letter exchange first
    dlq_exchange = await channel.declare_exchange(
        name=DLQ_EXCHANGE_NAME,
        type=ExchangeType.DIRECT,
        durable=True,
    )
    logger.info(f"Declared dead letter exchange: {DLQ_EXCHANGE_NAME}")
    
    # Declare dead letter queue
    dlq_queue = await channel.declare_queue(
        name=DLQ_QUEUE_NAME,
        durable=True,
        arguments={
            # Messages in DLQ will expire after 24 hours
            "x-message-ttl": 24 * 60 * 60 * 1000,  # 24 hours in milliseconds
        },
    )
    
    # Bind dead letter queue to dead letter exchange
    await dlq_queue.bind(dlq_exchange, routing_key=DLQ_ROUTING_KEY)
    logger.info(f"Declared and bound dead letter queue: {DLQ_QUEUE_NAME}")
    
    # Declare main exchange
    main_exchange = await channel.declare_exchange(
        name=EXCHANGE_NAME,
        type=ExchangeType.DIRECT,
        durable=True,
    )
    logger.info(f"Declared main exchange: {EXCHANGE_NAME}")
    
    # Declare main processing queue with dead letter configuration
    main_queue = await channel.declare_queue(
        name=QUEUE_NAME,
        durable=True,
        arguments={
            # Configure dead letter exchange
            "x-dead-letter-exchange": DLQ_EXCHANGE_NAME,
            "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
            # Optional: Set message TTL (uncomment if needed)
            # "x-message-ttl": 60 * 60 * 1000,  # 1 hour in milliseconds
        },
    )
    
    # Bind main queue to main exchange
    await main_queue.bind(main_exchange, routing_key=ROUTING_KEY)
    logger.info(f"Declared and bound main processing queue: {QUEUE_NAME}")
    
    logger.info("RabbitMQ topology setup completed successfully")
    
    return {
        "exchanges": {
            "main": main_exchange,
            "dlq": dlq_exchange,
        },
        "queues": {
            "main": main_queue,
            "dlq": dlq_queue,
        },
    }


async def setup_topology_with_client(client) -> Dict[str, Any]:
    """
    Set up topology using RabbitMQ client.
    
    Args:
        client: RabbitMQClient instance
        
    Returns:
        Dict containing created exchanges and queues
    """
    channel = await client.get_channel()
    return await setup_queue_topology(channel)
