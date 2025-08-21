"""RabbitMQ queue management for VSR API."""

from .client import RabbitMQClient, get_rabbitmq_client
from .publisher import JobPublisher
from .consumer import JobConsumer
from .topology import setup_queue_topology

__all__ = [
    "RabbitMQClient",
    "get_rabbitmq_client",
    "JobPublisher", 
    "JobConsumer",
    "setup_queue_topology",
]
