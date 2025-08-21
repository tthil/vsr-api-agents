"""Configuration settings for the VSR Worker."""

import os
from functools import lru_cache
from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Worker configuration settings."""
    
    # MongoDB settings
    mongodb_url: str = Field(
        default="mongodb://localhost:27017/vsr",
        description="MongoDB connection string"
    )
    mongodb_db_name: str = Field(default="vsr", description="MongoDB database name")
    
    # RabbitMQ settings
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="RabbitMQ connection string"
    )
    rabbitmq_queue: str = Field(default="vsr_jobs", description="RabbitMQ queue name")
    rabbitmq_exchange: str = Field(default="vsr", description="RabbitMQ exchange name")
    rabbitmq_routing_key: str = Field(default="job", description="RabbitMQ routing key")
    rabbitmq_dlx: str = Field(default="vsr.dlx", description="RabbitMQ dead letter exchange")
    rabbitmq_dlq: str = Field(default="vsr_jobs.dlq", description="RabbitMQ dead letter queue")
    
    # Spaces (S3) settings
    spaces_endpoint: str = Field(
        default="https://nyc3.digitaloceanspaces.com",
        description="DigitalOcean Spaces endpoint"
    )
    spaces_region: str = Field(default="nyc3", description="DigitalOcean Spaces region")
    spaces_key: str = Field(default="", description="DigitalOcean Spaces access key")
    spaces_secret: str = Field(default="", description="DigitalOcean Spaces secret key")
    spaces_bucket: str = Field(default="vsr-videos", description="DigitalOcean Spaces bucket name")
    
    # Worker settings
    worker_mode: str = Field(default="gpu", description="Worker mode: gpu or mock")
    worker_prefetch: int = Field(default=1, description="Number of jobs to prefetch")
    worker_polling_interval: int = Field(default=5, description="Polling interval in seconds")
    
    # GPU settings
    gpu_device: int = Field(default=0, description="GPU device ID")
    gpu_memory_limit: Optional[int] = Field(default=None, description="GPU memory limit in MB")
    
    # Mock worker settings
    mock_processing_time: int = Field(default=10, description="Mock processing time in seconds")
    mock_progress_interval: int = Field(default=1, description="Mock progress update interval in seconds")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()
