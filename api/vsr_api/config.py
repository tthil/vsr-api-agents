"""Configuration settings for the VSR API."""

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API configuration settings."""
    
    # API settings
    api_title: str = "Video Subtitle Removal API"
    api_description: str = "API for removing hardcoded subtitles from videos using AI"
    api_version: str = "0.1.0"
    
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
    
    # Spaces (S3) settings
    spaces_endpoint: str = Field(
        default="https://nyc3.digitaloceanspaces.com",
        description="DigitalOcean Spaces endpoint"
    )
    spaces_region: str = Field(default="nyc3", description="DigitalOcean Spaces region")
    spaces_key: str = Field(default="", description="DigitalOcean Spaces access key")
    spaces_secret: str = Field(default="", description="DigitalOcean Spaces secret key")
    spaces_bucket: str = Field(default="vsr-videos", description="DigitalOcean Spaces bucket name")
    
    # API keys
    api_keys: Dict[str, str] = Field(
        default={},
        description="API keys for authentication (name: key)"
    )
    
    # Webhook settings
    webhook_secret: str = Field(default="", description="Secret for signing webhooks")
    
    # Video processing settings
    max_video_duration_seconds: int = Field(
        default=60,
        description="Maximum video duration in seconds"
    )
    max_video_resolution: str = Field(
        default="1080p",
        description="Maximum video resolution"
    )
    
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
