"""API key utilities for VSR API."""

import secrets
import uuid
from datetime import datetime
from typing import Optional

from vsr_shared.models import ApiKey, ApiKeyUsage


def generate_api_key_string(length: int = 48) -> str:
    """
    Generate a secure random API key string.
    
    Args:
        length: Length of the API key in bytes (will be encoded to hex, so actual length is 2x)
        
    Returns:
        str: API key string
    """
    # Generate a secure random token and encode as hex
    return secrets.token_hex(length)


def create_api_key(
    name: str,
    key: Optional[str] = None,
    daily_limit: Optional[int] = None,
    monthly_limit: Optional[int] = None,
    active: bool = True,
) -> ApiKey:
    """
    Create a new API key model with default values.
    
    Args:
        name: Name of the API key
        key: Optional API key string (generated if not provided)
        daily_limit: Optional daily limit for API usage
        monthly_limit: Optional monthly limit for API usage
        active: Whether the API key is active
        
    Returns:
        ApiKey: API key model
    """
    now = datetime.utcnow()
    
    # Generate API key if not provided
    if key is None:
        key = generate_api_key_string()
    
    # Create default usage stats
    usage = ApiKeyUsage(
        jobs_created=0,
        jobs_completed=0,
        upload_count=0,
        download_count=0,
        processing_seconds=0.0,
        total_video_size_mb=0.0,
    )
    
    # Create API key model
    api_key = ApiKey(
        id=uuid.uuid4(),
        key=key,
        name=name,
        active=active,
        created_at=now,
        last_used_at=None,
        usage=usage,
        daily_limit=daily_limit,
        monthly_limit=monthly_limit,
    )
    
    return api_key
