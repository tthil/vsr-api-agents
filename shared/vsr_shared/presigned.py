"""Presigned URL helpers for S3/Spaces."""

import uuid
from datetime import datetime
from typing import Dict, Optional, Union

from vsr_shared.logging import get_logger
from vsr_shared.spaces import DEFAULT_EXPIRES, MAX_EXPIRES, get_spaces_client
from vsr_shared.spaces_keys import (
    KeyPrefix,
    generate_key,
    validate_key,
    get_upload_key,
    get_processed_key,
)

logger = get_logger(__name__)


def generate_presigned_put_url(
    key: Optional[str] = None,
    content_type: str = "video/mp4",
    expires_in: int = DEFAULT_EXPIRES,
    job_id: Optional[uuid.UUID] = None,
) -> Dict[str, Union[str, int]]:
    """
    Generate a presigned PUT URL for uploading a file.

    Args:
        key: Object key (if None, a key will be generated using job_id)
        content_type: Content type of the file
        expires_in: URL expiration time in seconds
        job_id: Job ID for key generation (if key is None)

    Returns:
        Dict with upload_url, key, and expires_in
    """
    # Validate content type
    if not content_type.startswith("video/"):
        raise ValueError("Content type must be a video format")

    # Generate key if not provided
    if key is None:
        if job_id is None:
            job_id = uuid.uuid4()
        key = get_upload_key(job_id)
    else:
        # Validate key format
        if not validate_key(key):
            raise ValueError(f"Invalid key format: {key}")

    # Ensure expires_in is within limits
    expires_in = min(expires_in, MAX_EXPIRES)

    # Get Spaces client and generate presigned URL
    client = get_spaces_client()
    result = client.generate_presigned_put_url(
        key=key,
        content_type=content_type,
        expires_in=expires_in,
    )

    logger.info(f"Generated presigned PUT URL for {key}")
    return result


def generate_presigned_get_url(
    key: str,
    expires_in: int = DEFAULT_EXPIRES,
) -> str:
    """
    Generate a presigned GET URL for downloading a file.

    Args:
        key: Object key
        expires_in: URL expiration time in seconds

    Returns:
        Presigned URL string
    """
    # Validate key format
    if not validate_key(key):
        raise ValueError(f"Invalid key format: {key}")

    # Ensure expires_in is within limits
    expires_in = min(expires_in, MAX_EXPIRES)

    # Get Spaces client and generate presigned URL
    client = get_spaces_client()
    url = client.generate_presigned_get_url(key=key, expires_in=expires_in)

    logger.info(f"Generated presigned GET URL for {key}")
    return url


def generate_upload_url(
    content_type: str = "video/mp4",
    expires_in: int = DEFAULT_EXPIRES,
    job_id: Optional[uuid.UUID] = None,
) -> Dict[str, Union[str, int]]:
    """
    Generate an upload URL for the API endpoint.

    Args:
        content_type: Content type of the file
        expires_in: URL expiration time in seconds
        job_id: Job ID for key generation

    Returns:
        Dict with upload_url, key, and expires_in
    """
    # Generate a job ID if not provided
    if job_id is None:
        job_id = uuid.uuid4()

    # Generate key using the job ID
    key = get_upload_key(job_id)

    # Generate presigned PUT URL
    result = generate_presigned_put_url(
        key=key,
        content_type=content_type,
        expires_in=expires_in,
    )

    logger.info(f"Generated upload URL for job {job_id}")
    return result
