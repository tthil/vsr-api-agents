"""Key scheme and validators for S3/Spaces object paths."""

import os
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple, Union

from vsr_shared.logging import get_logger

logger = get_logger(__name__)


class KeyPrefix(str, Enum):
    """S3/Spaces key prefixes."""

    UPLOADS = "uploads"
    PROCESSED = "processed"
    LOGS = "logs"


# Regex patterns for key validation
UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
DATE_PATTERN = r"\d{8}"  # YYYYMMDD
VIDEO_FILENAME_PATTERN = r"video\.mp4"

# Compiled regex patterns for key parsing
UPLOAD_KEY_REGEX = re.compile(
    f"^{KeyPrefix.UPLOADS}/({DATE_PATTERN})/({UUID_PATTERN})/({VIDEO_FILENAME_PATTERN})$"
)
PROCESSED_KEY_REGEX = re.compile(
    f"^{KeyPrefix.PROCESSED}/({DATE_PATTERN})/({UUID_PATTERN})/({VIDEO_FILENAME_PATTERN})$"
)
LOG_KEY_REGEX = re.compile(f"^{KeyPrefix.LOGS}/.*$")


def generate_key(
    prefix: Union[KeyPrefix, str],
    job_id: Optional[uuid.UUID] = None,
    date: Optional[datetime] = None,
    filename: str = "video.mp4",
) -> str:
    """
    Generate a key for S3/Spaces object.

    Args:
        prefix: Key prefix (uploads, processed, logs)
        job_id: Job ID (UUID)
        date: Date for the key (defaults to current UTC date)
        filename: Filename (defaults to video.mp4)

    Returns:
        str: Generated key
    """
    if prefix not in [KeyPrefix.UPLOADS, KeyPrefix.PROCESSED, KeyPrefix.LOGS]:
        raise ValueError(f"Invalid key prefix: {prefix}")

    # For logs, we don't need job_id and date
    if prefix == KeyPrefix.LOGS:
        if not filename:
            raise ValueError("Filename is required for logs")
        return f"{prefix}/{filename}"

    # For uploads and processed, we need job_id
    if not job_id:
        job_id = uuid.uuid4()

    # Use current UTC date if not provided
    if not date:
        date = datetime.utcnow()

    # Format date as YYYYMMDD
    date_str = date.strftime("%Y%m%d")

    return f"{prefix}/{date_str}/{job_id}/{filename}"


def validate_key(key: str) -> bool:
    """
    Validate if a key conforms to the expected patterns.

    Args:
        key: S3/Spaces object key

    Returns:
        bool: True if the key is valid, False otherwise
    """
    # Check for path traversal attempts
    if ".." in key or "//" in key:
        logger.warning(f"Path traversal attempt detected in key: {key}")
        return False

    # Check if the key matches any of the expected patterns
    if key.startswith(f"{KeyPrefix.UPLOADS}/"):
        return bool(UPLOAD_KEY_REGEX.match(key))
    elif key.startswith(f"{KeyPrefix.PROCESSED}/"):
        return bool(PROCESSED_KEY_REGEX.match(key))
    elif key.startswith(f"{KeyPrefix.LOGS}/"):
        return bool(LOG_KEY_REGEX.match(key))

    logger.warning(f"Key does not match any expected pattern: {key}")
    return False


def parse_key(key: str) -> Dict[str, str]:
    """
    Parse a key to extract its components.

    Args:
        key: S3/Spaces object key

    Returns:
        Dict with prefix, date, job_id, and filename
    """
    result = {"prefix": None, "date": None, "job_id": None, "filename": None}

    # Try to match upload key pattern
    match = UPLOAD_KEY_REGEX.match(key)
    if match:
        result["prefix"] = KeyPrefix.UPLOADS
        result["date"] = match.group(1)
        result["job_id"] = match.group(2)
        result["filename"] = match.group(3)
        return result

    # Try to match processed key pattern
    match = PROCESSED_KEY_REGEX.match(key)
    if match:
        result["prefix"] = KeyPrefix.PROCESSED
        result["date"] = match.group(1)
        result["job_id"] = match.group(2)
        result["filename"] = match.group(3)
        return result

    # Try to match log key pattern
    if key.startswith(f"{KeyPrefix.LOGS}/"):
        result["prefix"] = KeyPrefix.LOGS
        result["filename"] = key[len(f"{KeyPrefix.LOGS}/"):]
        return result

    raise ValueError(f"Invalid key format: {key}")


def get_upload_key(job_id: uuid.UUID) -> str:
    """
    Generate an upload key for a job.

    Args:
        job_id: Job ID

    Returns:
        str: Upload key
    """
    return generate_key(KeyPrefix.UPLOADS, job_id)


def get_processed_key(job_id: uuid.UUID) -> str:
    """
    Generate a processed key for a job.

    Args:
        job_id: Job ID

    Returns:
        str: Processed key
    """
    return generate_key(KeyPrefix.PROCESSED, job_id)


def get_log_key(job_id: uuid.UUID) -> str:
    """
    Generate a log key for a job.

    Args:
        job_id: Job ID

    Returns:
        str: Log key
    """
    return f"{KeyPrefix.LOGS}/{job_id}.log"
