"""Test script for Spaces client and presigned URLs with MinIO."""

import argparse
import asyncio
import os
import sys
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

import requests

from vsr_shared.logging import get_logger, setup_logging
from vsr_shared.spaces import SpacesClient, get_spaces_client
from vsr_shared.spaces_keys import (
    KeyPrefix,
    generate_key,
    validate_key,
    get_upload_key,
    get_processed_key,
)
from vsr_shared.presigned import (
    generate_presigned_put_url,
    generate_presigned_get_url,
    generate_upload_url,
)

logger = get_logger(__name__)

# Default MinIO settings for local testing
DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"
DEFAULT_BUCKET = "vsr-videos"


def create_test_video(size_kb: int = 10) -> BytesIO:
    """
    Create a test video file with random data.

    Args:
        size_kb: Size of the test file in KB

    Returns:
        BytesIO: File-like object with test data
    """
    # Create random data
    data = os.urandom(size_kb * 1024)
    return BytesIO(data)


def test_client_connection(client: SpacesClient) -> bool:
    """
    Test connection to Spaces/MinIO.

    Args:
        client: SpacesClient instance

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Check if bucket exists
        result = client._check_bucket()
        if not result:
            logger.error(f"Bucket {client.bucket_name} does not exist")
            return False
        
        logger.info(f"Successfully connected to {client.endpoint_url}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Spaces/MinIO: {e}")
        return False


def test_key_generation() -> None:
    """Test key generation and validation."""
    # Generate test job ID
    job_id = uuid.uuid4()
    
    # Test upload key
    upload_key = get_upload_key(job_id)
    logger.info(f"Generated upload key: {upload_key}")
    assert validate_key(upload_key), "Upload key validation failed"
    
    # Test processed key
    processed_key = get_processed_key(job_id)
    logger.info(f"Generated processed key: {processed_key}")
    assert validate_key(processed_key), "Processed key validation failed"
    
    # Test log key
    log_key = f"{KeyPrefix.LOGS}/{job_id}.log"
    logger.info(f"Generated log key: {log_key}")
    assert validate_key(log_key), "Log key validation failed"
    
    # Test invalid key
    invalid_key = f"invalid/{job_id}/video.mp4"
    logger.info(f"Testing invalid key: {invalid_key}")
    assert not validate_key(invalid_key), "Invalid key validation failed"
    
    logger.info("Key generation and validation tests passed")


def test_presigned_put_url(client: SpacesClient) -> str:
    """
    Test generating and using a presigned PUT URL.

    Args:
        client: SpacesClient instance

    Returns:
        str: Object key if successful
    """
    # Generate test job ID and key
    job_id = uuid.uuid4()
    key = get_upload_key(job_id)
    
    # Generate presigned URL
    result = generate_presigned_put_url(key=key, content_type="video/mp4")
    logger.info(f"Generated presigned PUT URL: {result['upload_url'][:100]}...")
    
    # Create test file
    test_file = create_test_video(size_kb=10)
    
    # Upload file using presigned URL
    headers = {"Content-Type": "video/mp4"}
    response = requests.put(
        result["upload_url"],
        data=test_file.getvalue(),
        headers=headers,
    )
    
    if response.status_code == 200:
        logger.info(f"Successfully uploaded test file to {key}")
        return key
    else:
        logger.error(f"Failed to upload test file: {response.status_code} {response.text}")
        raise RuntimeError(f"Failed to upload test file: {response.status_code}")


def test_presigned_get_url(client: SpacesClient, key: str) -> None:
    """
    Test generating and using a presigned GET URL.

    Args:
        client: SpacesClient instance
        key: Object key to get
    """
    # Generate presigned URL
    url = generate_presigned_get_url(key=key)
    logger.info(f"Generated presigned GET URL: {url[:100]}...")
    
    # Download file using presigned URL
    response = requests.get(url)
    
    if response.status_code == 200:
        logger.info(f"Successfully downloaded test file from {key} ({len(response.content)} bytes)")
    else:
        logger.error(f"Failed to download test file: {response.status_code} {response.text}")
        raise RuntimeError(f"Failed to download test file: {response.status_code}")


def test_upload_url_api() -> str:
    """
    Test the generate_upload_url API helper.

    Returns:
        str: Object key if successful
    """
    # Generate upload URL
    result = generate_upload_url(content_type="video/mp4")
    logger.info(f"Generated upload URL: {result['upload_url'][:100]}...")
    
    # Create test file
    test_file = create_test_video(size_kb=10)
    
    # Upload file using presigned URL
    headers = {"Content-Type": "video/mp4"}
    response = requests.put(
        result["upload_url"],
        data=test_file.getvalue(),
        headers=headers,
    )
    
    if response.status_code == 200:
        logger.info(f"Successfully uploaded test file to {result['key']}")
        return result["key"]
    else:
        logger.error(f"Failed to upload test file: {response.status_code} {response.text}")
        raise RuntimeError(f"Failed to upload test file: {response.status_code}")


def test_negative_cases(client: SpacesClient) -> None:
    """
    Test negative cases.

    Args:
        client: SpacesClient instance
    """
    # Test invalid content type
    try:
        generate_presigned_put_url(key="uploads/test/video.mp4", content_type="image/jpeg")
        logger.error("Failed: Invalid content type test passed")
    except ValueError as e:
        logger.info(f"Successfully caught invalid content type: {e}")
    
    # Test expired URL
    key = get_upload_key(uuid.uuid4())
    result = generate_presigned_put_url(key=key, content_type="video/mp4", expires_in=1)
    logger.info("Waiting for URL to expire...")
    time.sleep(2)  # Wait for URL to expire
    
    # Try to use expired URL
    test_file = create_test_video(size_kb=10)
    headers = {"Content-Type": "video/mp4"}
    response = requests.put(
        result["upload_url"],
        data=test_file.getvalue(),
        headers=headers,
    )
    
    if response.status_code >= 400:
        logger.info(f"Successfully tested expired URL: {response.status_code}")
    else:
        logger.error(f"Failed: Expired URL test failed: {response.status_code}")


def main() -> None:
    """Run the Spaces/MinIO test script."""
    # Set up logging
    setup_logging(level="INFO", json_format=False)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Spaces client with MinIO")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("SPACES_ENDPOINT", DEFAULT_ENDPOINT),
        help="Spaces/MinIO endpoint URL",
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("SPACES_KEY", DEFAULT_ACCESS_KEY),
        help="Spaces/MinIO access key",
    )
    parser.add_argument(
        "--secret-key",
        default=os.environ.get("SPACES_SECRET", DEFAULT_SECRET_KEY),
        help="Spaces/MinIO secret key",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("SPACES_BUCKET", DEFAULT_BUCKET),
        help="Spaces/MinIO bucket name",
    )
    parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="Create bucket if it doesn't exist",
    )
    
    args = parser.parse_args()
    
    # Set environment variables for client factory
    os.environ["SPACES_ENDPOINT"] = args.endpoint
    os.environ["SPACES_KEY"] = args.access_key
    os.environ["SPACES_SECRET"] = args.secret_key
    os.environ["SPACES_BUCKET"] = args.bucket
    
    # Create client
    client = get_spaces_client(force_new=True)
    
    # Test connection
    if not test_client_connection(client):
        if args.create_bucket:
            try:
                logger.info(f"Creating bucket {args.bucket}")
                client.client.create_bucket(Bucket=args.bucket)
                logger.info(f"Successfully created bucket {args.bucket}")
            except Exception as e:
                logger.error(f"Error creating bucket: {e}")
                sys.exit(1)
        else:
            logger.error(f"Bucket {args.bucket} does not exist. Use --create-bucket to create it.")
            sys.exit(1)
    
    # Run tests
    try:
        logger.info("Testing key generation and validation")
        test_key_generation()
        
        logger.info("Testing presigned PUT URL")
        key = test_presigned_put_url(client)
        
        logger.info("Testing presigned GET URL")
        test_presigned_get_url(client, key)
        
        logger.info("Testing upload URL API")
        key = test_upload_url_api()
        
        logger.info("Testing negative cases")
        test_negative_cases(client)
        
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
