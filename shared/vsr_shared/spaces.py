"""DigitalOcean Spaces (S3) client for VSR API."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vsr_shared.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BUCKET = "vsr-videos"
DEFAULT_REGION = "nyc3"
DEFAULT_EXPIRES = 3600  # 1 hour
MAX_EXPIRES = 7 * 24 * 3600  # 7 days


class SpacesClient:
    """DigitalOcean Spaces (S3) client."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: str = DEFAULT_BUCKET,
        check_bucket: bool = False,
    ):
        """
        Initialize the Spaces client.

        Args:
            endpoint_url: Spaces endpoint URL (e.g., https://nyc3.digitaloceanspaces.com)
            region_name: Region name (e.g., nyc3)
            access_key: Spaces access key
            secret_key: Spaces secret key
            bucket_name: Bucket name
            check_bucket: Whether to check if the bucket exists on initialization
        """
        self.endpoint_url = endpoint_url or os.environ.get("SPACES_ENDPOINT")
        if not self.endpoint_url:
            raise ValueError("Spaces endpoint URL is required")

        self.region_name = region_name or os.environ.get("SPACES_REGION", DEFAULT_REGION)
        self.access_key = access_key or os.environ.get("SPACES_KEY")
        self.secret_key = secret_key or os.environ.get("SPACES_SECRET")
        self.bucket_name = bucket_name or os.environ.get("SPACES_BUCKET", DEFAULT_BUCKET)

        if not self.access_key or not self.secret_key:
            raise ValueError("Spaces access key and secret key are required")

        # Configure boto3 client with retry settings
        config = Config(
            region_name=self.region_name,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=10,
        )

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=config,
        )

        if check_bucket:
            self._check_bucket()

    def _check_bucket(self) -> bool:
        """
        Check if the bucket exists.

        Returns:
            bool: True if the bucket exists, False otherwise
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                logger.error(f"Bucket {self.bucket_name} does not exist")
            else:
                logger.error(f"Error checking bucket: {e}")
            return False

    def generate_presigned_put_url(
        self,
        key: str,
        content_type: str = "video/mp4",
        expires_in: int = DEFAULT_EXPIRES,
    ) -> Dict[str, Any]:
        """
        Generate a presigned URL for uploading a file.

        Args:
            key: Object key
            content_type: Content type of the file
            expires_in: URL expiration time in seconds

        Returns:
            Dict with upload_url, key, and expires_in
        """
        # Validate content type
        if not content_type.startswith("video/"):
            raise ValueError("Content type must be a video format")

        # Ensure expires_in is within limits
        expires_in = min(expires_in, MAX_EXPIRES)

        # Generate presigned URL
        try:
            params = {
                "Bucket": self.bucket_name,
                "Key": key,
                "ContentType": content_type,
                "ServerSideEncryption": "AES256",
                "ACL": "private",
            }
            url = self.client.generate_presigned_url(
                "put_object", Params=params, ExpiresIn=expires_in
            )
            
            logger.info(f"Generated presigned PUT URL for {key}")
            return {
                "upload_url": url,
                "key": key,
                "expires_in": expires_in,
            }
        except ClientError as e:
            logger.error(f"Error generating presigned PUT URL: {e}")
            raise

    def generate_presigned_get_url(
        self, key: str, expires_in: int = DEFAULT_EXPIRES
    ) -> str:
        """
        Generate a presigned URL for downloading a file.

        Args:
            key: Object key
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        # Ensure expires_in is within limits
        expires_in = min(expires_in, MAX_EXPIRES)

        # Generate presigned URL
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
            logger.info(f"Generated presigned GET URL for {key}")
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned GET URL: {e}")
            raise

    def put_object(
        self,
        file_obj: Any,
        key: str,
        content_type: str = "video/mp4",
        content_length: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Upload a file object to Spaces.

        Args:
            file_obj: File-like object
            key: Object key
            content_type: Content type of the file
            content_length: Content length in bytes
            max_size: Maximum allowed size in bytes

        Returns:
            Dict with ETag and key
        """
        # Validate content type
        if not content_type.startswith("video/"):
            raise ValueError("Content type must be a video format")

        # Check file size if content_length and max_size are provided
        if content_length and max_size and content_length > max_size:
            raise ValueError(f"File size exceeds maximum allowed size of {max_size} bytes")

        # Upload file
        try:
            response = self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_obj,
                ContentType=content_type,
                ServerSideEncryption="AES256",
                ACL="private",
            )
            logger.info(f"Uploaded object {key}")
            return {
                "ETag": response.get("ETag", "").strip('"'),
                "key": key,
            }
        except ClientError as e:
            logger.error(f"Error uploading object: {e}")
            raise

    def delete_object(self, key: str) -> None:
        """
        Delete an object from Spaces.

        Args:
            key: Object key
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted object {key}")
        except ClientError as e:
            logger.error(f"Error deleting object: {e}")
            raise

    def check_object_exists(self, key: str) -> bool:
        """
        Check if an object exists in Spaces.

        Args:
            key: Object key

        Returns:
            bool: True if the object exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            logger.error(f"Error checking if object exists: {e}")
            raise


# Client factory
_spaces_client: Optional[SpacesClient] = None


def get_spaces_client(
    endpoint_url: Optional[str] = None,
    region_name: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
    check_bucket: bool = False,
    force_new: bool = False,
) -> SpacesClient:
    """
    Get a Spaces client instance.

    Args:
        endpoint_url: Spaces endpoint URL
        region_name: Region name
        access_key: Spaces access key
        secret_key: Spaces secret key
        bucket_name: Bucket name
        check_bucket: Whether to check if the bucket exists
        force_new: Force creation of a new client

    Returns:
        SpacesClient instance
    """
    global _spaces_client

    if _spaces_client is None or force_new:
        _spaces_client = SpacesClient(
            endpoint_url=endpoint_url,
            region_name=region_name,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name or DEFAULT_BUCKET,
            check_bucket=check_bucket,
        )

    return _spaces_client
