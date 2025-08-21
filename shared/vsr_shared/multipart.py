"""Multipart upload helpers for Spaces/S3."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union, BinaryIO

import boto3
from botocore.exceptions import ClientError

from vsr_shared.logging import get_logger
from vsr_shared.spaces import SpacesClient, get_spaces_client
from vsr_shared.spaces_keys import validate_key

logger = get_logger(__name__)


class MultipartUploadError(Exception):
    """Error during multipart upload."""

    pass


async def create_multipart_upload(
    key: str,
    content_type: str,
    spaces_client: Optional[SpacesClient] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a multipart upload.

    Args:
        key: Object key
        content_type: Content type of the object
        spaces_client: Optional SpacesClient instance
        metadata: Optional metadata to attach to the object

    Returns:
        str: Upload ID for the multipart upload

    Raises:
        ValueError: If key is invalid
        MultipartUploadError: If multipart upload creation fails
    """
    # Validate key
    if not validate_key(key):
        raise ValueError(f"Invalid key format: {key}")

    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # Create multipart upload
        params = {
            "Bucket": client.bucket_name,
            "Key": key,
            "ContentType": content_type,
            "ServerSideEncryption": "AES256",
            "ACL": "private",
        }

        # Add metadata if provided
        if metadata:
            params["Metadata"] = metadata

        response = await asyncio.to_thread(
            client.client.create_multipart_upload,
            **params
        )

        upload_id = response["UploadId"]
        logger.info(
            "Created multipart upload",
            key=key,
            upload_id=upload_id,
            content_type=content_type,
        )

        return upload_id

    except ClientError as e:
        logger.error(
            "Failed to create multipart upload",
            key=key,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to create multipart upload: {e}")


async def upload_part(
    key: str,
    upload_id: str,
    part_number: int,
    data: Union[bytes, BinaryIO],
    spaces_client: Optional[SpacesClient] = None,
) -> Dict[str, str]:
    """
    Upload a part in a multipart upload.

    Args:
        key: Object key
        upload_id: Upload ID from create_multipart_upload
        part_number: Part number (1-10000)
        data: Part data as bytes or file-like object
        spaces_client: Optional SpacesClient instance

    Returns:
        Dict[str, str]: Part information with ETag

    Raises:
        ValueError: If part_number is invalid
        MultipartUploadError: If part upload fails
    """
    # Validate part number
    if not 1 <= part_number <= 10000:
        raise ValueError(f"Invalid part number: {part_number}. Must be between 1 and 10000.")

    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # Upload part
        response = await asyncio.to_thread(
            client.client.upload_part,
            Bucket=client.bucket_name,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=data,
        )

        logger.debug(
            "Uploaded part",
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            etag=response["ETag"],
        )

        return {
            "PartNumber": part_number,
            "ETag": response["ETag"],
        }

    except ClientError as e:
        logger.error(
            "Failed to upload part",
            key=key,
            upload_id=upload_id,
            part_number=part_number,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to upload part {part_number}: {e}")


async def complete_multipart_upload(
    key: str,
    upload_id: str,
    parts: List[Dict[str, Union[int, str]]],
    spaces_client: Optional[SpacesClient] = None,
) -> str:
    """
    Complete a multipart upload.

    Args:
        key: Object key
        upload_id: Upload ID from create_multipart_upload
        parts: List of parts with PartNumber and ETag
        spaces_client: Optional SpacesClient instance

    Returns:
        str: Object URL

    Raises:
        MultipartUploadError: If multipart upload completion fails
    """
    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # Complete multipart upload
        response = await asyncio.to_thread(
            client.client.complete_multipart_upload,
            Bucket=client.bucket_name,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        logger.info(
            "Completed multipart upload",
            key=key,
            upload_id=upload_id,
            parts_count=len(parts),
            location=response.get("Location"),
        )

        return response.get("Location", f"{client.endpoint_url}/{client.bucket_name}/{key}")

    except ClientError as e:
        logger.error(
            "Failed to complete multipart upload",
            key=key,
            upload_id=upload_id,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to complete multipart upload: {e}")


async def abort_multipart_upload(
    key: str,
    upload_id: str,
    spaces_client: Optional[SpacesClient] = None,
) -> None:
    """
    Abort a multipart upload.

    Args:
        key: Object key
        upload_id: Upload ID from create_multipart_upload
        spaces_client: Optional SpacesClient instance

    Raises:
        MultipartUploadError: If multipart upload abort fails
    """
    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # Abort multipart upload
        await asyncio.to_thread(
            client.client.abort_multipart_upload,
            Bucket=client.bucket_name,
            Key=key,
            UploadId=upload_id,
        )

        logger.info(
            "Aborted multipart upload",
            key=key,
            upload_id=upload_id,
        )

    except ClientError as e:
        logger.error(
            "Failed to abort multipart upload",
            key=key,
            upload_id=upload_id,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to abort multipart upload: {e}")


async def upload_file_multipart(
    key: str,
    file_path: str,
    content_type: str,
    chunk_size: int = 5 * 1024 * 1024,  # 5 MB (S3 minimum)
    max_workers: int = 4,
    spaces_client: Optional[SpacesClient] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Upload a file using multipart upload.

    Args:
        key: Object key
        file_path: Path to file to upload
        content_type: Content type of the file
        chunk_size: Size of each part in bytes (minimum 5 MB for S3)
        max_workers: Maximum number of concurrent upload workers
        spaces_client: Optional SpacesClient instance
        metadata: Optional metadata to attach to the object

    Returns:
        str: Object URL

    Raises:
        FileNotFoundError: If file does not exist
        MultipartUploadError: If multipart upload fails
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file size
    file_size = os.path.getsize(file_path)

    # Get client
    client = spaces_client or get_spaces_client()

    # Create multipart upload
    upload_id = await create_multipart_upload(
        key=key,
        content_type=content_type,
        spaces_client=client,
        metadata=metadata,
    )

    try:
        # Calculate number of parts
        part_count = (file_size + chunk_size - 1) // chunk_size
        logger.info(
            "Starting multipart upload",
            key=key,
            upload_id=upload_id,
            file_size=file_size,
            chunk_size=chunk_size,
            part_count=part_count,
        )

        # Upload parts
        parts = []
        with open(file_path, "rb") as f:
            # Create a thread pool for concurrent uploads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit upload tasks
                futures = []
                for part_number in range(1, part_count + 1):
                    # Read chunk
                    data = f.read(chunk_size)
                    if not data:
                        break

                    # Submit upload task
                    future = asyncio.create_task(
                        upload_part(
                            key=key,
                            upload_id=upload_id,
                            part_number=part_number,
                            data=data,
                            spaces_client=client,
                        )
                    )
                    futures.append(future)

                # Wait for all uploads to complete
                parts = await asyncio.gather(*futures)

        # Complete multipart upload
        return await complete_multipart_upload(
            key=key,
            upload_id=upload_id,
            parts=parts,
            spaces_client=client,
        )

    except Exception as e:
        # Abort multipart upload on error
        logger.error(
            "Error during multipart upload, aborting",
            key=key,
            upload_id=upload_id,
            error=str(e),
        )
        await abort_multipart_upload(key=key, upload_id=upload_id, spaces_client=client)
        raise MultipartUploadError(f"Multipart upload failed: {e}")


async def upload_stream_multipart(
    key: str,
    stream: BinaryIO,
    content_type: str,
    chunk_size: int = 5 * 1024 * 1024,  # 5 MB (S3 minimum)
    max_workers: int = 4,
    spaces_client: Optional[SpacesClient] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Upload a stream using multipart upload.

    Args:
        key: Object key
        stream: File-like object to upload
        content_type: Content type of the stream
        chunk_size: Size of each part in bytes (minimum 5 MB for S3)
        max_workers: Maximum number of concurrent upload workers
        spaces_client: Optional SpacesClient instance
        metadata: Optional metadata to attach to the object

    Returns:
        str: Object URL

    Raises:
        MultipartUploadError: If multipart upload fails
    """
    # Get client
    client = spaces_client or get_spaces_client()

    # Create multipart upload
    upload_id = await create_multipart_upload(
        key=key,
        content_type=content_type,
        spaces_client=client,
        metadata=metadata,
    )

    try:
        # Upload parts
        parts = []
        part_number = 1
        
        while True:
            # Read chunk
            data = stream.read(chunk_size)
            if not data:
                break
            
            # Upload part
            part = await upload_part(
                key=key,
                upload_id=upload_id,
                part_number=part_number,
                data=data,
                spaces_client=client,
            )
            parts.append(part)
            part_number += 1
        
        if not parts:
            # No parts were uploaded (empty stream)
            logger.warning(
                "No parts uploaded (empty stream), aborting multipart upload",
                key=key,
                upload_id=upload_id,
            )
            await abort_multipart_upload(key=key, upload_id=upload_id, spaces_client=client)
            raise MultipartUploadError("No data to upload (empty stream)")
        
        # Complete multipart upload
        logger.info(
            "Completing multipart upload",
            key=key,
            upload_id=upload_id,
            parts_count=len(parts),
        )
        return await complete_multipart_upload(
            key=key,
            upload_id=upload_id,
            parts=parts,
            spaces_client=client,
        )
        
    except Exception as e:
        # Abort multipart upload on error
        logger.error(
            "Error during multipart upload, aborting",
            key=key,
            upload_id=upload_id,
            error=str(e),
        )
        await abort_multipart_upload(key=key, upload_id=upload_id, spaces_client=client)
        raise MultipartUploadError(f"Multipart upload failed: {e}")


async def list_multipart_uploads(
    prefix: Optional[str] = None,
    spaces_client: Optional[SpacesClient] = None,
) -> List[Dict]:
    """
    List in-progress multipart uploads.

    Args:
        prefix: Optional key prefix to filter uploads
        spaces_client: Optional SpacesClient instance

    Returns:
        List[Dict]: List of in-progress multipart uploads

    Raises:
        MultipartUploadError: If listing multipart uploads fails
    """
    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # List multipart uploads
        params = {"Bucket": client.bucket_name}
        if prefix:
            params["Prefix"] = prefix

        response = await asyncio.to_thread(
            client.client.list_multipart_uploads,
            **params
        )

        uploads = response.get("Uploads", [])
        logger.info(
            "Listed multipart uploads",
            count=len(uploads),
            prefix=prefix,
        )

        return uploads

    except ClientError as e:
        logger.error(
            "Failed to list multipart uploads",
            prefix=prefix,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to list multipart uploads: {e}")


async def clean_incomplete_multipart_uploads(
    prefix: Optional[str] = None,
    max_age_hours: int = 24,
    spaces_client: Optional[SpacesClient] = None,
) -> int:
    """
    Clean up incomplete multipart uploads.

    Args:
        prefix: Optional key prefix to filter uploads
        max_age_hours: Maximum age in hours for uploads to keep
        spaces_client: Optional SpacesClient instance

    Returns:
        int: Number of aborted uploads

    Raises:
        MultipartUploadError: If cleaning up multipart uploads fails
    """
    # Get client
    client = spaces_client or get_spaces_client()

    try:
        # List multipart uploads
        uploads = await list_multipart_uploads(prefix=prefix, spaces_client=client)
        
        if not uploads:
            logger.info("No incomplete multipart uploads found")
            return 0
        
        # Get current time
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        max_age = datetime.timedelta(hours=max_age_hours)
        
        # Abort old uploads
        aborted_count = 0
        for upload in uploads:
            key = upload.get("Key")
            upload_id = upload.get("UploadId")
            initiated = upload.get("Initiated")
            
            if not key or not upload_id:
                continue
            
            # Check age if initiated timestamp is available
            if initiated and now - initiated > max_age:
                logger.info(
                    "Aborting old multipart upload",
                    key=key,
                    upload_id=upload_id,
                    initiated=initiated,
                    age_hours=(now - initiated).total_seconds() / 3600,
                )
                
                await abort_multipart_upload(
                    key=key,
                    upload_id=upload_id,
                    spaces_client=client,
                )
                aborted_count += 1
        
        logger.info(
            "Cleaned up incomplete multipart uploads",
            found=len(uploads),
            aborted=aborted_count,
            prefix=prefix,
        )
        
        return aborted_count
        
    except ClientError as e:
        logger.error(
            "Failed to clean up multipart uploads",
            prefix=prefix,
            error=str(e),
            error_code=e.response.get("Error", {}).get("Code"),
        )
        raise MultipartUploadError(f"Failed to clean up multipart uploads: {e}")
