"""
Submit endpoints for video processing jobs.
Handles both direct upload and URL-based submission.
"""
import asyncio
import uuid
from typing import Optional
import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from vsr_shared.models import Job, JobStatus, ProcessingMode, SubtitleArea
from vsr_shared.db.client import get_db
from vsr_shared.spaces import get_spaces_client
from vsr_shared.spaces_keys import get_upload_key
from vsr_shared.queue.integration import publish_job_for_processing
from vsr_api.middleware.auth import get_current_api_key
from vsr_api.validation.validators import (
    VideoUploadRequest,
    VideoURLRequest,
    validate_video_file
)
from vsr_shared.models import ProcessingMode, SubtitleArea
from urllib.parse import urlparse

router = APIRouter(prefix="/api", tags=["submit"])

# Maximum file size: 500MB
MAX_FILE_SIZE = 500 * 1024 * 1024

@router.post("/upload-and-submit")
async def upload_and_submit(
    mode: str = Form(...),
    subtitle_area: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    video_file: UploadFile = File(...),
    db = Depends(get_db),
    spaces_client = Depends(get_spaces_client)
):
    """
    Upload video file directly and submit for processing.
    
    - **mode**: Processing mode (STTN, LAMA, PROPAINTER)
    - **subtitle_area**: Optional subtitle area coordinates [x1,y1,x2,y2]
    - **callback_url**: Optional webhook URL for completion notification
    - **video_file**: Video file to process
    """
    try:
        # Validate processing mode
        try:
            processing_mode = ProcessingMode(mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing mode. Must be one of: sttn, lama, propainter"
            )
        
        # Validate subtitle area if provided
        subtitle_area_obj = None
        if subtitle_area:
            # Parse subtitle area coordinates (x1,y1,x2,y2)
            try:
                coords = [int(x.strip()) for x in subtitle_area.split(',')]
                if len(coords) != 4:
                    raise ValueError("Must have 4 coordinates")
                subtitle_area_obj = SubtitleArea(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
            except (ValueError, IndexError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subtitle area must be 4 comma-separated coordinates: x1,y1,x2,y2"
                )
        
        # Validate callback URL if provided
        if callback_url:
            parsed = urlparse(callback_url)
            if parsed.scheme not in ['http', 'https']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Callback URL must use HTTP or HTTPS"
                )
        
        # Validate video file
        await validate_video_file(video_file, MAX_FILE_SIZE)
        
        # Generate unique job ID and video key
        job_id = uuid.uuid4()
        video_key = get_upload_key(job_id)
        
        # Upload video to Spaces
        try:
            # Reset file pointer
            await video_file.seek(0)
            
            # Upload with streaming using boto3 client
            spaces_client.client.upload_fileobj(
                video_file.file,
                spaces_client.bucket_name,
                video_key,
                ExtraArgs={
                    'ContentType': video_file.content_type,
                    'ACL': 'private'
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload video: {str(e)}"
            )
        
        # Get current queue length for ETA calculation
        queue_length = await db.jobs.count_documents({"status": JobStatus.PENDING.value})
        
        # Calculate ETA: base 2 minutes + 30 seconds per queued job
        eta_seconds = 120 + (queue_length * 30)
        
        # Create job record
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            mode=processing_mode,
            subtitle_area=subtitle_area_obj,
            video_key=video_key,
            callback_url=callback_url,
            queue_position=queue_length + 1
        )
        
        # Save job to database
        await db.jobs.insert_one(job.model_dump(by_alias=True))
        
        # Publish job to RabbitMQ
        await publish_job_for_processing(job)
        
        # Return job response
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "id": str(job_id),
                "status": job.status.value,
                "mode": job.mode.value,
                "subtitle_area": subtitle_area_obj.to_dict() if subtitle_area_obj else None,
                "video_key": video_key,
                "queue_position": job.queue_position,
                "eta_seconds": eta_seconds,
                "created_at": job.created_at.isoformat(),
                "callback_url": callback_url
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/submit-video")
async def submit_video_url(
    mode: str = Form(...),
    video_url: str = Form(...),
    subtitle_area: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    db = Depends(get_db),
    spaces_client = Depends(get_spaces_client)
):
    """
    Submit video URL for processing with server-side download.
    
    - **mode**: Processing mode (STTN, LAMA, PROPAINTER)
    - **video_url**: HTTPS URL of video to process
    - **subtitle_area**: Optional subtitle area coordinates [x1,y1,x2,y2]
    - **callback_url**: Optional webhook URL for completion notification
    """
    try:
        # Validate processing mode
        try:
            processing_mode = ProcessingMode(mode)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing mode. Must be one of: sttn, lama, propainter"
            )
        
        # Validate video URL
        parsed = urlparse(video_url)
        if parsed.scheme != 'https':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video URL must use HTTPS"
            )
        
        # Validate subtitle area if provided
        subtitle_area_obj = None
        if subtitle_area:
            # Parse subtitle area coordinates (x1,y1,x2,y2)
            try:
                coords = [int(x.strip()) for x in subtitle_area.split(',')]
                if len(coords) != 4:
                    raise ValueError("Must have 4 coordinates")
                subtitle_area_obj = SubtitleArea(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
            except (ValueError, IndexError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Subtitle area must be 4 comma-separated coordinates: x1,y1,x2,y2"
                )
        
        # Validate callback URL if provided
        if callback_url:
            parsed = urlparse(callback_url)
            if parsed.scheme not in ['http', 'https']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Callback URL must use HTTP or HTTPS"
                )
        
        # Validate video URL content-type with HEAD request
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                head_response = await client.head(video_url)
                head_response.raise_for_status()
                
                content_type = head_response.headers.get("content-type", "")
                if not content_type.startswith("video/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="URL does not point to a video file"
                    )
                
                # Check content length if available
                content_length = head_response.headers.get("content-length")
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video file too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                    
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to validate video URL: {str(e)}"
                )
        
        # Generate unique job ID and video key
        job_id = uuid.uuid4()
        video_key = get_upload_key(job_id)
        
        # Download and upload video to Spaces
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", video_url) as response:
                    response.raise_for_status()
                    
                    # Validate content type from actual response
                    actual_content_type = response.headers.get("content-type", "")
                    if not actual_content_type.startswith("video/"):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="URL does not serve a video file"
                        )
                    
                    # Stream upload to Spaces
                    upload_data = b""
                    total_size = 0
                    
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        total_size += len(chunk)
                        if total_size > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"Video file too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
                            )
                        upload_data += chunk
                    
                    # Upload to Spaces
                    from io import BytesIO
                    spaces_client.client.upload_fileobj(
                        BytesIO(upload_data),
                        spaces_client.bucket_name,
                        video_key,
                        ExtraArgs={
                            'ContentType': actual_content_type,
                            'ACL': 'private'
                        }
                    )
                    
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download video: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process video URL: {str(e)}"
            )
        
        # Get current queue length for ETA calculation
        queue_length = await db.jobs.count_documents({"status": JobStatus.PENDING.value})
        
        # Calculate ETA: base 2 minutes + 30 seconds per queued job
        eta_seconds = 120 + (queue_length * 30)
        
        # Create job record
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            mode=processing_mode,
            subtitle_area=subtitle_area_obj,
            video_key=video_key,
            callback_url=callback_url,
            queue_position=queue_length + 1
        )
        
        # Save job to database
        await db.jobs.insert_one(job.model_dump(by_alias=True))
        
        # Publish job to RabbitMQ
        await publish_job_for_processing(job)
        
        # Return job response
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "id": str(job_id),
                "status": job.status.value,
                "mode": job.mode.value,
                "subtitle_area": subtitle_area_obj.to_dict() if subtitle_area_obj else None,
                "video_key": video_key,
                "queue_position": job.queue_position,
                "eta_seconds": eta_seconds,
                "created_at": job.created_at.isoformat(),
                "callback_url": callback_url
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
