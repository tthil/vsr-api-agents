"""API routes for job management."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from vsr_api.config import get_settings
from vsr_api.middleware.auth import get_current_api_key
from vsr_shared.db.dal import JobsDAL, ApiKeysDAL
from vsr_shared.db.client import get_db
from vsr_shared.logging import get_logger
from vsr_shared.models import (
    Job,
    JobStatus,
    JobEvent,
    JobEventType,
    SubtitleArea,
    ApiKeyUsage,
    CreateJobRequest,
    CreateJobResponse,
    UploadAndSubmitResponse,
)
from vsr_shared.multipart import upload_file_multipart
from vsr_shared.spaces import get_spaces_client
from vsr_shared.spaces_keys import get_upload_key, validate_key
from vsr_shared.queue.integration import publish_job_for_processing

router = APIRouter(prefix="/api", tags=["Jobs"])
logger = get_logger(__name__)


@router.post(
    "/upload-and-submit-legacy",
    response_model=UploadAndSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload video and submit job in one step (Legacy)",
    description="Upload a video file and create a job to remove subtitles in one API call (Legacy with auth)",
)
async def upload_and_submit(
    request: Request,
    video: UploadFile = File(...),
    api_key: str = Form(...),
    subtitle_top: Optional[float] = Form(None),
    subtitle_left: Optional[float] = Form(None),
    subtitle_width: Optional[float] = Form(None),
    subtitle_height: Optional[float] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    """
    Upload a video file and create a job to remove subtitles in one API call.
    
    Args:
        request: FastAPI request
        video: Video file to upload
        api_key: API key for authentication
        subtitle_top: Top position of subtitle area (0-1)
        subtitle_left: Left position of subtitle area (0-1)
        subtitle_width: Width of subtitle area (0-1)
        subtitle_height: Height of subtitle area (0-1)
        callback_url: Optional URL to call when job is complete
        
    Returns:
        UploadAndSubmitResponse: Response with job details
    """
    # Check content type
    content_type = video.content_type
    if not content_type or not content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {content_type}",
        )
    
    # Validate API key
    db = get_db()
    api_keys_dal = ApiKeysDAL(db)
    
    api_key_doc = await api_keys_dal.get_by_key(api_key)
    if not api_key_doc or not api_key_doc.active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    # Create job ID and upload key
    job_id = uuid.uuid4()
    key = get_upload_key(job_id)
    
    try:
        # Upload file to Spaces
        spaces_client = get_spaces_client()
        
        # Get file object from UploadFile
        file = video.file
        
        # Upload file using multipart upload
        await upload_file_multipart(
            key=key,
            file_path=None,  # We're using a stream instead
            content_type=content_type,
            spaces_client=spaces_client,
            metadata={
                "job_id": str(job_id),
                "original_filename": video.filename,
            },
            stream=file,  # Pass the file stream directly
        )
        
        # Create subtitle area if coordinates provided
        subtitle_area = None
        if all(x is not None for x in [subtitle_top, subtitle_left, subtitle_width, subtitle_height]):
            subtitle_area = SubtitleArea(
                top=subtitle_top,
                left=subtitle_left,
                width=subtitle_width,
                height=subtitle_height,
            )
        
        # Create job
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            video_key=key,
            subtitle_area=subtitle_area,
            callback_url=callback_url,
            api_key_id=api_key_doc.id,
        )
        
        # Save job to database
        jobs_dal = JobsDAL(db)
        await jobs_dal.create(job)
        
        # Create job event
        event = JobEvent(
            job_id=job_id,
            type=JobEventType.JOB_CREATED,
            message="Job created and video uploaded",
        )
        await jobs_dal.add_event(event)
        
        # Publish job for processing via RabbitMQ
        trace_id = request.headers.get("X-Request-ID")
        try:
            await publish_job_for_processing(job, trace_id=trace_id)
            logger.info(f"Published job {job_id} for processing")
            
            # Update job status to queued
            await jobs_dal.update_status(job_id, JobStatus.QUEUED)
            
            # Add queued event
            queued_event = JobEvent(
                job_id=job_id,
                type=JobEventType.JOB_QUEUED,
                message="Job queued for processing",
            )
            await jobs_dal.add_event(queued_event)
            
        except Exception as e:
            logger.error(f"Failed to publish job {job_id} for processing: {e}")
            # Job remains in PENDING status if publishing fails
        
        # Update API key usage
        await api_keys_dal.increment_usage(
            api_key_doc.id,
            ApiKeyUsage(jobs_created=1)
        )
        
        # Return response
        return UploadAndSubmitResponse(
            job_id=job_id,
            status=job.status,
            message="Job created and video uploaded successfully",
        )
        
    except Exception as e:
        logger.error(
            "Error uploading video and creating job",
            error=str(e),
            job_id=job_id,
            key=key,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading video and creating job: {str(e)}",
        )


@router.post(
    "/jobs",
    response_model=CreateJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new job",
    description="Create a new job to remove subtitles from a video",
)
async def create_job(
    request: Request,
    job_request: CreateJobRequest,
):
    """
    Create a new job to remove subtitles from a video.
    
    Args:
        request: FastAPI request
        job_request: Job creation request
        
    Returns:
        CreateJobResponse: Response with job details
    """
    # Validate API key
    db = get_db()
    api_keys_dal = ApiKeysDAL(db)
    
    api_key_doc = await api_keys_dal.get_by_key(job_request.api_key)
    if not api_key_doc or not api_key_doc.active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    
    # Validate video key
    if not validate_key(job_request.video_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video key format: {job_request.video_key}",
        )
    
    # Check if video exists
    spaces_client = get_spaces_client()
    if not await spaces_client.exists(job_request.video_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video not found with key: {job_request.video_key}",
        )
    
    # Create job ID
    job_id = uuid.uuid4()
    
    try:
        # Create job
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            video_key=job_request.video_key,
            subtitle_area=job_request.subtitle_area,
            callback_url=job_request.callback_url,
            api_key_id=api_key_doc.id,
        )
        
        # Save job to database
        jobs_dal = JobsDAL(db)
        await jobs_dal.create(job)
        
        # Create job event
        event = JobEvent(
            job_id=job_id,
            type=JobEventType.JOB_CREATED,
            message="Job created",
        )
        await jobs_dal.add_event(event)
        
        # Publish job for processing via RabbitMQ
        trace_id = request.headers.get("X-Request-ID")
        try:
            await publish_job_for_processing(job, trace_id=trace_id)
            logger.info(f"Published job {job_id} for processing")
            
            # Update job status to queued
            await jobs_dal.update_status(job_id, JobStatus.QUEUED)
            
            # Add queued event
            queued_event = JobEvent(
                job_id=job_id,
                type=JobEventType.JOB_QUEUED,
                message="Job queued for processing",
            )
            await jobs_dal.add_event(queued_event)
            
        except Exception as e:
            logger.error(f"Failed to publish job {job_id} for processing: {e}")
            # Job remains in PENDING status if publishing fails
        
        # Update API key usage
        await api_keys_dal.increment_usage(
            api_key_doc.id,
            ApiKeyUsage(jobs_created=1)
        )
        
        # Return response
        return CreateJobResponse(
            job_id=job_id,
            status=job.status,
            message="Job created successfully",
        )
        
    except Exception as e:
        logger.error(
            "Error creating job",
            error=str(e),
            job_id=job_id,
            video_key=job_request.video_key,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating job: {str(e)}",
        )


@router.get(
    "/job-status/{job_id}",
    summary="Get job status",
    description="Get the status of a job with optional presigned URL for processed video",
)
async def get_job_status(
    job_id: uuid.UUID,
    db = Depends(get_db),
    spaces_client = Depends(get_spaces_client)
):
    """
    Get the status of a job.
    
    Args:
        job_id: Job ID
        user_id: Authenticated user ID from middleware
        
    Returns:
        JSONResponse: Job status and details with presigned URL if completed
    """
    try:
        # Get job from database
        job_doc = await db.jobs.find_one({"_id": job_id})
        
        if not job_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}"
            )
        
        # Convert to Job model
        job = Job(**job_doc)
        
        # Get current queue position for pending jobs
        queue_position = None
        if job.status == JobStatus.PENDING:
            queue_position = await db.jobs.count_documents({
                "status": JobStatus.PENDING.value,
                "created_at": {"$lt": job.created_at}
            }) + 1
        
        # Generate presigned URL for processed video if completed
        processed_video_url = None
        if job.status == JobStatus.COMPLETED and job.processed_video_key:
            from vsr_shared.presigned import generate_presigned_get_url
            try:
                processed_video_url = await generate_presigned_get_url(
                    spaces_client,
                    job.processed_video_key,
                    expires_in=3600  # 1 hour
                )
            except Exception as e:
                logger.warning(f"Failed to generate presigned URL for job {job_id}: {e}")
        
        # Calculate processing time if completed
        processing_time_seconds = None
        if job.completed_at and job.started_at:
            processing_time_seconds = (job.completed_at - job.started_at).total_seconds()
        
        # Return comprehensive job status
        return JSONResponse(
            content={
                "id": str(job.id),
                "status": job.status.value,
                "mode": job.mode.value if job.mode else None,
                "subtitle_area": job.subtitle_area.to_dict() if job.subtitle_area else None,
                "progress": job.progress,
                "video_key": job.video_key,
                "processed_video_key": job.processed_video_key,
                "processed_video_url": processed_video_url,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
                "callback_url": job.callback_url,
                "processing_time_seconds": processing_time_seconds,
                "queue_position": queue_position
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
