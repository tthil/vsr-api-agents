"""Shared Pydantic models for VSR API."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(str, Enum):
    """Video subtitle removal processing mode."""

    STTN = "sttn"  # Spatial-Temporal Transformer Network
    LAMA = "lama"  # Large Mask Inpainting
    PROPAINTER = "propainter"  # Propagation-based Inpainting


class JobEventType(str, Enum):
    """Job event types."""

    JOB_CREATED = "job_created"
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    WEBHOOK_SENT = "webhook_sent"
    WEBHOOK_FAILED = "webhook_failed"


class SubtitleArea(BaseModel):
    """Subtitle area coordinates."""

    top: float = Field(ge=0.0, le=1.0)
    left: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "SubtitleArea":
        """Validate that coordinates are valid."""
        if self.left + self.width > 1.0:
            raise ValueError("left + width must be <= 1.0")
        if self.top + self.height > 1.0:
            raise ValueError("top + height must be <= 1.0")
        return self

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format for storage."""
        return {
            "top": self.top,
            "left": self.left,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, coords: Dict[str, float]) -> "SubtitleArea":
        """Create from dictionary format."""
        return cls(
            top=coords.get("top", 0.0),
            left=coords.get("left", 0.0),
            width=coords.get("width", 1.0),
            height=coords.get("height", 1.0),
        )


class Job(BaseModel):
    """Video subtitle removal job."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, alias="_id")
    status: JobStatus = Field(default=JobStatus.PENDING)
    mode: ProcessingMode = Field(default=ProcessingMode.STTN)
    subtitle_area: Optional[SubtitleArea] = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    video_key: str
    processed_video_key: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    queue_position: Optional[int] = None
    api_key_id: Optional[uuid.UUID] = None

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={uuid.UUID: str},
        arbitrary_types_allowed=False,
    )

    @field_validator("video_key", "processed_video_key")
    @classmethod
    def validate_video_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate that video keys don't contain full URLs."""
        if v is None:
            return None
        if v.startswith("http://") or v.startswith("https://"):
            raise ValueError("Video key should not contain full URL")
        return v


class JobEvent(BaseModel):
    """Job processing event."""

    id: Any = Field(default=None, alias="_id")  # ObjectId in MongoDB
    job_id: uuid.UUID
    type: JobEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={uuid.UUID: str},
        arbitrary_types_allowed=True,
    )


class ApiKeyUsage(BaseModel):
    """API key usage statistics."""

    jobs_created: int = 0
    jobs_completed: int = 0
    upload_count: int = 0
    download_count: int = 0
    processing_seconds: float = 0.0
    total_video_size_mb: float = 0.0


class ApiKey(BaseModel):
    """API key for authentication and usage tracking."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, alias="_id")
    key: str = Field(min_length=32)
    name: str
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    usage: ApiKeyUsage = Field(default_factory=ApiKeyUsage)
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=False,
    )


# Request/Response models for API endpoints

class GenerateUploadUrlRequest(BaseModel):
    """Request for generate upload URL endpoint."""

    content_type: str = Field(default="video/mp4")
    api_key: str


class GenerateUploadUrlResponse(BaseModel):
    """Response for generate upload URL endpoint."""

    upload_url: str
    key: str
    expires_in: int


class CreateJobRequest(BaseModel):
    """Request for create job endpoint."""

    api_key: str
    video_key: str
    subtitle_area: Optional[SubtitleArea] = None
    callback_url: Optional[str] = None
    mode: ProcessingMode = Field(default=ProcessingMode.STTN)


class CreateJobResponse(BaseModel):
    """Response for create job endpoint."""

    job_id: uuid.UUID
    status: JobStatus
    message: str


class UploadAndSubmitResponse(BaseModel):
    """Response for upload and submit endpoint."""

    job_id: uuid.UUID
    status: JobStatus
    message: str


class JobResponse(BaseModel):
    """Job response for API endpoints."""

    id: uuid.UUID
    status: str
    mode: str
    subtitle_area: Optional[Dict[str, float]] = None
    progress: float
    video_key: str
    processed_video_key: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    queue_position: Optional[int] = None

    model_config = ConfigDict(
        json_encoders={uuid.UUID: str},
    )

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        """Create JobResponse from Job model."""
        return cls(
            id=job.id,
            status=job.status.value,
            mode=job.mode.value,
            subtitle_area=job.subtitle_area.to_dict() if job.subtitle_area else None,
            progress=job.progress,
            video_key=job.video_key,
            processed_video_key=job.processed_video_key,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            callback_url=job.callback_url,
            processing_time_seconds=job.processing_time_seconds,
            queue_position=job.queue_position,
        )


class JobEventResponse(BaseModel):
    """Job event response for API endpoints."""

    job_id: uuid.UUID
    type: str
    timestamp: datetime
    progress: float
    message: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_encoders={uuid.UUID: str},
    )

    @classmethod
    def from_job_event(cls, event: JobEvent) -> "JobEventResponse":
        """Create JobEventResponse from JobEvent model."""
        return cls(
            job_id=event.job_id,
            type=event.type.value,
            timestamp=event.timestamp,
            progress=event.progress,
            message=event.message,
            metadata=event.metadata,
        )


# RabbitMQ Message Models
class JobMessage(BaseModel):
    """Message schema for RabbitMQ job processing queue."""
    
    job_id: uuid.UUID = Field(..., description="Job ID")
    mode: ProcessingMode = Field(..., description="Processing mode")
    video_key_in: str = Field(..., description="S3 key for input video")
    video_url: Optional[str] = Field(None, description="Alternative video URL (deprecated)")
    subtitle_area: Optional[SubtitleArea] = Field(None, description="Subtitle area coordinates")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL")
    requested_at: datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc), description="Request timestamp")
    trace_id: Optional[str] = Field(None, description="Trace ID for request tracking")
    
    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    )


class JobMessageResponse(BaseModel):
    """Response from worker after processing job message."""
    
    job_id: uuid.UUID = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Job processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processed_video_key: Optional[str] = Field(None, description="S3 key for processed video")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time in seconds")
    trace_id: Optional[str] = Field(None, description="Trace ID for request tracking")
    
    model_config = ConfigDict(
        json_encoders={
            uuid.UUID: str,
        }
    )
