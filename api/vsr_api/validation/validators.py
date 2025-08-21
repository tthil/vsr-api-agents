"""Request validation framework for VSR API."""

import os
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel, Field, validator

from vsr_shared.logging import get_logger
from vsr_shared.models import ProcessingMode, SubtitleArea

logger = get_logger(__name__)

# Environment-based configuration
ALLOW_HTTP_WEBHOOKS = os.getenv("ALLOW_HTTP_WEBHOOKS", "false").lower() == "true"

# Constants for validation
HD_WIDTH = 1920
HD_HEIGHT = 1080
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_VIDEO_TYPES = [
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",  # AVI
    "video/x-ms-wmv",   # WMV
]


class ValidationError(Exception):
    """Custom validation error."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class VideoUploadRequest(BaseModel):
    """Request model for video upload validation."""
    
    mode: ProcessingMode = Field(..., description="Processing mode")
    subtitle_area: Optional[SubtitleArea] = Field(None, description="Subtitle area coordinates")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL")
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate processing mode."""
        if v not in [ProcessingMode.STTN, ProcessingMode.LAMA, ProcessingMode.PROPAINTER]:
            raise ValueError(f"Invalid processing mode. Must be one of: {[m.value for m in ProcessingMode]}")
        return v
    
    @validator('subtitle_area')
    def validate_subtitle_area(cls, v):
        """Validate subtitle area coordinates."""
        if v is None:
            return v
            
        # Validate coordinates are within HD frame bounds
        if not (0 <= v.x1 < v.x2 <= HD_WIDTH):
            raise ValueError(f"Invalid x coordinates. Must be within 0-{HD_WIDTH} and x1 < x2")
        
        if not (0 <= v.y1 < v.y2 <= HD_HEIGHT):
            raise ValueError(f"Invalid y coordinates. Must be within 0-{HD_HEIGHT} and y1 < y2")
            
        # Validate minimum area size (at least 10x10 pixels)
        if (v.x2 - v.x1) < 10 or (v.y2 - v.y1) < 10:
            raise ValueError("Subtitle area must be at least 10x10 pixels")
            
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        """Validate callback URL format."""
        if v is None:
            return v
            
        # Must be HTTPS
        parsed = urlparse(v)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Callback URL must use HTTP or HTTPS")
            
        if not parsed.netloc:
            raise ValueError("Invalid callback URL format")
            
        return v


class VideoURLRequest(BaseModel):
    """Request model for video URL submission."""
    
    video_url: str = Field(..., description="HTTPS URL to video file")
    mode: ProcessingMode = Field(..., description="Processing mode")
    subtitle_area: Optional[SubtitleArea] = Field(None, description="Subtitle area coordinates")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL")
    
    @validator('video_url')
    def validate_video_url(cls, v):
        """Validate video URL format."""
        # Must be HTTPS
        parsed = urlparse(v)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Video URL must use HTTPS")
            
        if not parsed.netloc:
            raise ValueError("Invalid video URL format")
            
        return v
    
    @validator('mode')
    def validate_mode(cls, v):
        """Validate processing mode."""
        if v not in [ProcessingMode.STTN, ProcessingMode.LAMA, ProcessingMode.PROPAINTER]:
            raise ValueError(f"Invalid processing mode. Must be one of: {[m.value for m in ProcessingMode]}")
        return v
    
    @validator('subtitle_area')
    def validate_subtitle_area(cls, v):
        """Validate subtitle area coordinates."""
        if v is None:
            return v
            
        # Validate coordinates are within HD frame bounds
        if not (0 <= v.x1 < v.x2 <= HD_WIDTH):
            raise ValueError(f"Invalid x coordinates. Must be within 0-{HD_WIDTH} and x1 < x2")
        
        if not (0 <= v.y1 < v.y2 <= HD_HEIGHT):
            raise ValueError(f"Invalid y coordinates. Must be within 0-{HD_HEIGHT} and y1 < y2")
            
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        """Validate callback URL format."""
        if v is None:
            return v
            
        # Must be HTTPS
        parsed = urlparse(v)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Callback URL must use HTTP or HTTPS")
            
        if not parsed.netloc:
            raise ValueError("Invalid callback URL format")
            
        return v


class UploadURLRequest(BaseModel):
    """Request model for presigned upload URL generation."""
    
    filename: str = Field(..., description="Video filename")
    content_type: str = Field(..., description="Video content type")
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Filename cannot be empty")
            
        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError("Invalid filename: path traversal not allowed")
            
        # Check file extension
        valid_extensions = ['.mp4', '.mpeg', '.mov', '.avi', '.wmv']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Invalid file extension. Supported: {valid_extensions}")
            
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type."""
        if v not in SUPPORTED_VIDEO_TYPES:
            raise ValueError(f"Unsupported content type. Supported: {SUPPORTED_VIDEO_TYPES}")
        return v


async def validate_video_url_accessibility(video_url: str) -> Dict[str, Any]:
    """
    Validate that video URL is accessible and has correct content type.
    
    Args:
        video_url: URL to validate
        
    Returns:
        Dict with validation results
        
    Raises:
        ValidationError: If URL is not accessible or invalid
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Perform HEAD request to check accessibility and content type
            response = await client.head(video_url, follow_redirects=True)
            
            # Check if URL is accessible
            if response.status_code not in [200, 206]:  # 206 for partial content
                raise ValidationError(
                    f"Video URL not accessible: HTTP {response.status_code}",
                    "VIDEO_URL_NOT_ACCESSIBLE"
                )
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('video/'):
                raise ValidationError(
                    f"Invalid content type: {content_type}. Expected video/*",
                    "INVALID_CONTENT_TYPE"
                )
            
            # Check content length if available
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > MAX_VIDEO_SIZE_MB:
                    raise ValidationError(
                        f"Video too large: {size_mb:.1f}MB. Maximum: {MAX_VIDEO_SIZE_MB}MB",
                        "VIDEO_TOO_LARGE"
                    )
            
            return {
                "accessible": True,
                "content_type": content_type,
                "content_length": content_length,
                "size_mb": int(content_length) / (1024 * 1024) if content_length else None,
            }
            
    except httpx.TimeoutException:
        raise ValidationError(
            "Video URL validation timeout",
            "VIDEO_URL_TIMEOUT"
        )
    except httpx.RequestError as e:
        raise ValidationError(
            f"Error accessing video URL: {str(e)}",
            "VIDEO_URL_ERROR"
        )


def validate_video_file_size(file_size: int) -> None:
    """
    Validate uploaded video file size.
    
    Args:
        file_size: File size in bytes
        
    Raises:
        ValidationError: If file is too large
    """
    max_size_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        raise ValidationError(
            f"Video file too large: {size_mb:.1f}MB. Maximum: {MAX_VIDEO_SIZE_MB}MB",
            "VIDEO_TOO_LARGE"
        )


def validate_video_content_type(content_type: str) -> None:
    """
    Validate video content type.
    
    Args:
        content_type: MIME content type
        
    Raises:
        ValidationError: If content type is not supported
    """
    if not content_type or content_type not in SUPPORTED_VIDEO_TYPES:
        raise ValidationError(
            f"Unsupported video format: {content_type}. Supported: {SUPPORTED_VIDEO_TYPES}",
            "UNSUPPORTED_VIDEO_FORMAT"
        )


class ValidationErrorHandler:
    """Handler for validation errors with standardized error responses."""
    
    ERROR_CODES = {
        "VALIDATION_ERROR": {
            "status_code": status.HTTP_400_BAD_REQUEST,
            "message": "Validation failed"
        },
        "VIDEO_URL_NOT_ACCESSIBLE": {
            "status_code": status.HTTP_400_BAD_REQUEST,
            "message": "Video URL is not accessible"
        },
        "INVALID_CONTENT_TYPE": {
            "status_code": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "message": "Unsupported media type"
        },
        "VIDEO_TOO_LARGE": {
            "status_code": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "message": "Video file too large"
        },
        "VIDEO_URL_TIMEOUT": {
            "status_code": status.HTTP_400_BAD_REQUEST,
            "message": "Video URL validation timeout"
        },
        "VIDEO_URL_ERROR": {
            "status_code": status.HTTP_400_BAD_REQUEST,
            "message": "Error accessing video URL"
        },
        "UNSUPPORTED_VIDEO_FORMAT": {
            "status_code": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "message": "Unsupported video format"
        },
    }
    
    @classmethod
    def handle_validation_error(cls, error: ValidationError) -> HTTPException:
        """
        Convert ValidationError to HTTPException.
        
        Args:
            error: ValidationError to convert
            
        Returns:
            HTTPException with appropriate status code and message
        """
        error_info = cls.ERROR_CODES.get(error.error_code, cls.ERROR_CODES["VALIDATION_ERROR"])
        
        return HTTPException(
            status_code=error_info["status_code"],
            detail={
                "error": error_info["message"],
                "error_code": error.error_code,
                "details": error.message,
            }
        )
    
    @classmethod
    def handle_pydantic_error(cls, error: Exception) -> HTTPException:
        """
        Convert Pydantic validation error to HTTPException.
        
        Args:
            error: Pydantic validation error
            
        Returns:
            HTTPException with validation details
        """
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Request validation failed",
                "error_code": "VALIDATION_ERROR",
                "details": str(error),
            }
        )


# Validation decorators
def validate_request(request_model):
    """
    Decorator to validate request using Pydantic model.
    
    Args:
        request_model: Pydantic model class for validation
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValidationError as e:
                raise ValidationErrorHandler.handle_validation_error(e)
            except Exception as e:
                if "validation" in str(e).lower():
                    raise ValidationErrorHandler.handle_pydantic_error(e)
                raise
        return wrapper
    return decorator


async def validate_video_file(video_file: UploadFile, max_size: int) -> None:
    """
    Validate uploaded video file.
    
    Args:
        video_file: Uploaded video file
        max_size: Maximum allowed file size in bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Check content type
    if not video_file.content_type or not video_file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video format"
        )
    
    # Check file size if available
    if hasattr(video_file, 'size') and video_file.size:
        if video_file.size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {max_size // (1024*1024)}MB"
            )
    
    # Check filename extension
    if video_file.filename:
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        file_ext = video_file.filename.lower().split('.')[-1]
        if f'.{file_ext}' not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported video format: {file_ext}"
            )
