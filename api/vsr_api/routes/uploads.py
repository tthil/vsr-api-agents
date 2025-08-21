"""Upload routes for VSR API."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from vsr_api.middleware.auth import get_current_api_key
from vsr_shared.logging import get_logger
from vsr_shared.models import GenerateUploadUrlResponse
from vsr_shared.presigned import generate_upload_url

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["uploads"])


class GenerateUploadUrlRequest(BaseModel):
    """Request for generate upload URL endpoint."""

    content_type: str = Field(default="video/mp4")

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate that content type is a video format."""
        if not v.startswith("video/"):
            raise ValueError("Content type must be a video format")
        return v


@router.post("/generate-upload-url", response_model=GenerateUploadUrlResponse)
async def generate_upload_url_endpoint(
    request: Optional[GenerateUploadUrlRequest] = None,
    content_type: Optional[str] = Query(None, description="Content type of the video file")
) -> GenerateUploadUrlResponse:
    """
    Generate a presigned URL for uploading a video file.

    Args:
        request: Request body with content_type
        content_type: Content type query parameter (alternative to request body)

    Returns:
        GenerateUploadUrlResponse: Response with upload_url, key, and expires_in
    """
    # Get content_type from either request body or query parameter
    content_type_value = None
    if request:
        content_type_value = request.content_type
    elif content_type:
        content_type_value = content_type
    else:
        content_type_value = "video/mp4"  # Default

    # Validate content type
    if not content_type_value.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Content type must be a video format",
        )

    try:
        # Generate presigned URL
        result = generate_upload_url(content_type=content_type_value)
        
        # Return response
        return GenerateUploadUrlResponse(
            upload_url=result["upload_url"],
            key=result["key"],
            expires_in=result["expires_in"],
        )
    except ValueError as e:
        logger.error(f"Error generating upload URL: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error generating upload URL: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )
