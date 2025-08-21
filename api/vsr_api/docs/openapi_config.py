"""
OpenAPI configuration and documentation for VSR API.
Includes comprehensive examples, tags, and security definitions.
"""
from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with comprehensive documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Custom OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Video Subtitle Removal API",
        version="1.0.0",
        description="""
# Video Subtitle Removal API

A powerful API for removing hardcoded subtitles from videos using advanced AI models.

## Features

- **Multiple AI Models**: Choose from STTN, LAMA, or ProPainter for optimal results
- **Flexible Input**: Upload files directly or provide video URLs
- **Real-time Status**: Track job progress with detailed status updates
- **Secure Storage**: Videos stored securely in DigitalOcean Spaces
- **Rate Limited**: Built-in rate limiting for fair usage

## Authentication

All endpoints require API key authentication using Bearer token:

```
Authorization: Bearer your-api-key-here
```

## Processing Modes

- **STTN**: Spatial-Temporal Transformer Network - Best for temporal consistency
- **LAMA**: Large Mask Inpainting - Fast processing with good quality
- **ProPainter**: Advanced flow-guided propagation - Highest quality results

## Rate Limits

- **Upload endpoints**: 100 requests per hour
- **General endpoints**: 60 requests per minute  
- **Status endpoints**: 1000 requests per hour

## Error Codes

The API uses standardized error codes for consistent error handling:

- `invalid_api_key`: Authentication failed
- `invalid_processing_mode`: Unsupported processing mode
- `invalid_video_url`: Video URL validation failed
- `file_too_large`: File exceeds size limits
- `rate_limit_exceeded`: Rate limit exceeded
- `job_not_found`: Requested job not found
- `processing_failed`: Video processing error
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "API key authentication using Bearer token"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    # Add comprehensive examples
    openapi_schema["components"]["examples"] = get_api_examples()
    
    # Add response schemas
    openapi_schema["components"]["schemas"].update(get_response_schemas())
    
    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "submit",
            "description": "Video submission endpoints for processing jobs"
        },
        {
            "name": "jobs", 
            "description": "Job status and management endpoints"
        },
        {
            "name": "uploads",
            "description": "File upload and presigned URL endpoints"
        },
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_api_examples() -> Dict[str, Any]:
    """Get comprehensive API examples for documentation."""
    return {
        "upload_and_submit_request": {
            "summary": "Upload and submit video for processing",
            "description": "Example request for uploading a video file and submitting it for subtitle removal",
            "value": {
                "mode": "LAMA",
                "subtitle_area": "[100,400,800,500]",
                "callback_url": "https://your-app.com/webhook"
            }
        },
        "submit_video_url_request": {
            "summary": "Submit video URL for processing", 
            "description": "Example request for processing a video from a URL",
            "value": {
                "mode": "STTN",
                "video_url": "https://example.com/video.mp4",
                "subtitle_area": "[50,350,900,450]",
                "callback_url": "https://your-app.com/webhook"
            }
        },
        "job_response": {
            "summary": "Job creation response",
            "description": "Response after successfully creating a processing job",
            "value": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "mode": "LAMA", 
                "subtitle_area": {
                    "x1": 100,
                    "y1": 400,
                    "x2": 800,
                    "y2": 500
                },
                "video_key": "uploads/20241212/550e8400-e29b-41d4-a716-446655440000/video.mp4",
                "queue_position": 3,
                "eta_seconds": 210,
                "created_at": "2024-12-12T10:30:00Z",
                "callback_url": "https://your-app.com/webhook"
            }
        },
        "job_status_response": {
            "summary": "Job status response",
            "description": "Detailed job status with processing information",
            "value": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "mode": "LAMA",
                "subtitle_area": {
                    "x1": 100,
                    "y1": 400, 
                    "x2": 800,
                    "y2": 500
                },
                "progress": 100,
                "video_key": "uploads/20241212/550e8400-e29b-41d4-a716-446655440000/video.mp4",
                "processed_video_key": "processed/20241212/550e8400-e29b-41d4-a716-446655440000/video.mp4",
                "processed_video_url": "https://spaces.example.com/processed/video.mp4?signature=...",
                "created_at": "2024-12-12T10:30:00Z",
                "updated_at": "2024-12-12T10:35:00Z",
                "started_at": "2024-12-12T10:31:00Z",
                "completed_at": "2024-12-12T10:35:00Z",
                "processing_time_seconds": 240,
                "queue_position": None
            }
        },
        "upload_url_response": {
            "summary": "Presigned upload URL response",
            "description": "Response containing presigned URL for direct upload",
            "value": {
                "upload_url": "https://spaces.example.com/uploads/video.mp4?signature=...",
                "key": "uploads/20241212/550e8400-e29b-41d4-a716-446655440000/video.mp4",
                "expires_in": 3600
            }
        },
        "error_response": {
            "summary": "Error response",
            "description": "Standard error response format",
            "value": {
                "error": "invalid_processing_mode",
                "message": "Invalid processing mode. Must be one of: STTN, LAMA, PROPAINTER",
                "details": "Received mode: 'INVALID_MODE'"
            }
        }
    }


def get_response_schemas() -> Dict[str, Any]:
    """Get additional response schemas for documentation."""
    return {
        "JobResponse": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique job identifier"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in-progress", "completed", "failed", "cancelled"],
                    "description": "Current job status"
                },
                "mode": {
                    "type": "string", 
                    "enum": ["STTN", "LAMA", "PROPAINTER"],
                    "description": "AI model used for processing"
                },
                "subtitle_area": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer", "description": "Left coordinate"},
                        "y1": {"type": "integer", "description": "Top coordinate"},
                        "x2": {"type": "integer", "description": "Right coordinate"},
                        "y2": {"type": "integer", "description": "Bottom coordinate"}
                    },
                    "description": "Subtitle area coordinates (optional)"
                },
                "progress": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Processing progress percentage"
                },
                "video_key": {
                    "type": "string",
                    "description": "Storage key for input video"
                },
                "processed_video_key": {
                    "type": "string",
                    "description": "Storage key for processed video (when completed)"
                },
                "processed_video_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "Presigned URL for downloading processed video"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Job creation timestamp"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time", 
                    "description": "Last update timestamp"
                },
                "started_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Processing start timestamp"
                },
                "completed_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Processing completion timestamp"
                },
                "error_message": {
                    "type": "string",
                    "description": "Error message if processing failed"
                },
                "callback_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "Webhook URL for completion notification"
                },
                "processing_time_seconds": {
                    "type": "number",
                    "description": "Total processing time in seconds"
                },
                "queue_position": {
                    "type": "integer",
                    "description": "Position in processing queue (null if not queued)"
                },
                "eta_seconds": {
                    "type": "integer",
                    "description": "Estimated time to completion in seconds"
                }
            },
            "required": ["id", "status", "mode", "created_at"]
        },
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "Error code identifier"
                },
                "message": {
                    "type": "string", 
                    "description": "Human-readable error message"
                },
                "details": {
                    "type": "string",
                    "description": "Additional error details (optional)"
                }
            },
            "required": ["error", "message"]
        },
        "UploadUrlResponse": {
            "type": "object",
            "properties": {
                "upload_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "Presigned URL for uploading video file"
                },
                "key": {
                    "type": "string",
                    "description": "Storage key for the uploaded file"
                },
                "expires_in": {
                    "type": "integer",
                    "description": "URL expiration time in seconds"
                }
            },
            "required": ["upload_url", "key", "expires_in"]
        }
    }


def configure_openapi_docs(app: FastAPI) -> None:
    """
    Configure OpenAPI documentation for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Set custom OpenAPI function
    app.openapi = lambda: custom_openapi(app)
    
    # Configure docs URLs (can be disabled in production)
    app.docs_url = "/docs"
    app.redoc_url = "/redoc"
    app.openapi_url = "/openapi.json"
