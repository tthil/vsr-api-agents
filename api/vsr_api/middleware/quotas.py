"""
Quotas and request guards middleware for VSR API.

Implements resource protection through video duration/resolution limits,
daily job quotas per API key, and business hours processing controls.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from collections import defaultdict
import subprocess
import json
import os

from fastapi import Request, HTTPException, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger(__name__)


class QuotaManager:
    """
    Manages API quotas and usage tracking.
    
    Tracks daily job limits per API key and enforces business hours restrictions.
    """
    
    def __init__(self):
        # In-memory quota tracking (in production, use Redis or database)
        self.daily_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_reset = datetime.now().date()
        
        # Configuration
        self.default_daily_limit = int(os.getenv("DAILY_JOB_LIMIT", "30"))
        self.business_hours_only = os.getenv("BUSINESS_HOURS_ONLY", "false").lower() == "true"
        self.business_start_hour = int(os.getenv("BUSINESS_START_HOUR", "9"))
        self.business_end_hour = int(os.getenv("BUSINESS_END_HOUR", "17"))
        
    def _reset_daily_usage_if_needed(self):
        """Reset daily usage counters if it's a new day."""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_usage.clear()
            self.last_reset = current_date
            logger.info("Daily usage counters reset", date=str(current_date))
    
    def check_daily_quota(self, api_key: str) -> Tuple[bool, int, int]:
        """
        Check if API key has remaining daily quota.
        
        Args:
            api_key: API key to check
            
        Returns:
            Tuple of (has_quota, used_count, limit)
        """
        self._reset_daily_usage_if_needed()
        
        today = str(datetime.now().date())
        used = self.daily_usage[api_key][today]
        limit = self.default_daily_limit
        
        return used < limit, used, limit
    
    def increment_usage(self, api_key: str):
        """Increment usage counter for API key."""
        self._reset_daily_usage_if_needed()
        
        today = str(datetime.now().date())
        self.daily_usage[api_key][today] += 1
        
        logger.info(
            "API usage incremented",
            api_key=api_key[:8] + "...",
            usage=self.daily_usage[api_key][today],
            limit=self.default_daily_limit
        )
    
    def check_business_hours(self) -> Tuple[bool, str]:
        """
        Check if current time is within business hours.
        
        Returns:
            Tuple of (is_business_hours, reason)
        """
        if not self.business_hours_only:
            return True, "Business hours restriction disabled"
        
        now = datetime.now()
        current_hour = now.hour
        
        if self.business_start_hour <= current_hour < self.business_end_hour:
            return True, "Within business hours"
        else:
            return False, f"Outside business hours ({self.business_start_hour}:00-{self.business_end_hour}:00)"


class VideoValidator:
    """
    Validates video files against duration and resolution limits.
    
    Uses ffprobe to extract video metadata and enforce limits.
    """
    
    def __init__(self):
        self.max_duration_seconds = int(os.getenv("MAX_VIDEO_DURATION", "60"))
        self.max_width = int(os.getenv("MAX_VIDEO_WIDTH", "1920"))
        self.max_height = int(os.getenv("MAX_VIDEO_HEIGHT", "1080"))
    
    async def validate_video_file(self, file: UploadFile) -> Tuple[bool, str, Dict]:
        """
        Validate video file against duration and resolution limits.
        
        Args:
            file: Uploaded video file
            
        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        try:
            # Save file temporarily for ffprobe analysis
            temp_path = f"/tmp/video_validation_{int(time.time())}.mp4"
            
            with open(temp_path, "wb") as temp_file:
                content = await file.read()
                temp_file.write(content)
                # Reset file pointer for later use
                await file.seek(0)
            
            try:
                # Extract video metadata using ffprobe
                metadata = await self._get_video_metadata(temp_path)
                
                # Validate duration
                duration = metadata.get("duration", 0)
                if duration > self.max_duration_seconds:
                    return False, f"Video duration ({duration}s) exceeds limit ({self.max_duration_seconds}s)", metadata
                
                # Validate resolution
                width = metadata.get("width", 0)
                height = metadata.get("height", 0)
                
                if width > self.max_width or height > self.max_height:
                    return False, f"Video resolution ({width}x{height}) exceeds limit ({self.max_width}x{self.max_height})", metadata
                
                return True, "Video validation passed", metadata
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error("Video validation error", error=str(e))
            return False, f"Video validation failed: {str(e)}", {}
    
    async def _get_video_metadata(self, file_path: str) -> Dict:
        """
        Extract video metadata using ffprobe.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"ffprobe failed: {stderr.decode()}")
            
            probe_data = json.loads(stdout.decode())
            
            # Extract relevant metadata
            video_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise Exception("No video stream found")
            
            format_info = probe_data.get("format", {})
            
            return {
                "duration": float(format_info.get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "codec": video_stream.get("codec_name", "unknown"),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "size_bytes": int(format_info.get("size", 0))
            }
            
        except Exception as e:
            logger.error("Failed to extract video metadata", error=str(e))
            raise


class QuotasMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enforcing quotas and request guards.
    
    Applies quotas and validation rules to incoming requests.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.quota_manager = QuotaManager()
        self.video_validator = VideoValidator()
        
        # Paths that require quota checking
        self.quota_paths = ["/api/upload-and-submit", "/api/submit-video"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip quota checking for non-quota paths
        if not any(request.url.path.startswith(path) for path in self.quota_paths):
            return await call_next(request)
        
        # Check if authentication is disabled for local development
        disable_auth = os.getenv("DISABLE_AUTH", "false").lower() == "true"
        if disable_auth:
            logger.info("Authentication disabled for local development")
            return await call_next(request)
        
        try:
            # Extract API key from request
            api_key = self._extract_api_key(request)
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required")
            
            # Check business hours
            is_business_hours, hours_reason = self.quota_manager.check_business_hours()
            if not is_business_hours:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "outside_business_hours",
                        "message": hours_reason,
                        "retry_after": "Next business day"
                    }
                )
            
            # Check daily quota
            has_quota, used, limit = self.quota_manager.check_daily_quota(api_key)
            if not has_quota:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "daily_quota_exceeded",
                        "message": f"Daily limit of {limit} jobs exceeded",
                        "used": used,
                        "limit": limit,
                        "reset_time": "00:00 UTC"
                    }
                )
            
            # For video upload endpoints, validate video file
            if request.url.path.startswith("/api/upload-and-submit"):
                # This validation will be applied in the route handler
                # since we need access to the uploaded file
                pass
            
            # Process request
            response = await call_next(request)
            
            # Increment usage counter on successful job submission
            if response.status_code in [200, 201, 202]:
                self.quota_manager.increment_usage(api_key)
            
            # Add quota headers
            response.headers["X-Daily-Quota-Used"] = str(used + 1)
            response.headers["X-Daily-Quota-Limit"] = str(limit)
            response.headers["X-Daily-Quota-Remaining"] = str(limit - used - 1)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Quota middleware error", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers or query parameters."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter
        return request.query_params.get("api_key")


# Global instances
quota_manager = QuotaManager()
video_validator = VideoValidator()


async def validate_video_upload(file: UploadFile) -> Tuple[bool, str, Dict]:
    """
    Validate uploaded video file.
    
    Args:
        file: Uploaded video file
        
    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    return await video_validator.validate_video_file(file)


def get_quota_status(api_key: str) -> Dict:
    """
    Get current quota status for API key.
    
    Args:
        api_key: API key to check
        
    Returns:
        Dictionary with quota information
    """
    has_quota, used, limit = quota_manager.check_daily_quota(api_key)
    is_business_hours, hours_reason = quota_manager.check_business_hours()
    
    return {
        "api_key": api_key[:8] + "...",
        "daily_quota": {
            "used": used,
            "limit": limit,
            "remaining": limit - used,
            "has_quota": has_quota
        },
        "business_hours": {
            "is_business_hours": is_business_hours,
            "reason": hours_reason
        }
    }
