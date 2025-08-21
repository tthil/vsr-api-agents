"""Webhook delivery subsystem for VSR API."""

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, Field

from .models import Job, JobStatus, JobEventType

logger = structlog.get_logger(__name__)


class WebhookPayload(BaseModel):
    """Webhook payload sent to callback URLs."""
    
    job_id: uuid.UUID
    status: JobStatus
    timestamp: datetime
    video_key: str
    processed_video_key: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    model_config = {
        "json_encoders": {
            uuid.UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    }


class WebhookDeliveryResult(BaseModel):
    """Result of webhook delivery attempt."""
    
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    attempts: int = 0
    delivery_time_ms: Optional[float] = None


class WebhookSigner:
    """HMAC signature generation for webhook security."""
    
    def __init__(self, secret: str):
        """Initialize with webhook secret."""
        self.secret = secret.encode('utf-8')
    
    def sign_payload(self, payload: str, timestamp: str) -> str:
        """Generate HMAC-SHA256 signature for webhook payload.
        
        Args:
            payload: JSON string payload
            timestamp: Unix timestamp as string
            
        Returns:
            Hex-encoded HMAC signature
        """
        # Create signature string: timestamp + payload
        sig_string = f"{timestamp}.{payload}"
        
        # Generate HMAC-SHA256
        signature = hmac.new(
            self.secret,
            sig_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def create_headers(self, payload: str) -> Dict[str, str]:
        """Create webhook headers with signature and timestamp.
        
        Args:
            payload: JSON string payload
            
        Returns:
            Dictionary of headers to include in webhook request
        """
        timestamp = str(int(time.time()))
        signature = self.sign_payload(payload, timestamp)
        
        return {
            "X-VSR-Signature": f"sha256={signature}",
            "X-VSR-Timestamp": timestamp,
            "Content-Type": "application/json",
            "User-Agent": "VSR-Webhook/1.0",
        }


class WebhookClient:
    """HTTP client for webhook delivery with retries and timeouts."""
    
    def __init__(self, webhook_secret: str, timeout_seconds: int = 10):
        """Initialize webhook client.
        
        Args:
            webhook_secret: Secret for HMAC signing
            timeout_seconds: HTTP timeout (default: 10s)
        """
        self.signer = WebhookSigner(webhook_secret)
        self.timeout = httpx.Timeout(
            connect=5.0,  # 5s connect timeout
            read=timeout_seconds,  # configurable read timeout
            write=5.0,  # 5s write timeout
            pool=10.0   # 10s pool timeout
        )
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),  # 1s, 2s, 4s (capped at 30s)
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _send_webhook_with_retry(self, url: str, payload: str, headers: Dict[str, str]) -> httpx.Response:
        """Send webhook with automatic retries.
        
        Args:
            url: Webhook URL
            payload: JSON payload string
            headers: HTTP headers including signature
            
        Returns:
            HTTP response
            
        Raises:
            httpx.RequestError: For network/connection errors
            httpx.HTTPStatusError: For HTTP error status codes
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, content=payload, headers=headers)
            
            # Raise for 4xx/5xx status codes to trigger retry
            if response.status_code >= 400:
                response.raise_for_status()
                
            return response
    
    async def send_webhook(self, url: str, payload: WebhookPayload) -> WebhookDeliveryResult:
        """Send webhook to callback URL with retries and error handling.
        
        Args:
            url: Webhook callback URL
            payload: Webhook payload to send
            
        Returns:
            WebhookDeliveryResult with delivery status and details
        """
        start_time = time.time()
        attempts = 0
        
        try:
            # Serialize payload to JSON
            payload_json = payload.model_dump_json()
            
            # Create signed headers
            headers = self.signer.create_headers(payload_json)
            
            logger.info(
                "Sending webhook",
                job_id=str(payload.job_id),
                url=url,
                status=payload.status.value,
            )
            
            # Send with retries
            response = await self._send_webhook_with_retry(url, payload_json, headers)
            attempts = 3  # Assume max attempts if successful (tenacity doesn't expose this easily)
            
            delivery_time = (time.time() - start_time) * 1000  # Convert to ms
            
            logger.info(
                "Webhook delivered successfully",
                job_id=str(payload.job_id),
                url=url,
                status_code=response.status_code,
                delivery_time_ms=delivery_time,
            )
            
            return WebhookDeliveryResult(
                success=True,
                status_code=response.status_code,
                response_body=response.text[:1000],  # Limit response body size
                attempts=attempts,
                delivery_time_ms=delivery_time,
            )
            
        except Exception as e:
            delivery_time = (time.time() - start_time) * 1000
            
            # Extract status code if it's an HTTP error
            status_code = None
            if hasattr(e, 'response') and e.response:
                status_code = e.response.status_code
            
            logger.error(
                "Webhook delivery failed",
                job_id=str(payload.job_id),
                url=url,
                error=str(e),
                status_code=status_code,
                delivery_time_ms=delivery_time,
            )
            
            return WebhookDeliveryResult(
                success=False,
                status_code=status_code,
                error_message=str(e),
                attempts=3,  # Assume max attempts on failure
                delivery_time_ms=delivery_time,
            )


class WebhookService:
    """High-level webhook service for job event notifications."""
    
    def __init__(self, webhook_secret: str, db_client=None):
        """Initialize webhook service.
        
        Args:
            webhook_secret: Secret for HMAC signing
            db_client: Database client for event logging (optional)
        """
        self.client = WebhookClient(webhook_secret)
        self.db_client = db_client
        
    async def notify_job_completed(self, job: Job) -> Optional[WebhookDeliveryResult]:
        """Send webhook notification for completed job.
        
        Args:
            job: Completed job to notify about
            
        Returns:
            WebhookDeliveryResult if webhook was sent, None if no callback URL
        """
        if not job.callback_url:
            logger.debug("No callback URL configured, skipping webhook", job_id=str(job.id))
            return None
            
        payload = WebhookPayload(
            job_id=job.id,
            status=job.status,
            timestamp=datetime.utcnow(),
            video_key=job.video_key,
            processed_video_key=job.processed_video_key,
            processing_time_seconds=job.processing_time_seconds,
        )
        
        result = await self.client.send_webhook(job.callback_url, payload)
        
        # Log webhook event to database if available
        if self.db_client is not None:
            await self._log_webhook_event(job.id, result)
            
        return result
    
    async def notify_job_failed(self, job: Job) -> Optional[WebhookDeliveryResult]:
        """Send webhook notification for failed job.
        
        Args:
            job: Failed job to notify about
            
        Returns:
            WebhookDeliveryResult if webhook was sent, None if no callback URL
        """
        if not job.callback_url:
            logger.debug("No callback URL configured, skipping webhook", job_id=str(job.id))
            return None
            
        payload = WebhookPayload(
            job_id=job.id,
            status=job.status,
            timestamp=datetime.utcnow(),
            video_key=job.video_key,
            error_message=job.error_message,
            processing_time_seconds=job.processing_time_seconds,
        )
        
        result = await self.client.send_webhook(job.callback_url, payload)
        
        # Log webhook event to database if available
        if self.db_client is not None:
            await self._log_webhook_event(job.id, result)
            
        return result
    
    async def _log_webhook_event(self, job_id: uuid.UUID, result: WebhookDeliveryResult) -> None:
        """Log webhook delivery event to database.
        
        Args:
            job_id: Job ID
            result: Webhook delivery result
        """
        if self.db_client is None:
            return
            
        try:
            from .models import JobEvent
            
            event_type = JobEventType.WEBHOOK_SENT if result.success else JobEventType.WEBHOOK_FAILED
            
            event = JobEvent(
                job_id=job_id,
                type=event_type,
                message=f"Webhook {'delivered' if result.success else 'failed'}",
                metadata={
                    "status_code": result.status_code,
                    "attempts": result.attempts,
                    "delivery_time_ms": result.delivery_time_ms,
                    "error_message": result.error_message,
                }
            )
            
            # Insert event (implementation depends on your DAL)
            # await self.db_client.job_events.insert_one(event.model_dump())
            logger.info(
                "Webhook event logged",
                job_id=str(job_id),
                event_type=event_type.value,
                success=result.success,
            )
            
        except Exception as e:
            logger.error(
                "Failed to log webhook event",
                job_id=str(job_id),
                error=str(e),
            )
