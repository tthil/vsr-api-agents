"""
Webhook notification system for job completion callbacks.
Handles secure webhook delivery with retries and validation.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import hmac

import httpx
from vsr_shared.models import Job, JobStatus


logger = logging.getLogger(__name__)


class WebhookNotifier:
    """
    Webhook notification system for job completion callbacks.
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: int = 5):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def send_completion_notification(self, job: Job) -> bool:
        """
        Send job completion notification to callback URL.
        
        Args:
            job: Completed job to notify about
            
        Returns:
            True if notification was sent successfully
        """
        if not job.callback_url:
            logger.debug(f"No callback URL for job {job.id}, skipping notification")
            return True
        
        try:
            payload = self._create_webhook_payload(job)
            headers = self._create_webhook_headers(payload)
            
            success = await self._send_with_retries(
                job.callback_url,
                payload,
                headers
            )
            
            if success:
                logger.info(f"Successfully sent webhook notification for job {job.id}")
            else:
                logger.error(f"Failed to send webhook notification for job {job.id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending webhook notification for job {job.id}: {e}")
            return False
    
    def _create_webhook_payload(self, job: Job) -> Dict[str, Any]:
        """
        Create webhook payload from job data.
        
        Args:
            job: Job to create payload for
            
        Returns:
            Webhook payload dictionary
        """
        payload = {
            "event": "job.completed" if job.status == JobStatus.COMPLETED else "job.failed",
            "timestamp": datetime.utcnow().isoformat(),
            "job": {
                "id": str(job.id),
                "status": job.status.value,
                "mode": job.mode.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "processing_time_seconds": job.processing_time_seconds
            }
        }
        
        # Add processed video information if completed successfully
        if job.status == JobStatus.COMPLETED and job.processed_video_key:
            payload["job"]["processed_video_key"] = job.processed_video_key
        
        # Add error information if failed
        if job.status == JobStatus.FAILED and job.error_message:
            payload["job"]["error_message"] = job.error_message
        
        return payload
    
    def _create_webhook_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Create webhook headers including signature.
        
        Args:
            payload: Webhook payload
            
        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VSR-API-Webhook/1.0",
            "X-VSR-Event": payload["event"],
            "X-VSR-Timestamp": payload["timestamp"]
        }
        
        # Add signature if webhook secret is configured
        webhook_secret = os.getenv("VSR_WEBHOOK_SECRET")
        if webhook_secret:
            signature = self._generate_signature(payload, webhook_secret)
            headers["X-VSR-Signature"] = signature
        
        return headers
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """
        Generate HMAC signature for webhook payload.
        
        Args:
            payload: Webhook payload
            secret: Webhook secret key
            
        Returns:
            HMAC signature
        """
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    async def _send_with_retries(self, 
                                url: str, 
                                payload: Dict[str, Any], 
                                headers: Dict[str, str]) -> bool:
        """
        Send webhook with retry logic.
        
        Args:
            url: Webhook URL
            payload: Payload to send
            headers: Request headers
            
        Returns:
            True if sent successfully
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                
                # Consider 2xx status codes as success
                if 200 <= response.status_code < 300:
                    logger.debug(f"Webhook sent successfully (attempt {attempt + 1}): {response.status_code}")
                    return True
                
                logger.warning(f"Webhook failed with status {response.status_code} (attempt {attempt + 1})")
                
                # Don't retry for client errors (4xx)
                if 400 <= response.status_code < 500:
                    logger.error(f"Client error, not retrying: {response.status_code}")
                    return False
                
            except httpx.TimeoutException:
                logger.warning(f"Webhook timeout (attempt {attempt + 1})")
            except httpx.RequestError as e:
                logger.warning(f"Webhook request error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected webhook error (attempt {attempt + 1}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        return False
    
    async def validate_webhook_url(self, url: str) -> bool:
        """
        Validate webhook URL by sending a test ping.
        
        Args:
            url: Webhook URL to validate
            
        Returns:
            True if URL is valid and reachable
        """
        try:
            test_payload = {
                "event": "webhook.test",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "This is a test webhook from VSR API"
            }
            
            headers = self._create_webhook_headers(test_payload)
            
            response = await self.client.post(
                url,
                json=test_payload,
                headers=headers
            )
            
            return 200 <= response.status_code < 300
            
        except Exception as e:
            logger.error(f"Webhook validation failed for {url}: {e}")
            return False
    
    async def cleanup(self):
        """Clean up HTTP client resources."""
        await self.client.aclose()


# Utility functions
def validate_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Validate webhook signature.
    
    Args:
        payload: Raw payload bytes
        signature: Received signature
        secret: Webhook secret
        
    Returns:
        True if signature is valid
    """
    if not signature.startswith('sha256='):
        return False
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    received_signature = signature[7:]  # Remove 'sha256=' prefix
    
    return hmac.compare_digest(expected_signature, received_signature)


async def send_test_webhook(url: str) -> bool:
    """
    Send a test webhook to validate URL.
    
    Args:
        url: Webhook URL to test
        
    Returns:
        True if test was successful
    """
    notifier = WebhookNotifier()
    try:
        return await notifier.validate_webhook_url(url)
    finally:
        await notifier.cleanup()
