"""Worker webhook integration for job completion notifications."""

import asyncio
import os
from typing import Optional

import structlog
from vsr_shared.db.client import get_db
from vsr_shared.models import Job, JobStatus
from vsr_shared.webhooks import WebhookService

logger = structlog.get_logger(__name__)


class WorkerWebhookNotifier:
    """Webhook notifier for worker job completion events."""
    
    def __init__(self):
        """Initialize webhook notifier with configuration from environment."""
        self.webhook_secret = os.getenv("WEBHOOK_SECRET", "default-webhook-secret-change-in-production")
        self.webhook_service = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize webhook service with database connection."""
        if self._initialized:
            return
            
        try:
            db_client = await get_db()
            self.webhook_service = WebhookService(self.webhook_secret, db_client)
            self._initialized = True
            logger.info("Webhook notifier initialized")
        except Exception as e:
            logger.error("Failed to initialize webhook notifier", error=str(e))
            # Continue without webhooks rather than failing
            self.webhook_service = WebhookService(self.webhook_secret)
            self._initialized = True
    
    async def notify_job_completed(self, job: Job) -> None:
        """Notify webhook of job completion.
        
        Args:
            job: Completed job
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.webhook_service:
            logger.warning("Webhook service not available, skipping notification", job_id=str(job.id))
            return
            
        try:
            result = await self.webhook_service.notify_job_completed(job)
            if result:
                logger.info(
                    "Job completion webhook sent",
                    job_id=str(job.id),
                    success=result.success,
                    status_code=result.status_code,
                )
        except Exception as e:
            logger.error(
                "Failed to send job completion webhook",
                job_id=str(job.id),
                error=str(e),
            )
    
    async def notify_job_failed(self, job: Job) -> None:
        """Notify webhook of job failure.
        
        Args:
            job: Failed job
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.webhook_service:
            logger.warning("Webhook service not available, skipping notification", job_id=str(job.id))
            return
            
        try:
            result = await self.webhook_service.notify_job_failed(job)
            if result:
                logger.info(
                    "Job failure webhook sent",
                    job_id=str(job.id),
                    success=result.success,
                    status_code=result.status_code,
                )
        except Exception as e:
            logger.error(
                "Failed to send job failure webhook",
                job_id=str(job.id),
                error=str(e),
            )


# Global webhook notifier instance
webhook_notifier = WorkerWebhookNotifier()


async def notify_job_completion(job: Job) -> None:
    """Convenience function to notify job completion.
    
    Args:
        job: Job that completed or failed
    """
    if job.status == JobStatus.COMPLETED:
        await webhook_notifier.notify_job_completed(job)
    elif job.status == JobStatus.FAILED:
        await webhook_notifier.notify_job_failed(job)
    else:
        logger.debug(
            "Job status not final, skipping webhook notification",
            job_id=str(job.id),
            status=job.status.value,
        )
