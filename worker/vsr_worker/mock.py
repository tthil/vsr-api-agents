"""Mock worker implementation for local development."""

import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict

from vsr_shared.logging import get_logger
from vsr_worker.config import Settings

logger = get_logger(__name__)


def run_mock_worker(settings: Settings) -> None:
    """
    Run the mock worker that simulates video processing.
    
    Args:
        settings: Worker configuration settings
    """
    logger.info("Starting mock worker", settings=settings.model_dump())
    
    # In a real implementation, this would connect to RabbitMQ and MongoDB
    # For now, we'll just simulate the processing loop
    
    while True:
        try:
            # Simulate waiting for a job
            logger.info("Waiting for job...")
            time.sleep(settings.worker_polling_interval)
            
            # Simulate processing a job
            job_id = str(uuid.uuid4())
            process_mock_job(job_id, settings)
            
        except KeyboardInterrupt:
            logger.info("Mock worker stopped by user")
            break
        except Exception as e:
            logger.exception("Error in mock worker", error=str(e))
            time.sleep(settings.worker_polling_interval)


def process_mock_job(job_id: str, settings: Settings) -> None:
    """
    Process a mock job.
    
    Args:
        job_id: Job ID
        settings: Worker configuration settings
    """
    logger.info("Processing mock job", job_id=job_id)
    
    # Simulate job processing with progress updates
    total_steps = settings.mock_processing_time
    
    for step in range(total_steps + 1):
        progress = int(step * 100 / total_steps)
        
        # Log progress
        logger.info(
            "Job progress",
            job_id=job_id,
            progress=progress,
            step=step,
            total_steps=total_steps,
        )
        
        # In a real implementation, this would update the job status in MongoDB
        
        # Simulate processing time
        time.sleep(settings.mock_progress_interval)
    
    # Simulate job completion
    logger.info("Job completed", job_id=job_id)
    
    # In a real implementation, this would:
    # 1. Update the job status in MongoDB
    # 2. Upload the processed video to Spaces
    # 3. Send a webhook notification if configured
