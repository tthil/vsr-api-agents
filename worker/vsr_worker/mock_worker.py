"""Mock worker implementation for local development and testing."""

import asyncio
import uuid
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

from vsr_shared.logging import get_logger
from vsr_shared.models import JobMessage, JobMessageResponse, JobStatus, ProcessingMode
from vsr_shared.queue.consumer import create_job_consumer
from vsr_shared.spaces import get_spaces_client
from vsr_shared.spaces_keys import get_processed_key
from vsr_shared.db.client import get_mongodb_client
from vsr_shared.db.dal import JobsDAL, JobEvent, JobEventType

logger = get_logger(__name__)


class MockWorker:
    """Mock worker for local development and testing."""

    def __init__(self, worker_id: Optional[str] = None):
        """
        Initialize mock worker.

        Args:
            worker_id: Unique worker identifier
        """
        self.worker_id = worker_id or f"mock-worker-{uuid.uuid4().hex[:8]}"
        self.consumer = None
        self.running = False
        self.processing_times = {
            ProcessingMode.STTN: (15, 30),      # 15-30 seconds
            ProcessingMode.LAMA: (20, 40),      # 20-40 seconds  
            ProcessingMode.PROPAINTER: (30, 60) # 30-60 seconds
        }

    async def start(self) -> None:
        """Start mock worker to consume messages."""
        if self.running:
            logger.warning("Mock worker is already running")
            return

        try:
            logger.info(f"Starting mock worker {self.worker_id}")
            
            # Create consumer with message handler
            self.consumer = await create_job_consumer(
                message_handler=self._handle_job_message
            )
            
            # Start consuming messages
            await self.consumer.start_consuming()
            self.running = True
            
            logger.info(f"Mock worker {self.worker_id} started consuming messages")
            
        except Exception as e:
            logger.error(f"Failed to start mock worker: {e}")
            raise

    async def stop(self) -> None:
        """Stop mock worker."""
        if not self.running:
            return

        try:
            logger.info(f"Stopping mock worker {self.worker_id}")
            
            if self.consumer:
                await self.consumer.stop_consuming()
                await self.consumer.close()
                
            self.running = False
            logger.info(f"Mock worker {self.worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping mock worker: {e}")

    async def _handle_job_message(self, job_message: JobMessage) -> JobMessageResponse:
        """
        Handle incoming job message with mock processing.

        Args:
            job_message: Job message to process

        Returns:
            JobMessageResponse: Processing result
        """
        job_id = job_message.job_id
        start_time = datetime.now(timezone.utc)
        
        logger.info(
            f"Mock worker {self.worker_id} processing job {job_id} "
            f"with mode {job_message.mode.value}"
        )

        try:
            # Update job status to processing
            await self._update_job_status(job_id, JobStatus.PROCESSING)
            
            # Simulate processing with progress updates
            processing_result = await self._simulate_processing(job_message)
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update job status to completed
            await self._update_job_status(
                job_id, 
                JobStatus.COMPLETED,
                processed_video_key=processing_result.get("processed_video_key"),
                processing_time_seconds=processing_time
            )
            
            logger.info(
                f"Mock worker {self.worker_id} completed job {job_id} "
                f"in {processing_time:.2f}s"
            )
            
            return JobMessageResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                processed_video_key=processing_result.get("processed_video_key"),
                processing_time_seconds=processing_time,
                trace_id=job_message.trace_id,
            )
            
        except Exception as e:
            logger.error(f"Mock worker {self.worker_id} failed to process job {job_id}: {e}")
            
            # Update job status to failed
            await self._update_job_status(
                job_id, 
                JobStatus.FAILED,
                error_message=str(e)
            )
            
            return JobMessageResponse(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                trace_id=job_message.trace_id,
            )

    async def _simulate_processing(self, job_message: JobMessage) -> Dict[str, Any]:
        """
        Simulate video processing with realistic timing and progress updates.
        
        Args:
            job_message: Job message with processing details
            
        Returns:
            Dict: Processing result with processed_video_key
        """
        # Get processing time range for the mode
        min_time, max_time = self.processing_times.get(
            job_message.mode, 
            (20, 40)
        )
        
        # Random processing time within range
        total_processing_time = random.uniform(min_time, max_time)
        
        # Simulate processing with progress updates
        progress_steps = 10
        step_time = total_processing_time / progress_steps
        
        for i in range(progress_steps):
            await asyncio.sleep(step_time)
            progress = ((i + 1) / progress_steps) * 100
            await self._update_job_progress(job_message.job_id, progress)
            
            logger.debug(f"Job {job_message.job_id} progress: {progress:.1f}%")
        
        # Simulate creating processed video file
        processed_key = await self._create_mock_processed_video(job_message)
        
        return {
            "processed_video_key": processed_key,
            "processing_mode": job_message.mode.value,
            "mock_processing": True,
        }

    async def _create_mock_processed_video(self, job_message: JobMessage) -> str:
        """
        Create mock processed video by copying input video.
        
        Args:
            job_message: Job message
            
        Returns:
            Processed video key
        """
        try:
            # Generate processed video key
            processed_key = get_processed_key(job_message.job_id)
            
            # Get Spaces client
            spaces_client = get_spaces_client()
            
            # In a real implementation, we would:
            # 1. Download the input video from job_message.video_key_in
            # 2. Process it to remove subtitles
            # 3. Upload the processed video
            
            # For mock, we'll copy the input video to processed location
            # This simulates the processing pipeline without actual video processing
            
            try:
                # Copy input video to processed location
                copy_source = {
                    'Bucket': spaces_client.bucket_name,
                    'Key': job_message.video_key_in
                }
                
                spaces_client.client.copy_object(
                    CopySource=copy_source,
                    Bucket=spaces_client.bucket_name,
                    Key=processed_key,
                    ServerSideEncryption='AES256',
                    ACL='private'
                )
                
                logger.info(f"Mock processed video created: {processed_key}")
                
            except Exception as e:
                logger.warning(f"Could not copy video file (expected in local dev): {e}")
                # In local development with MinIO, the file might not exist
                # This is expected and doesn't affect the mock functionality
            
            return processed_key
            
        except Exception as e:
            logger.error(f"Error creating mock processed video: {e}")
            # Return a mock key even if upload fails
            return get_processed_key(job_message.job_id)

    async def _update_job_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        processed_video_key: Optional[str] = None,
        processing_time_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update job status in database."""
        try:
            # Get MongoDB client and DAL
            mongo_client = get_mongodb_client()
            db = mongo_client.get_database()
            jobs_dal = JobsDAL(db)
            
            # Update job status
            await jobs_dal.update_status(job_id, status)
            
            # Update additional fields if provided
            if processed_video_key:
                await jobs_dal.update_processed_video_key(job_id, processed_video_key)
                
            if processing_time_seconds:
                await jobs_dal.update_processing_time(job_id, processing_time_seconds)
                
            if error_message:
                await jobs_dal.update_error_message(job_id, error_message)
                
            # Add status change event
            event_messages = {
                JobStatus.PROCESSING: "Job processing started by mock worker",
                JobStatus.COMPLETED: "Job completed successfully by mock worker",
                JobStatus.FAILED: f"Job failed: {error_message}" if error_message else "Job failed",
            }
            
            event = JobEvent(
                job_id=job_id,
                type=JobEventType.JOB_STATUS_CHANGED,
                message=event_messages.get(status, f"Status changed to {status.value}"),
                metadata={
                    "worker_id": self.worker_id,
                    "worker_type": "mock",
                    "status": status.value,
                }
            )
            await jobs_dal.add_event(event)
                
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")

    async def _update_job_progress(self, job_id: uuid.UUID, progress: float) -> None:
        """Update job progress in database."""
        try:
            # Get MongoDB client and DAL
            mongo_client = get_mongodb_client()
            db = mongo_client.get_database()
            jobs_dal = JobsDAL(db)
            
            # Update progress
            await jobs_dal.update_progress(job_id, progress)
            
            # Add progress event every 20% or at completion
            if progress % 20 == 0 or progress >= 100:
                event = JobEvent(
                    job_id=job_id,
                    type=JobEventType.JOB_PROGRESS,
                    progress=progress,
                    message=f"Mock processing progress: {progress}%",
                    metadata={
                        "worker_id": self.worker_id,
                        "worker_type": "mock",
                    }
                )
                await jobs_dal.add_event(event)
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id} progress: {e}")

    @property
    def is_running(self) -> bool:
        """Check if mock worker is currently running."""
        return self.running

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on mock worker."""
        return {
            "healthy": True,
            "worker_id": self.worker_id,
            "worker_type": "mock",
            "running": self.running,
            "processing_times": {
                mode.value: f"{min_t}-{max_t}s" 
                for mode, (min_t, max_t) in self.processing_times.items()
            },
            "capabilities": ["video_processing_simulation", "progress_updates", "minIO_integration"],
        }


async def create_mock_worker(worker_id: Optional[str] = None) -> MockWorker:
    """
    Create and initialize mock worker.

    Args:
        worker_id: Optional worker identifier

    Returns:
        MockWorker instance
    """
    worker = MockWorker(worker_id=worker_id)
    return worker


async def run_mock_worker(worker_id: Optional[str] = None) -> None:
    """
    Run mock worker indefinitely.

    Args:
        worker_id: Optional worker identifier
    """
    worker = await create_mock_worker(worker_id)
    
    try:
        await worker.start()
        
        # Keep running until interrupted
        while worker.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Mock worker interrupted by user")
    except Exception as e:
        logger.error(f"Mock worker error: {e}")
    finally:
        await worker.stop()


if __name__ == "__main__":
    # Run mock worker when executed directly
    asyncio.run(run_mock_worker())
