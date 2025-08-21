"""RabbitMQ queue handler for VSR worker."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from vsr_shared.logging import get_logger
from vsr_shared.models import JobMessage, JobMessageResponse, JobStatus
from vsr_shared.queue.consumer import create_job_consumer
from vsr_shared.queue.client import get_rabbitmq_client
from vsr_shared.db.client import get_mongodb_client
from vsr_shared.db.dal import JobsDAL

logger = get_logger(__name__)


class WorkerQueueHandler:
    """Handles RabbitMQ job processing for the worker."""

    def __init__(self, worker_id: Optional[str] = None):
        """
        Initialize worker queue handler.

        Args:
            worker_id: Unique worker identifier
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.consumer = None
        self.running = False

    async def start(self) -> None:
        """Start consuming messages from the job queue."""
        if self.running:
            logger.warning("Worker queue handler is already running")
            return

        try:
            logger.info(f"Starting worker queue handler {self.worker_id}")
            
            # Create consumer with message handler
            self.consumer = await create_job_consumer(
                message_handler=self._handle_job_message
            )
            
            # Start consuming messages
            await self.consumer.start_consuming()
            self.running = True
            
            logger.info(f"Worker {self.worker_id} started consuming messages")
            
        except Exception as e:
            logger.error(f"Failed to start worker queue handler: {e}")
            raise

    async def stop(self) -> None:
        """Stop consuming messages."""
        if not self.running:
            return

        try:
            logger.info(f"Stopping worker queue handler {self.worker_id}")
            
            if self.consumer:
                await self.consumer.stop_consuming()
                await self.consumer.close()
                
            self.running = False
            logger.info(f"Worker {self.worker_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping worker queue handler: {e}")

    async def _handle_job_message(self, job_message: JobMessage) -> JobMessageResponse:
        """
        Handle incoming job message.

        Args:
            job_message: Job message to process

        Returns:
            JobMessageResponse: Processing result
        """
        job_id = job_message.job_id
        start_time = datetime.now(timezone.utc)
        
        logger.info(
            f"Worker {self.worker_id} processing job {job_id} "
            f"with mode {job_message.mode.value}"
        )

        try:
            # Update job status to processing
            await self._update_job_status(job_id, JobStatus.PROCESSING)
            
            # Simulate video processing (replace with actual processing logic)
            processing_result = await self._process_video(job_message)
            
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
                f"Worker {self.worker_id} completed job {job_id} "
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
            logger.error(f"Worker {self.worker_id} failed to process job {job_id}: {e}")
            
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

    async def _process_video(self, job_message: JobMessage) -> dict:
        """
        Process video to remove subtitles.
        
        This is a placeholder implementation. In a real worker, this would:
        1. Download video from S3/Spaces using job_message.video_key_in
        2. Apply subtitle removal using the specified mode
        3. Upload processed video to S3/Spaces
        4. Return the processed video key
        
        Args:
            job_message: Job message with processing details
            
        Returns:
            dict: Processing result with processed_video_key
        """
        # Simulate processing time based on mode
        processing_times = {
            "sttn": 30,      # STTN is faster
            "lama": 45,      # LAMA is medium
            "propainter": 60 # ProPainter is slower
        }
        
        processing_time = processing_times.get(job_message.mode.value, 30)
        
        # Simulate processing with progress updates
        for i in range(5):
            await asyncio.sleep(processing_time / 5)
            progress = (i + 1) * 20
            await self._update_job_progress(job_message.job_id, progress)
            
        # Generate processed video key
        from vsr_shared.spaces_keys import get_processed_key
        processed_key = get_processed_key(job_message.job_id)
        
        return {
            "processed_video_key": processed_key,
            "processing_mode": job_message.mode.value,
        }

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
            
            # Add progress event
            from vsr_shared.models import JobEvent, JobEventType
            event = JobEvent(
                job_id=job_id,
                type=JobEventType.JOB_PROGRESS,
                progress=progress,
                message=f"Processing progress: {progress}%",
                metadata={"worker_id": self.worker_id}
            )
            await jobs_dal.add_event(event)
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id} progress: {e}")

    @property
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self.running


async def create_worker_queue_handler(worker_id: Optional[str] = None) -> WorkerQueueHandler:
    """
    Create and initialize worker queue handler.

    Args:
        worker_id: Optional worker identifier

    Returns:
        WorkerQueueHandler instance
    """
    handler = WorkerQueueHandler(worker_id=worker_id)
    return handler
