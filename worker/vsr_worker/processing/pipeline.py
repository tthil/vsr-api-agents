"""
Video processing pipeline for subtitle removal.
Coordinates model inference, I/O operations, and progress tracking.
"""
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable, AsyncGenerator
from contextlib import asynccontextmanager
import torch
import cv2
import numpy as np
from datetime import datetime

from vsr_shared.models import Job, JobStatus, ProcessingMode
from vsr_shared.db.client import get_database
from vsr_shared.spaces import get_spaces_client
from vsr_worker.models import load_sttn_model, load_lama_model, load_propainter_model
from vsr_worker.io.video_io import VideoReader, VideoWriter, VideoMetadata
from vsr_worker.io.frame_processor import FrameProcessor


logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Custom exception for processing errors."""
    pass


class VideoProcessingPipeline:
    """
    Main video processing pipeline that coordinates all processing steps.
    """
    
    def __init__(self):
        self.db = None
        self.spaces_client = None
        self.models = {}
        self.temp_dir = None
        
    async def initialize(self):
        """Initialize pipeline components."""
        try:
            self.db = await get_database()
            self.spaces_client = get_spaces_client()
            
            # Create temporary directory for processing
            self.temp_dir = Path(tempfile.mkdtemp(prefix="vsr_processing_"))
            logger.info(f"Initialized pipeline with temp dir: {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise ProcessingError(f"Pipeline initialization failed: {e}")
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        try:
            # Clean up temporary directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp dir: {self.temp_dir}")
                
            # Unload models to free GPU memory
            for model_name, model in self.models.items():
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                del model
                logger.info(f"Unloaded model: {model_name}")
            
            self.models.clear()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
    
    async def load_model(self, mode: ProcessingMode) -> Any:
        """
        Load the appropriate AI model for processing.
        
        Args:
            mode: Processing mode (STTN, LAMA, PROPAINTER)
            
        Returns:
            Loaded model instance
        """
        if mode.value in self.models:
            return self.models[mode.value]
        
        try:
            logger.info(f"Loading model: {mode.value}")
            
            if mode == ProcessingMode.STTN:
                model = await load_sttn_model()
            elif mode == ProcessingMode.LAMA:
                model = await load_lama_model()
            elif mode == ProcessingMode.PROPAINTER:
                model = await load_propainter_model()
            else:
                raise ProcessingError(f"Unsupported processing mode: {mode.value}")
            
            self.models[mode.value] = model
            logger.info(f"Successfully loaded model: {mode.value}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {mode.value}: {e}")
            raise ProcessingError(f"Model loading failed: {e}")
    
    async def download_video(self, job: Job) -> Path:
        """
        Download video from storage to local temp directory.
        
        Args:
            job: Job containing video key
            
        Returns:
            Path to downloaded video file
        """
        try:
            video_path = self.temp_dir / f"input_{job.id}.mp4"
            
            logger.info(f"Downloading video: {job.video_key}")
            
            # Download from Spaces
            with open(video_path, 'wb') as f:
                await self.spaces_client.download_fileobj(
                    job.video_key,
                    f
                )
            
            logger.info(f"Downloaded video to: {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"Failed to download video {job.video_key}: {e}")
            raise ProcessingError(f"Video download failed: {e}")
    
    async def upload_processed_video(self, job: Job, processed_path: Path) -> str:
        """
        Upload processed video to storage.
        
        Args:
            job: Job to update with processed video key
            processed_path: Path to processed video file
            
        Returns:
            Storage key for processed video
        """
        try:
            # Generate processed video key
            date_str = datetime.now().strftime("%Y%m%d")
            processed_key = f"processed/{date_str}/{job.id}/video.mp4"
            
            logger.info(f"Uploading processed video: {processed_key}")
            
            # Upload to Spaces
            with open(processed_path, 'rb') as f:
                await self.spaces_client.upload_fileobj(
                    f,
                    processed_key,
                    ExtraArgs={
                        'ContentType': 'video/mp4',
                        'ServerSideEncryption': 'AES256',
                        'ACL': 'private'
                    }
                )
            
            logger.info(f"Uploaded processed video: {processed_key}")
            return processed_key
            
        except Exception as e:
            logger.error(f"Failed to upload processed video: {e}")
            raise ProcessingError(f"Video upload failed: {e}")
    
    async def update_job_progress(self, job_id: str, progress: int, status: Optional[JobStatus] = None):
        """
        Update job progress in database.
        
        Args:
            job_id: Job ID to update
            progress: Progress percentage (0-100)
            status: Optional status update
        """
        try:
            update_data = {
                "progress": progress,
                "updated_at": datetime.utcnow()
            }
            
            if status:
                update_data["status"] = status.value
                if status == JobStatus.IN_PROGRESS and progress == 0:
                    update_data["started_at"] = datetime.utcnow()
                elif status == JobStatus.COMPLETED:
                    update_data["completed_at"] = datetime.utcnow()
            
            await self.db.jobs.update_one(
                {"_id": job_id},
                {"$set": update_data}
            )
            
            logger.info(f"Updated job {job_id}: progress={progress}%, status={status}")
            
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")
    
    async def process_video(
        self, 
        job: Job, 
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> str:
        """
        Main video processing method.
        
        Args:
            job: Job to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Storage key for processed video
        """
        try:
            logger.info(f"Starting video processing for job {job.id}")
            
            # Update job status to in-progress
            await self.update_job_progress(job.id, 0, JobStatus.IN_PROGRESS)
            
            # Step 1: Download video (10% progress)
            input_path = await self.download_video(job)
            await self.update_job_progress(job.id, 10)
            if progress_callback:
                progress_callback(10)
            
            # Step 2: Load model (20% progress)
            model = await self.load_model(job.mode)
            await self.update_job_progress(job.id, 20)
            if progress_callback:
                progress_callback(20)
            
            # Step 3: Process video frames (20% -> 80% progress)
            output_path = await self._process_frames(
                input_path, 
                model, 
                job,
                progress_callback
            )
            
            # Step 4: Upload processed video (90% progress)
            processed_key = await self.upload_processed_video(job, output_path)
            await self.update_job_progress(job.id, 90)
            if progress_callback:
                progress_callback(90)
            
            # Step 5: Update job completion (100% progress)
            await self.db.jobs.update_one(
                {"_id": job.id},
                {
                    "$set": {
                        "processed_video_key": processed_key,
                        "progress": 100,
                        "status": JobStatus.COMPLETED.value,
                        "completed_at": datetime.utcnow(),
                        "processing_time_seconds": (
                            datetime.utcnow() - job.started_at
                        ).total_seconds() if job.started_at else None
                    }
                }
            )
            
            if progress_callback:
                progress_callback(100)
            
            logger.info(f"Completed video processing for job {job.id}")
            return processed_key
            
        except Exception as e:
            logger.error(f"Video processing failed for job {job.id}: {e}")
            
            # Update job status to failed
            await self.db.jobs.update_one(
                {"_id": job.id},
                {
                    "$set": {
                        "status": JobStatus.FAILED.value,
                        "error_message": str(e),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            raise ProcessingError(f"Video processing failed: {e}")
    
    async def _process_frames(
        self, 
        input_path: Path, 
        model: Any, 
        job: Job,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Path:
        """
        Process video frames using the loaded model.
        
        Args:
            input_path: Path to input video
            model: Loaded AI model
            job: Job with processing parameters
            progress_callback: Progress callback function
            
        Returns:
            Path to processed video file
        """
        output_path = self.temp_dir / f"output_{job.id}.mp4"
        
        try:
            # Initialize video I/O
            reader = VideoReader(input_path)
            metadata = await reader.get_metadata()
            
            writer = VideoWriter(
                output_path,
                metadata.fps,
                (metadata.width, metadata.height),
                metadata.codec
            )
            
            # Initialize frame processor
            processor = FrameProcessor(model, job.mode)
            
            # Process frames in batches
            total_frames = metadata.frame_count
            processed_frames = 0
            batch_size = self._get_batch_size(job.mode)
            
            logger.info(f"Processing {total_frames} frames in batches of {batch_size}")
            
            async for frame_batch in reader.read_frames_batch(batch_size):
                # Process batch with model
                processed_batch = await processor.process_batch(
                    frame_batch,
                    job.subtitle_area
                )
                
                # Write processed frames
                for frame in processed_batch:
                    await writer.write_frame(frame)
                
                processed_frames += len(frame_batch)
                
                # Update progress (20% -> 80% range)
                progress = 20 + int((processed_frames / total_frames) * 60)
                await self.update_job_progress(job.id, progress)
                
                if progress_callback:
                    progress_callback(progress)
                
                logger.debug(f"Processed {processed_frames}/{total_frames} frames")
            
            # Finalize video
            await writer.finalize()
            await reader.close()
            
            logger.info(f"Frame processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise ProcessingError(f"Frame processing failed: {e}")
    
    def _get_batch_size(self, mode: ProcessingMode) -> int:
        """
        Get optimal batch size based on processing mode and available GPU memory.
        
        Args:
            mode: Processing mode
            
        Returns:
            Optimal batch size
        """
        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if mode == ProcessingMode.STTN:
                # STTN needs more memory for temporal processing
                return max(1, min(8, int(gpu_memory_gb / 4)))
            elif mode == ProcessingMode.LAMA:
                # LAMA is more memory efficient
                return max(1, min(16, int(gpu_memory_gb / 2)))
            elif mode == ProcessingMode.PROPAINTER:
                # ProPainter needs significant memory for flow estimation
                return max(1, min(4, int(gpu_memory_gb / 6)))
        
        return 1  # Fallback for CPU processing
    
    @asynccontextmanager
    async def processing_context(self):
        """Context manager for safe pipeline operations."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()


class BatchProcessor:
    """
    Batch processor for handling multiple jobs efficiently.
    """
    
    def __init__(self, max_concurrent_jobs: int = 2):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = set()
        self.pipeline = VideoProcessingPipeline()
    
    async def process_job_batch(self, jobs: list[Job]) -> Dict[str, Any]:
        """
        Process multiple jobs with concurrency control.
        
        Args:
            jobs: List of jobs to process
            
        Returns:
            Processing results summary
        """
        results = {
            "completed": [],
            "failed": [],
            "total_processing_time": 0
        }
        
        start_time = datetime.utcnow()
        
        # Process jobs with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        async def process_single_job(job: Job):
            async with semaphore:
                try:
                    async with self.pipeline.processing_context() as pipeline:
                        processed_key = await pipeline.process_video(job)
                        results["completed"].append({
                            "job_id": job.id,
                            "processed_key": processed_key
                        })
                except Exception as e:
                    results["failed"].append({
                        "job_id": job.id,
                        "error": str(e)
                    })
        
        # Execute all jobs concurrently
        await asyncio.gather(*[process_single_job(job) for job in jobs])
        
        results["total_processing_time"] = (
            datetime.utcnow() - start_time
        ).total_seconds()
        
        return results


# Utility functions for pipeline operations
async def estimate_processing_time(job: Job) -> int:
    """
    Estimate processing time for a job based on video metadata and mode.
    
    Args:
        job: Job to estimate processing time for
        
    Returns:
        Estimated processing time in seconds
    """
    try:
        # Base processing times per mode (seconds per minute of video)
        base_times = {
            ProcessingMode.LAMA: 30,      # Fastest
            ProcessingMode.STTN: 45,      # Medium
            ProcessingMode.PROPAINTER: 60  # Highest quality, slowest
        }
        
        # For estimation, assume average 2-minute video
        # In production, this would analyze actual video metadata
        estimated_duration_minutes = 2
        base_time = base_times.get(job.mode, 45)
        
        return int(estimated_duration_minutes * base_time)
        
    except Exception:
        # Fallback estimate
        return 120


async def get_processing_queue_info() -> Dict[str, Any]:
    """
    Get information about the current processing queue.
    
    Returns:
        Queue information including length and estimated wait time
    """
    try:
        db = await get_database()
        
        # Count pending jobs
        pending_count = await db.jobs.count_documents({
            "status": JobStatus.PENDING.value
        })
        
        # Count in-progress jobs
        in_progress_count = await db.jobs.count_documents({
            "status": JobStatus.IN_PROGRESS.value
        })
        
        # Estimate average processing time
        avg_processing_time = 180  # 3 minutes average
        
        return {
            "pending_jobs": pending_count,
            "in_progress_jobs": in_progress_count,
            "estimated_wait_time": pending_count * avg_processing_time,
            "queue_length": pending_count + in_progress_count
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue info: {e}")
        return {
            "pending_jobs": 0,
            "in_progress_jobs": 0,
            "estimated_wait_time": 0,
            "queue_length": 0
        }
