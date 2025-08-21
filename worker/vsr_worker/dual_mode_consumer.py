"""
Dual-mode RabbitMQ consumer for processing video jobs.

Supports both CPU (local development) and GPU (production) processing modes
with automatic environment detection and adaptive model loading.
"""

import asyncio
import json
import uuid
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID
import numpy as np

import logging
from vsr_shared.models import Job, JobStatus
from vsr_shared.db.client import get_db
from vsr_shared.queue.client import RabbitMQClient, QUEUE_NAME
from vsr_shared.queue.topology import setup_queue_topology
from vsr_worker.webhooks import notify_job_completion

# Dual-mode imports (with fallbacks)
try:
    from vsr_worker.gpu.environment import WorkerEnvironment
    from vsr_worker.models.adaptive import AdaptiveModelFactory
    from vsr_worker.config.dual_mode import (
        get_config_manager, 
        print_config_summary, 
        validate_environment,
        WorkerMode
    )
    DUAL_MODE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dual-mode imports failed: {e}. Falling back to basic CPU processing.")
    DUAL_MODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DualModeVideoJobConsumer:
    """Dual-mode consumer for video processing jobs supporting CPU and GPU modes."""
    
    def __init__(self):
        """Initialize dual-mode consumer."""
        self.client = RabbitMQClient()
        self.running = False
        self.worker_env = None
        self.model_factory = None
        self.config_manager = get_config_manager()
        
    async def initialize(self) -> bool:
        """Initialize worker environment and models."""
        try:
            logger.info("Initializing dual-mode worker...")
            
            # Print configuration summary
            print_config_summary()
            
            # Validate environment configuration
            if not validate_environment():
                logger.error("Environment validation failed")
                return False
            
            # Initialize worker environment
            self.worker_env = WorkerEnvironment()
            env_info = await self.worker_env.initialize()
            
            logger.info("Worker environment initialized", **env_info)
            
            # Initialize model factory
            self.model_factory = AdaptiveModelFactory(self.worker_env)
            
            logger.info(f"Dual-mode worker initialized in {self.worker_env.mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            return False
    
    async def start_consuming(self):
        """Start consuming messages from the job queue."""
        if not await self.initialize():
            logger.error("Failed to initialize worker, exiting")
            return
        
        logger.info("Starting dual-mode job consumer...")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Set up queue topology
            await setup_queue_topology(self.client)
            
            # Start consuming
            await self.client.start_consuming(
                queue_name=QUEUE_NAME,
                callback=self._process_message
            )
            
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            await self.cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def _process_message(self, message: Dict[str, Any]) -> bool:
        """Process a job message from the queue."""
        try:
            job_id = message.get("job_id")
            if not job_id:
                logger.error("No job_id in message")
                return False
            
            bind_request_id(str(uuid.uuid4()))
            logger.info(f"Processing job: {job_id}")
            
            # Process the job
            success = await self.process_job(message)
            
            if success:
                logger.info(f"Job {job_id} completed successfully")
            else:
                logger.error(f"Job {job_id} failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return False
    
    async def process_job(self, job_data: Dict[str, Any]) -> bool:
        """Process a single video job using dual-mode architecture."""
        job_id = job_data.get("job_id")
        mode = job_data.get("mode", "sttn")
        
        db = await get_db()
        job = None
        
        try:
            # Update job status to processing
            await db.jobs.update_one(
                {"_id": UUID(job_id)},
                {
                    "$set": {
                        "status": JobStatus.PROCESSING.value,
                        "updated_at": datetime.utcnow(),
                        "started_at": datetime.utcnow(),
                        "worker_mode": self.worker_env.mode,
                        "processing_node": os.getenv("HOSTNAME", "unknown")
                    }
                }
            )
            
            # Log job event
            await self._log_job_event(
                job_id, 
                JobEventType.PROCESSING_STARTED, 
                {"worker_mode": self.worker_env.mode}
            )
            
            # Get processing configuration
            processing_config = self.config_manager.get_processing_config(
                WorkerMode(self.worker_env.mode)
            )
            
            logger.info(f"Job {job_id}: Starting {self.worker_env.mode} processing with {mode} model")
            processing_start = datetime.utcnow()
            
            # Load appropriate model
            model = await self.model_factory.create_model(mode)
            
            # Process video
            result = await self._process_video(job_data, model, processing_config)
            
            processing_end = datetime.utcnow()
            processing_time = (processing_end - processing_start).total_seconds()
            
            # Update job status to completed
            await db.jobs.update_one(
                {"_id": UUID(job_id)},
                {
                    "$set": {
                        "status": JobStatus.COMPLETED.value,
                        "updated_at": processing_end,
                        "completed_at": processing_end,
                        "processing_time_seconds": processing_time,
                        "processed_video_key": result.get("processed_video_key"),
                        "model_used": mode,
                        "quality_metrics": result.get("quality_metrics", {})
                    }
                }
            )
            
            # Log completion event
            await self._log_job_event(
                job_id, 
                JobEventType.PROCESSING_COMPLETED, 
                {
                    "processing_time": processing_time,
                    "model_used": mode,
                    "worker_mode": self.worker_env.mode
                }
            )
            
            # Fetch updated job for webhook notification
            job_doc = await db.jobs.find_one({"_id": UUID(job_id)})
            if job_doc:
                # Convert MongoDB document to Job model
                job_doc["id"] = job_doc.pop("_id")
                job = Job(**job_doc)
                
                # Send webhook notification
                await notify_job_completion(job)
            
            logger.info(f"Job {job_id} completed in {processing_time:.2f}s using {self.worker_env.mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Job {job_id} processing failed: {e}")
            
            # Update job status to failed
            await db.jobs.update_one(
                {"_id": UUID(job_id)},
                {
                    "$set": {
                        "status": JobStatus.FAILED.value,
                        "updated_at": datetime.utcnow(),
                        "error_message": str(e),
                        "failed_at": datetime.utcnow()
                    }
                }
            )
            
            # Log failure event
            await self._log_job_event(
                job_id, 
                JobEventType.PROCESSING_FAILED, 
                {"error": str(e), "worker_mode": self.worker_env.mode}
            )
            
            # Send failure webhook notification
            if job:
                await notify_job_completion(job)
            
            return False
    
    async def _process_video(
        self, 
        job_data: Dict[str, Any], 
        model, 
        processing_config
    ) -> Dict[str, Any]:
        """Process video using the loaded model."""
        job_id = job_data.get("job_id")
        video_key = job_data.get("video_key")
        subtitle_area = job_data.get("subtitle_area", {})
        
        # Get Spaces client
        spaces_client = get_spaces_client()
        
        try:
            # Download video from Spaces
            logger.info(f"Job {job_id}: Downloading video from {video_key}")
            
            # For now, simulate video processing
            # In a real implementation, you would:
            # 1. Download video from Spaces
            # 2. Extract frames
            # 3. Process frames with the model
            # 4. Reconstruct video
            # 5. Upload processed video
            
            # Simulate processing time based on mode
            if self.worker_env.mode == "cpu":
                # CPU processing is slower
                processing_time = min(processing_config.timeout_seconds, 120)
                await asyncio.sleep(min(processing_time / 10, 12))  # Simulate 10% of max time
            else:
                # GPU processing is faster
                processing_time = min(processing_config.timeout_seconds, 30)
                await asyncio.sleep(min(processing_time / 10, 3))  # Simulate 10% of max time
            
            # Generate processed video key
            processed_key = f"processed/{job_id}.mp4"
            
            # Simulate quality metrics
            quality_metrics = {
                "psnr": 35.2 if self.worker_env.mode == "gpu" else 28.5,
                "ssim": 0.92 if self.worker_env.mode == "gpu" else 0.85,
                "processing_mode": self.worker_env.mode,
                "model_quality": self.config_manager.config.model.quality.value
            }
            
            logger.info(f"Job {job_id}: Video processing completed")
            
            return {
                "processed_video_key": processed_key,
                "quality_metrics": quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Video processing failed for job {job_id}: {e}")
            raise
    
    async def _log_job_event(
        self, 
        job_id: str, 
        event_type: JobEventType, 
        details: Dict[str, Any]
    ):
        """Log a job event to the database."""
        try:
            db = await get_db()
            
            event = JobEvent(
                job_id=UUID(job_id),
                event_type=event_type,
                timestamp=datetime.utcnow(),
                details=details
            )
            
            await db.job_events.insert_one(event.dict())
            
        except Exception as e:
            logger.warning(f"Failed to log job event: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("Cleaning up worker resources...")
            
            # Unload models
            if self.model_factory:
                await self.model_factory.unload_all_models()
            
            # Close RabbitMQ connection
            if self.client:
                await self.client.close()
            
            logger.info("Worker cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main entry point for the dual-mode worker."""
    logger.info("Starting dual-mode video processing worker...")
    
    consumer = DualModeVideoJobConsumer()
    
    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
    finally:
        await consumer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
