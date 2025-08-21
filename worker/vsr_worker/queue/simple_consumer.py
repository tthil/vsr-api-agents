"""Dual-mode RabbitMQ consumer for processing video jobs."""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional

from vsr_shared.logging import get_logger, bind_request_id
from vsr_shared.db import get_db
from vsr_shared.models import Job, JobStatus, JobEvent, JobEventType
from vsr_shared.queue.client import get_rabbitmq_client
from vsr_shared.storage import get_spaces_client
from vsr_worker.webhooks import notify_job_completion
from vsr_worker.gpu.environment import WorkerEnvironment
from vsr_worker.models.adaptive import AdaptiveModelFactory
from vsr_worker.config.dual_mode import get_config_manager, print_config_summary

logger = get_logger(__name__)


class SimpleVideoJobConsumer:
    """Simple consumer for video processing jobs."""
    
    def __init__(self):
        self.client = get_rabbitmq_client()
        self.client = RabbitMQClient()
        self.running = False
        
    async def process_job(self, job_data: dict) -> None:
        """Process a single job."""
        job_id = job_data.get("job_id")
        logger.info(f"Processing job: {job_id}")
        
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
                        "started_at": datetime.utcnow()
                    }
                }
            )
            
            # Simulate processing (replace with real processing later)
            logger.info(f"Job {job_id}: Starting video processing...")
            processing_start = datetime.utcnow()
            await asyncio.sleep(5)  # Simulate processing time
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
                        "processed_video_key": f"processed/{job_id}.mp4"
                    }
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
            
            logger.info(f"Job {job_id}: Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Processing failed: {e}")
            
            # Update job status to failed
            await db.jobs.update_one(
                {"_id": UUID(job_id)},
                {
                    "$set": {
                        "status": JobStatus.FAILED.value,
                        "updated_at": datetime.utcnow(),
                        "error_message": str(e)
                    }
                }
            )
            
            # Fetch updated job for webhook notification
            job_doc = await db.jobs.find_one({"_id": UUID(job_id)})
            if job_doc:
                # Convert MongoDB document to Job model
                job_doc["id"] = job_doc.pop("_id")
                job = Job(**job_doc)
                
                # Send webhook notification for failure
                await notify_job_completion(job)
    
    async def message_handler(self, message):
        """Handle incoming messages."""
        async with message.process():
            try:
                job_data = json.loads(message.body.decode())
                await self.process_job(job_data)
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                raise
    
    async def start_consuming(self):
        """Start consuming messages."""
        logger.info("Starting video job consumer...")
        
        # Connect to RabbitMQ
        await self.client.connect()
        
        # Setup topology
        await setup_queue_topology(self.client.channel)
        
        # Get the processing queue
        queue = await self.client.channel.get_queue(QUEUE_NAME)
        
        # Start consuming
        await queue.consume(self.message_handler)
        
        self.running = True
        logger.info(f"Consumer started, waiting for messages on queue: {QUEUE_NAME}")
        
        # Keep running until stopped
        while self.running:
            await asyncio.sleep(1)
    
    async def stop(self):
        """Stop the consumer."""
        logger.info("Stopping consumer...")
        self.running = False
        await self.client.disconnect()


async def main():
    """Main entry point."""
    consumer = SimpleVideoJobConsumer()
    
    # Setup signal handlers
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(consumer.stop())
    
    # Register signal handlers
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        sys.exit(1)
    finally:
        await consumer.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    asyncio.run(main())
