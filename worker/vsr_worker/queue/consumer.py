"""
RabbitMQ consumer for processing video jobs.
"""

import asyncio
import json
import logging
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from uuid import UUID

import aio_pika
from aio_pika import Message, DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

from vsr_shared.models import Job, JobStatus, ProcessingMode
from vsr_shared.db.client import get_db
from vsr_shared.queue.client import RabbitMQClient, QUEUE_NAME
from vsr_shared.queue.topology import setup_queue_topology


logger = logging.getLogger(__name__)


class JobConsumer:
    """
    RabbitMQ consumer for processing video subtitle removal jobs.
    """
    
    def __init__(self):
        self.connection: Optional[AbstractRobustConnection] = None
        self.channel: Optional[AbstractRobustChannel] = None
        self.pipeline = VideoProcessingPipeline()
        self.webhook_notifier = WebhookNotifier()
        self.is_running = False
        self.current_job_id: Optional[str] = None
        
    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(
                host=RABBITMQ_CONFIG["host"],
                port=RABBITMQ_CONFIG["port"],
                login=RABBITMQ_CONFIG["username"],
                password=RABBITMQ_CONFIG["password"],
                virtualhost=RABBITMQ_CONFIG["vhost"],
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)  # Process one job at a time
            
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        try:
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    async def setup_queues(self) -> None:
        """Setup required queues and exchanges."""
        try:
            # Declare main processing queue
            processing_queue = await self.channel.declare_queue(
                RABBITMQ_CONFIG["queues"]["processing"],
                durable=True,
                arguments={
                    "x-message-ttl": 3600000,  # 1 hour TTL
                    "x-dead-letter-exchange": RABBITMQ_CONFIG["exchanges"]["dlx"],
                    "x-dead-letter-routing-key": "failed"
                }
            )
            
            # Declare dead letter exchange and queue
            dlx_exchange = await self.channel.declare_exchange(
                RABBITMQ_CONFIG["exchanges"]["dlx"],
                aio_pika.ExchangeType.DIRECT,
                durable=True
            )
            
            failed_queue = await self.channel.declare_queue(
                RABBITMQ_CONFIG["queues"]["failed"],
                durable=True
            )
            
            await failed_queue.bind(dlx_exchange, "failed")
            
            # Declare retry queue with delay
            retry_queue = await self.channel.declare_queue(
                RABBITMQ_CONFIG["queues"]["retry"],
                durable=True,
                arguments={
                    "x-message-ttl": 300000,  # 5 minutes delay
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": RABBITMQ_CONFIG["queues"]["processing"]
                }
            )
            
            logger.info("RabbitMQ queues setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup queues: {e}")
            raise
    
    async def process_message(self, message: IncomingMessage) -> None:
        """
        Process a single job message.
        
        Args:
            message: RabbitMQ message containing job data
        """
        job_data = None
        
        try:
            # Parse message
            job_data = json.loads(message.body.decode())
            job_id = job_data.get("job_id")
            
            if not job_id:
                logger.error("Message missing job_id")
                await message.nack(requeue=False)
                return
            
            self.current_job_id = job_id
            logger.info(f"Processing job: {job_id}")
            
            # Fetch job from database
            db = await get_db()
            job_doc = await db.jobs.find_one({"_id": job_id})
            
            if not job_doc:
                logger.error(f"Job not found in database: {job_id}")
                await message.nack(requeue=False)
                return
            
            # Convert to Job model
            job = Job.from_dict(job_doc)
            
            # Check if job is still pending
            if job.status != JobStatus.PENDING:
                logger.warning(f"Job {job_id} is not pending, skipping (status: {job.status})")
                await message.ack()
                return
            
            # Process the video
            await self._process_job(job)
            
            # Send webhook notification if callback URL provided
            if job.callback_url:
                await self.webhook_notifier.send_completion_notification(job)
            
            # Acknowledge message
            await message.ack()
            logger.info(f"Successfully processed job: {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_data.get('job_id') if job_data else 'unknown'}: {e}")
            
            # Handle retry logic
            retry_count = message.headers.get("x-retry-count", 0) if message.headers else 0
            
            if retry_count < RABBITMQ_CONFIG["max_retries"]:
                # Send to retry queue
                await self._send_to_retry_queue(job_data, retry_count + 1)
                await message.ack()
                logger.info(f"Sent job to retry queue (attempt {retry_count + 1})")
            else:
                # Max retries exceeded, send to DLQ
                await message.nack(requeue=False)
                logger.error(f"Max retries exceeded for job, sent to DLQ")
                
                # Update job status to failed
                if job_data and job_data.get("job_id"):
                    await self._mark_job_failed(job_data["job_id"], str(e))
        
        finally:
            self.current_job_id = None
    
    async def _process_job(self, job: Job) -> None:
        """
        Process a single job using the video processing pipeline.
        
        Args:
            job: Job to process
        """
        try:
            # Initialize pipeline
            await self.pipeline.initialize()
            
            # Process video with progress tracking
            def progress_callback(progress: int):
                logger.debug(f"Job {job.id} progress: {progress}%")
            
            processed_key = await self.pipeline.process_video(job, progress_callback)
            
            logger.info(f"Job {job.id} completed successfully, processed video: {processed_key}")
            
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            raise
        finally:
            # Always cleanup pipeline resources
            await self.pipeline.cleanup()
    
    async def _send_to_retry_queue(self, job_data: Dict[str, Any], retry_count: int) -> None:
        """
        Send job to retry queue with updated retry count.
        
        Args:
            job_data: Job data to retry
            retry_count: Current retry attempt number
        """
        try:
            retry_message = Message(
                json.dumps(job_data).encode(),
                headers={"x-retry-count": retry_count},
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            
            await self.channel.default_exchange.publish(
                retry_message,
                routing_key=RABBITMQ_CONFIG["queues"]["retry"]
            )
            
        except Exception as e:
            logger.error(f"Failed to send job to retry queue: {e}")
    
    async def _mark_job_failed(self, job_id: str, error_message: str) -> None:
        """
        Mark job as failed in database.
        
        Args:
            job_id: Job ID to mark as failed
            error_message: Error message to store
        """
        try:
            db = await get_db()
            await db.jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": JobStatus.FAILED.value,
                        "error_message": error_message,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            logger.info(f"Marked job {job_id} as failed")
            
        except Exception as e:
            logger.error(f"Failed to mark job as failed: {e}")
    
    async def start_consuming(self) -> None:
        """Start consuming messages from the processing queue."""
        try:
            await self.connect()
            await self.setup_queues()
            
            processing_queue = await self.channel.get_queue(
                RABBITMQ_CONFIG["queues"]["processing"]
            )
            
            # Start consuming
            await processing_queue.consume(self.process_message)
            
            self.is_running = True
            logger.info("Started consuming messages from processing queue")
            
            # Keep the consumer running
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            raise
        finally:
            await self.disconnect()
    
    async def stop_consuming(self) -> None:
        """Stop consuming messages gracefully."""
        logger.info("Stopping consumer...")
        self.is_running = False
        
        # If currently processing a job, wait for it to complete
        if self.current_job_id:
            logger.info(f"Waiting for current job {self.current_job_id} to complete...")
            while self.current_job_id:
                await asyncio.sleep(1)
        
        await self.disconnect()
        logger.info("Consumer stopped")


class ConsumerManager:
    """
    Manager for multiple consumer instances with health monitoring.
    """
    
    def __init__(self, num_consumers: int = 1):
        self.num_consumers = num_consumers
        self.consumers: list[JobConsumer] = []
        self.consumer_tasks: list[asyncio.Task] = []
        self.is_running = False
    
    async def start(self) -> None:
        """Start all consumer instances."""
        try:
            logger.info(f"Starting {self.num_consumers} consumer instances")
            
            for i in range(self.num_consumers):
                consumer = JobConsumer()
                self.consumers.append(consumer)
                
                task = asyncio.create_task(
                    consumer.start_consuming(),
                    name=f"consumer-{i}"
                )
                self.consumer_tasks.append(task)
            
            self.is_running = True
            
            # Wait for all consumers to complete
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error starting consumers: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop all consumer instances gracefully."""
        logger.info("Stopping all consumers...")
        self.is_running = False
        
        # Stop all consumers
        stop_tasks = [consumer.stop_consuming() for consumer in self.consumers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel remaining tasks
        for task in self.consumer_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        logger.info("All consumers stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all consumers.
        
        Returns:
            Health status information
        """
        healthy_consumers = 0
        total_consumers = len(self.consumers)
        
        for i, consumer in enumerate(self.consumers):
            try:
                # Check if consumer is connected and running
                if consumer.connection and not consumer.connection.is_closed:
                    healthy_consumers += 1
            except Exception as e:
                logger.warning(f"Consumer {i} health check failed: {e}")
        
        return {
            "healthy_consumers": healthy_consumers,
            "total_consumers": total_consumers,
            "is_healthy": healthy_consumers == total_consumers,
            "timestamp": datetime.utcnow().isoformat()
        }


# Signal handlers for graceful shutdown
def setup_signal_handlers(consumer_manager: ConsumerManager):
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(consumer_manager.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Main entry point
async def main():
    """Main entry point for the worker consumer."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create consumer manager
    num_consumers = int(os.getenv("VSR_WORKER_CONSUMERS", "1"))
    consumer_manager = ConsumerManager(num_consumers)
    
    # Setup signal handlers
    setup_signal_handlers(consumer_manager)
    
    try:
        logger.info("Starting VSR Worker Consumer")
        await consumer_manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        sys.exit(1)
    finally:
        await consumer_manager.stop()
        logger.info("VSR Worker Consumer shutdown complete")


if __name__ == "__main__":
    import os
    asyncio.run(main())
