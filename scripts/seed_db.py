#!/usr/bin/env python3
"""
Seed script for MongoDB database with test data.
This script creates test API keys, jobs, and job events for development.
"""

import argparse
import asyncio
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add the project root and shared directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'shared'))

from vsr_shared.db.client import get_mongodb_client
from vsr_shared.db.collections import create_collections_with_validation
from vsr_shared.db.dal import ApiKeysDAL, JobsDAL
from vsr_shared.logging import get_logger, setup_logging
from vsr_shared.models import (
    Job,
    JobEvent,
    JobEventType,
    JobStatus,
    ProcessingMode,
    SubtitleArea,
)
from vsr_shared.spaces_keys import get_upload_key, get_processed_key
from vsr_shared.utils.api_keys import create_api_key

logger = get_logger(__name__)

# Default test data
DEFAULT_API_KEYS = [
    {
        "key": "vsr_test_key_1234567890abcdef1234567890abcdef",
        "name": "Test API Key",
    },
    {
        "key": "vsr_dev_key_0987654321fedcba0987654321fedcba",
        "name": "Development API Key",
    },
]


# Removed hash_api_key function as it's no longer needed with the updated model


def generate_test_subtitle_area() -> SubtitleArea:
    """
    Generate a random subtitle area for testing.
    
    Returns:
        SubtitleArea: Random subtitle area
    """
    # Generate random coordinates in normalized format (0-1)
    top = round(random.uniform(0.7, 0.85), 2)  # Bottom of screen
    left = round(random.uniform(0.1, 0.2), 2)
    width = round(random.uniform(0.6, 0.8), 2)
    height = round(random.uniform(0.05, 0.15), 2)
    
    # Ensure coordinates are valid
    if left + width > 1.0:
        width = 1.0 - left
    if top + height > 1.0:
        height = 1.0 - top
    
    return SubtitleArea(
        top=top,
        left=left,
        width=width,
        height=height,
    )


def generate_test_jobs(api_key_id: uuid.UUID, count: int = 5) -> List[Job]:
    """
    Generate test jobs for development.
    
    Args:
        api_key_id: API key ID to associate with jobs
        count: Number of jobs to generate
        
    Returns:
        List[Job]: List of generated jobs
    """
    jobs = []
    
    # Generate jobs with different statuses
    statuses = list(JobStatus)
    modes = list(ProcessingMode)
    
    for i in range(count):
        job_id = uuid.uuid4()
        status = statuses[i % len(statuses)]
        mode = modes[i % len(modes)]
        
        # Create job
        job = Job(
            id=job_id,
            status=status,
            mode=mode,
            subtitle_area=generate_test_subtitle_area(),
            progress=random.uniform(0, 100) if status != JobStatus.PENDING else 0,
            video_key=get_upload_key(job_id),
            processed_video_key=get_processed_key(job_id) if status == JobStatus.COMPLETED else None,
            created_at=datetime.utcnow() - timedelta(days=random.randint(0, 5)),
            updated_at=datetime.utcnow() - timedelta(hours=random.randint(0, 24)),
            started_at=datetime.utcnow() - timedelta(hours=random.randint(0, 12)) if status != JobStatus.PENDING else None,
            completed_at=datetime.utcnow() - timedelta(hours=random.randint(0, 6)) if status == JobStatus.COMPLETED else None,
            error_message="Test error message" if status == JobStatus.FAILED else None,
            callback_url="https://example.com/webhook" if i % 2 == 0 else None,
            processing_time_seconds=random.uniform(60, 300) if status == JobStatus.COMPLETED else None,
            queue_position=i if status == JobStatus.QUEUED else None,
            api_key_id=api_key_id,
        )
        
        jobs.append(job)
    
    return jobs


def generate_test_job_events(job: Job, count: int = 3) -> List[JobEvent]:
    """
    Generate test job events for a job.
    
    Args:
        job: Job to generate events for
        count: Number of events to generate
        
    Returns:
        List[JobEvent]: List of generated job events
    """
    events = []
    
    # Generate events based on job status
    if job.status == JobStatus.PENDING:
        # For pending jobs, just create a submission event
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_CREATED,
                timestamp=job.created_at,
                progress=0,
                message="Job submitted and waiting for processing",
                metadata={"source": "api"}
            )
        )
    elif job.status == JobStatus.QUEUED:
        # For queued jobs, create submission and queued events
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_CREATED,
                timestamp=job.created_at,
                progress=0,
                message="Job submitted and waiting for processing",
                metadata={"source": "api"}
            )
        )
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_QUEUED,
                timestamp=job.updated_at,
                progress=0,
                message=f"Job queued for processing at position {job.queue_position}",
                metadata={"queue_position": job.queue_position}
            )
        )
    elif job.status == JobStatus.PROCESSING:
        # For processing jobs, create submission, started, and progress events
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_CREATED,
                timestamp=job.created_at,
                progress=0,
                message="Job submitted and waiting for processing",
                metadata={"source": "api"}
            )
        )
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_STARTED,
                timestamp=job.started_at or (job.created_at + timedelta(minutes=random.randint(1, 10))),
                progress=0,
                message="Job processing started",
                metadata={"worker_id": f"test-worker-{random.randint(1, 5)}"}
            )
        )
        # Add some progress events
        for i in range(count):
            progress = min(100, (i + 1) * 25)
            events.append(
                JobEvent(
                    job_id=job.id,
                    type=JobEventType.JOB_PROGRESS,
                    timestamp=job.updated_at - timedelta(minutes=random.randint(1, 30)),
                    progress=progress,
                    message=f"Processing progress: {progress}%",
                    metadata={"frame": i * 100, "total_frames": count * 100}
                )
            )
    elif job.status == JobStatus.COMPLETED:
        # For completed jobs, create submission, started, progress, and completed events
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_CREATED,
                timestamp=job.created_at,
                progress=0,
                message="Job submitted and waiting for processing",
                metadata={"source": "api"}
            )
        )
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_STARTED,
                timestamp=job.started_at or (job.created_at + timedelta(minutes=random.randint(1, 10))),
                progress=0,
                message="Job processing started",
                metadata={"worker_id": f"test-worker-{random.randint(1, 5)}"}
            )
        )
        # Add some progress events
        for i in range(count - 1):
            progress = min(95, (i + 1) * 30)
            events.append(
                JobEvent(
                    job_id=job.id,
                    type=JobEventType.JOB_PROGRESS,
                    timestamp=job.updated_at - timedelta(minutes=random.randint(5, 30)),
                    progress=progress,
                    message=f"Processing progress: {progress}%",
                    metadata={"frame": i * 100, "total_frames": count * 100}
                )
            )
        # Add completed event
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_COMPLETED,
                timestamp=job.completed_at or job.updated_at,
                progress=100,
                message="Job completed successfully",
                metadata={
                    "processing_time_seconds": job.processing_time_seconds,
                    "output_file_size": random.randint(1000000, 10000000)
                }
            )
        )
    elif job.status == JobStatus.FAILED:
        # For failed jobs, create submission, started, and failed events
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_CREATED,
                timestamp=job.created_at,
                progress=0,
                message="Job submitted and waiting for processing",
                metadata={"source": "api"}
            )
        )
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_STARTED,
                timestamp=job.started_at or (job.created_at + timedelta(minutes=random.randint(1, 10))),
                progress=0,
                message="Job processing started",
                metadata={"worker_id": f"test-worker-{random.randint(1, 5)}"}
            )
        )
        # Add some progress events
        for i in range(count - 1):
            progress = min(80, (i + 1) * 20)
            events.append(
                JobEvent(
                    job_id=job.id,
                    type=JobEventType.JOB_PROGRESS,
                    timestamp=job.updated_at - timedelta(minutes=random.randint(5, 30)),
                    progress=progress,
                    message=f"Processing progress: {progress}%",
                    metadata={"frame": i * 100, "total_frames": count * 100}
                )
            )
        # Add failed event
        events.append(
            JobEvent(
                job_id=job.id,
                type=JobEventType.JOB_FAILED,
                timestamp=job.updated_at,
                progress=job.progress,
                message=f"Job failed: {job.error_message or 'Unknown error'}",
                metadata={"error_code": "PROCESSING_ERROR", "stack_trace": "Test stack trace"}
            )
        )
    
    return events


async def seed_api_keys(api_keys_dal: ApiKeysDAL, keys: List[Dict[str, str]]) -> Dict[str, uuid.UUID]:
    """
    Seed API keys into the database.
    
    Args:
        api_keys_dal: API keys data access layer
        keys: List of API keys to seed
        
    Returns:
        Dict[str, uuid.UUID]: Mapping of API key to API key ID
    """
    key_map = {}
    
    for key_data in keys:
        # Check if key already exists
        existing_key = await api_keys_dal.get_by_key(key_data["key"])
        if existing_key:
            logger.info(f"API key '{key_data['name']}' already exists")
            key_map[key_data["key"]] = existing_key.id
            continue
        
        # Create API key using helper function
        api_key = create_api_key(
            name=key_data["name"],
            key=key_data["key"],
            daily_limit=100,  # Add reasonable limits for test keys
            monthly_limit=1000,
        )
        
        # Save API key
        api_key_id = await api_keys_dal.create(api_key)
        logger.info(f"Created API key '{key_data['name']}'")
        
        key_map[key_data["key"]] = api_key.id
    
    return key_map


async def seed_jobs(jobs_dal: JobsDAL, api_key_id: uuid.UUID, count: int = 5) -> None:
    """
    Seed jobs and job events into the database.
    
    Args:
        jobs_dal: Jobs data access layer
        api_key_id: API key ID to associate with jobs
        count: Number of jobs to generate
    """
    # Generate test jobs
    jobs = generate_test_jobs(api_key_id, count)
    
    for job in jobs:
        # Check if job already exists
        existing_job = await jobs_dal.get(job.id)
        if existing_job:
            logger.info(f"Job {job.id} already exists")
            continue
        
        # Save job
        await jobs_dal.create(job)
        logger.info(f"Created job {job.id} with status {job.status.value}")
        
        # Generate and save job events
        events = generate_test_job_events(job)
        for event in events:
            await jobs_dal.add_event(event)
        
        logger.info(f"Created {len(events)} events for job {job.id}")


async def main(args: argparse.Namespace) -> None:
    """
    Main function to seed the database.
    
    Args:
        args: Command-line arguments
    """
    # Connect to MongoDB with authentication
    mongo_uri = "mongodb://root:example@localhost:27017/admin"
    client = get_mongodb_client(uri=mongo_uri, db_name=args.db_name)
    await client.connect()
    
    try:
        # Check connection
        if not await client.ping():
            logger.error("MongoDB connection failed")
            sys.exit(1)
        
        # Get database
        db = client.get_database()
        
        # Create collections with validation if needed
        if args.create_collections:
            await create_collections_with_validation(db, drop_existing=args.drop_existing)
        
        # Create DAL instances
        api_keys_dal = ApiKeysDAL(db)
        jobs_dal = JobsDAL(db)
        
        # Seed API keys
        key_map = await seed_api_keys(api_keys_dal, DEFAULT_API_KEYS)
        
        # Seed jobs for each API key
        for key, key_id in key_map.items():
            await seed_jobs(jobs_dal, key_id, args.job_count)
        
        logger.info("Database seeding completed successfully")
    
    finally:
        # Close connection
        await client.close()


if __name__ == "__main__":
    # Set up logging
    setup_logging(level="INFO", json_format=False)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Seed MongoDB database with test data")
    parser.add_argument(
        "--db-name",
        help="Database name override",
    )
    parser.add_argument(
        "--create-collections",
        action="store_true",
        help="Create collections with validation before seeding",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collections before creating new ones",
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=5,
        help="Number of jobs to create per API key",
    )
    
    args = parser.parse_args()
    
    # Run seeding
    asyncio.run(main(args))
