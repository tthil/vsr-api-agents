"""MongoDB collections and schemas for VSR API."""

import enum
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pymongo
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING

from vsr_shared.db.client import get_db
from vsr_shared.logging import get_logger
from vsr_shared.models import Job, JobEvent, ApiKey, JobStatus, JobEventType

logger = get_logger(__name__)

class Collections(enum.Enum):
    """Collection names for MongoDB."""
    
    JOBS = "jobs"
    JOB_EVENTS = "job_events"
    API_KEYS = "api_keys"


class CollectionsManager:
    """MongoDB collections for VSR API."""

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize collections.

        Args:
            db: MongoDB database
        """
        self.db = db
        self.jobs = db[Collections.JOBS.value]
        self.job_events = db[Collections.JOB_EVENTS.value]
        self.api_keys = db[Collections.API_KEYS.value]

    async def create_indexes(self) -> None:
        """Create indexes for all collections."""
        await self._create_jobs_indexes()
        await self._create_job_events_indexes()
        await self._create_api_keys_indexes()

    async def _create_jobs_indexes(self) -> None:
        """Create indexes for jobs collection."""
        indexes = [
            IndexModel([("status", ASCENDING)], background=True),
            IndexModel([("created_at", DESCENDING)], background=True),
            IndexModel([("updated_at", DESCENDING)], background=True),
            IndexModel(
                [("completed_at", ASCENDING)],
                name="completed_jobs_ttl",
                expireAfterSeconds=30 * 24 * 60 * 60,  # 30 days TTL
                background=True,
                partialFilterExpression={"status": JobStatus.COMPLETED.value},
            ),
            IndexModel(
                [("completed_at", ASCENDING)],
                name="failed_jobs_ttl",
                expireAfterSeconds=7 * 24 * 60 * 60,  # 7 days TTL
                background=True,
                partialFilterExpression={"status": JobStatus.FAILED.value},
            ),
            IndexModel([("video_key", ASCENDING)], background=True),
            IndexModel([("processed_video_key", ASCENDING)], background=True),
            IndexModel([("api_key_id", ASCENDING)], background=True),
        ]

        await self.jobs.create_indexes(indexes)
        logger.info(f"Created indexes for {Collections.JOBS.value} collection")

    async def _create_job_events_indexes(self) -> None:
        """Create indexes for job_events collection."""
        indexes = [
            IndexModel([("job_id", ASCENDING)], background=True),
            IndexModel([("timestamp", DESCENDING)], background=True),
            IndexModel([("type", ASCENDING)], background=True),
            # TTL index to automatically remove old events
            IndexModel(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=7 * 24 * 60 * 60,  # 7 days TTL
                background=True,
            ),
        ]

        await self.job_events.create_indexes(indexes)
        logger.info(f"Created indexes for {Collections.JOB_EVENTS.value} collection")

    async def _create_api_keys_indexes(self) -> None:
        """Create indexes for api_keys collection."""
        indexes = [
            IndexModel([("key", ASCENDING)], unique=True, background=True),
            IndexModel([("name", ASCENDING)], unique=True, background=True),
            IndexModel([("active", ASCENDING)], background=True),
            IndexModel([("last_used_at", DESCENDING)], background=True),
        ]

        await self.api_keys.create_indexes(indexes)
        logger.info(f"Created indexes for {Collections.API_KEYS.value} collection")


async def get_collections() -> CollectionsManager:
    """
    Get MongoDB collections.

    Returns:
        CollectionsManager: MongoDB collections
    """
    db = await get_db()
    return CollectionsManager(db)


# BSON schemas for collections
# These are used for validation when creating collections

JOBS_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "status", "mode", "progress", "video_key", "created_at", "updated_at", "api_key_id"],
    "properties": {
        "_id": {"bsonType": "binData"},  # UUID as binary
        "status": {
            "bsonType": "string",
            "enum": [s.value for s in JobStatus],
        },
        "mode": {"bsonType": "string"},
        "subtitle_area": {
            "bsonType": ["object", "null"],
            "properties": {
                "top": {"bsonType": "double", "minimum": 0, "maximum": 1},
                "left": {"bsonType": "double", "minimum": 0, "maximum": 1},
                "width": {"bsonType": "double", "minimum": 0, "maximum": 1},
                "height": {"bsonType": "double", "minimum": 0, "maximum": 1},
            },
        },
        "progress": {"bsonType": "double", "minimum": 0, "maximum": 100},
        "video_key": {"bsonType": "string"},
        "processed_video_key": {"bsonType": ["string", "null"]},
        "created_at": {"bsonType": "date"},
        "updated_at": {"bsonType": "date"},
        "started_at": {"bsonType": ["date", "null"]},
        "completed_at": {"bsonType": ["date", "null"]},
        "error_message": {"bsonType": ["string", "null"]},
        "callback_url": {"bsonType": ["string", "null"]},
        "processing_time_seconds": {"bsonType": ["double", "null"]},
        "queue_position": {"bsonType": ["int", "null"]},
        "api_key_id": {"bsonType": "binData"},  # UUID as binary
    },
}

JOB_EVENTS_SCHEMA = {
    "bsonType": "object",
    "required": ["job_id", "type", "timestamp", "progress", "message"],
    "properties": {
        "_id": {"bsonType": "objectId"},
        "job_id": {"bsonType": "binData"},  # UUID as binary
        "type": {
            "bsonType": "string",
            "enum": [t.value for t in JobEventType],
        },
        "timestamp": {"bsonType": "date"},
        "progress": {"bsonType": "double", "minimum": 0, "maximum": 100},
        "message": {"bsonType": "string"},
        "metadata": {"bsonType": ["object", "null"]},
    },
}

API_KEYS_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "key", "name", "active", "created_at", "usage"],
    "properties": {
        "_id": {"bsonType": "binData"},  # UUID as binary
        "key": {"bsonType": "string", "minLength": 32},
        "name": {"bsonType": "string"},
        "active": {"bsonType": "bool"},
        "created_at": {"bsonType": "date"},
        "last_used_at": {"bsonType": ["date", "null"]},
        "usage": {
            "bsonType": "object",
            "properties": {
                "jobs_created": {"bsonType": "int"},
                "jobs_completed": {"bsonType": "int"},
                "upload_count": {"bsonType": "int"},
                "download_count": {"bsonType": "int"},
                "processing_seconds": {"bsonType": "double"},
                "total_video_size_mb": {"bsonType": "double"},
            },
        },
        "daily_limit": {"bsonType": ["int", "null"]},
        "monthly_limit": {"bsonType": ["int", "null"]},
    },
}


async def create_collections_with_validation(db: AsyncIOMotorDatabase, drop_existing: bool = False) -> None:
    """
    Create collections with validation schemas if they don't exist.

    Args:
        db: MongoDB database
        drop_existing: Whether to drop existing collections
    """
    # Check if collections exist
    existing_collections = await db.list_collection_names()

    # Drop collections if requested
    if drop_existing:
        for collection in Collections:
            if collection.value in existing_collections:
                await db.drop_collection(collection.value)
                logger.info(f"Dropped {collection.value} collection")
        existing_collections = []

    # Create jobs collection if it doesn't exist
    if Collections.JOBS.value not in existing_collections:
        await db.create_collection(
            Collections.JOBS.value,
            validator={"$jsonSchema": JOBS_SCHEMA},
            validationLevel="strict",
        )
        logger.info(f"Created {Collections.JOBS.value} collection with validation")

    # Create job_events collection if it doesn't exist
    if Collections.JOB_EVENTS.value not in existing_collections:
        await db.create_collection(
            Collections.JOB_EVENTS.value,
            validator={"$jsonSchema": JOB_EVENTS_SCHEMA},
            validationLevel="strict",
        )
        logger.info(f"Created {Collections.JOB_EVENTS.value} collection with validation")

    # Create api_keys collection if it doesn't exist
    if Collections.API_KEYS.value not in existing_collections:
        await db.create_collection(
            Collections.API_KEYS.value,
            validator={"$jsonSchema": API_KEYS_SCHEMA},
            validationLevel="strict",
        )
        logger.info(f"Created {Collections.API_KEYS.value} collection with validation")

    # Create indexes
    collections = CollectionsManager(db)
    await collections.create_indexes()


# Helper functions for document conversion

def job_to_document(job: Job) -> Dict[str, Any]:
    """
    Convert Job model to MongoDB document.

    Args:
        job: Job model

    Returns:
        Dict: MongoDB document
    """
    doc = job.model_dump(by_alias=True)
    
    # Convert subtitle_area to dict
    if "subtitle_area" in doc and doc["subtitle_area"] is not None and hasattr(doc["subtitle_area"], "to_dict"):
        doc["subtitle_area"] = doc["subtitle_area"].to_dict()
    
    # Convert enum values to strings
    if "status" in doc and hasattr(doc["status"], "value"):
        doc["status"] = doc["status"].value
    
    if "mode" in doc and hasattr(doc["mode"], "value"):
        doc["mode"] = doc["mode"].value
    
    return doc


def document_to_job(doc: Dict[str, Any]) -> Job:
    """
    Convert MongoDB document to Job model.

    Args:
        doc: MongoDB document

    Returns:
        Job: Job model
    """
    # Create a copy to avoid modifying the original
    doc_copy = doc.copy()
    
    # Ensure _id is a UUID
    if "_id" in doc_copy and not isinstance(doc_copy["_id"], uuid.UUID):
        if isinstance(doc_copy["_id"], str):
            doc_copy["_id"] = uuid.UUID(doc_copy["_id"])
            
    # Ensure api_key_id is a UUID
    if "api_key_id" in doc_copy and not isinstance(doc_copy["api_key_id"], uuid.UUID):
        if isinstance(doc_copy["api_key_id"], str):
            doc_copy["api_key_id"] = uuid.UUID(doc_copy["api_key_id"])
    
    return Job(**doc_copy)


def job_event_to_document(event: JobEvent) -> Dict[str, Any]:
    """
    Convert JobEvent model to MongoDB document.

    Args:
        event: JobEvent model

    Returns:
        Dict: MongoDB document
    """
    doc = event.model_dump(by_alias=True, exclude={"_id"} if event.id is None else set())
    
    # Convert enum values to strings
    if "type" in doc and hasattr(doc["type"], "value"):
        doc["type"] = doc["type"].value
    
    # Add ObjectId for _id if not present
    if "_id" not in doc or doc["_id"] is None:
        doc["_id"] = ObjectId()
        
    return doc


def document_to_job_event(doc: Dict[str, Any]) -> JobEvent:
    """
    Convert MongoDB document to JobEvent model.

    Args:
        doc: MongoDB document

    Returns:
        JobEvent: JobEvent model
    """
    return JobEvent(**doc)


def api_key_to_document(api_key: ApiKey) -> Dict[str, Any]:
    """
    Convert ApiKey model to MongoDB document.

    Args:
        api_key: ApiKey model

    Returns:
        Dict: MongoDB document
    """
    doc = api_key.model_dump(by_alias=True)
    return doc


def document_to_api_key(doc: Dict[str, Any]) -> ApiKey:
    """
    Convert MongoDB document to ApiKey model.

    Args:
        doc: MongoDB document

    Returns:
        ApiKey: ApiKey model
    """
    # Create a copy to avoid modifying the original
    doc_copy = doc.copy()
    
    # Ensure _id is a UUID
    if "_id" in doc_copy and not isinstance(doc_copy["_id"], uuid.UUID):
        if isinstance(doc_copy["_id"], str):
            doc_copy["_id"] = uuid.UUID(doc_copy["_id"])
    
    return ApiKey(**doc_copy)
