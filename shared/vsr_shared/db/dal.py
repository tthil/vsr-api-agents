"""Data Access Layer for MongoDB."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from vsr_shared.db.client import get_db
from vsr_shared.db.collections import (
    Collections,
    get_collections,
    job_to_document,
    document_to_job,
    job_event_to_document,
    document_to_job_event,
    api_key_to_document,
    document_to_api_key,
)
from vsr_shared.logging import get_logger
from vsr_shared.models import Job, JobEvent, ApiKey, JobStatus

logger = get_logger(__name__)


class JobsDAL:
    """Data Access Layer for jobs collection."""

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize JobsDAL.

        Args:
            db: MongoDB database
        """
        self.collection = db[Collections.JOBS.value]
        self.events_collection = db[Collections.JOB_EVENTS.value]

    async def create(self, job: Job) -> uuid.UUID:
        """
        Create a new job.

        Args:
            job: Job model

        Returns:
            uuid.UUID: Job ID
        """
        # Convert Job model to MongoDB document
        doc = job_to_document(job)
        
        try:
            # Insert document
            result = await self.collection.insert_one(doc)
            logger.info(f"Created job {job.id}")
            return job.id
        except DuplicateKeyError:
            logger.error(f"Job {job.id} already exists")
            raise
        except PyMongoError as e:
            logger.error(f"Error creating job: {e}")
            raise

    async def get(self, job_id: uuid.UUID) -> Optional[Job]:
        """
        Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job: Job model or None if not found
        """
        try:
            doc = await self.collection.find_one({"_id": job_id})
            if doc:
                logger.debug(f"Found job {job_id}")
                return document_to_job(doc)
            logger.debug(f"Job {job_id} not found")
            return None
        except PyMongoError as e:
            logger.error(f"Error getting job {job_id}: {e}")
            raise

    async def update(
        self, job_id: uuid.UUID, update: Dict[str, Any], upsert: bool = False
    ) -> Optional[Job]:
        """
        Update a job.

        Args:
            job_id: Job ID
            update: Update document
            upsert: Whether to insert if not exists

        Returns:
            Job: Updated job or None if not found
        """
        # Always update the updated_at field
        if "updated_at" not in update:
            update["updated_at"] = datetime.utcnow()

        try:
            # Update document and return the updated version
            doc = await self.collection.find_one_and_update(
                {"_id": job_id},
                {"$set": update},
                return_document=ReturnDocument.AFTER,
                upsert=upsert,
            )
            if doc:
                logger.info(f"Updated job {job_id}")
                return document_to_job(doc)
            logger.warning(f"Job {job_id} not found for update")
            return None
        except PyMongoError as e:
            logger.error(f"Error updating job {job_id}: {e}")
            raise

    async def update_status(
        self, job_id: uuid.UUID, status: JobStatus, error_message: Optional[str] = None
    ) -> Optional[Job]:
        """
        Update job status.

        Args:
            job_id: Job ID
            status: New status
            error_message: Error message (for failed status)

        Returns:
            Job: Updated job or None if not found
        """
        update: Dict[str, Any] = {"status": status.value, "updated_at": datetime.utcnow()}

        # Add status-specific fields
        if status == JobStatus.PROCESSING:
            update["started_at"] = datetime.utcnow()
        elif status == JobStatus.COMPLETED:
            now = datetime.utcnow()
            update["completed_at"] = now
            # Calculate processing time if started_at is set
            job = await self.get(job_id)
            if job and job.started_at:
                processing_time = (now - job.started_at).total_seconds()
                update["processing_time_seconds"] = processing_time
        elif status == JobStatus.FAILED:
            update["error_message"] = error_message or "Unknown error"

        return await self.update(job_id, update)

    async def update_progress(
        self, job_id: uuid.UUID, progress: float
    ) -> Optional[Job]:
        """
        Update job progress.

        Args:
            job_id: Job ID
            progress: Progress value (0-100)

        Returns:
            Job: Updated job or None if not found
        """
        update = {"progress": progress, "updated_at": datetime.utcnow()}
        return await self.update(job_id, update)

    async def set_processed_video_key(
        self, job_id: uuid.UUID, processed_video_key: str
    ) -> Optional[Job]:
        """
        Set the processed video key.

        Args:
            job_id: Job ID
            processed_video_key: Processed video key

        Returns:
            Job: Updated job or None if not found
        """
        update = {"processed_video_key": processed_video_key}
        return await self.update(job_id, update)

    async def list(
        self,
        status: Optional[Union[JobStatus, List[JobStatus]]] = None,
        api_key_id: Optional[uuid.UUID] = None,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "created_at",
        sort_direction: int = -1,  # -1 for descending, 1 for ascending
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status or list of statuses
            api_key_id: Filter by API key ID
            limit: Maximum number of jobs to return
            skip: Number of jobs to skip
            sort_by: Field to sort by
            sort_direction: Sort direction (-1 for descending, 1 for ascending)

        Returns:
            List[Job]: List of jobs
        """
        # Build query
        query = {}
        if status is not None:
            if isinstance(status, list):
                query["status"] = {"$in": [s.value for s in status]}
            else:
                query["status"] = status.value
                
        if api_key_id is not None:
            query["api_key_id"] = api_key_id

        try:
            # Execute query
            cursor = (
                self.collection.find(query)
                .sort(sort_by, sort_direction)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert documents to Job models
            jobs = [document_to_job(doc) async for doc in cursor]
            logger.debug(f"Listed {len(jobs)} jobs")
            return jobs
        except PyMongoError as e:
            logger.error(f"Error listing jobs: {e}")
            raise

    async def count(
        self, status: Optional[Union[JobStatus, List[JobStatus]]] = None, api_key_id: Optional[uuid.UUID] = None
    ) -> int:
        """
        Count jobs with optional filtering.

        Args:
            status: Filter by status or list of statuses
            api_key_id: Filter by API key ID

        Returns:
            int: Number of jobs
        """
        # Build query
        query = {}
        if status is not None:
            if isinstance(status, list):
                query["status"] = {"$in": [s.value for s in status]}
            else:
                query["status"] = status.value
                
        if api_key_id is not None:
            query["api_key_id"] = api_key_id

        try:
            # Execute query
            count = await self.collection.count_documents(query)
            logger.debug(f"Counted {count} jobs")
            return count
        except PyMongoError as e:
            logger.error(f"Error counting jobs: {e}")
            raise

    async def delete(self, job_id: uuid.UUID) -> bool:
        """
        Delete a job.

        Args:
            job_id: Job ID

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            result = await self.collection.delete_one({"_id": job_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted job {job_id}")
                # Also delete job events
                await self.events_collection.delete_many({"job_id": job_id})
                return True
            logger.warning(f"Job {job_id} not found for deletion")
            return False
        except PyMongoError as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            raise
            
    async def add_event(self, event: JobEvent) -> ObjectId:
        """
        Add a job event.
        
        Args:
            event: Job event
            
        Returns:
            ObjectId: Event ID
        """
        try:
            # Convert event to document
            doc = job_event_to_document(event)
            
            # Insert document
            result = await self.events_collection.insert_one(doc)
            logger.debug(f"Added event for job {event.job_id}")
            return result.inserted_id
        except PyMongoError as e:
            logger.error(f"Error adding event for job {event.job_id}: {e}")
            raise
            
    async def get_events(self, job_id: uuid.UUID, limit: int = 100, skip: int = 0) -> List[JobEvent]:
        """
        Get events for a job.
        
        Args:
            job_id: Job ID
            limit: Maximum number of events to return
            skip: Number of events to skip
            
        Returns:
            List[JobEvent]: List of job events
        """
        try:
            # Execute query
            cursor = (
                self.events_collection.find({"job_id": job_id})
                .sort("timestamp", -1)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert documents to JobEvent models
            events = [document_to_job_event(doc) async for doc in cursor]
            logger.debug(f"Found {len(events)} events for job {job_id}")
            return events
        except PyMongoError as e:
            logger.error(f"Error getting events for job {job_id}: {e}")
            raise


class JobEventsDAL:
    """Data Access Layer for job events collection."""

    def __init__(self, collection: AsyncIOMotorCollection):
        """
        Initialize JobEventsDAL.

        Args:
            collection: MongoDB collection for job events
        """
        self.collection = collection

    async def create_event(self, event: JobEvent) -> JobEvent:
        """
        Create a new job event.

        Args:
            event: JobEvent model

        Returns:
            JobEvent: Created job event
        """
        # Convert JobEvent model to MongoDB document
        doc = job_event_to_document(event)
        
        try:
            # Insert document
            result = await self.collection.insert_one(doc)
            # Set the ObjectId
            event.id = result.inserted_id
            logger.info(f"Created job event for job {event.job_id}")
            return event
        except PyMongoError as e:
            logger.error(f"Error creating job event: {e}")
            raise

    async def get_events(
        self,
        job_id: uuid.UUID,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "ts",
        sort_direction: int = -1,  # -1 for descending, 1 for ascending
    ) -> List[JobEvent]:
        """
        Get events for a job.

        Args:
            job_id: Job ID
            limit: Maximum number of events to return
            skip: Number of events to skip
            sort_by: Field to sort by
            sort_direction: Sort direction (1 for ascending, -1 for descending)

        Returns:
            List[JobEvent]: List of job events
        """
        try:
            # Execute query
            cursor = (
                self.collection.find({"job_id": job_id})
                .sort(sort_by, sort_direction)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert documents to JobEvent models
            events = [document_to_job_event(doc) async for doc in cursor]
            logger.debug(f"Listed {len(events)} events for job {job_id}")
            return events
        except PyMongoError as e:
            logger.error(f"Error getting events for job {job_id}: {e}")
            raise

    async def count_events(self, job_id: uuid.UUID) -> int:
        """
        Count events for a job.

        Args:
            job_id: Job ID

        Returns:
            int: Number of events
        """
        try:
            count = await self.collection.count_documents({"job_id": job_id})
            logger.debug(f"Counted {count} events for job {job_id}")
            return count
        except PyMongoError as e:
            logger.error(f"Error counting events for job {job_id}: {e}")
            raise

    async def delete_events(self, job_id: uuid.UUID) -> int:
        """
        Delete all events for a job.

        Args:
            job_id: Job ID

        Returns:
            int: Number of deleted events
        """
        try:
            result = await self.collection.delete_many({"job_id": job_id})
            logger.info(f"Deleted {result.deleted_count} events for job {job_id}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting events for job {job_id}: {e}")
            raise


class ApiKeysDAL:
    """Data Access Layer for API keys collection."""

    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize ApiKeysDAL.

        Args:
            db: MongoDB database
        """
        self.collection = db[Collections.API_KEYS.value]

    async def create(self, api_key: ApiKey) -> uuid.UUID:
        """
        Create a new API key.

        Args:
            api_key: ApiKey model

        Returns:
            uuid.UUID: API key ID
        """
        # Convert ApiKey model to MongoDB document
        doc = api_key_to_document(api_key)
        
        try:
            # Insert document
            result = await self.collection.insert_one(doc)
            logger.info(f"Created API key {api_key.id} with name '{api_key.name}'")
            return api_key.id
        except DuplicateKeyError:
            logger.error(f"API key {api_key.id} already exists")
            raise
        except PyMongoError as e:
            logger.error(f"Error creating API key: {e}")
            raise

    async def get_by_id(self, api_key_id: uuid.UUID) -> Optional[ApiKey]:
        """
        Get an API key by ID.

        Args:
            api_key_id: API key ID

        Returns:
            ApiKey: API key model or None if not found
        """
        try:
            doc = await self.collection.find_one({"_id": api_key_id})
            if doc:
                logger.debug(f"Found API key {api_key_id}")
                return document_to_api_key(doc)
            logger.debug(f"API key {api_key_id} not found")
            return None
        except PyMongoError as e:
            logger.error(f"Error getting API key: {e}")
            raise
            
    async def get_by_key(self, key: str) -> Optional[ApiKey]:
        """
        Get an API key by the key string.

        Args:
            key: API key string

        Returns:
            ApiKey: API key model or None if not found
        """
        try:
            doc = await self.collection.find_one({"key": key})
            if doc:
                logger.debug(f"Found API key for '{key[:8]}...'")
                return document_to_api_key(doc)
            logger.debug(f"API key for '{key[:8]}...' not found")
            return None
        except PyMongoError as e:
            logger.error(f"Error getting API key: {e}")
            raise

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[ApiKey]:
        """
        Get an API key by hash.

        Args:
            key_hash: API key hash

        Returns:
            ApiKey: API key model or None if not found
        """
        try:
            doc = await self.collection.find_one({"key_hash": key_hash})
            if doc:
                logger.debug(f"Found API key with hash {key_hash[:8]}...")
                return document_to_api_key(doc)
            logger.debug(f"API key with hash {key_hash[:8]}... not found")
            return None
        except PyMongoError as e:
            logger.error(f"Error getting API key: {e}")
            raise

    async def get_api_key_by_name(self, name: str) -> Optional[ApiKey]:
        """
        Get an API key by name.

        Args:
            name: API key name

        Returns:
            ApiKey: API key model or None if not found
        """
        try:
            doc = await self.collection.find_one({"name": name})
            if doc:
                logger.debug(f"Found API key {name}")
                return document_to_api_key(doc)
            logger.debug(f"API key {name} not found")
            return None
        except PyMongoError as e:
            logger.error(f"Error getting API key: {e}")
            raise

    async def update(
        self, api_key_id: uuid.UUID, update: Dict[str, Any]
    ) -> Optional[ApiKey]:
        """
        Update an API key.

        Args:
            api_key_id: API key ID
            update: Update document

        Returns:
            ApiKey: Updated API key or None if not found
        """
        try:
            # Update document and return the updated version
            doc = await self.collection.find_one_and_update(
                {"_id": api_key_id},
                {"$set": update},
                return_document=ReturnDocument.AFTER,
            )
            if doc:
                logger.info(f"Updated API key {api_key_id}")
                return document_to_api_key(doc)
            logger.warning(f"API key {api_key_id} not found for update")
            return None
        except PyMongoError as e:
            logger.error(f"Error updating API key: {e}")
            raise

    async def increment_job_count(self, api_key_id: uuid.UUID) -> Optional[ApiKey]:
        """
        Increment jobs_created counter for an API key.

        Args:
            api_key_id: API key ID

        Returns:
            ApiKey: Updated API key or None if not found
        """
        try:
            # Update document and return the updated version
            doc = await self.collection.find_one_and_update(
                {"_id": api_key_id},
                {
                    "$inc": {"usage.jobs_created": 1},
                    "$set": {"last_used_at": datetime.utcnow()}
                },
                return_document=ReturnDocument.AFTER,
            )
            if doc:
                logger.debug(f"Incremented job count for API key {api_key_id}")
                return document_to_api_key(doc)
            logger.warning(f"API key {api_key_id} not found for usage increment")
            return None
        except PyMongoError as e:
            logger.error(f"Error incrementing API key job count: {e}")
            raise
            
    async def increment_upload_count(self, api_key_id: uuid.UUID, size_mb: float = 0.0) -> Optional[ApiKey]:
        """
        Increment upload counter for an API key.

        Args:
            api_key_id: API key ID
            size_mb: Size of uploaded file in MB

        Returns:
            ApiKey: Updated API key or None if not found
        """
        try:
            # Update document and return the updated version
            update = {
                "$inc": {
                    "usage.upload_count": 1,
                    "usage.total_video_size_mb": size_mb
                },
                "$set": {"last_used_at": datetime.utcnow()}
            }
            
            doc = await self.collection.find_one_and_update(
                {"_id": api_key_id},
                update,
                return_document=ReturnDocument.AFTER,
            )
            if doc:
                logger.debug(f"Incremented upload count for API key {api_key_id}")
                return document_to_api_key(doc)
            logger.warning(f"API key {api_key_id} not found for usage increment")
            return None
        except PyMongoError as e:
            logger.error(f"Error incrementing API key upload count: {e}")
            raise

    async def list(
        self,
        active_only: bool = True,
        limit: int = 100,
        skip: int = 0,
    ) -> List[ApiKey]:
        """
        List API keys.

        Args:
            active_only: Whether to return only active keys
            limit: Maximum number of keys to return
            skip: Number of keys to skip

        Returns:
            List[ApiKey]: List of API keys
        """
        # Build query
        query = {}
        if active_only:
            query["active"] = True

        try:
            # Execute query
            cursor = (
                self.collection.find(query)
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit)
            )
            
            # Convert documents to ApiKey models
            keys = [document_to_api_key(doc) async for doc in cursor]
            logger.debug(f"Listed {len(keys)} API keys")
            return keys
        except PyMongoError as e:
            logger.error(f"Error listing API keys: {e}")
            raise

    async def delete(self, api_key_id: uuid.UUID) -> bool:
        """
        Delete an API key.

        Args:
            api_key_id: API key ID

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            result = await self.collection.delete_one({"_id": api_key_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted API key {api_key_id}")
                return True
            logger.warning(f"API key {api_key_id} not found for deletion")
            return False
        except PyMongoError as e:
            logger.error(f"Error deleting API key: {e}")
            raise

    async def validate_key(self, api_key: str) -> Optional[ApiKey]:
        """
        Validate an API key and update last_used_at timestamp.
        
        Args:
            api_key: API key string
            
        Returns:
            ApiKey: API key model if valid, None if invalid or inactive
        """
        try:
            # Find and update in one operation
            doc = await self.collection.find_one_and_update(
                {"key": api_key, "active": True},
                {"$set": {"last_used_at": datetime.utcnow()}},
                return_document=ReturnDocument.AFTER
            )
            
            if doc:
                logger.debug(f"Validated API key '{api_key[:8]}...'")
                return document_to_api_key(doc)
            
            # Check if key exists but is inactive
            doc = await self.collection.find_one({"key": api_key})
            if doc:
                logger.warning(f"API key '{api_key[:8]}...' is inactive")
            else:
                logger.warning(f"API key '{api_key[:8]}...' not found")
                
            return None
        except PyMongoError as e:
            logger.error(f"Error validating API key: {e}")
            raise


class DAL:
    """Data Access Layer for VSR API."""

    def __init__(self, collections: Collections):
        """
        Initialize DAL.

        Args:
            collections: MongoDB collections
        """
        self.collections = collections
        self.jobs = JobsDAL(collections.jobs)
        self.job_events = JobEventsDAL(collections.job_events)
        self.api_keys = ApiKeysDAL(collections.api_keys)


async def get_dal() -> DAL:
    """
    Get DAL instance.

    Returns:
        DAL: Data Access Layer
    """
    collections = await get_collections()
    return DAL(collections)
