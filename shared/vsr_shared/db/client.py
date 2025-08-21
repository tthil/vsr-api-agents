"""MongoDB client factory for VSR API."""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson.codec_options import CodecOptions
from pymongo.read_preferences import ReadPreference
from bson.binary import UuidRepresentation

from vsr_shared.logging import get_logger

logger = get_logger(__name__)

# Default MongoDB connection settings
DEFAULT_DB_NAME = "vsr_api"
DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_APP_NAME = "vsr-api"

# Connection timeouts
CONNECT_TIMEOUT_MS = 5000
SERVER_SELECTION_TIMEOUT_MS = 5000


class MongoDBClient:
    """MongoDB client wrapper for VSR API."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        app_name: Optional[str] = None,
        connect_timeout_ms: int = CONNECT_TIMEOUT_MS,
        server_selection_timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
        retry_writes: bool = True,
        read_preference: Optional[str] = None,
    ):
        """
        Initialize the MongoDB client.

        Args:
            uri: MongoDB connection URI
            db_name: Database name
            app_name: Application name
            connect_timeout_ms: Connection timeout in milliseconds
            server_selection_timeout_ms: Server selection timeout in milliseconds
            retry_writes: Whether to retry writes
            read_preference: Read preference (primary, secondary, etc.)
        """
        self.uri = uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGODB_URL", DEFAULT_URI)
        self.db_name = db_name or os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME)
        self.app_name = app_name or os.environ.get("MONGODB_APP_NAME", DEFAULT_APP_NAME)

        # Connection options
        self.options = {
            "connectTimeoutMS": connect_timeout_ms,
            "serverSelectionTimeoutMS": server_selection_timeout_ms,
            "retryWrites": retry_writes,
            "appname": self.app_name,
            "uuidRepresentation": "standard",
        }

        # Add read preference if provided
        if read_preference:
            self.options["readPreference"] = read_preference

        # Initialize client
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> AsyncIOMotorClient:
        """
        Connect to MongoDB.

        Returns:
            AsyncIOMotorClient: MongoDB client
        """
        if self.client is None:
            logger.info(f"Connecting to MongoDB at {self._sanitize_uri(self.uri)}")
            self.client = AsyncIOMotorClient(self.uri, **self.options)
            self.db = self.client[self.db_name]

        return self.client

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            logger.info("Closing MongoDB connection")
            self.client.close()
            self.client = None
            self.db = None

    async def ping(self) -> bool:
        """
        Ping the MongoDB server to check connectivity.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        if not self.client:
            await self.connect()

        try:
            # Use admin command to ping the server
            await self.client.admin.command("ping")
            logger.debug("MongoDB ping successful")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the MongoDB database.

        Returns:
            AsyncIOMotorDatabase: MongoDB database
        """
        if self.db is None:
            raise RuntimeError("MongoDB client not connected")
        return self.db

    def _sanitize_uri(self, uri: str) -> str:
        """
        Sanitize the MongoDB URI for logging.

        Args:
            uri: MongoDB URI

        Returns:
            str: Sanitized URI with password redacted
        """
        if "@" in uri:
            # Simple redaction of password in URI
            parts = uri.split("@")
            auth_parts = parts[0].split(":")
            if len(auth_parts) > 2:
                # Handle special case where there are multiple colons in the auth part
                auth_parts[2] = "****"
                parts[0] = ":".join(auth_parts)
            elif len(auth_parts) == 2:
                # Standard case: username:password
                auth_parts[1] = "****"
                parts[0] = ":".join(auth_parts)
            return "@".join(parts)
        return uri


# Client factory
_mongo_client: Optional[MongoDBClient] = None


def get_mongodb_client(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    app_name: Optional[str] = None,
    connect_timeout_ms: int = CONNECT_TIMEOUT_MS,
    server_selection_timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
    retry_writes: bool = True,
    read_preference: Optional[str] = None,
    force_new: bool = False,
) -> MongoDBClient:
    """
    Get a MongoDB client instance.

    Args:
        uri: MongoDB connection URI
        db_name: Database name
        app_name: Application name
        connect_timeout_ms: Connection timeout in milliseconds
        server_selection_timeout_ms: Server selection timeout in milliseconds
        retry_writes: Whether to retry writes
        read_preference: Read preference (primary, secondary, etc.)
        force_new: Force creation of a new client

    Returns:
        MongoDBClient instance
    """
    global _mongo_client

    if _mongo_client is None or force_new:
        _mongo_client = MongoDBClient(
            uri=uri,
            db_name=db_name,
            app_name=app_name,
            connect_timeout_ms=connect_timeout_ms,
            server_selection_timeout_ms=server_selection_timeout_ms,
            retry_writes=retry_writes,
            read_preference=read_preference,
        )

    return _mongo_client


async def get_db() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database.

    Returns:
        AsyncIOMotorDatabase: MongoDB database
    """
    client = get_mongodb_client()
    await client.connect()
    return client.get_database()


@asynccontextmanager
async def mongodb_lifespan():
    """
    MongoDB lifespan context manager for FastAPI.

    Yields:
        None
    """
    client = get_mongodb_client()
    try:
        await client.connect()
        # Verify connection with ping
        if await client.ping():
            logger.info("MongoDB connection established")
        else:
            logger.error("MongoDB connection failed")
        yield
    finally:
        await client.close()
