"""Database package for VSR API."""

from vsr_shared.db.client import (
    MongoDBClient,
    get_db,
    get_mongodb_client,
    mongodb_lifespan,
)

__all__ = ["MongoDBClient", "get_db", "get_mongodb_client", "mongodb_lifespan"]
