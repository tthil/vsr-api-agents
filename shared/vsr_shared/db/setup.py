"""MongoDB setup script for VSR API."""

import asyncio
import sys
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from vsr_shared.db.client import get_db, get_mongodb_client
from vsr_shared.db.collections import create_collections_with_validation
from vsr_shared.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def setup_mongodb(db_name: Optional[str] = None) -> None:
    """
    Set up MongoDB collections and indexes.

    Args:
        db_name: Optional database name override
    """
    # Connect to MongoDB
    client = get_mongodb_client(db_name=db_name)
    await client.connect()
    
    try:
        # Check connection
        if not await client.ping():
            logger.error("MongoDB connection failed")
            sys.exit(1)
        
        # Get database
        db = client.get_database()
        
        # Create collections with validation
        await create_collections_with_validation(db)
        
        logger.info("MongoDB setup completed successfully")
    finally:
        # Close connection
        await client.close()


async def drop_collections(db_name: Optional[str] = None) -> None:
    """
    Drop all collections in the database.
    
    WARNING: This will delete all data!

    Args:
        db_name: Optional database name override
    """
    # Connect to MongoDB
    client = get_mongodb_client(db_name=db_name)
    await client.connect()
    
    try:
        # Check connection
        if not await client.ping():
            logger.error("MongoDB connection failed")
            sys.exit(1)
        
        # Get database
        db = client.get_database()
        
        # List collections
        collections = await db.list_collection_names()
        
        # Drop each collection
        for collection_name in collections:
            logger.warning(f"Dropping collection {collection_name}")
            await db.drop_collection(collection_name)
        
        logger.info(f"Dropped {len(collections)} collections")
    finally:
        # Close connection
        await client.close()


def main() -> None:
    """Run the MongoDB setup script."""
    # Set up logging
    setup_logging(level="INFO", json_format=False)
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="MongoDB setup script for VSR API")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all collections before setup (WARNING: This will delete all data!)",
    )
    parser.add_argument(
        "--db-name",
        help="Database name override",
    )
    
    args = parser.parse_args()
    
    # Run setup
    if args.drop:
        logger.warning("Dropping all collections")
        asyncio.run(drop_collections(db_name=args.db_name))
    
    logger.info("Setting up MongoDB collections and indexes")
    asyncio.run(setup_mongodb(db_name=args.db_name))


if __name__ == "__main__":
    main()
