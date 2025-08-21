#!/usr/bin/env python3
"""
Data retention and cleanup system for VSR API.

Implements storage management with Spaces lifecycle rules, cleanup scripts,
and automated backup procedures for MongoDB and file storage.
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import pymongo
from pymongo import MongoClient
import subprocess
import json
import structlog

logger = structlog.get_logger(__name__)


class StorageCleanupManager:
    """
    Manages data retention and cleanup for VSR API storage systems.
    
    Handles cleanup of temporary files, old processed videos, database records,
    and implements automated backup procedures.
    """
    
    def __init__(self):
        # Spaces/S3 configuration
        self.spaces_endpoint = os.getenv("SPACES_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
        self.spaces_key = os.getenv("SPACES_KEY")
        self.spaces_secret = os.getenv("SPACES_SECRET")
        self.spaces_bucket = os.getenv("SPACES_BUCKET", "vsr-videos")
        self.backup_bucket = os.getenv("BACKUP_BUCKET", "vsr-backups")
        
        # MongoDB configuration
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/vsr")
        
        # Retention policies (days)
        self.temp_file_retention = int(os.getenv("TEMP_FILE_RETENTION_DAYS", "1"))
        self.processed_video_retention = int(os.getenv("PROCESSED_VIDEO_RETENTION_DAYS", "30"))
        self.job_record_retention = int(os.getenv("JOB_RECORD_RETENTION_DAYS", "90"))
        self.backup_retention = int(os.getenv("BACKUP_RETENTION_DAYS", "365"))
        
        # Initialize clients
        self.s3_client = None
        self.mongo_client = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize storage clients."""
        try:
            # Initialize S3/Spaces client
            if self.spaces_key and self.spaces_secret:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.spaces_endpoint,
                    aws_access_key_id=self.spaces_key,
                    aws_secret_access_key=self.spaces_secret,
                    region_name='nyc3'
                )
                logger.info("Spaces client initialized")
            
            # Initialize MongoDB client
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client.get_default_database()
            logger.info("MongoDB client initialized")
            
        except Exception as e:
            logger.error("Failed to initialize clients", error=str(e))
            raise
    
    async def run_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive cleanup process.
        
        Args:
            dry_run: If True, only report what would be cleaned up
            
        Returns:
            Dictionary with cleanup results
        """
        logger.info("Starting cleanup process", dry_run=dry_run)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": dry_run,
            "cleanup_results": {}
        }
        
        try:
            # Clean up temporary files
            temp_results = await self._cleanup_temp_files(dry_run)
            results["cleanup_results"]["temp_files"] = temp_results
            
            # Clean up old processed videos
            video_results = await self._cleanup_old_videos(dry_run)
            results["cleanup_results"]["processed_videos"] = video_results
            
            # Clean up old job records
            job_results = await self._cleanup_old_jobs(dry_run)
            results["cleanup_results"]["job_records"] = job_results
            
            # Clean up old backups
            backup_results = await self._cleanup_old_backups(dry_run)
            results["cleanup_results"]["old_backups"] = backup_results
            
            # Verify lifecycle rules
            lifecycle_results = await self._verify_lifecycle_rules()
            results["cleanup_results"]["lifecycle_verification"] = lifecycle_results
            
            logger.info("Cleanup process completed", results=results)
            
        except Exception as e:
            logger.error("Cleanup process failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def _cleanup_temp_files(self, dry_run: bool) -> Dict[str, Any]:
        """Clean up temporary files older than retention period."""
        logger.info("Cleaning up temporary files", retention_days=self.temp_file_retention)
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.temp_file_retention)
        temp_prefixes = ["temp/", "uploads/tmp/", "processing/tmp/"]
        
        deleted_count = 0
        deleted_size = 0
        errors = []
        
        try:
            for prefix in temp_prefixes:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.spaces_bucket, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    for obj in page['Contents']:
                        if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                            if not dry_run:
                                try:
                                    self.s3_client.delete_object(
                                        Bucket=self.spaces_bucket,
                                        Key=obj['Key']
                                    )
                                    logger.debug("Deleted temp file", key=obj['Key'])
                                except ClientError as e:
                                    errors.append(f"Failed to delete {obj['Key']}: {str(e)}")
                                    continue
                            
                            deleted_count += 1
                            deleted_size += obj['Size']
        
        except Exception as e:
            errors.append(f"Temp file cleanup error: {str(e)}")
        
        return {
            "files_deleted": deleted_count,
            "bytes_freed": deleted_size,
            "errors": errors
        }
    
    async def _cleanup_old_videos(self, dry_run: bool) -> Dict[str, Any]:
        """Clean up old processed videos beyond retention period."""
        logger.info("Cleaning up old processed videos", retention_days=self.processed_video_retention)
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.processed_video_retention)
        
        deleted_count = 0
        deleted_size = 0
        errors = []
        
        try:
            # Find old processed videos
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.spaces_bucket, Prefix="processed/")
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        # Check if job still exists in database
                        job_id = self._extract_job_id_from_key(obj['Key'])
                        if job_id and self._job_exists(job_id):
                            # Skip if job record still exists (not expired)
                            continue
                        
                        if not dry_run:
                            try:
                                self.s3_client.delete_object(
                                    Bucket=self.spaces_bucket,
                                    Key=obj['Key']
                                )
                                logger.debug("Deleted old video", key=obj['Key'])
                            except ClientError as e:
                                errors.append(f"Failed to delete {obj['Key']}: {str(e)}")
                                continue
                        
                        deleted_count += 1
                        deleted_size += obj['Size']
        
        except Exception as e:
            errors.append(f"Video cleanup error: {str(e)}")
        
        return {
            "videos_deleted": deleted_count,
            "bytes_freed": deleted_size,
            "errors": errors
        }
    
    async def _cleanup_old_jobs(self, dry_run: bool) -> Dict[str, Any]:
        """Clean up old job records from MongoDB."""
        logger.info("Cleaning up old job records", retention_days=self.job_record_retention)
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.job_record_retention)
        
        deleted_jobs = 0
        deleted_events = 0
        errors = []
        
        try:
            # Find old completed jobs
            old_jobs_query = {
                "created_at": {"$lt": cutoff_date},
                "status": {"$in": ["completed", "failed", "cancelled"]}
            }
            
            if not dry_run:
                # Get job IDs before deletion for event cleanup
                old_job_ids = [
                    job["_id"] for job in 
                    self.db.jobs.find(old_jobs_query, {"_id": 1})
                ]
                
                # Delete old jobs
                result = self.db.jobs.delete_many(old_jobs_query)
                deleted_jobs = result.deleted_count
                
                # Delete associated job events
                if old_job_ids:
                    event_result = self.db.job_events.delete_many({
                        "job_id": {"$in": old_job_ids}
                    })
                    deleted_events = event_result.deleted_count
            else:
                # Count what would be deleted
                deleted_jobs = self.db.jobs.count_documents(old_jobs_query)
                
                old_job_ids = [
                    job["_id"] for job in 
                    self.db.jobs.find(old_jobs_query, {"_id": 1})
                ]
                
                if old_job_ids:
                    deleted_events = self.db.job_events.count_documents({
                        "job_id": {"$in": old_job_ids}
                    })
        
        except Exception as e:
            errors.append(f"Job cleanup error: {str(e)}")
        
        return {
            "jobs_deleted": deleted_jobs,
            "events_deleted": deleted_events,
            "errors": errors
        }
    
    async def _cleanup_old_backups(self, dry_run: bool) -> Dict[str, Any]:
        """Clean up old backup files."""
        logger.info("Cleaning up old backups", retention_days=self.backup_retention)
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention)
        
        deleted_count = 0
        deleted_size = 0
        errors = []
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.backup_bucket)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        if not dry_run:
                            try:
                                self.s3_client.delete_object(
                                    Bucket=self.backup_bucket,
                                    Key=obj['Key']
                                )
                                logger.debug("Deleted old backup", key=obj['Key'])
                            except ClientError as e:
                                errors.append(f"Failed to delete backup {obj['Key']}: {str(e)}")
                                continue
                        
                        deleted_count += 1
                        deleted_size += obj['Size']
        
        except Exception as e:
            errors.append(f"Backup cleanup error: {str(e)}")
        
        return {
            "backups_deleted": deleted_count,
            "bytes_freed": deleted_size,
            "errors": errors
        }
    
    async def _verify_lifecycle_rules(self) -> Dict[str, Any]:
        """Verify that Spaces lifecycle rules are properly configured."""
        logger.info("Verifying lifecycle rules")
        
        try:
            response = self.s3_client.get_bucket_lifecycle_configuration(
                Bucket=self.spaces_bucket
            )
            
            rules = response.get('Rules', [])
            
            return {
                "lifecycle_rules_configured": len(rules) > 0,
                "rules_count": len(rules),
                "rules": [
                    {
                        "id": rule.get('ID'),
                        "status": rule.get('Status'),
                        "prefix": rule.get('Filter', {}).get('Prefix', ''),
                        "expiration_days": rule.get('Expiration', {}).get('Days')
                    }
                    for rule in rules
                ]
            }
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
                return {
                    "lifecycle_rules_configured": False,
                    "error": "No lifecycle configuration found"
                }
            else:
                return {
                    "lifecycle_rules_configured": False,
                    "error": str(e)
                }
    
    def _extract_job_id_from_key(self, key: str) -> Optional[str]:
        """Extract job ID from S3 object key."""
        try:
            # Assuming format: processed/{job_id}.mp4
            if key.startswith("processed/"):
                filename = key.split("/")[-1]
                job_id = filename.split(".")[0]
                return job_id
        except Exception:
            pass
        return None
    
    def _job_exists(self, job_id: str) -> bool:
        """Check if job record still exists in database."""
        try:
            return self.db.jobs.find_one({"_id": job_id}) is not None
        except Exception:
            return False
    
    async def create_backup(self) -> Dict[str, Any]:
        """Create backup of MongoDB data."""
        logger.info("Creating MongoDB backup")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mongodb_backup_{timestamp}"
        
        try:
            # Create MongoDB dump
            dump_command = [
                "mongodump",
                "--uri", self.mongodb_uri,
                "--out", f"/tmp/{backup_name}"
            ]
            
            result = subprocess.run(dump_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"mongodump failed: {result.stderr}")
            
            # Compress backup
            tar_command = [
                "tar", "-czf", f"/tmp/{backup_name}.tar.gz",
                "-C", "/tmp", backup_name
            ]
            
            subprocess.run(tar_command, check=True)
            
            # Upload to backup bucket
            with open(f"/tmp/{backup_name}.tar.gz", "rb") as backup_file:
                self.s3_client.upload_fileobj(
                    backup_file,
                    self.backup_bucket,
                    f"mongodb/{backup_name}.tar.gz"
                )
            
            # Cleanup local files
            subprocess.run(["rm", "-rf", f"/tmp/{backup_name}"], check=True)
            subprocess.run(["rm", f"/tmp/{backup_name}.tar.gz"], check=True)
            
            logger.info("Backup created successfully", backup_name=backup_name)
            
            return {
                "success": True,
                "backup_name": backup_name,
                "timestamp": timestamp
            }
        
        except Exception as e:
            logger.error("Backup creation failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """Main cleanup script entry point."""
    parser = argparse.ArgumentParser(description="VSR API Storage Cleanup")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be cleaned up without actually doing it")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before cleanup")
    parser.add_argument("--backup-only", action="store_true",
                       help="Only create backup, skip cleanup")
    
    args = parser.parse_args()
    
    cleanup_manager = StorageCleanupManager()
    
    try:
        # Create backup if requested
        if args.backup or args.backup_only:
            backup_result = await cleanup_manager.create_backup()
            print(json.dumps(backup_result, indent=2))
            
            if not backup_result["success"]:
                sys.exit(1)
        
        # Run cleanup unless backup-only
        if not args.backup_only:
            cleanup_result = await cleanup_manager.run_cleanup(dry_run=args.dry_run)
            print(json.dumps(cleanup_result, indent=2))
            
            if cleanup_result.get("error"):
                sys.exit(1)
    
    except Exception as e:
        logger.error("Script execution failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
