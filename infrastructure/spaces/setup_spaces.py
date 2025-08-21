"""
DigitalOcean Spaces configuration and setup script.
Configures bucket policies, CORS, lifecycle rules, and CDN settings.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config


logger = logging.getLogger(__name__)


class SpacesConfigurator:
    """
    DigitalOcean Spaces configuration manager.
    """
    
    def __init__(self, 
                 endpoint_url: str,
                 access_key: str,
                 secret_key: str,
                 region: str = "nyc3"):
        """
        Initialize Spaces configurator.
        
        Args:
            endpoint_url: DigitalOcean Spaces endpoint URL
            access_key: Spaces access key
            secret_key: Spaces secret key
            region: Spaces region
        """
        self.endpoint_url = endpoint_url
        self.region = region
        
        # Configure boto3 client for DigitalOcean Spaces
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version='s3v4',
                s3={
                    'addressing_style': 'virtual'
                }
            )
        )
        
        self.cdn_client = boto3.client(
            'cloudfront',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'  # CloudFront requires us-east-1
        )
    
    async def create_bucket(self, bucket_name: str) -> bool:
        """
        Create a new Spaces bucket if it doesn't exist.
        
        Args:
            bucket_name: Name of the bucket to create
            
        Returns:
            True if bucket was created or already exists
        """
        try:
            # Check if bucket already exists
            try:
                self.client.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket {bucket_name} already exists")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise
            
            # Create bucket
            if self.region == 'nyc3':
                # Default region doesn't need LocationConstraint
                self.client.create_bucket(Bucket=bucket_name)
            else:
                self.client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            logger.info(f"Created bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            return False
    
    async def configure_bucket_policy(self, bucket_name: str) -> bool:
        """
        Configure bucket policy for secure access.
        
        Args:
            bucket_name: Name of the bucket to configure
            
        Returns:
            True if policy was configured successfully
        """
        try:
            # Define bucket policy for private access with presigned URLs
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "DenyDirectPublicAccess",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*",
                        "Condition": {
                            "Bool": {
                                "aws:SecureTransport": "false"
                            }
                        }
                    },
                    {
                        "Sid": "AllowPresignedURLAccess",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject"
                        ],
                        "Resource": f"arn:aws:s3:::{bucket_name}/*",
                        "Condition": {
                            "Bool": {
                                "aws:SecureTransport": "true"
                            }
                        }
                    }
                ]
            }
            
            # Apply bucket policy
            self.client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
            
            logger.info(f"Configured bucket policy for: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure bucket policy for {bucket_name}: {e}")
            return False
    
    async def configure_cors(self, bucket_name: str) -> bool:
        """
        Configure CORS settings for the bucket.
        
        Args:
            bucket_name: Name of the bucket to configure
            
        Returns:
            True if CORS was configured successfully
        """
        try:
            cors_configuration = {
                'CORSRules': [
                    {
                        'AllowedHeaders': [
                            'Authorization',
                            'Content-Type',
                            'Content-Length',
                            'Content-MD5',
                            'x-amz-date',
                            'x-amz-content-sha256'
                        ],
                        'AllowedMethods': [
                            'GET',
                            'PUT',
                            'POST',
                            'DELETE',
                            'HEAD'
                        ],
                        'AllowedOrigins': [
                            'https://*.brightonsolutions.com',
                            'https://localhost:3000',
                            'https://localhost:8000'
                        ],
                        'ExposeHeaders': [
                            'ETag',
                            'x-amz-request-id'
                        ],
                        'MaxAgeSeconds': 3600
                    }
                ]
            }
            
            self.client.put_bucket_cors(
                Bucket=bucket_name,
                CORSConfiguration=cors_configuration
            )
            
            logger.info(f"Configured CORS for bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure CORS for {bucket_name}: {e}")
            return False
    
    async def configure_lifecycle_policy(self, bucket_name: str) -> bool:
        """
        Configure lifecycle policy for automatic cleanup.
        
        Args:
            bucket_name: Name of the bucket to configure
            
        Returns:
            True if lifecycle policy was configured successfully
        """
        try:
            lifecycle_configuration = {
                'Rules': [
                    {
                        'ID': 'DeleteIncompleteMultipartUploads',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'AbortIncompleteMultipartUpload': {
                            'DaysAfterInitiation': 1
                        }
                    },
                    {
                        'ID': 'DeleteTempFiles',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': 'temp/'},
                        'Expiration': {
                            'Days': 1
                        }
                    },
                    {
                        'ID': 'ArchiveProcessedVideos',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': 'processed/'},
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            }
                        ]
                    },
                    {
                        'ID': 'DeleteOldUploads',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': 'uploads/'},
                        'Expiration': {
                            'Days': 7
                        }
                    }
                ]
            }
            
            self.client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration=lifecycle_configuration
            )
            
            logger.info(f"Configured lifecycle policy for bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure lifecycle policy for {bucket_name}: {e}")
            return False
    
    async def configure_encryption(self, bucket_name: str) -> bool:
        """
        Configure server-side encryption for the bucket.
        
        Args:
            bucket_name: Name of the bucket to configure
            
        Returns:
            True if encryption was configured successfully
        """
        try:
            encryption_configuration = {
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
            
            self.client.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration=encryption_configuration
            )
            
            logger.info(f"Configured encryption for bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure encryption for {bucket_name}: {e}")
            return False
    
    async def configure_versioning(self, bucket_name: str, enabled: bool = False) -> bool:
        """
        Configure versioning for the bucket.
        
        Args:
            bucket_name: Name of the bucket to configure
            enabled: Whether to enable versioning
            
        Returns:
            True if versioning was configured successfully
        """
        try:
            status = 'Enabled' if enabled else 'Suspended'
            
            self.client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': status}
            )
            
            logger.info(f"Configured versioning ({status}) for bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure versioning for {bucket_name}: {e}")
            return False
    
    async def setup_cdn(self, bucket_name: str, domain_name: Optional[str] = None) -> Optional[str]:
        """
        Setup CDN distribution for the bucket.
        
        Args:
            bucket_name: Name of the bucket
            domain_name: Optional custom domain name
            
        Returns:
            CDN distribution domain name if successful
        """
        try:
            # Note: DigitalOcean Spaces CDN is automatically enabled
            # This method would configure CloudFront if using AWS S3
            
            # For DigitalOcean Spaces, CDN is available at:
            cdn_endpoint = f"{bucket_name}.{self.region}.cdn.digitaloceanspaces.com"
            
            logger.info(f"CDN endpoint for {bucket_name}: {cdn_endpoint}")
            return cdn_endpoint
            
        except Exception as e:
            logger.error(f"Failed to setup CDN for {bucket_name}: {e}")
            return None
    
    async def validate_configuration(self, bucket_name: str) -> Dict[str, Any]:
        """
        Validate bucket configuration.
        
        Args:
            bucket_name: Name of the bucket to validate
            
        Returns:
            Validation results
        """
        results = {
            "bucket_exists": False,
            "cors_configured": False,
            "lifecycle_configured": False,
            "encryption_configured": False,
            "policy_configured": False,
            "errors": []
        }
        
        try:
            # Check bucket existence
            self.client.head_bucket(Bucket=bucket_name)
            results["bucket_exists"] = True
            
            # Check CORS configuration
            try:
                self.client.get_bucket_cors(Bucket=bucket_name)
                results["cors_configured"] = True
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchCORSConfiguration':
                    results["errors"].append(f"CORS check failed: {e}")
            
            # Check lifecycle configuration
            try:
                self.client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                results["lifecycle_configured"] = True
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchLifecycleConfiguration':
                    results["errors"].append(f"Lifecycle check failed: {e}")
            
            # Check encryption configuration
            try:
                self.client.get_bucket_encryption(Bucket=bucket_name)
                results["encryption_configured"] = True
            except ClientError as e:
                if e.response['Error']['Code'] != 'ServerSideEncryptionConfigurationNotFoundError':
                    results["errors"].append(f"Encryption check failed: {e}")
            
            # Check bucket policy
            try:
                self.client.get_bucket_policy(Bucket=bucket_name)
                results["policy_configured"] = True
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchBucketPolicy':
                    results["errors"].append(f"Policy check failed: {e}")
            
        except Exception as e:
            results["errors"].append(f"Validation failed: {e}")
        
        return results
    
    async def setup_complete_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """
        Setup a complete bucket with all configurations.
        
        Args:
            bucket_name: Name of the bucket to setup
            
        Returns:
            Setup results
        """
        results = {
            "bucket_created": False,
            "policy_configured": False,
            "cors_configured": False,
            "lifecycle_configured": False,
            "encryption_configured": False,
            "versioning_configured": False,
            "cdn_endpoint": None,
            "errors": []
        }
        
        try:
            logger.info(f"Setting up complete bucket configuration for: {bucket_name}")
            
            # Step 1: Create bucket
            if await self.create_bucket(bucket_name):
                results["bucket_created"] = True
            else:
                results["errors"].append("Failed to create bucket")
                return results
            
            # Step 2: Configure bucket policy
            if await self.configure_bucket_policy(bucket_name):
                results["policy_configured"] = True
            else:
                results["errors"].append("Failed to configure bucket policy")
            
            # Step 3: Configure CORS
            if await self.configure_cors(bucket_name):
                results["cors_configured"] = True
            else:
                results["errors"].append("Failed to configure CORS")
            
            # Step 4: Configure lifecycle policy
            if await self.configure_lifecycle_policy(bucket_name):
                results["lifecycle_configured"] = True
            else:
                results["errors"].append("Failed to configure lifecycle policy")
            
            # Step 5: Configure encryption
            if await self.configure_encryption(bucket_name):
                results["encryption_configured"] = True
            else:
                results["errors"].append("Failed to configure encryption")
            
            # Step 6: Configure versioning (disabled by default)
            if await self.configure_versioning(bucket_name, enabled=False):
                results["versioning_configured"] = True
            else:
                results["errors"].append("Failed to configure versioning")
            
            # Step 7: Setup CDN
            cdn_endpoint = await self.setup_cdn(bucket_name)
            if cdn_endpoint:
                results["cdn_endpoint"] = cdn_endpoint
            
            logger.info(f"Completed bucket setup for: {bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup bucket {bucket_name}: {e}")
            results["errors"].append(f"Setup failed: {e}")
        
        return results


async def main():
    """Main setup function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration from environment
    endpoint_url = os.getenv('VSR_SPACES_ENDPOINT')
    access_key = os.getenv('VSR_SPACES_ACCESS_KEY')
    secret_key = os.getenv('VSR_SPACES_SECRET_KEY')
    bucket_name = os.getenv('VSR_SPACES_BUCKET')
    region = os.getenv('VSR_SPACES_REGION', 'nyc3')
    
    if not all([endpoint_url, access_key, secret_key, bucket_name]):
        logger.error("Missing required environment variables")
        logger.error("Required: VSR_SPACES_ENDPOINT, VSR_SPACES_ACCESS_KEY, VSR_SPACES_SECRET_KEY, VSR_SPACES_BUCKET")
        return
    
    # Initialize configurator
    configurator = SpacesConfigurator(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        region=region
    )
    
    try:
        # Setup complete bucket configuration
        logger.info("Starting DigitalOcean Spaces configuration...")
        results = await configurator.setup_complete_bucket(bucket_name)
        
        # Print results
        logger.info("Configuration Results:")
        for key, value in results.items():
            if key != "errors":
                logger.info(f"  {key}: {value}")
        
        if results["errors"]:
            logger.error("Errors encountered:")
            for error in results["errors"]:
                logger.error(f"  - {error}")
        
        # Validate configuration
        logger.info("Validating configuration...")
        validation = await configurator.validate_configuration(bucket_name)
        
        logger.info("Validation Results:")
        for key, value in validation.items():
            if key != "errors":
                logger.info(f"  {key}: {value}")
        
        if validation["errors"]:
            logger.error("Validation errors:")
            for error in validation["errors"]:
                logger.error(f"  - {error}")
        
        logger.info("DigitalOcean Spaces configuration completed!")
        
    except Exception as e:
        logger.error(f"Configuration failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
