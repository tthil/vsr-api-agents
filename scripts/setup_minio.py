#!/usr/bin/env python3
"""
Setup script for MinIO bucket and lifecycle policies for local development.
This script creates the vsr-videos bucket and configures lifecycle policies.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional

# Default MinIO settings
DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"
DEFAULT_BUCKET = "vsr-videos"


def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command.

    Args:
        cmd: Command to run
        check: Whether to check for errors

    Returns:
        CompletedProcess: Result of the command
    """
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def check_mc_installed() -> bool:
    """
    Check if MinIO Client (mc) is installed.

    Returns:
        bool: True if mc is installed, False otherwise
    """
    try:
        result = run_command(["mc", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def setup_mc_alias(
    alias: str = "minio",
    endpoint: str = DEFAULT_ENDPOINT,
    access_key: str = DEFAULT_ACCESS_KEY,
    secret_key: str = DEFAULT_SECRET_KEY,
) -> bool:
    """
    Set up MinIO Client alias.

    Args:
        alias: Alias name
        endpoint: MinIO endpoint URL
        access_key: MinIO access key
        secret_key: MinIO secret key

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        run_command(["mc", "alias", "set", alias, endpoint, access_key, secret_key])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up MinIO alias: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def create_bucket(
    alias: str = "minio", bucket: str = DEFAULT_BUCKET
) -> bool:
    """
    Create MinIO bucket.

    Args:
        alias: MinIO alias
        bucket: Bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if bucket exists
        result = run_command(["mc", "ls", f"{alias}/{bucket}"], check=False)
        if result.returncode == 0:
            print(f"Bucket {bucket} already exists")
            return True

        # Create bucket
        run_command(["mc", "mb", f"{alias}/{bucket}"])
        print(f"Created bucket {bucket}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating bucket: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def setup_lifecycle_policies(
    alias: str = "minio", bucket: str = DEFAULT_BUCKET
) -> bool:
    """
    Set up lifecycle policies for MinIO bucket.

    Args:
        alias: MinIO alias
        bucket: Bucket name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Add lifecycle rules
        run_command(["mc", "ilm", "add", "--expiry-days", "7", "--prefix", "uploads/", f"{alias}/{bucket}"])
        print("Added lifecycle rule for uploads/ (7 days)")

        run_command(["mc", "ilm", "add", "--expiry-days", "30", "--prefix", "processed/", f"{alias}/{bucket}"])
        print("Added lifecycle rule for processed/ (30 days)")

        run_command(["mc", "ilm", "add", "--expiry-days", "90", "--prefix", "logs/", f"{alias}/{bucket}"])
        print("Added lifecycle rule for logs/ (90 days)")

        # List lifecycle rules
        result = run_command(["mc", "ilm", "ls", f"{alias}/{bucket}"])
        print("\nConfigured lifecycle rules:")
        print(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up lifecycle policies: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main() -> None:
    """Run the MinIO setup script."""
    parser = argparse.ArgumentParser(description="Set up MinIO bucket and lifecycle policies")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MINIO_ENDPOINT", DEFAULT_ENDPOINT),
        help="MinIO endpoint URL",
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("MINIO_ACCESS_KEY", DEFAULT_ACCESS_KEY),
        help="MinIO access key",
    )
    parser.add_argument(
        "--secret-key",
        default=os.environ.get("MINIO_SECRET_KEY", DEFAULT_SECRET_KEY),
        help="MinIO secret key",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET),
        help="MinIO bucket name",
    )
    parser.add_argument(
        "--alias",
        default="minio",
        help="MinIO alias name",
    )
    parser.add_argument(
        "--skip-lifecycle",
        action="store_true",
        help="Skip setting up lifecycle policies",
    )

    args = parser.parse_args()

    # Check if mc is installed
    if not check_mc_installed():
        print("Error: MinIO Client (mc) is not installed.")
        print("Please install it from https://min.io/docs/minio/linux/reference/minio-mc.html")
        sys.exit(1)

    # Set up MinIO alias
    if not setup_mc_alias(
        alias=args.alias,
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
    ):
        sys.exit(1)

    # Create bucket
    if not create_bucket(alias=args.alias, bucket=args.bucket):
        sys.exit(1)

    # Set up lifecycle policies
    if not args.skip_lifecycle:
        if not setup_lifecycle_policies(alias=args.alias, bucket=args.bucket):
            sys.exit(1)

    print("\nMinIO setup completed successfully!")


if __name__ == "__main__":
    main()
