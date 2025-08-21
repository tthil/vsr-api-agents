"""Main worker module for video subtitle removal."""

import argparse
import os
import sys
import time
from typing import Dict, Optional

from vsr_shared.logging import get_logger, setup_logging
from vsr_worker.config import get_settings
from vsr_worker.mock import run_mock_worker


# Logger for this module
logger = get_logger(__name__)


def main() -> None:
    """Main entry point for the worker."""
    parser = argparse.ArgumentParser(description="Video Subtitle Removal Worker")
    parser.add_argument(
        "--mode",
        choices=["gpu", "mock"],
        default=os.environ.get("WORKER_MODE", "gpu"),
        help="Worker mode: gpu or mock",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = args.log_level.upper()
    setup_logging(level=log_level, json_format=True)

    # Load settings
    settings = get_settings()
    
    logger.info(
        "Starting worker",
        mode=args.mode,
        version=__import__("vsr_worker").__version__,
    )

    try:
        if args.mode == "mock":
            run_mock_worker(settings)
        else:
            # This will be implemented in Task 7
            logger.error("GPU mode not yet implemented")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.exception("Worker failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
