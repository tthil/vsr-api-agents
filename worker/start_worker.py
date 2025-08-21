#!/usr/bin/env python3
"""
Dual-mode worker startup script.

Automatically detects environment and starts the appropriate worker mode.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add worker directory to Python path
worker_dir = Path(__file__).parent
sys.path.insert(0, str(worker_dir))

from vsr_worker.gpu.environment import WorkerEnvironment
from vsr_worker.config.dual_mode import get_config_manager, print_config_summary
from vsr_worker.dual_mode_consumer import DualModeVideoJobConsumer
from vsr_shared.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main entry point for worker startup."""
    print("=" * 60)
    print("VSR Dual-Mode Worker Starting...")
    print("=" * 60)
    
    # Print configuration summary
    print_config_summary()
    
    # Initialize and start consumer
    consumer = DualModeVideoJobConsumer()
    
    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Worker startup failed: {e}")
        sys.exit(1)
    finally:
        await consumer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
