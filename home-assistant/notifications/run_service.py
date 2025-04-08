#!/usr/bin/env python3

import asyncio
import argparse
import logging
import signal
import os
import sys

from notification_service import AINotificationService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('notification_service.log')
    ]
)
logger = logging.getLogger("run_service")

# Global flag for shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle termination signals"""
    global shutdown_requested
    logger.info(f"Received signal {sig}, shutting down...")
    shutdown_requested = True

async def run_service(config_path: str):
    """Run the notification service with graceful shutdown"""
    global shutdown_requested

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create the service
    service = AINotificationService(config_path)

    try:
        # Initialize connections
        await service.triton_client.initialize()
        await service.load_user_preferences()

        logger.info("AI Notification Service started")

        # Main service loop
        while not shutdown_requested:
            try:
                await service.process_events()

                # Check if shutdown was requested during processing
                if shutdown_requested:
                    break

                await asyncio.sleep(service.config.general["polling_interval"])

            except Exception as e:
                logger.error(f"Error in service loop: {e}", exc_info=True)
                # Brief pause on error before retrying
                await asyncio.sleep(5)

        logger.info("Service shutdown initiated")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Close connections
        logger.info("Closing connections...")
        await service.triton_client.close()
        await service.ha_client.close()

    logger.info("AI Notification Service stopped")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Notification Service for Home Assistant")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    exit_code = asyncio.run(run_service(args.config))
    sys.exit(exit_code)
