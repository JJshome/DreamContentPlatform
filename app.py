#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Content Platform - Main Application Entry Point

This script initializes and runs the Dream Content Platform API server.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add src directory to the Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import API module
from src.api.api import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_file=None):
    """
    Load configuration from file or environment variables.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dict with configuration
    """
    # Default configuration
    config = {
        "HOST": "0.0.0.0",
        "PORT": 5000,
        "DEBUG": False,
        "EEG_SAMPLING_RATE": 1000,
        "EEG_CHANNELS": ["Fp1", "Fp2"],
        "UPLOAD_DIR": os.path.join(os.path.dirname(__file__), "uploads"),
        "CONTENT_OUTPUT_DIR": os.path.join(os.path.dirname(__file__), "output"),
        "MARKETPLACE_DATA_DIR": os.path.join(os.path.dirname(__file__), "marketplace_data"),
        "BLOCKCHAIN_ENABLED": False,
        "BLOCKCHAIN_PROVIDER_URL": "http://localhost:8545",
        "NFT_CONTRACT_ADDRESS": None,
        "NFT_CONTRACT_ABI": None
    }
    
    # Load from configuration file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
    
    # Override with environment variables
    for key in config.keys():
        env_key = f"DREAM_PLATFORM_{key}"
        if env_key in os.environ:
            value = os.environ[env_key]
            
            # Convert string to appropriate type
            if key in ["PORT", "EEG_SAMPLING_RATE"]:
                config[key] = int(value)
            elif key in ["DEBUG", "BLOCKCHAIN_ENABLED"]:
                config[key] = value.lower() in ["true", "1", "yes"]
            elif key in ["EEG_CHANNELS"]:
                config[key] = value.split(",")
            else:
                config[key] = value
                
            logger.info(f"Loaded {key} from environment variable")
    
    # Create required directories
    for dir_key in ["UPLOAD_DIR", "CONTENT_OUTPUT_DIR", "MARKETPLACE_DATA_DIR"]:
        os.makedirs(config[dir_key], exist_ok=True)
    
    return config


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Dream Content Platform")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--blockchain", 
        action="store_true", 
        help="Enable blockchain integration"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    if args.host:
        config["HOST"] = args.host
    
    if args.port:
        config["PORT"] = args.port
    
    if args.debug:
        config["DEBUG"] = True
    
    if args.blockchain:
        config["BLOCKCHAIN_ENABLED"] = True
    
    # Create and configure the application
    app = create_app(config)
    
    # Run the application
    logger.info(f"Starting Dream Content Platform API on {config['HOST']}:{config['PORT']}")
    app.run(
        host=config["HOST"],
        port=config["PORT"],
        debug=config["DEBUG"]
    )


if __name__ == "__main__":
    main()
