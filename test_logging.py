#!/usr/bin/env python3
"""
Test script to verify logging configuration
"""

import logging
import os
import sys

def test_logging_config():
    """Test the logging configuration"""
    print("🧪 Testing logging configuration...")
    
    # Clear any existing log file
    if os.path.exists('log.txt'):
        os.remove('log.txt')
    
    # Configure logging exactly like the trader
    logger = logging.getLogger('scanner')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add only file handler
    file_handler = logging.FileHandler('log.txt', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Also disable root logger console output
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.CRITICAL)  # Only show critical errors
    
    # Test logging
    print("📝 Testing log messages...")
    logger.info("This should only appear in log.txt")
    logger.warning("This warning should only appear in log.txt")
    logger.error("This error should only appear in log.txt")
    
    # Test console output
    print("✅ This should appear in console")
    print("🔍 This should also appear in console")
    
    # Check if log file was created
    if os.path.exists('log.txt'):
        print("✅ Log file created successfully")
        with open('log.txt', 'r') as f:
            content = f.read()
            print(f"📄 Log file content:\n{content}")
    else:
        print("❌ Log file was not created")
    
    print("\n🎯 Test complete!")
    print("Console should only show the test messages above, not the logger messages.")

if __name__ == '__main__':
    test_logging_config()
