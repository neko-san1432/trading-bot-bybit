#!/usr/bin/env python3
"""
CUDA Error Handler - Safely handles CUDA errors and disables GPU if needed
"""

import os
import sys

# Global flag to track if CUDA is working
CUDA_WORKING = True

def safe_cuda_operation(func, *args, **kwargs):
    """Safely execute a CUDA operation with error handling"""
    global CUDA_WORKING
    
    if not CUDA_WORKING:
        return None
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Log the error but don't print to console
        try:
            import logging
            logger = logging.getLogger('scanner')
            logger.error(f"CUDA operation failed: {e}")
        except:
            pass
        
        # Disable CUDA for future operations
        CUDA_WORKING = False
        return None

def check_cuda_availability():
    """Check if CUDA is available and working"""
    global CUDA_WORKING
    
    if not CUDA_WORKING:
        return False
    
    try:
        import cupy as cp
        # Test basic CUDA operation
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        return True
    except Exception:
        CUDA_WORKING = False
        return False

def disable_cuda():
    """Disable CUDA operations"""
    global CUDA_WORKING
    CUDA_WORKING = False

def is_cuda_working():
    """Check if CUDA is currently working"""
    return CUDA_WORKING
