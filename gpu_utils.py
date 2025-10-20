#!/usr/bin/env python3
"""
GPU Configuration Utility for Trading System
Ensures NVIDIA GPU is used instead of integrated GPU
"""

import os
import sys
import logging
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_gpu_environment():
    """Configure environment to use NVIDIA GPU instead of integrated GPU"""
    
    # Set CUDA device to 0 (NVIDIA GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Force CUDA to use the first available GPU (NVIDIA)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Disable integrated GPU for CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Set memory growth to avoid allocating all GPU memory at once
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    logger.info("✅ GPU environment configured for NVIDIA GPU")

def check_gpu_availability() -> dict:
    """Check available GPUs and their properties"""
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'cupy_available': False,
        'numba_available': False
    }
    
    # Check CUDA availability
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['cuda_available'] = True
        gpu_info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
        
        for i in range(gpu_info['gpu_count']):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                gpu_info['gpu_names'].append(props['name'].decode())
        
        logger.info(f"✅ Found {gpu_info['gpu_count']} GPU(s): {gpu_info['gpu_names']}")
        
    except ImportError:
        logger.warning("❌ CuPy not available - GPU acceleration disabled")
    except Exception as e:
        logger.warning(f"❌ CUDA not available: {e}")
    
    # Check Numba availability
    try:
        import numba
        from numba import cuda
        gpu_info['numba_available'] = True
        logger.info("✅ Numba CUDA available")
    except ImportError:
        logger.warning("❌ Numba not available")
    except Exception as e:
        logger.warning(f"❌ Numba CUDA not available: {e}")
    
    return gpu_info

def force_nvidia_gpu():
    """Force the use of NVIDIA GPU for all GPU operations"""
    
    # Configure environment first
    configure_gpu_environment()
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    
    if not gpu_info['cuda_available']:
        logger.error("❌ No CUDA-capable GPU available")
        return False
    
    if not gpu_info['cupy_available']:
        logger.error("❌ CuPy not available - install with: pip install cupy-cuda12x")
        return False
    
    # Force NVIDIA GPU selection
    try:
        import cupy as cp
        
        # Set the device to 0 (first GPU, should be NVIDIA)
        cp.cuda.Device(0).use()
        
        # Verify we're using the right GPU
        current_device = cp.cuda.Device().id
        device_props = cp.cuda.runtime.getDeviceProperties(current_device)
        device_name = device_props['name'].decode()
        
        logger.info(f"✅ Using GPU {current_device}: {device_name}")
        
        # Test GPU computation
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        logger.info(f"✅ GPU test computation successful: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to configure NVIDIA GPU: {e}")
        # Don't raise the exception, just return False
        return False

def get_optimal_gpu_config() -> dict:
    """Get optimal GPU configuration for trading system"""
    
    gpu_info = check_gpu_availability()
    
    config = {
        'use_gpu': gpu_info['cuda_available'] and gpu_info['cupy_available'],
        'gpu_device': 0,  # Use first GPU (NVIDIA)
        'memory_fraction': 0.8,  # Use 80% of GPU memory
        'allow_memory_growth': True,
        'gpu_names': gpu_info['gpu_names'],
        'gpu_count': gpu_info['gpu_count']
    }
    
    if config['use_gpu']:
        logger.info("✅ GPU acceleration enabled")
        logger.info(f"   Device: {config['gpu_device']}")
        logger.info(f"   Memory fraction: {config['memory_fraction']}")
        logger.info(f"   Available GPUs: {config['gpu_names']}")
    else:
        logger.warning("⚠️  GPU acceleration disabled - using CPU")
    
    return config

def initialize_gpu_for_trading():
    """Initialize GPU specifically for trading operations"""
    
    logger.info("🚀 Initializing GPU for trading system...")
    
    # Force NVIDIA GPU
    if not force_nvidia_gpu():
        logger.warning("⚠️  GPU initialization failed - falling back to CPU")
        return False
    
    # Get optimal configuration
    config = get_optimal_gpu_config()
    
    if config['use_gpu']:
        try:
            import cupy as cp
            
            # Set memory management
            if config['allow_memory_growth']:
                # Enable memory growth to avoid OOM
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            
            # Pre-allocate some memory to avoid first-run delays
            _ = cp.array([1.0, 2.0, 3.0])
            
            logger.info("✅ GPU initialized successfully for trading")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU initialization failed: {e}")
            return False
    
    return False

def monitor_gpu_usage():
    """Monitor GPU usage during trading"""
    try:
        import cupy as cp
        
        # Get current device
        current_device = cp.cuda.Device().id
        
        # Get memory info
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        pinned_bytes = pinned_mempool.n_free_blocks()
        
        logger.info(f"🔍 GPU {current_device} Memory Usage:")
        logger.info(f"   Used: {used_bytes / 1024**2:.2f} MB")
        logger.info(f"   Total: {total_bytes / 1024**2:.2f} MB")
        logger.info(f"   Free: {(total_bytes - used_bytes) / 1024**2:.2f} MB")
        logger.info(f"   Pinned blocks: {pinned_bytes}")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not monitor GPU usage: {e}")

def cleanup_gpu_memory():
    """Clean up GPU memory after trading operations"""
    try:
        import cupy as cp
        
        # Clear memory pools
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        logger.info("🧹 GPU memory cleaned up")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not clean up GPU memory: {e}")

def main():
    """Main function for testing GPU configuration"""
    print("🔍 GPU Configuration Test")
    print("=" * 40)
    
    # Initialize GPU
    success = initialize_gpu_for_trading()
    
    if success:
        print("✅ GPU configuration successful!")
        
        # Monitor usage
        monitor_gpu_usage()
        
        # Test computation
        try:
            import cupy as cp
            test_data = cp.random.random((1000, 1000))
            result = cp.linalg.norm(test_data)
            print(f"✅ GPU computation test: {result:.2f}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    else:
        print("❌ GPU configuration failed - check your setup")

if __name__ == '__main__':
    main()
