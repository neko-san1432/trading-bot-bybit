#!/usr/bin/env python3
"""
Safe CUDA Test - Tests CUDA and disables if there are errors
"""

import os
import sys

def test_cuda_safely():
    """Test CUDA safely and disable if there are errors"""
    print("🧪 Testing CUDA safely...")
    
    try:
        import cupy as cp
        print("✅ CuPy imported successfully")
        
        # Test basic CUDA operation
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"✅ CUDA computation successful: {result}")
        
        # Test device selection
        cp.cuda.Device(0).use()
        current_device = cp.cuda.Device().id
        print(f"✅ Using CUDA device: {current_device}")
        
        return True
        
    except ImportError as e:
        print(f"❌ CuPy not available: {e}")
        print("💡 Install with: pip install cupy-cuda12x")
        return False
        
    except Exception as e:
        print(f"❌ CUDA error: {e}")
        print("💡 CUDA will be disabled for trading")
        return False

def disable_cuda_environment():
    """Disable CUDA by setting environment variables"""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print("🔧 CUDA disabled via environment variables")

if __name__ == '__main__':
    if not test_cuda_safely():
        print("\n🔧 Disabling CUDA to prevent errors...")
        disable_cuda_environment()
        print("✅ CUDA disabled - trading will use CPU only")
    else:
        print("✅ CUDA working - GPU acceleration enabled")
