#!/usr/bin/env python3
"""
Safe Bot Runner - Runs the trading bot with CUDA error handling
"""

import os
import sys

def main():
    """Main function with CUDA error handling"""
    print("🚀 Starting Safe Trading Bot")
    
    # Test CUDA first
    try:
        import cupy as cp
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        print("✅ CUDA working - GPU acceleration enabled")
    except Exception as e:
        print(f"⚠️  CUDA error detected: {e}")
        print("🔧 Disabling CUDA to prevent errors...")
        
        # Disable CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['CUDA_DEVICE_ORDER'] = ''
        
        # Set environment to force CPU mode
        os.environ['BOT_GPU_DISABLED'] = '1'
        
        print("✅ CUDA disabled - using CPU only")
    
    # Import and run the bot
    try:
        from live.trend_scalping_trader_clean import main as run_bot
        run_bot()
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
