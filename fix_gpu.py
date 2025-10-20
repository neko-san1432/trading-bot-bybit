#!/usr/bin/env python3
"""
GPU Fix Script - Forces NVIDIA GPU usage for trading system
"""

import os
import sys
import subprocess
import platform

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"❌ Error checking NVIDIA GPU: {e}")
        return False

def check_cuda_installation():
    """Check CUDA installation"""
    try:
        # Check CUDA version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ CUDA compiler found:")
            print(result.stdout)
            return True
        else:
            print("❌ nvcc not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA toolkit may not be installed")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_python_gpu_packages():
    """Check if Python GPU packages are installed"""
    packages = ['cupy', 'numba', 'cudf']
    installed = []
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    print(f"✅ Installed GPU packages: {installed}")
    if missing:
        print(f"❌ Missing GPU packages: {missing}")
    
    return len(missing) == 0

def fix_gpu_environment():
    """Fix GPU environment variables"""
    print("🔧 Setting GPU environment variables...")
    
    # Force NVIDIA GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Disable integrated GPU for CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Memory management
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    print("✅ Environment variables set")

def test_gpu_computation():
    """Test GPU computation"""
    print("🧪 Testing GPU computation...")
    
    try:
        import cupy as cp
        
        # Set device to 0 (NVIDIA GPU)
        cp.cuda.Device(0).use()
        
        # Test computation
        test_array = cp.random.random((1000, 1000))
        result = cp.linalg.norm(test_array)
        
        print(f"✅ GPU computation successful: {result:.2f}")
        
        # Check which device we're using
        current_device = cp.cuda.Device().id
        device_props = cp.cuda.runtime.getDeviceProperties(current_device)
        device_name = device_props['name'].decode()
        
        print(f"✅ Using GPU {current_device}: {device_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def install_gpu_packages():
    """Install required GPU packages"""
    print("📦 Installing GPU packages...")
    
    packages = [
        'cupy-cuda12x',  # For CUDA 12.x
        'numba',
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def main():
    """Main function"""
    print("🚀 GPU Fix Script for Trading System")
    print("=" * 50)
    
    # Check system
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    
    # Check NVIDIA GPU
    print("\n1️⃣ Checking NVIDIA GPU...")
    nvidia_ok = check_nvidia_gpu()
    
    # Check CUDA
    print("\n2️⃣ Checking CUDA installation...")
    cuda_ok = check_cuda_installation()
    
    # Check Python packages
    print("\n3️⃣ Checking Python GPU packages...")
    packages_ok = check_python_gpu_packages()
    
    # Fix environment
    print("\n4️⃣ Fixing GPU environment...")
    fix_gpu_environment()
    
    # Test GPU computation
    print("\n5️⃣ Testing GPU computation...")
    gpu_ok = test_gpu_computation()
    
    # Summary
    print("\n📊 Summary:")
    print(f"   NVIDIA GPU: {'✅' if nvidia_ok else '❌'}")
    print(f"   CUDA: {'✅' if cuda_ok else '❌'}")
    print(f"   Python packages: {'✅' if packages_ok else '❌'}")
    print(f"   GPU computation: {'✅' if gpu_ok else '❌'}")
    
    if gpu_ok:
        print("\n🎉 GPU is working correctly!")
        print("Your trading system should now use the NVIDIA GPU.")
    else:
        print("\n❌ GPU setup incomplete.")
        print("Please install NVIDIA drivers and CUDA toolkit.")
        
        # Offer to install packages
        if not packages_ok:
            install = input("\nInstall missing Python packages? (y/N): ").lower().strip() == 'y'
            if install:
                install_gpu_packages()

if __name__ == '__main__':
    main()
