# GPU Fix Guide - Force NVIDIA GPU Usage

This guide will help you fix the issue where your trading system is using the integrated GPU instead of your NVIDIA GPU.

## 🚨 Problem
Your trading system is using the integrated GPU instead of the NVIDIA GPU, which can cause:
- Slower performance
- Higher CPU usage
- Potential trading delays
- Inefficient resource utilization

## 🔧 Quick Fix (Windows)

### Method 1: Run the Fix Script
```bash
python fix_gpu.py
```

### Method 2: Use Batch File
```bash
# Double-click or run:
set_gpu_env.bat
```

### Method 3: Use PowerShell (Recommended)
```powershell
# Run as Administrator:
.\configure_gpu.ps1
```

## 🛠️ Manual Fix

### 1. Set Environment Variables
Add these to your system environment variables or run before starting the bot:

```bash
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set TF_FORCE_GPU_ALLOW_GROWTH=true
```

### 2. Install Required Packages
```bash
pip install cupy-cuda12x numba
```

### 3. Verify GPU Detection
```python
import cupy as cp
cp.cuda.Device(0).use()
print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
```

## 🔍 Troubleshooting

### Check NVIDIA GPU
```bash
nvidia-smi
```

### Check CUDA Installation
```bash
nvcc --version
```

### Test GPU Computation
```python
import cupy as cp
cp.cuda.Device(0).use()
test_array = cp.random.random((1000, 1000))
result = cp.linalg.norm(test_array)
print(f"GPU computation: {result}")
```

## 📋 Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Drivers** (latest version)
3. **CUDA Toolkit** (version 11.x or 12.x)
4. **Python packages**: `cupy-cuda12x`, `numba`

## 🚀 How It Works

The fix works by:

1. **Environment Variables**: Forces CUDA to use GPU 0 (NVIDIA)
2. **Device Selection**: Explicitly sets the CUDA device
3. **Memory Management**: Optimizes GPU memory usage
4. **Package Configuration**: Ensures proper GPU package installation

## 📊 Verification

After applying the fix, you should see:
```
✅ NVIDIA GPU configured for trading
✅ GPU initialized for trading
Using GPU 0: NVIDIA GeForce RTX [Your GPU Name]
```

## 🔄 Integration

The fix is automatically integrated into your trading system:

- **Strategy**: `backtest/trend_scalping_strategy_clean.py`
- **Trader**: `live/trend_scalping_trader_clean.py`
- **Utilities**: `gpu_utils.py`

## ⚡ Performance Benefits

With NVIDIA GPU acceleration:
- **Faster calculations**: 10-100x speedup for technical indicators
- **Lower CPU usage**: Offloads computation to GPU
- **Better responsiveness**: Reduced trading delays
- **Scalability**: Handle more trading pairs simultaneously

## 🐛 Common Issues

### Issue: "No CUDA-capable GPU available"
**Solution**: Install NVIDIA drivers and CUDA toolkit

### Issue: "CuPy not available"
**Solution**: Install with `pip install cupy-cuda12x`

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or enable memory growth

### Issue: "Still using integrated GPU"
**Solution**: Check environment variables and restart terminal

## 📞 Support

If you're still having issues:

1. Run `python fix_gpu.py` for diagnostics
2. Check the GPU monitoring output in your trading logs
3. Verify NVIDIA drivers are up to date
4. Ensure CUDA toolkit is properly installed

## 🎯 Expected Results

After applying the fix:
- Trading system uses NVIDIA GPU for all computations
- Faster technical indicator calculations
- Lower CPU usage
- Better overall performance
- GPU memory monitoring in trading logs

The system will automatically fall back to CPU if GPU is unavailable, ensuring trading continues even if there are GPU issues.
