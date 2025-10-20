@echo off
echo Setting GPU environment variables for NVIDIA GPU...

REM Force NVIDIA GPU (device 0)
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

REM Disable integrated GPU for CUDA operations
set CUDA_VISIBLE_DEVICES=0

REM Memory management
set TF_FORCE_GPU_ALLOW_GROWTH=true

REM Additional GPU optimizations
set CUDA_CACHE_DISABLE=0
set CUDA_CACHE_MAXSIZE=2147483648

echo Environment variables set:
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo CUDA_DEVICE_ORDER=%CUDA_DEVICE_ORDER%
echo TF_FORCE_GPU_ALLOW_GROWTH=%TF_FORCE_GPU_ALLOW_GROWTH%

echo.
echo Now run your trading script:
echo python run_clean_bot.py
echo.
pause
