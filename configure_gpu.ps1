# PowerShell script to configure GPU for trading system
# Run as Administrator for best results

Write-Host "🚀 GPU Configuration Script for Trading System" -ForegroundColor Green
Write-Host "=" * 50

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "⚠️  Warning: Not running as Administrator. Some settings may not apply." -ForegroundColor Yellow
}

# Set environment variables
Write-Host "🔧 Setting GPU environment variables..." -ForegroundColor Cyan

# Force NVIDIA GPU
$env:CUDA_VISIBLE_DEVICES = "0"
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"

# Disable integrated GPU for CUDA
$env:CUDA_VISIBLE_DEVICES = "0"

# Memory management
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# Additional optimizations
$env:CUDA_CACHE_DISABLE = "0"
$env:CUDA_CACHE_MAXSIZE = "2147483648"

# Set for current session
[Environment]::SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0", "User")
[Environment]::SetEnvironmentVariable("CUDA_DEVICE_ORDER", "PCI_BUS_ID", "User")
[Environment]::SetEnvironmentVariable("TF_FORCE_GPU_ALLOW_GROWTH", "true", "User")

Write-Host "✅ Environment variables set for current session and user profile" -ForegroundColor Green

# Check NVIDIA GPU
Write-Host "🔍 Checking NVIDIA GPU..." -ForegroundColor Cyan
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    Write-Host "✅ NVIDIA GPU found:" -ForegroundColor Green
    Write-Host $gpuInfo
} catch {
    Write-Host "❌ nvidia-smi not found. Please install NVIDIA drivers." -ForegroundColor Red
}

# Check CUDA
Write-Host "🔍 Checking CUDA installation..." -ForegroundColor Cyan
try {
    $nvcc = Get-Command nvcc -ErrorAction Stop
    $cudaVersion = & nvcc --version | Select-String "release"
    Write-Host "✅ CUDA found: $cudaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ nvcc not found. Please install CUDA toolkit." -ForegroundColor Red
}

# Check Python GPU packages
Write-Host "🔍 Checking Python GPU packages..." -ForegroundColor Cyan
$packages = @("cupy", "numba")
$missingPackages = @()

foreach ($package in $packages) {
    try {
        $result = python -c "import $package; print('OK')" 2>$null
        if ($result -eq "OK") {
            Write-Host "✅ $package is installed" -ForegroundColor Green
        } else {
            $missingPackages += $package
        }
    } catch {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "❌ Missing packages: $($missingPackages -join ', ')" -ForegroundColor Red
    $install = Read-Host "Install missing packages? (y/N)"
    if ($install -eq "y" -or $install -eq "Y") {
        foreach ($package in $missingPackages) {
            if ($package -eq "cupy") {
                Write-Host "Installing cupy-cuda12x..." -ForegroundColor Cyan
                python -m pip install cupy-cuda12x
            } else {
                Write-Host "Installing $package..." -ForegroundColor Cyan
                python -m pip install $package
            }
        }
    }
}

# Test GPU computation
Write-Host "🧪 Testing GPU computation..." -ForegroundColor Cyan
try {
    $testScript = @"
import cupy as cp
cp.cuda.Device(0).use()
test_array = cp.random.random((100, 100))
result = cp.linalg.norm(test_array)
print(f'GPU computation successful: {result:.2f}')
current_device = cp.cuda.Device().id
device_props = cp.cuda.runtime.getDeviceProperties(current_device)
device_name = device_props['name'].decode()
print(f'Using GPU {current_device}: {device_name}')
"@
    
    $testScript | python
    Write-Host "✅ GPU computation test passed!" -ForegroundColor Green
} catch {
    Write-Host "❌ GPU computation test failed: $_" -ForegroundColor Red
}

Write-Host "`n🎉 GPU configuration complete!" -ForegroundColor Green
Write-Host "You can now run your trading system with GPU acceleration." -ForegroundColor Green
Write-Host "`nTo run the trading bot:" -ForegroundColor Cyan
Write-Host "python run_clean_bot.py" -ForegroundColor White

Read-Host "Press Enter to continue"
