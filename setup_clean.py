#!/usr/bin/env python3
"""
Clean Setup Script - Sets up the production environment
"""

import os
import subprocess
import sys

def main():
    print("🧹 CLEANING UP AND SETTING UP PRODUCTION BOT")
    print("=" * 60)
    
    # 1. Remove debug files
    print("\n1️⃣ Removing debug and test files...")
    debug_files = [
        "examples/expanded_coverage_demo.py",
        "examples/evaluation_logs_demo.py", 
        "examples/leverage_based_demo.py",
        "examples/trend_scalping_demo.py",
        "examples/dynamic_strategy_demo.py",
        "examples/pair_coverage_demo.py"
    ]
    
    for file in debug_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   ✅ Removed {file}")
    
    # 2. Install dependencies
    print("\n2️⃣ Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("   ✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error installing dependencies: {e}")
        return
    
    # 3. Check API credentials (new env vars)
    print("\n3️⃣ Checking API credentials (new env vars)...")
    env_testnet = os.getenv('BYBIT_TESTNET', '').lower() in ('1', 'true', 'yes')
    demo_key = os.getenv('BYBIT_API_KEY_DEMO')
    demo_secret = os.getenv('BYBIT_API_SECRET_DEMO')
    real_key = os.getenv('BYBIT_API_KEY_REAL')
    real_secret = os.getenv('BYBIT_API_SECRET_REAL')
    
    if env_testnet:
        if demo_key and demo_secret:
            print("   ✅ Demo (Testnet) credentials found")
        else:
            print("   ⚠️  Demo credentials not found")
            print("   Set:")
            print("   BYBIT_TESTNET=1")
            print("   BYBIT_API_KEY_DEMO=your_demo_key")
            print("   BYBIT_API_SECRET_DEMO=your_demo_secret")
    else:
        if real_key and real_secret:
            print("   ✅ Real (Mainnet) credentials found")
        else:
            print("   ⚠️  Real credentials not found")
            print("   Set:")
            print("   BYBIT_TESTNET=0")
            print("   BYBIT_API_KEY_REAL=your_real_key")
            print("   BYBIT_API_SECRET_REAL=your_real_secret")
    
    # 4. Show usage
    print("\n4️⃣ Setup complete!")
    print("\n🚀 TO RUN THE BOT:")
    print("   python run_clean_bot.py")
    print("\n📁 CLEAN FILES CREATED:")
    print("   - backtest/trend_scalping_strategy_clean.py")
    print("   - live/trend_scalping_trader_clean.py")
    print("   - run_clean_bot.py")
    print("\n✨ Ready for production trading!")

if __name__ == "__main__":
    main()
