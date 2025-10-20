#!/usr/bin/env python3
"""
Clean Bot Runner - Simple script to run the production bot
"""

import os
import sys
from live.trend_scalping_trader_clean import TrendScalpingTraderClean

def main():
    print("🚀 CLEAN TREND SCALPING BOT")
    print("=" * 50)
    
    # Check for API credentials using new env vars
    # BYBIT_TESTNET controls which set to use when not passing explicitly
    env_testnet = os.getenv('BYBIT_TESTNET', '').lower() in ('1', 'true', 'yes')
    api_key = os.getenv('BYBIT_API_KEY_DEMO') if env_testnet else os.getenv('BYBIT_API_KEY_REAL')
    api_secret = os.getenv('BYBIT_API_SECRET_DEMO') if env_testnet else os.getenv('BYBIT_API_SECRET_REAL')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found!")
        print("Please set the new environment variables:")
        print("\nDemo (Testnet):")
        print("  BYBIT_TESTNET=1")
        print("  BYBIT_API_KEY_DEMO=your_demo_key")
        print("  BYBIT_API_SECRET_DEMO=your_demo_secret")
        print("\nReal (Mainnet):")
        print("  BYBIT_TESTNET=0")
        print("  BYBIT_API_KEY_REAL=your_real_key")
        print("  BYBIT_API_SECRET_REAL=your_real_secret")
        return
    
    print("✅ API credentials found")
    print("🔄 Starting bot...")
    print("\nPress Ctrl+C to stop the bot")
    print("-" * 50)
    
    try:
        # Initialize and run trader
        trader = TrendScalpingTraderClean(
            api_key=api_key,
            api_secret=api_secret,
            testnet=env_testnet  # Controlled by BYBIT_TESTNET
        )
        
        trader.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
