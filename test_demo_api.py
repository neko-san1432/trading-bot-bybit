#!/usr/bin/env python3
"""
Test script to verify the demo API configuration is working correctly
"""

import os
import sys
from pybit.unified_trading import HTTP

def test_demo_api_config():
    """Test the demo API configuration"""
    print("Testing Demo API Configuration")
    print("=" * 50)
    
    # Test 1: Check if we can create a client with demo API
    try:
        print("1. Creating demo API client...")
        demo_client = HTTP(
            demo=True,  # Demo uses demo=True
            api_key="test_key",
            api_secret="test_secret"
        )
        print("   Demo API client created successfully")
        print(f"   Endpoint: {demo_client.endpoint}")
        print(f"   Demo: {demo_client.demo}")
    except Exception as e:
        print(f"   Error creating demo client: {e}")
        return False
    
    # Test 2: Check if we can create a mainnet client
    try:
        print("\n2. Creating mainnet API client...")
        mainnet_client = HTTP(
            testnet=False,
            api_key="test_key",
            api_secret="test_secret"
        )
        print("   Mainnet API client created successfully")
        print(f"   Endpoint: {mainnet_client.endpoint}")
        print(f"   Testnet: {mainnet_client.testnet}")
    except Exception as e:
        print(f"   Error creating mainnet client: {e}")
        return False
    
    # Test 3: Verify the endpoints are different
    print("\n3. Verifying API endpoints...")
    if demo_client.endpoint != mainnet_client.endpoint:
        print(f"   Demo API: {demo_client.endpoint}")
        print(f"   Mainnet API: {mainnet_client.endpoint}")
        print("   Endpoints are correctly different")
    else:
        print("   Endpoints are the same - configuration error!")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! Demo API configuration is correct.")
    print("\nTo use the trading bot:")
    print("1. Set your API keys:")
    print("   - BYBIT_API_KEY_DEMO=your_demo_key")
    print("   - BYBIT_API_SECRET_DEMO=your_demo_secret")
    print("   - BYBIT_API_KEY_REAL=your_real_key")
    print("   - BYBIT_API_SECRET_REAL=your_real_secret")
    print("\n2. Run the bot:")
    print("   python run_momentum_bot.py --demo --balance 1.7 --target 200 --yes")
    print("   python run_momentum_bot.py --mainnet --balance 100 --target 200 --yes")
    
    return True

if __name__ == "__main__":
    test_demo_api_config()
