#!/usr/bin/env python3
"""
Standalone script to test Bybit connectivity ping functionality
"""

import os
import sys
from dotenv import load_dotenv
from ping_utils import BybitPingTester

# Load environment variables
load_dotenv()

def main():
    """Main function to test ping functionality"""
    print("🚀 Bybit Connectivity Ping Test")
    print("=" * 50)
    
    # Check if we have API credentials
    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ No API credentials found in environment variables")
        print("Please set BYBIT_API_KEY and BYBIT_API_SECRET in your .env file")
        return
    
    # Ask user for testnet preference
    use_testnet = input("Use testnet? (y/N): ").lower().strip() == 'y'
    
    # Initialize ping tester
    tester = BybitPingTester(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet
    )
    
    print(f"\n🌐 Testing {'TESTNET' if use_testnet else 'MAINNET'} environment")
    print("=" * 50)
    
    # Run comprehensive test
    results = tester.comprehensive_ping_test(include_auth=True)
    
    # Print connection statistics
    print("\n📊 Connection Statistics:")
    stats = tester.get_connection_stats()
    print(f"   Total tests: {stats['total_tests']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average latencies: {stats['average_latencies']}")
    
    # Ask if user wants to monitor connection
    monitor = input("\nMonitor connection health? (y/N): ").lower().strip() == 'y'
    if monitor:
        duration = input("Duration in minutes (default 5): ").strip()
        try:
            duration = int(duration) if duration else 5
        except ValueError:
            duration = 5
        
        print(f"\n🔍 Monitoring connection for {duration} minutes...")
        tester.monitor_connection_health(duration_minutes=duration, interval_seconds=30)

if __name__ == '__main__':
    main()
