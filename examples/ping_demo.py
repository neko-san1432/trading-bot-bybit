#!/usr/bin/env python3
"""
Example script demonstrating Bybit ping test functionality
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ping_utils import BybitPingTester
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demo_quick_ping():
    """Demonstrate quick ping test"""
    print("🔍 Quick Ping Test Demo")
    print("-" * 30)
    
    tester = BybitPingTester(use_testnet=True)
    
    # Quick ping test
    result = tester.quick_ping_test()
    print(f"Quick ping result: {'✅ Success' if result else '❌ Failed'}")
    
    return result

def demo_comprehensive_test():
    """Demonstrate comprehensive ping test"""
    print("\n🔍 Comprehensive Ping Test Demo")
    print("-" * 40)
    
    tester = BybitPingTester(use_testnet=True)
    
    # Run comprehensive test
    results = tester.comprehensive_ping_test(include_auth=False)  # Skip auth for demo
    
    print(f"\nOverall status: {results['overall_status']}")
    print(f"Recommendations: {len(results['recommendations'])}")
    
    return results

def demo_individual_tests():
    """Demonstrate individual ping tests"""
    print("\n🔍 Individual Ping Tests Demo")
    print("-" * 35)
    
    tester = BybitPingTester(use_testnet=True)
    
    # Test DNS resolution
    print("1. DNS Resolution:")
    dns_result = tester.ping_dns_resolution()
    print(f"   Success: {dns_result['success']}")
    print(f"   Latency: {dns_result['latency_ms']}ms")
    
    # Test HTTP connectivity
    print("\n2. HTTP Connectivity:")
    http_result = tester.ping_http_connectivity()
    print(f"   Success: {http_result['success']}")
    print(f"   Latency: {http_result['latency_ms']}ms")
    
    # Test server time
    print("\n3. Server Time:")
    server_result = tester.ping_server_time()
    print(f"   Success: {server_result['success']}")
    print(f"   Latency: {server_result['latency_ms']}ms")
    
    # Test market data
    print("\n4. Market Data:")
    market_result = tester.ping_market_data("BTCUSDT")
    print(f"   Success: {market_result['success']}")
    print(f"   Latency: {market_result['latency_ms']}ms")

def demo_connection_monitoring():
    """Demonstrate connection monitoring"""
    print("\n🔍 Connection Monitoring Demo")
    print("-" * 35)
    
    tester = BybitPingTester(use_testnet=True)
    
    print("Monitoring connection for 2 minutes with 30-second intervals...")
    print("(This will run for 2 minutes - press Ctrl+C to stop early)")
    
    try:
        results = tester.monitor_connection_health(duration_minutes=2, interval_seconds=30)
        
        # Analyze results
        healthy_checks = sum(1 for r in results if r['healthy'])
        total_checks = len(results)
        
        print(f"\n📊 Monitoring Results:")
        print(f"   Total checks: {total_checks}")
        print(f"   Healthy checks: {healthy_checks}")
        print(f"   Health rate: {healthy_checks/total_checks:.1%}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")

def demo_connection_stats():
    """Demonstrate connection statistics"""
    print("\n🔍 Connection Statistics Demo")
    print("-" * 35)
    
    tester = BybitPingTester(use_testnet=True)
    
    # Run a few tests to build history
    print("Running multiple tests to build connection history...")
    for i in range(3):
        print(f"   Test {i+1}/3...")
        tester.comprehensive_ping_test(include_auth=False)
        time.sleep(2)
    
    # Get statistics
    stats = tester.get_connection_stats()
    
    print(f"\n📊 Connection Statistics:")
    print(f"   Total tests: {stats['total_tests']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average latencies: {stats['average_latencies']}")

def main():
    """Main demo function"""
    print("🚀 Bybit Ping Test Demo")
    print("=" * 50)
    print("This demo shows various ping test functionalities")
    print("All tests will use TESTNET for safety")
    print("=" * 50)
    
    try:
        # Demo 1: Quick ping
        demo_quick_ping()
        
        # Demo 2: Comprehensive test
        demo_comprehensive_test()
        
        # Demo 3: Individual tests
        demo_individual_tests()
        
        # Demo 4: Connection monitoring (optional)
        monitor = input("\nRun connection monitoring demo? (y/N): ").lower().strip() == 'y'
        if monitor:
            demo_connection_monitoring()
        
        # Demo 5: Connection statistics
        demo_connection_stats()
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")

if __name__ == '__main__':
    main()
