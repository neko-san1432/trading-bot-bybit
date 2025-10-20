import time
import requests
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pybit.unified_trading import HTTP
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BybitPingTester:
    """Comprehensive ping test utility for Bybit server connectivity"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = False):
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.use_testnet = use_testnet
        
        # Bybit API endpoints
        if use_testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_url = "wss://stream-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_url = "wss://stream.bybit.com"
        
        # Initialize HTTP client for API tests
        self.session = HTTP(
            testnet=use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        # Connection history for monitoring
        self.connection_history = []
        self.max_history = 100
        
        # Ping thresholds
        self.good_latency_threshold = 200  # ms
        self.warning_latency_threshold = 500  # ms
        self.critical_latency_threshold = 1000  # ms
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def ping_server_time(self) -> Dict[str, any]:
        """Test basic server time endpoint (no auth required)"""
        start_time = time.time()
        
        try:
            response = self.session.get_server_time()
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.get('retCode') == 0:
                server_time = response['result']['timeSecond']
                return {
                    'success': True,
                    'latency_ms': round(latency, 2),
                    'server_time': server_time,
                    'status': self._get_latency_status(latency),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'latency_ms': round(latency, 2),
                    'server_time': None,
                    'status': 'error',
                    'error': response.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'server_time': None,
                'status': 'error',
                'error': str(e)
            }
    
    def ping_authenticated_endpoint(self) -> Dict[str, any]:
        """Test authenticated endpoint (wallet balance)"""
        start_time = time.time()
        
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            if response.get('retCode') == 0:
                balance = float(response['result']['list'][0]['totalWalletBalance'])
                return {
                    'success': True,
                    'latency_ms': round(latency, 2),
                    'balance': balance,
                    'status': self._get_latency_status(latency),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'latency_ms': round(latency, 2),
                    'balance': None,
                    'status': 'error',
                    'error': response.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'balance': None,
                'status': 'error',
                'error': str(e)
            }
    
    def ping_market_data(self, symbol: str = "BTCUSDT") -> Dict[str, any]:
        """Test market data endpoint"""
        start_time = time.time()
        
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval="1",
                limit=1
            )
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            if response.get('retCode') == 0:
                kline_data = response['result']['list'][0] if response['result']['list'] else None
                return {
                    'success': True,
                    'latency_ms': round(latency, 2),
                    'symbol': symbol,
                    'kline_data': kline_data,
                    'status': self._get_latency_status(latency),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'latency_ms': round(latency, 2),
                    'symbol': symbol,
                    'kline_data': None,
                    'status': 'error',
                    'error': response.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'symbol': symbol,
                'kline_data': None,
                'status': 'error',
                'error': str(e)
            }
    
    def ping_trading_endpoint(self) -> Dict[str, any]:
        """Test trading endpoint (get positions - read-only)"""
        start_time = time.time()
        
        try:
            response = self.session.get_positions(
                category="linear",
                settleCoin="USDT",
                limit=10
            )
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            if response.get('retCode') == 0:
                positions = response['result']['list']
                return {
                    'success': True,
                    'latency_ms': round(latency, 2),
                    'positions_count': len(positions),
                    'status': self._get_latency_status(latency),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'latency_ms': round(latency, 2),
                    'positions_count': 0,
                    'status': 'error',
                    'error': response.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'positions_count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def ping_dns_resolution(self) -> Dict[str, any]:
        """Test DNS resolution for Bybit domains"""
        start_time = time.time()
        
        try:
            # Extract hostname from base URL
            hostname = self.base_url.replace('https://', '').replace('http://', '')
            ip_address = socket.gethostbyname(hostname)
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            return {
                'success': True,
                'latency_ms': round(latency, 2),
                'hostname': hostname,
                'ip_address': ip_address,
                'status': self._get_latency_status(latency),
                'error': None
            }
            
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'hostname': hostname if 'hostname' in locals() else 'unknown',
                'ip_address': None,
                'status': 'error',
                'error': str(e)
            }
    
    def ping_http_connectivity(self) -> Dict[str, any]:
        """Test basic HTTP connectivity"""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/v5/market/time", timeout=10)
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code,
                    'status': self._get_latency_status(latency),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'latency_ms': round(latency, 2),
                    'status_code': response.status_code,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return {
                'success': False,
                'latency_ms': round(latency, 2),
                'status_code': None,
                'status': 'error',
                'error': str(e)
            }
    
    def comprehensive_ping_test(self, include_auth: bool = True) -> Dict[str, any]:
        """Run comprehensive ping test suite"""
        print("🔍 Running comprehensive Bybit connectivity test...")
        print(f"🌐 Environment: {'TESTNET' if self.use_testnet else 'MAINNET'}")
        print(f"🔗 Base URL: {self.base_url}")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'testnet' if self.use_testnet else 'mainnet',
            'base_url': self.base_url,
            'tests': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Test 1: DNS Resolution
        print("1️⃣ Testing DNS resolution...")
        dns_result = self.ping_dns_resolution()
        results['tests']['dns'] = dns_result
        self._print_test_result("DNS Resolution", dns_result)
        
        # Test 2: HTTP Connectivity
        print("\n2️⃣ Testing HTTP connectivity...")
        http_result = self.ping_http_connectivity()
        results['tests']['http'] = http_result
        self._print_test_result("HTTP Connectivity", http_result)
        
        # Test 3: Server Time (No Auth)
        print("\n3️⃣ Testing server time endpoint...")
        server_time_result = self.ping_server_time()
        results['tests']['server_time'] = server_time_result
        self._print_test_result("Server Time", server_time_result)
        
        # Test 4: Market Data
        print("\n4️⃣ Testing market data endpoint...")
        market_result = self.ping_market_data()
        results['tests']['market_data'] = market_result
        self._print_test_result("Market Data", market_result)
        
        # Test 5: Trading Endpoint
        print("\n5️⃣ Testing trading endpoint...")
        trading_result = self.ping_trading_endpoint()
        results['tests']['trading'] = trading_result
        self._print_test_result("Trading Endpoint", trading_result)
        
        # Test 6: Authenticated Endpoint (if requested)
        if include_auth:
            print("\n6️⃣ Testing authenticated endpoint...")
            auth_result = self.ping_authenticated_endpoint()
            results['tests']['authenticated'] = auth_result
            self._print_test_result("Authenticated Endpoint", auth_result)
        
        # Calculate overall status
        results['overall_status'] = self._calculate_overall_status(results['tests'])
        results['recommendations'] = self._generate_recommendations(results['tests'])
        
        # Store in history
        self._add_to_history(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def quick_ping_test(self) -> bool:
        """Quick ping test for trading decisions"""
        try:
            result = self.ping_server_time()
            return result['success'] and result['status'] in ['good', 'warning']
        except:
            return False
    
    def monitor_connection_health(self, duration_minutes: int = 60, interval_seconds: int = 30) -> List[Dict]:
        """Monitor connection health over time"""
        print(f"🔍 Starting connection health monitoring for {duration_minutes} minutes...")
        print(f"⏱️  Check interval: {interval_seconds} seconds")
        print("=" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        monitoring_results = []
        
        while datetime.now() < end_time:
            current_time = datetime.now()
            print(f"\n🕐 {current_time.strftime('%H:%M:%S')} - Health Check")
            
            # Quick ping test
            ping_result = self.ping_server_time()
            monitoring_results.append({
                'timestamp': current_time.isoformat(),
                'ping_result': ping_result,
                'healthy': ping_result['success'] and ping_result['status'] in ['good', 'warning']
            })
            
            # Print status
            if ping_result['success']:
                status_emoji = "✅" if ping_result['status'] == 'good' else "⚠️"
                print(f"{status_emoji} Latency: {ping_result['latency_ms']}ms ({ping_result['status']})")
            else:
                print(f"❌ Connection failed: {ping_result['error']}")
            
            # Wait for next check
            time.sleep(interval_seconds)
        
        print(f"\n📊 Monitoring completed. Total checks: {len(monitoring_results)}")
        return monitoring_results
    
    def _get_latency_status(self, latency: float) -> str:
        """Determine latency status based on thresholds"""
        if latency <= self.good_latency_threshold:
            return 'good'
        elif latency <= self.warning_latency_threshold:
            return 'warning'
        elif latency <= self.critical_latency_threshold:
            return 'critical'
        else:
            return 'error'
    
    def _print_test_result(self, test_name: str, result: Dict[str, any]):
        """Print formatted test result"""
        if result['success']:
            status_emoji = "✅" if result['status'] == 'good' else "⚠️" if result['status'] == 'warning' else "🔴"
            print(f"   {status_emoji} {test_name}: {result['latency_ms']}ms ({result['status']})")
        else:
            print(f"   ❌ {test_name}: FAILED - {result['error']}")
    
    def _calculate_overall_status(self, tests: Dict[str, Dict]) -> str:
        """Calculate overall connection status"""
        critical_tests = ['dns', 'http', 'server_time']
        all_critical_passed = all(tests[test]['success'] for test in critical_tests if test in tests)
        
        if not all_critical_passed:
            return 'critical'
        
        # Check for any critical latency issues
        has_critical_latency = any(
            test['status'] == 'critical' for test in tests.values() 
            if test['success']
        )
        
        if has_critical_latency:
            return 'warning'
        
        return 'good'
    
    def _generate_recommendations(self, tests: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # DNS issues
        if not tests.get('dns', {}).get('success', True):
            recommendations.append("🔧 Check your internet connection and DNS settings")
        
        # HTTP connectivity issues
        if not tests.get('http', {}).get('success', True):
            recommendations.append("🌐 Check firewall settings and proxy configuration")
        
        # High latency
        high_latency_tests = [
            test for test in tests.values() 
            if test.get('success') and test.get('status') in ['warning', 'critical']
        ]
        if high_latency_tests:
            recommendations.append("⚡ High latency detected - consider using a VPN or different network")
        
        # Authentication issues
        if not tests.get('authenticated', {}).get('success', True):
            recommendations.append("🔑 Check API credentials and permissions")
        
        # Trading endpoint issues
        if not tests.get('trading', {}).get('success', True):
            recommendations.append("📊 Check trading permissions and account status")
        
        if not recommendations:
            recommendations.append("✅ All systems operational - ready for trading!")
        
        return recommendations
    
    def _print_summary(self, results: Dict[str, any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 CONNECTIVITY TEST SUMMARY")
        print("=" * 60)
        
        status_emoji = {
            'good': '✅',
            'warning': '⚠️',
            'critical': '🔴',
            'error': '❌'
        }
        
        overall_status = results['overall_status']
        print(f"🎯 Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        
        print(f"\n📈 Test Results:")
        for test_name, test_result in results['tests'].items():
            if test_result['success']:
                print(f"   ✅ {test_name}: {test_result['latency_ms']}ms")
            else:
                print(f"   ❌ {test_name}: FAILED")
        
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   {rec}")
        
        print("=" * 60)
    
    def _add_to_history(self, results: Dict[str, any]):
        """Add results to connection history"""
        self.connection_history.append(results)
        if len(self.connection_history) > self.max_history:
            self.connection_history.pop(0)
    
    def get_connection_stats(self) -> Dict[str, any]:
        """Get connection statistics from history"""
        if not self.connection_history:
            return {'message': 'No connection history available'}
        
        recent_tests = self.connection_history[-10:]  # Last 10 tests
        
        total_tests = len(recent_tests)
        successful_tests = sum(1 for test in recent_tests if test['overall_status'] == 'good')
        warning_tests = sum(1 for test in recent_tests if test['overall_status'] == 'warning')
        failed_tests = total_tests - successful_tests - warning_tests
        
        # Calculate average latencies
        avg_latencies = {}
        for test in recent_tests:
            for test_name, test_result in test['tests'].items():
                if test_result['success']:
                    if test_name not in avg_latencies:
                        avg_latencies[test_name] = []
                    avg_latencies[test_name].append(test_result['latency_ms'])
        
        avg_latencies = {
            test_name: sum(latencies) / len(latencies) 
            for test_name, latencies in avg_latencies.items()
        }
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'average_latencies': avg_latencies,
            'last_test_time': recent_tests[-1]['timestamp'] if recent_tests else None
        }


def main():
    """Main function for standalone ping testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bybit Connectivity Ping Test')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    parser.add_argument('--monitor', type=int, metavar='MINUTES', help='Monitor connection for specified minutes')
    parser.add_argument('--quick', action='store_true', help='Run quick ping test only')
    parser.add_argument('--no-auth', action='store_true', help='Skip authenticated tests')
    
    args = parser.parse_args()
    
    # Initialize ping tester
    tester = BybitPingTester(use_testnet=args.testnet)
    
    if args.quick:
        # Quick test
        result = tester.quick_ping_test()
        print("✅ Connection OK" if result else "❌ Connection Failed")
    elif args.monitor:
        # Monitor mode
        tester.monitor_connection_health(duration_minutes=args.monitor)
    else:
        # Comprehensive test
        tester.comprehensive_ping_test(include_auth=not args.no_auth)


if __name__ == '__main__':
    main()
