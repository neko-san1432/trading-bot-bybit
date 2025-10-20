#!/usr/bin/env python3
"""
Aggressive Trading Bot Runner
Run different aggressive trading strategies
"""

import os
import sys
import argparse
from datetime import datetime
from aggressive_configs import get_aggressive_config, list_available_strategies
from live.aggressive_trader import AggressiveTrader

def main():
    parser = argparse.ArgumentParser(description='Aggressive Trading Bot')
    parser.add_argument('--strategy', '-s', 
                       choices=['scalping', 'momentum', 'breakout', 'news', 'grid', 'volatility'],
                       default='scalping',
                       help='Trading strategy to use')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Use demo API (https://api-demo.bybit.com)')
    parser.add_argument('--mainnet', '-m', action='store_true', default=True,
                       help='Use mainnet API (https://api.bybit.com) - default')
    parser.add_argument('--balance', '-b', type=float, default=10000.0,
                       help='Account balance (default: 10000)')
    parser.add_argument('--max-positions', '-p', type=int, default=None,
                       help='Maximum positions (overrides strategy default)')
    parser.add_argument('--leverage', '-l', type=float, default=None,
                       help='Leverage (overrides strategy default)')
    parser.add_argument('--risk', '-r', type=float, default=None,
                       help='Risk per trade (overrides strategy default)')
    parser.add_argument('--list', action='store_true',
                       help='List available strategies and exit')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (less console output)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_strategies()
        return
    
    # Set environment variables
    if args.quiet:
        os.environ['BOT_QUIET'] = '1'
    
    # Determine API endpoint
    demo = args.demo
    
    print("🚀 Aggressive Trading Bot")
    print("=" * 50)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Strategy: {args.strategy.upper()}")
    print(f"🌐 API Endpoint: {'https://api-demo.bybit.com (Demo)' if demo else 'https://api.bybit.com (Mainnet)'}")
    print(f"💰 Balance: ${args.balance:,.2f}")
    print("=" * 50)
    
    try:
        # Get strategy configuration
        config = get_aggressive_config(args.strategy)
        
        # Override with command line arguments
        if args.balance:
            config.account_balance = args.balance
        if args.max_positions:
            config.max_positions = args.max_positions
        if args.leverage:
            config.leverage = args.leverage
        if args.risk:
            config.risk_per_trade = args.risk
        
        # Create and run trader
        trader = AggressiveTrader(demo=demo)
        trader.config = config  # Override with custom config
        trader.strategy = trader.strategy.__class__(config)  # Update strategy with new config
        
        print(f"⚡ Max Positions: {config.max_positions}")
        print(f"📈 Leverage: {config.leverage}x")
        print(f"🎯 Risk per Trade: {config.risk_per_trade*100}%")
        print(f"🛡️ Max Total Risk: {config.max_total_risk*100}%")
        print(f"⏱️ Timeframes: {', '.join(config.timeframes)}")
        print("=" * 50)
        
        # Run the trader
        trader.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
