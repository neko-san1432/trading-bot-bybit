#!/usr/bin/env python3
"""
High Momentum Bot Runner
Trade 50-100% gainers with small balance ($1.7 -> $200)
"""

import os
import sys
import argparse
from datetime import datetime
from live.momentum_trader import MomentumTrader

def main():
    parser = argparse.ArgumentParser(description='High Momentum Trading Bot')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Use demo API (https://api-demo.bybit.com)')
    parser.add_argument('--mainnet', '-m', action='store_true', default=True,
                       help='Use mainnet API (https://api.bybit.com) - default')
    parser.add_argument('--balance', '-b', type=float, default=1.7,
                       help='Starting balance (default: 1.7)')
    parser.add_argument('--target', '-g', type=float, default=200.0,
                       help='Target balance (default: 200)')
    parser.add_argument('--leverage', '-l', type=float, default=50.0,
                       help='Leverage (default: 50)')
    parser.add_argument('--risk', '-r', type=float, default=0.10,
                       help='Risk per trade (default: 0.10 = 10%)')
    parser.add_argument('--max-positions', '-p', type=int, default=1,
                       help='Max positions (default: 1)')
    parser.add_argument('--min-gain', type=float, default=0.50,
                       help='Minimum 24h gain (default: 0.50 = 50%)')
    parser.add_argument('--max-gain', type=float, default=2.00,
                       help='Maximum 24h gain (default: 2.00 = 200%)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode')
    parser.add_argument('--calculator', '-c', action='store_true',
                       help='Run position size calculator instead')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    if args.calculator:
        # Run position calculator
        from position_calculator import main as calc_main
        calc_main()
        return
    
    # Set environment variables
    if args.quiet:
        os.environ['BOT_QUIET'] = '1'
    
    # Determine API endpoint
    demo = args.demo
    
    print("High Momentum Trading Bot")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Endpoint: {'https://api-demo.bybit.com (Demo)' if demo else 'https://api.bybit.com (Mainnet)'}")
    print(f"Starting Balance: ${args.balance}")
    print(f"Target Balance: ${args.target}")
    print(f"Leverage: {args.leverage}x")
    print(f"Risk per Trade: {args.risk*100}%")
    print(f"Max Positions: {args.max_positions}")
    print(f"Min 24h Gain: {args.min_gain*100}%")
    print(f"Max 24h Gain: {args.max_gain*100}%")
    print("=" * 60)
    
    # Warning for small balance
    if args.balance < 5.0 and not args.yes:
        print("WARNING: Very small balance detected!")
        print("   - High risk of liquidation")
        print("   - Consider starting with $5+ for safety")
        print("   - This bot is designed for high-risk, high-reward trading")
        print()
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Trading cancelled")
            return
    
    try:
        # Create and run trader
        trader = MomentumTrader(demo=demo)
        
        # Override configuration
        trader.config.account_balance = args.balance
        trader.config.target_balance = args.target
        trader.config.leverage = args.leverage
        trader.config.risk_per_trade = args.risk
        trader.config.max_positions = args.max_positions
        trader.config.min_24h_gain = args.min_gain
        trader.config.max_24h_gain = args.max_gain
        
        # Update strategy with new config
        trader.strategy = trader.strategy.__class__(trader.config)
        
        print(f"⚙️ Configuration Updated:")
        print(f"   API: {'Demo' if demo else 'Mainnet'}")
        print(f"   Balance: ${trader.config.account_balance}")
        print(f"   Target: ${trader.config.target_balance}")
        print(f"   Leverage: {trader.config.leverage}x")
        print(f"   Risk: {trader.config.risk_per_trade*100}%")
        print(f"   Positions: {trader.config.max_positions}")
        print(f"   24h Gain Range: {trader.config.min_24h_gain*100}%-{trader.config.max_24h_gain*100}%")
        print("=" * 60)
        
        # Run the trader
        trader.run()
        
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
