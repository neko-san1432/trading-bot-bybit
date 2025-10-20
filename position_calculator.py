#!/usr/bin/env python3
"""
Position Size Calculator - Debug margin and position sizing issues
Helps understand why trades fail with "insufficient balance"
"""

import os
import sys
from pybit.unified_trading import HTTP
from high_momentum_strategy import create_micro_balance_config, HighMomentumStrategy

def calculate_position_sizes():
    """Calculate position sizes for different scenarios"""
    
    print("Position Size Calculator")
    print("=" * 60)
    
    # Initialize API client
    try:
        # Try mainnet first, fallback to demo
        api_key = os.getenv('BYBIT_API_KEY_REAL')
        api_secret = os.getenv('BYBIT_API_SECRET_REAL')
        use_demo = False
        
        if not api_key or not api_secret:
            print("Warning: Mainnet API keys not found, using demo keys...")
            api_key = os.getenv('BYBIT_API_KEY_DEMO')
            api_secret = os.getenv('BYBIT_API_SECRET_DEMO')
            use_demo = True
            
        if not api_key or not api_secret:
            print("Error: No API keys found. Please set BYBIT_API_KEY_REAL and BYBIT_API_SECRET_REAL")
            return
            
        if use_demo:
            # Use demo API with demo endpoint
            client = HTTP(
                demo=True,
                api_key=api_key,
                api_secret=api_secret
            )
            print(f"Using API: Demo (https://api-demo.bybit.com)")
        else:
            # Use mainnet API
            client = HTTP(
                testnet=False,
                api_key=api_key,
                api_secret=api_secret
            )
            print(f"Using API: Mainnet (https://api.bybit.com)")
        
    except Exception as e:
        print(f"API Error: {e}")
        return
    
    # Get account balance
    try:
        response = client.get_wallet_balance(accountType="UNIFIED")
        if response['retCode'] == 0:
            balance = float(response['result']['list'][0]['totalWalletBalance'])
            print(f"Account Balance: ${balance:.2f}")
        else:
            print(f"Failed to get balance: {response['retMsg']}")
            balance = 1.7  # Use default
    except Exception as e:
        print(f"Balance error: {e}")
        balance = 1.7
    
    # Create strategy with actual balance
    config = create_micro_balance_config()
    config.account_balance = balance
    strategy = HighMomentumStrategy(config)
    
    print(f"Strategy Configuration:")
    print(f"   Leverage: {config.leverage}x")
    print(f"   Risk per Trade: {config.risk_per_trade*100}%")
    print(f"   Max Total Risk: {config.max_total_risk*100}%")
    print(f"   Min Position Value: ${config.min_position_value}")
    print(f"   Max Position Value: ${config.max_position_value}")
    print("=" * 60)
    
    # Test different scenarios
    scenarios = [
        # (symbol, entry_price, stop_price, signal_strength, description)
        ("BTCUSDT", 50000, 49000, 8.0, "BTC - 2% stop"),
        ("ETHUSDT", 3000, 2940, 7.0, "ETH - 2% stop"),
        ("DOGEUSDT", 0.08, 0.078, 6.0, "DOGE - 2.5% stop"),
        ("XRPUSDT", 0.5, 0.49, 5.0, "XRP - 2% stop"),
        ("ADAUSDT", 0.4, 0.392, 6.0, "ADA - 2% stop"),
        ("SOLUSDT", 100, 98, 7.0, "SOL - 2% stop"),
        ("MATICUSDT", 0.8, 0.784, 5.0, "MATIC - 2% stop"),
        ("AVAXUSDT", 25, 24.5, 6.0, "AVAX - 2% stop"),
    ]
    
    for symbol, entry, stop, strength, desc in scenarios:
        print(f"\n{symbol} - {desc}")
        print("-" * 40)
        
        calc = strategy.get_position_size_calculation(symbol, entry, stop, strength)
        
        if 'error' not in calc:
            print(f"   Entry Price: ${calc['entry_price']:.4f}")
            print(f"   Stop Price: ${calc['stop_price']:.4f}")
            print(f"   Stop Distance: {calc['stop_distance_pct']:.2%}")
            print(f"   Base Risk Amount: ${calc['base_risk_amount']:.2f}")
            print(f"   Strength Multiplier: {calc['strength_multiplier']:.2f}")
            print(f"   Adjusted Risk: ${calc['adjusted_risk_amount']:.2f}")
            print(f"   Position Size: {calc['position_size']:.6f}")
            print(f"   Position Value: ${calc['position_value']:.2f}")
            print(f"   Leveraged Size: {calc['leveraged_position_size']:.6f}")
            print(f"   Leveraged Value: ${calc['leveraged_position_value']:.2f}")
            print(f"   Required Margin: ${calc['required_margin']:.2f}")
            print(f"   Available Balance: ${calc['available_balance']:.2f}")
            print(f"   Can Trade: {'YES' if calc['can_trade'] else 'NO'}")
            
            if not calc['can_trade']:
                shortfall = calc['required_margin'] - calc['available_balance']
                print(f"   Shortfall: ${shortfall:.2f}")
                
                # Suggest solutions
                suggested_balance = calc['required_margin'] * 1.2  # 20% buffer
                print(f"   Suggested Balance: ${suggested_balance:.2f}")
                
                # Calculate minimum viable position
                min_position_value = calc['available_balance'] * 0.8  # Use 80% of available
                min_position_size = min_position_value / entry
                min_leveraged_size = min_position_size * config.leverage
                min_required_margin = min_position_value / config.leverage
                
                print(f"   Minimum Viable Position:")
                print(f"      Size: {min_leveraged_size:.6f}")
                print(f"      Value: ${min_position_value:.2f}")
                print(f"      Required Margin: ${min_required_margin:.2f}")
        else:
            print(f"   Error: {calc['error']}")
    
    print("\n" + "=" * 60)
    print("SOLUTIONS FOR INSUFFICIENT BALANCE:")
    print("=" * 60)
    print("1. Increase Account Balance")
    print("   - Add more funds to your account")
    print("   - Minimum recommended: $5-10")
    print()
    print("2. Reduce Position Sizes")
    print("   - Lower risk_per_trade from 20% to 10%")
    print("   - Reduce max_position_value")
    print("   - Use smaller stop distances")
    print()
    print("3. Adjust Strategy Settings")
    print("   - Lower leverage (50x instead of 100x)")
    print("   - Increase stop distances (3% instead of 2%)")
    print("   - Trade higher-priced coins")
    print()
    print("4. Focus on High-Value Coins")
    print("   - Trade BTC, ETH instead of low-price coins")
    print("   - Higher prices = smaller position sizes needed")
    print()
    print("5. Optimize Risk Management")
    print("   - Use 10% risk per trade instead of 20%")
    print("   - Set max_position_value to $10-20")
    print("   - Use 2-3% stop distances")

def suggest_optimal_settings():
    """Suggest optimal settings for small balance"""
    print("\nOPTIMAL SETTINGS FOR $1.7 BALANCE")
    print("=" * 60)
    
    balance = 1.7
    
    print("Recommended Configuration:")
    print(f"   Account Balance: ${balance}")
    print(f"   Leverage: 50x (instead of 100x)")
    print(f"   Risk per Trade: 10% (instead of 20%)")
    print(f"   Max Position Value: $10 (instead of $25)")
    print(f"   Stop Distance: 3% (instead of 2%)")
    print(f"   Max Positions: 1 (instead of 2)")
    print()
    
    print("Position Size Examples:")
    print(f"   BTC at $50,000 with 3% stop:")
    print(f"   - Risk Amount: ${balance * 0.10:.2f}")
    print(f"   - Stop Distance: $1,500")
    print(f"   - Position Size: {balance * 0.10 / 1500:.6f} BTC")
    print(f"   - Leveraged Size: {balance * 0.10 / 1500 * 50:.6f} BTC")
    print(f"   - Required Margin: ${balance * 0.10 / 50:.2f}")
    print(f"   - Can Trade: {'YES' if balance * 0.10 / 50 <= balance * 0.8 else 'NO'}")
    print()
    
    print("Strategy Recommendations:")
    print("1. Start with 1 position only")
    print("2. Use 50x leverage maximum")
    print("3. Risk 10% per trade")
    print("4. Use 3% stop distances")
    print("5. Target 2-3% profits")
    print("6. Focus on high-volume coins")
    print("7. Trade during high volatility periods")

def main():
    """Main function"""
    print("Position Size Calculator & Margin Debugger")
    print("=" * 60)
    
    # Check if API keys are set
    if not os.getenv('BYBIT_API_KEY_DEMO'):
        print("Warning: BYBIT_API_KEY_DEMO not set")
        print("   Using mock calculations...")
    
    # Calculate position sizes
    calculate_position_sizes()
    
    # Suggest optimal settings
    suggest_optimal_settings()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
