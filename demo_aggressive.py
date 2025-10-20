#!/usr/bin/env python3
"""
Aggressive Trading Demo
Demonstrates different aggressive trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from aggressive_strategy import AggressiveStrategy, create_aggressive_config
from aggressive_configs import get_aggressive_config, list_available_strategies

def create_sample_data(symbol: str, periods: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    base_price = 100.0
    prices = [base_price]
    
    for i in range(periods - 1):
        # Random walk with trend
        change = np.random.normal(0, 0.02)  # 2% volatility
        if i > 20:  # Add some trend after 20 periods
            change += 0.001  # Slight upward bias
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': datetime.now() - timedelta(minutes=periods-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def demo_strategy(strategy_name: str):
    """Demo a specific aggressive strategy"""
    print(f"\n🚀 Demo: {strategy_name.upper()} Strategy")
    print("=" * 50)
    
    # Get strategy configuration
    config = get_aggressive_config(strategy_name)
    strategy = AggressiveStrategy(config)
    
    # Create sample data for multiple timeframes
    timeframes = ['1m', '5m', '15m']
    data_dict = {}
    
    for tf in timeframes:
        periods = 100 if tf == '1m' else 50 if tf == '5m' else 30
        data_dict[tf] = create_sample_data(f"BTCUSDT", periods)
    
    # Analyze with strategy
    analysis = strategy.analyze_multi_timeframe("BTCUSDT", data_dict)
    
    print(f"📊 Analysis Results:")
    print(f"   Action: {analysis['action'].upper()}")
    print(f"   Strength: {analysis['strength']:.1f}")
    print(f"   Bullish Signals: {analysis['bullish_count']}")
    print(f"   Bearish Signals: {analysis['bearish_count']}")
    print(f"   Total Signals: {len(analysis['signals'])}")
    
    # Get entry signal
    signal = strategy.get_aggressive_entry_signal("BTCUSDT", analysis)
    
    if signal:
        print(f"\n🎯 Entry Signal Generated:")
        print(f"   Symbol: {signal['symbol']}")
        print(f"   Side: {signal['side'].upper()}")
        print(f"   Entry Price: ${signal['entry_price']:.4f}")
        print(f"   Stop Loss: ${signal['stop_price']:.4f}")
        print(f"   Take Profit: ${signal['take_profit']:.4f}")
        print(f"   Position Size: {signal['position_size']:.6f}")
        print(f"   Leverage: {signal['leverage']}x")
        print(f"   Risk Amount: ${signal['risk_amount']:.2f}")
        print(f"   Signal Type: {signal['signal_type']}")
        print(f"   Strategy: {signal['strategy']}")
    else:
        print(f"\n❌ No entry signal generated")
    
    # Show strategy configuration
    print(f"\n⚙️ Strategy Configuration:")
    print(f"   Max Positions: {config.max_positions}")
    print(f"   Leverage: {config.leverage}x")
    print(f"   Risk per Trade: {config.risk_per_trade*100}%")
    print(f"   Max Total Risk: {config.max_total_risk*100}%")
    print(f"   Timeframes: {', '.join(config.timeframes)}")
    print(f"   Risk-Reward Ratio: {config.risk_reward_ratio}")
    print(f"   Max Hold Time: {config.max_hold_time}s")
    print(f"   Min Profit Target: {config.min_profit_target*100}%")

def demo_all_strategies():
    """Demo all available strategies"""
    print("🚀 Aggressive Trading Strategies Demo")
    print("=" * 60)
    
    strategies = ['scalping', 'momentum', 'breakout', 'news', 'grid', 'volatility']
    
    for strategy in strategies:
        try:
            demo_strategy(strategy)
        except Exception as e:
            print(f"❌ Error demoing {strategy}: {e}")
    
    print(f"\n✅ Demo completed for all strategies!")

def demo_custom_strategy():
    """Demo custom strategy configuration"""
    print(f"\n🔧 Demo: Custom Strategy Configuration")
    print("=" * 50)
    
    # Create custom configuration
    config = create_aggressive_config()
    
    # Modify settings for custom strategy
    config.max_positions = 8
    config.leverage = 75.0
    config.risk_per_trade = 0.04
    config.max_total_risk = 0.25
    config.timeframes = ["1m", "3m", "5m"]
    config.ema_ultra_fast = 2
    config.ema_fast = 5
    config.rsi_period = 7
    config.risk_reward_ratio = 1.8
    
    strategy = AggressiveStrategy(config)
    
    print(f"📊 Custom Configuration:")
    print(f"   Max Positions: {config.max_positions}")
    print(f"   Leverage: {config.leverage}x")
    print(f"   Risk per Trade: {config.risk_per_trade*100}%")
    print(f"   Max Total Risk: {config.max_total_risk*100}%")
    print(f"   Timeframes: {', '.join(config.timeframes)}")
    print(f"   EMAs: {config.ema_ultra_fast}, {config.ema_fast}, {config.ema_medium}, {config.ema_slow}")
    print(f"   RSI Period: {config.rsi_period}")
    print(f"   Risk-Reward: {config.risk_reward_ratio}")
    
    # Test with sample data
    data_dict = {
        '1m': create_sample_data("ETHUSDT", 100),
        '3m': create_sample_data("ETHUSDT", 50),
        '5m': create_sample_data("ETHUSDT", 30)
    }
    
    analysis = strategy.analyze_multi_timeframe("ETHUSDT", data_dict)
    signal = strategy.get_aggressive_entry_signal("ETHUSDT", analysis)
    
    if signal:
        print(f"\n🎯 Custom Strategy Signal:")
        print(f"   Action: {signal['side'].upper()}")
        print(f"   Strength: {signal['strength']:.1f}")
        print(f"   Position Size: {signal['position_size']:.6f}")
        print(f"   Risk Amount: ${signal['risk_amount']:.2f}")
    else:
        print(f"\n❌ No signal generated with custom configuration")

def main():
    """Main demo function"""
    print("🚀 Aggressive Trading Strategies Demo")
    print("=" * 60)
    
    # List available strategies
    list_available_strategies()
    
    # Demo all strategies
    demo_all_strategies()
    
    # Demo custom strategy
    demo_custom_strategy()
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\nTo run live trading:")
    print(f"   python run_aggressive_bot.py --strategy scalping --testnet")
    print(f"   python run_aggressive_bot.py --strategy momentum --balance 10000")
    print(f"   python run_aggressive_bot.py --strategy breakout --mainnet")

if __name__ == "__main__":
    main()
