#!/usr/bin/env python3
"""
Enhanced Crypto Trading Strategy Demo

This script demonstrates how to use the enhanced crypto trading strategy
with synthetic data to show the different components working together.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.enhanced_strategy import EnhancedStrategyConfig, EnhancedCryptoStrategy


def create_synthetic_data(n_bars: int = 1000, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Create synthetic OHLCV data for demonstration"""
    print(f"Creating {n_bars} bars of synthetic data for {symbol}...")
    
    # Generate realistic price movement
    np.random.seed(42)  # For reproducible results
    
    # Base price
    base_price = 45000 if symbol == "BTCUSDT" else 3000
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0, 0.002, n_bars)  # 0.2% average volatility
    prices = [base_price]
    
    for i in range(1, n_bars):
        # Add some trend and mean reversion
        trend = 0.0001 if i < n_bars // 2 else -0.0001
        price = prices[-1] * (1 + returns[i] + trend)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.001))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000
        volume_multiplier = 1 + volatility * 10
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
        
        # Create timestamp
        timestamp = datetime.now() - timedelta(minutes=n_bars-i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def demonstrate_strategy_components():
    """Demonstrate the different strategy components"""
    print("\n" + "="*60)
    print("ENHANCED CRYPTO TRADING STRATEGY DEMONSTRATION")
    print("="*60)
    
    # Create synthetic data
    df = create_synthetic_data(500, "BTCUSDT")
    print(f"✅ Created {len(df)} bars of synthetic data")
    
    # Create strategy configurations for different styles
    styles = ["scalping", "day", "swing"]
    
    for style in styles:
        print(f"\n📊 Testing {style.upper()} strategy...")
        print("-" * 40)
        
        # Create configuration
        config = EnhancedStrategyConfig(trading_style=style)
        
        # Create strategy
        strategy = EnhancedCryptoStrategy(config)
        
        # Run backtest
        results = strategy.run_backtest(df, verbose=False)
        
        # Print results
        print(f"Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total PnL: ${results['total_pnl']:,.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.1%}")
        
        if results['trades']:
            # Show recent trades
            recent_trades = results['trades'][-3:]
            print(f"\nRecent trades:")
            for i, trade in enumerate(recent_trades, 1):
                print(f"  {i}. {trade['direction']} @ ${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} | PnL: ${trade['pnl']:.2f}")


def demonstrate_signal_analysis():
    """Demonstrate signal analysis in detail"""
    print("\n" + "="*60)
    print("SIGNAL ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create data
    df = create_synthetic_data(200, "ETHUSDT")
    
    # Create strategy
    config = EnhancedStrategyConfig(trading_style="swing")
    strategy = EnhancedCryptoStrategy(config)
    
    # Prepare data
    df = strategy.prepare(df)
    
    print(f"Analyzing {len(df)} bars of data...")
    
    # Analyze each bar for signals
    signals_found = 0
    for i in range(50, len(df)):  # Start from bar 50 to have enough history
        row = df.iloc[i]
        signal = strategy.get_entry_signal("ETHUSDT", row)
        
        if signal:
            signals_found += 1
            print(f"\n🎯 Signal #{signals_found} at bar {i}:")
            print(f"   Direction: {signal['direction'].upper()}")
            print(f"   Price: ${signal['entry_price']:.2f}")
            print(f"   Signal Strength: {signal['signal_strength']}")
            print(f"   Combined Score: {signal['combined_score']:.3f}")
            print(f"   Technical Signals: {', '.join(signal['technical_signals'])}")
            print(f"   Sentiment: {signal['sentiment_analysis']['combined']:.3f}")
            print(f"   Fundamental: {signal['fundamental_analysis']['score']:.3f}")
            
            if signals_found >= 5:  # Limit to first 5 signals
                break
    
    if signals_found == 0:
        print("No signals found in the data. This is normal for synthetic data.")
    else:
        print(f"\n✅ Found {signals_found} trading signals")


def demonstrate_risk_management():
    """Demonstrate risk management calculations"""
    print("\n" + "="*60)
    print("RISK MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create strategy
    config = EnhancedStrategyConfig(trading_style="swing")
    strategy = EnhancedCryptoStrategy(config)
    
    # Simulate different account balances
    balances = [1000, 5000, 10000, 50000]
    
    print("Position sizing for different account balances:")
    print("-" * 50)
    
    for balance in balances:
        # Create a mock signal
        mock_signal = {
            'direction': 'long',
            'entry_price': 45000,
            'signal_strength': 4,
            'combined_score': 0.7
        }
        
        # Calculate position size
        position_info = strategy.calculate_position_size(mock_signal, balance)
        
        print(f"\nAccount Balance: ${balance:,}")
        print(f"  Position Size: {position_info['position_size']:.4f} BTC")
        print(f"  Notional Value: ${position_info['notional_value']:,.2f}")
        print(f"  Margin Required: ${position_info['margin_required']:,.2f}")
        print(f"  Leverage Used: {position_info['leverage_used']:.1f}x")
        print(f"  Stop Loss: ${position_info['stop_loss_price']:,.2f}")
        print(f"  Take Profit: ${position_info['take_profit_price']:,.2f}")


def demonstrate_parameter_optimization():
    """Demonstrate parameter optimization"""
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create data
    df = create_synthetic_data(300, "BTCUSDT")
    
    # Test different parameter combinations
    tp_values = [0.01, 0.015, 0.02]
    sl_values = [0.005, 0.01, 0.015]
    
    best_score = 0
    best_params = None
    best_result = None
    
    print("Testing parameter combinations...")
    print("-" * 40)
    
    for tp in tp_values:
        for sl in sl_values:
            # Create config with test parameters
            config = EnhancedStrategyConfig(
                trading_style="swing",
                technical=type('TechnicalConfig', (), {
                    'ema_fast': 9,
                    'ema_slow': 20,
                    'ema_trend': 50,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'vol_ma': 20,
                    'vol_spike_threshold': 1.5,
                    'lookback_pivot': 5,
                    'take_profit': tp,
                    'stop_loss': sl,
                    'max_holding_bars': 20,
                    'risk_per_trade': 0.01,
                    'leverage': 5.0
                })()
            )
            
            # Run backtest
            strategy = EnhancedCryptoStrategy(config)
            result = strategy.run_backtest(df, verbose=False)
            
            # Calculate score (combination of win rate and Sharpe ratio)
            if result['total_trades'] > 0:
                score = result['win_rate'] * result['sharpe_ratio'] * (1 - result['max_drawdown'])
                
                print(f"TP={tp:.1%}, SL={sl:.1%} -> Trades={result['total_trades']}, Win={result['win_rate']:.1%}, Score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = (tp, sl)
                    best_result = result
    
    if best_params:
        tp, sl = best_params
        print(f"\n🏆 Best parameters found:")
        print(f"   Take Profit: {tp:.1%}")
        print(f"   Stop Loss: {sl:.1%}")
        print(f"   Score: {best_score:.3f}")
        print(f"   Trades: {best_result['total_trades']}")
        print(f"   Win Rate: {best_result['win_rate']:.1%}")
        print(f"   Final Equity: ${best_result['final_equity']:,.2f}")


def main():
    """Main demonstration function"""
    print("🚀 Enhanced Crypto Trading Strategy Demo")
    print("This demo shows how the strategy works with synthetic data")
    
    try:
        # Demonstrate strategy components
        demonstrate_strategy_components()
        
        # Demonstrate signal analysis
        demonstrate_signal_analysis()
        
        # Demonstrate risk management
        demonstrate_risk_management()
        
        # Demonstrate parameter optimization
        demonstrate_parameter_optimization()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("\nNext steps:")
        print("1. Run backtest with real data: python backtest/run_enhanced_backtest.py --file your_data.csv")
        print("2. Test live trading on testnet: python live/enhanced_trader.py --testnet")
        print("3. Read the full guide: ENHANCED_STRATEGY_GUIDE.md")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check your Python environment and dependencies.")


if __name__ == "__main__":
    main()
