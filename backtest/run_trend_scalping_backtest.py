import pandas as pd
import click
import numpy as np
from trend_scalping_strategy import TrendScalpingConfig, TrendScalpingStrategy
import os
from datetime import datetime
import json


def load_csv(path: str) -> pd.DataFrame:
    """Load and prepare CSV data for backtesting"""
    df = pd.read_csv(path)
    
    # Normalize timestamp
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    else:
        # Assume first column is timestamp
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    
    # Convert numeric columns
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    return df


def print_strategy_summary(config: TrendScalpingConfig):
    """Print a summary of the trend scalping strategy configuration"""
    print("=" * 60)
    print("TREND SCALPING STRATEGY")
    print("=" * 60)
    print(f"Strategy Type: TREND SCALPING")
    print(f"Max Positions: 8 (default)")
    print(f"Max Daily Trades: 50 (default)")
    print(f"Max Daily Loss: 5% (default)")
    print()
    
    print("TREND DETECTION:")
    print(f"  • ADX Period: {config.adx_period} | Threshold: {config.adx_threshold}")
    print(f"  • EMAs: {config.ema_fast}/{config.ema_medium}/{config.ema_slow}/{config.ema_trend}/{config.ema_long}")
    print(f"  • Entry Signals: Pullback({config.pullback_weight:.1f}) + Breakout({config.breakout_weight:.1f}) + Momentum({config.momentum_weight:.1f})")
    print()
    
    print("PROFIT TARGETS:")
    print(f"  • Base Target: {config.base_take_profit:.1%}")
    print(f"  • Range: {config.min_take_profit:.1%} - {config.max_take_profit:.1%}")
    print(f"  • Dynamic: Volatility({config.volatility_multiplier:.1f}x) + Trend({config.trend_multiplier:.1f}x)")
    print()
    
    print("RISK MANAGEMENT:")
    print(f"  • ATR Multiplier: {config.atr_multiplier:.1f}x (Range: {config.min_atr_multiplier:.1f}-{config.max_atr_multiplier:.1f})")
    print(f"  • Risk per Trade: {config.risk_per_trade:.1%}")
    print(f"  • Leverage: {config.leverage:.1f}x (Range: {config.min_leverage:.1f}-{config.max_leverage:.1f})")
    print(f"  • Hold Time: {config.min_holding_bars}-{config.max_holding_bars} minutes")
    print()
    
    print("VOLATILITY FILTERS:")
    print(f"  • Min Volatility: {config.min_volatility:.1%}")
    print(f"  • Max Volatility: {config.max_volatility:.1%}")
    print(f"  • Volume Spike: {config.volume_spike_threshold:.1f}x average")
    print("=" * 60)


def print_trade_analysis(results: dict):
    """Print detailed trade analysis for trend scalping"""
    trades = results['trades']
    
    if not trades:
        print("No trades executed during backtest period.")
        return
    
    print("\nTRADE ANALYSIS:")
    print("-" * 50)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['total_trades'] - results['wins']}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total PnL: ${results['total_pnl']:,.2f}")
    print(f"Average Win: ${results['avg_win']:,.2f}")
    print(f"Average Loss: ${results['avg_loss']:,.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    
    # Scalping-specific metrics
    print(f"\nSCALPING METRICS:")
    print(f"Average Hold Time: {results['avg_hold_time']:.1f} minutes")
    print(f"Max Consecutive Losses: {results['max_consecutive_losses']}")
    print(f"Average Risk-Reward: {results['avg_risk_reward']:.1f}")
    
    # Signal analysis
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    
    print(f"\nLONG TRADES: {len(long_trades)}")
    if long_trades:
        long_win_rate = sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades)
        print(f"  Win Rate: {long_win_rate:.1%}")
        print(f"  Avg PnL: ${np.mean([t['pnl'] for t in long_trades]):,.2f}")
        print(f"  Avg Hold Time: {np.mean([t['hold_time'] for t in long_trades]):.1f} min")
    
    print(f"\nSHORT TRADES: {len(short_trades)}")
    if short_trades:
        short_win_rate = sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades)
        print(f"  Win Rate: {short_win_rate:.1%}")
        print(f"  Avg PnL: ${np.mean([t['pnl'] for t in short_trades]):,.2f}")
        print(f"  Avg Hold Time: {np.mean([t['hold_time'] for t in short_trades]):.1f} min")
    
    # Entry signal analysis
    signal_counts = {}
    for trade in trades:
        for signal in trade['entry_signals']:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    print(f"\nENTRY SIGNAL ANALYSIS:")
    for signal, count in sorted(signal_counts.items(), key=lambda x: x[1], reverse=True):
        win_count = sum(1 for t in trades if signal in t['entry_signals'] and t['pnl'] > 0)
        total_count = sum(1 for t in trades if signal in t['entry_signals'])
        win_rate = win_count / total_count if total_count > 0 else 0
        print(f"  {signal}: {count} trades, {win_rate:.1%} win rate")
    
    # Trend direction analysis
    trend_counts = {}
    for trade in trades:
        trend = trade['trend_direction']
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
    
    print(f"\nTREND DIRECTION ANALYSIS:")
    for trend, count in sorted(trend_counts.items(), key=lambda x: x[1], reverse=True):
        win_count = sum(1 for t in trades if t['trend_direction'] == trend and t['pnl'] > 0)
        win_rate = win_count / count if count > 0 else 0
        print(f"  {trend}: {count} trades, {win_rate:.1%} win rate")


def print_recent_trades(trades: list, limit: int = 15):
    """Print recent trades for analysis"""
    print(f"\nRECENT TRADES (Last {min(limit, len(trades))}):")
    print("-" * 100)
    print(f"{'#':<3} {'Time':<12} {'Dir':<4} {'Entry':<8} {'Exit':<8} {'PnL':<8} {'Hold':<4} {'Signals':<20} {'Trend':<8} {'R:R':<5}")
    print("-" * 100)
    
    for i, trade in enumerate(trades[-limit:]):
        time_str = str(trade['time'])[:12] if 'time' in trade else f"#{trade['index']}"
        signals_str = ', '.join(trade['entry_signals'][:2])  # Show first 2 signals
        print(f"{i+1:<3} {time_str:<12} {trade['direction']:<4} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl']:<8.2f} {trade['hold_time']:<4} {signals_str:<20} "
              f"{trade['trend_direction']:<8} {trade['risk_reward_ratio']:<5.1f}")


@click.command()
@click.option('--file', 'file_path', required=True, help='Path to OHLCV CSV file')
@click.option('--symbol', default='BTCUSDT', help='Trading symbol')
@click.option('--adx-threshold', default=25.0, help='ADX threshold for trend strength')
@click.option('--base-tp', default=0.015, help='Base take profit percentage')
@click.option('--atr-multiplier', default=2.0, help='ATR multiplier for stop loss')
@click.option('--leverage', default=30.0, help='Base leverage')
@click.option('--risk', default=0.003, help='Risk per trade (as decimal)')
@click.option('--verbose', is_flag=True, help='Show detailed trade information')
@click.option('--optimize', is_flag=True, help='Run parameter optimization')
@click.option('--save-results', is_flag=True, help='Save results to JSON file')
def main(file_path, symbol, adx_threshold, base_tp, atr_multiplier, leverage, risk, verbose, optimize, save_results):
    """Trend scalping strategy backtester"""
    
    # Load data
    print(f"Loading data from {file_path}...")
    try:
        df = load_csv(file_path)
        print(f"Loaded {len(df)} bars of data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create strategy configuration
    config = TrendScalpingConfig(
        adx_threshold=adx_threshold,
        base_take_profit=base_tp,
        atr_multiplier=atr_multiplier,
        leverage=leverage,
        risk_per_trade=risk
    )
    
    # Print strategy summary
    print_strategy_summary(config)
    
    if optimize:
        print("\nRunning parameter optimization...")
        best_result = None
        best_config = None
        
        # Parameter ranges for optimization
        adx_values = [20, 25, 30, 35]
        tp_values = [0.01, 0.015, 0.02, 0.025]
        atr_values = [1.5, 2.0, 2.5, 3.0]
        leverage_values = [20, 30, 40, 50]
        
        total_combinations = len(adx_values) * len(tp_values) * len(atr_values) * len(leverage_values)
        current = 0
        
        for adx in adx_values:
            for tp in tp_values:
                for atr in atr_values:
                    for lev in leverage_values:
                        current += 1
                        print(f"Testing combination {current}/{total_combinations}: ADX={adx}, TP={tp:.1%}, ATR={atr:.1f}x, Lev={lev}x")
                        
                        # Create test config
                        test_config = TrendScalpingConfig(
                            adx_threshold=adx,
                            base_take_profit=tp,
                            atr_multiplier=atr,
                            leverage=lev,
                            risk_per_trade=risk
                        )
                        
                        # Run backtest
                        strategy = TrendScalpingStrategy(test_config)
                        result = strategy.run_backtest(df, verbose=False)
                        
                        # Evaluate result (using Sharpe ratio and win rate as primary metrics)
                        if result['total_trades'] > 10:  # Need minimum trades
                            score = result['sharpe_ratio'] * result['win_rate'] * (1 - result['max_drawdown'])
                            
                            if best_result is None or score > best_result['score']:
                                best_result = {
                                    'score': score,
                                    'result': result,
                                    'config': test_config
                                }
                                best_config = test_config
        
        if best_result:
            print(f"\nBest configuration found:")
            print(f"  ADX Threshold: {best_config.adx_threshold}")
            print(f"  Base TP: {best_config.base_take_profit:.1%}")
            print(f"  ATR Multiplier: {best_config.atr_multiplier:.1f}x")
            print(f"  Leverage: {best_config.leverage:.1f}x")
            print(f"  Score: {best_result['score']:.3f}")
            
            # Run final backtest with best config
            strategy = TrendScalpingStrategy(best_config)
            results = strategy.run_backtest(df, verbose=verbose)
        else:
            print("No valid configurations found during optimization.")
            return
    else:
        # Run single backtest
        strategy = TrendScalpingStrategy(config)
        results = strategy.run_backtest(df, verbose=verbose)
    
    # Print results
    print_trade_analysis(results)
    print_recent_trades(results['trades'], limit=15)
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trend_scalping_results_{symbol}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'config': {
                'adx_threshold': config.adx_threshold,
                'base_take_profit': config.base_take_profit,
                'atr_multiplier': config.atr_multiplier,
                'leverage': config.leverage,
                'risk_per_trade': config.risk_per_trade
            },
            'performance': {
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'final_equity': results['final_equity'],
                'total_pnl': results['total_pnl'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'avg_hold_time': results['avg_hold_time'],
                'max_consecutive_losses': results['max_consecutive_losses'],
                'avg_risk_reward': results['avg_risk_reward']
            },
            'trades': [
                {
                    'index': t['index'],
                    'direction': t['direction'],
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'pnl': t['pnl'],
                    'hold_time': t['hold_time'],
                    'signal_strength': t['signal_strength'],
                    'entry_signals': t['entry_signals'],
                    'trend_direction': t['trend_direction'],
                    'risk_reward_ratio': t['risk_reward_ratio']
                }
                for t in results['trades']
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


if __name__ == '__main__':
    main()
