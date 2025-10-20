import pandas as pd
import click
import numpy as np
from enhanced_strategy import EnhancedStrategyConfig, EnhancedCryptoStrategy
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


def print_strategy_summary(config: EnhancedStrategyConfig):
    """Print a summary of the strategy configuration"""
    print("=" * 60)
    print("ENHANCED CRYPTO TRADING STRATEGY")
    print("=" * 60)
    print(f"Trading Style: {config.trading_style.upper()}")
    print(f"Max Positions: {config.max_positions}")
    print(f"Max Daily Trades: {config.max_daily_trades}")
    print(f"Max Daily Loss: {config.max_daily_loss:.1%}")
    print()
    
    print("TECHNICAL ANALYSIS:")
    print(f"  • EMA Fast/Slow: {config.technical.ema_fast}/{config.technical.ema_slow}")
    print(f"  • RSI Period: {config.technical.rsi_period}")
    print(f"  • Take Profit: {config.technical.take_profit:.1%}")
    print(f"  • Stop Loss: {config.technical.stop_loss:.1%}")
    print(f"  • Risk per Trade: {config.technical.risk_per_trade:.1%}")
    print(f"  • Leverage: {config.technical.leverage}x")
    print()
    
    print("SENTIMENT ANALYSIS:")
    print(f"  • News Weight: {config.sentiment.news_weight:.1%}")
    print(f"  • Social Weight: {config.sentiment.social_weight:.1%}")
    print(f"  • Technical Weight: {config.sentiment.technical_weight:.1%}")
    print(f"  • Positive Threshold: {config.sentiment.positive_sentiment_threshold}")
    print(f"  • Negative Threshold: {config.sentiment.negative_sentiment_threshold}")
    print()
    
    print("FUNDAMENTAL ANALYSIS:")
    print(f"  • Check Token Burns: {config.fundamental.check_token_burns}")
    print(f"  • Check Token Unlocks: {config.fundamental.check_token_unlocks}")
    print(f"  • Check Exchange Listings: {config.fundamental.check_exchange_listings}")
    print(f"  • Avoid Regulatory Uncertainty: {config.fundamental.avoid_regulatory_uncertainty}")
    print("=" * 60)


def print_trade_analysis(results: dict):
    """Print detailed trade analysis"""
    trades = results['trades']
    
    if not trades:
        print("No trades executed during backtest period.")
        return
    
    print("\nTRADE ANALYSIS:")
    print("-" * 40)
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
    
    # Signal analysis
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    
    print(f"\nLONG TRADES: {len(long_trades)}")
    if long_trades:
        long_win_rate = sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades)
        print(f"  Win Rate: {long_win_rate:.1%}")
        print(f"  Avg PnL: ${np.mean([t['pnl'] for t in long_trades]):,.2f}")
    
    print(f"\nSHORT TRADES: {len(short_trades)}")
    if short_trades:
        short_win_rate = sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades)
        print(f"  Win Rate: {short_win_rate:.1%}")
        print(f"  Avg PnL: ${np.mean([t['pnl'] for t in short_trades]):,.2f}")
    
    # Signal strength analysis
    signal_strengths = [t['signal_strength'] for t in trades]
    print(f"\nSIGNAL STRENGTH ANALYSIS:")
    print(f"  Average: {np.mean(signal_strengths):.1f}")
    print(f"  Min: {min(signal_strengths)}")
    print(f"  Max: {max(signal_strengths)}")
    
    # Score analysis
    combined_scores = [t['combined_score'] for t in trades]
    print(f"\nCOMBINED SCORE ANALYSIS:")
    print(f"  Average: {np.mean(combined_scores):.3f}")
    print(f"  Min: {min(combined_scores):.3f}")
    print(f"  Max: {max(combined_scores):.3f}")


def print_recent_trades(trades: list, limit: int = 10):
    """Print recent trades for analysis"""
    print(f"\nRECENT TRADES (Last {min(limit, len(trades))}):")
    print("-" * 80)
    print(f"{'#':<3} {'Time':<12} {'Dir':<4} {'Entry':<8} {'Exit':<8} {'PnL':<8} {'Score':<6} {'Signals':<20}")
    print("-" * 80)
    
    for i, trade in enumerate(trades[-limit:]):
        time_str = str(trade['time'])[:12] if 'time' in trade else f"#{trade['index']}"
        signals_str = ', '.join(trade['technical_signals'][:2])  # Show first 2 signals
        print(f"{i+1:<3} {time_str:<12} {trade['direction']:<4} "
              f"{trade['entry_price']:<8.2f} {trade['exit_price']:<8.2f} "
              f"{trade['pnl']:<8.2f} {trade['combined_score']:<6.3f} {signals_str:<20}")


@click.command()
@click.option('--file', 'file_path', required=True, help='Path to OHLCV CSV file')
@click.option('--symbol', default='BTCUSDT', help='Trading symbol')
@click.option('--style', type=click.Choice(['scalping', 'day', 'swing']), default='swing', help='Trading style')
@click.option('--leverage', default=5.0, help='Leverage multiplier')
@click.option('--risk', default=0.01, help='Risk per trade (as decimal)')
@click.option('--tp', default=0.015, help='Take profit percentage')
@click.option('--sl', default=0.01, help='Stop loss percentage')
@click.option('--verbose', is_flag=True, help='Show detailed trade information')
@click.option('--optimize', is_flag=True, help='Run parameter optimization')
@click.option('--save-results', is_flag=True, help='Save results to JSON file')
def main(file_path, symbol, style, leverage, risk, tp, sl, verbose, optimize, save_results):
    """Enhanced crypto trading strategy backtester"""
    
    # Load data
    print(f"Loading data from {file_path}...")
    try:
        df = load_csv(file_path)
        print(f"Loaded {len(df)} bars of data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create strategy configuration
    config = EnhancedStrategyConfig(
        trading_style=style,
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
            'liquidity_sweep_threshold': 0.001,
            'take_profit': tp,
            'stop_loss': sl,
            'risk_reward_ratio': tp / sl,
            'max_holding_bars': 20 if style == 'swing' else (5 if style == 'day' else 3),
            'risk_per_trade': risk,
            'leverage': leverage,
            'max_leverage': leverage * 2
        })(),
        sentiment=type('SentimentConfig', (), {
            'positive_sentiment_threshold': 0.6,
            'negative_sentiment_threshold': -0.6,
            'neutral_sentiment_threshold': 0.2,
            'news_lookback_hours': 24,
            'social_lookback_hours': 6,
            'news_weight': 0.4,
            'social_weight': 0.3,
            'technical_weight': 0.3
        })(),
        fundamental=type('FundamentalConfig', (), {
            'check_token_burns': True,
            'check_token_unlocks': True,
            'check_staking_rewards': True,
            'check_supply_changes': True,
            'check_regulatory_news': True,
            'check_exchange_listings': True,
            'check_partnerships': True,
            'check_influencer_tweets': True,
            'avoid_high_unlock_periods': True,
            'avoid_regulatory_uncertainty': True,
            'avoid_high_volatility_periods': True
        })()
    )
    
    # Print strategy summary
    print_strategy_summary(config)
    
    if optimize:
        print("\nRunning parameter optimization...")
        best_result = None
        best_config = None
        
        # Parameter ranges for optimization
        tp_values = [0.01, 0.015, 0.02, 0.025]
        sl_values = [0.005, 0.01, 0.015, 0.02]
        leverage_values = [3, 5, 7, 10]
        
        total_combinations = len(tp_values) * len(sl_values) * len(leverage_values)
        current = 0
        
        for tp_val in tp_values:
            for sl_val in sl_values:
                for lev_val in leverage_values:
                    current += 1
                    print(f"Testing combination {current}/{total_combinations}: TP={tp_val:.1%}, SL={sl_val:.1%}, Lev={lev_val}x")
                    
                    # Create test config
                    test_config = EnhancedStrategyConfig(
                        trading_style=style,
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
                            'liquidity_sweep_threshold': 0.001,
                            'take_profit': tp_val,
                            'stop_loss': sl_val,
                            'risk_reward_ratio': tp_val / sl_val,
                            'max_holding_bars': 20 if style == 'swing' else (5 if style == 'day' else 3),
                            'risk_per_trade': risk,
                            'leverage': lev_val,
                            'max_leverage': lev_val * 2
                        })()
                    )
                    
                    # Run backtest
                    strategy = EnhancedCryptoStrategy(test_config)
                    result = strategy.run_backtest(df, verbose=False)
                    
                    # Evaluate result (using Sharpe ratio as primary metric)
                    if result['total_trades'] > 0:
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
            print(f"  TP: {best_config.technical.take_profit:.1%}")
            print(f"  SL: {best_config.technical.stop_loss:.1%}")
            print(f"  Leverage: {best_config.technical.leverage}x")
            print(f"  Score: {best_result['score']:.3f}")
            
            # Run final backtest with best config
            strategy = EnhancedCryptoStrategy(best_config)
            results = strategy.run_backtest(df, verbose=verbose)
        else:
            print("No valid configurations found during optimization.")
            return
    else:
        # Run single backtest
        strategy = EnhancedCryptoStrategy(config)
        results = strategy.run_backtest(df, verbose=verbose)
    
    # Print results
    print_trade_analysis(results)
    print_recent_trades(results['trades'], limit=10)
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{symbol}_{style}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'config': {
                'trading_style': config.trading_style,
                'leverage': config.technical.leverage,
                'risk_per_trade': config.technical.risk_per_trade,
                'take_profit': config.technical.take_profit,
                'stop_loss': config.technical.stop_loss
            },
            'performance': {
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'final_equity': results['final_equity'],
                'total_pnl': results['total_pnl'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio']
            },
            'trades': [
                {
                    'index': t['index'],
                    'direction': t['direction'],
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'pnl': t['pnl'],
                    'signal_strength': t['signal_strength'],
                    'combined_score': t['combined_score']
                }
                for t in results['trades']
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


if __name__ == '__main__':
    main()
