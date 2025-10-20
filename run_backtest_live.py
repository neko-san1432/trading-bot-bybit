#!/usr/bin/env python3
"""
Run backtesting using testnet (live data) instead of historical CSV files
"""
import click
from live.trader import LiveTrader

@click.command()
@click.option('--symbols', '-s', multiple=True, help='Trading pairs to scan (default: BTCUSDT,ETHUSDT)')
@click.option('--max-positions', default=3, help='Maximum number of open positions')
@click.option('--max-daily-trades', default=20, help='Maximum number of trades per day')
@click.option('--max-loss-percent', default=5.0, help='Maximum daily loss percentage before stopping')
@click.option('--duration', default=60, help='Backtest duration in minutes')
def main(symbols, max_positions, max_daily_trades, max_loss_percent, duration):
    """Run backtesting using testnet live data"""
    symbols = list(symbols) if symbols else ['BTCUSDT', 'ETHUSDT']
    
    print(f"Starting backtest with live testnet data:")
    print(f"Environment: TESTNET (Backtesting)")
    print(f"Symbols: {symbols}")
    print(f"Max positions: {max_positions}")
    print(f"Max daily trades: {max_daily_trades}")
    print(f"Max loss %: {max_loss_percent}")
    print(f"Duration: {duration} minutes")
    print(f"⚠️  This will use testnet for safe backtesting")
    
    trader = LiveTrader(
        symbols=symbols,
        max_positions=max_positions,
        max_daily_trades=max_daily_trades,
        max_loss_percent=max_loss_percent,
        use_testnet=True  # Always use testnet for backtesting
    )
    
    try:
        trader.scan_and_trade()
    except KeyboardInterrupt:
        print("\nStopping backtest...")

if __name__ == '__main__':
    main()
