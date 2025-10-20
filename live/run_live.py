import click
from trader import LiveTrader

@click.command()
@click.option('--symbols', '-s', multiple=True, help='Trading pairs to scan (default: BTCUSDT,ETHUSDT)')
@click.option('--max-positions', default=3, help='Maximum number of open positions')
@click.option('--max-daily-trades', default=20, help='Maximum number of trades per day')
@click.option('--max-loss-percent', default=5.0, help='Maximum daily loss percentage before stopping')
@click.option('--testnet/--mainnet', default=False, help='Use testnet for backtesting or mainnet for live trading')
def main(symbols, max_positions, max_daily_trades, max_loss_percent, testnet):
    """Run the automated crypto scalping strategy"""
    symbols = list(symbols) if symbols else ['BTCUSDT', 'ETHUSDT']
    
    env_name = "TESTNET (Backtesting)" if testnet else "MAINNET (Live Trading)"
    print(f"Starting live trader with settings:")
    print(f"Environment: {env_name}")
    print(f"Symbols: {symbols}")
    print(f"Max positions: {max_positions}")
    print(f"Max daily trades: {max_daily_trades}")
    print(f"Max loss %: {max_loss_percent}")
    
    trader = LiveTrader(
        symbols=symbols,
        max_positions=max_positions,
        max_daily_trades=max_daily_trades,
        max_loss_percent=max_loss_percent,
        use_testnet=testnet
    )
    
    try:
        trader.scan_and_trade()
    except KeyboardInterrupt:
        print("\nStopping trader...")

if __name__ == '__main__':
    main()