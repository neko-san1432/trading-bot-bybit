import pandas as pd
import click
from strategy import ScalperConfig, ScalperStrategy
import os


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # try to normalize timestamp
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    else:
        # assume first column is timestamp
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

    # ensure columns exist
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    return df


@click.command()
@click.option('--file', 'file_path', required=True, help='Path to OHLCV CSV')
@click.option('--symbol', default='BTCUSDT')
@click.option('--sweep', is_flag=True, help='Run a small parameter sweep')
def main(file_path, symbol, sweep):
    df = load_csv(file_path)
    cfg = ScalperConfig()
    strat = ScalperStrategy(cfg)
    res = strat.run_backtest(df)
    print(f"Symbol: {symbol}  Trades: {res['total_trades']}  Win rate: {res['win_rate']:.2%}  Final equity: {res['final_equity']:.2f}")

    if sweep:
        print('Running small sweep over TP/SL...')
        best = None
        for tp in [0.005, 0.0075, 0.01]:
            for sl in [0.002, 0.003, 0.005]:
                cfg = ScalperConfig(take_profit=tp, stop_loss=sl)
                strat = ScalperStrategy(cfg)
                r = strat.run_backtest(df)
                if best is None or r['win_rate'] > best['win_rate']:
                    best = {'cfg': (tp, sl), 'res': r}
        if best:
            tp, sl = best['cfg']
            print(f"Best sweep TP={tp:.4f} SL={sl:.4f} -> Trades={best['res']['total_trades']} Win={best['res']['win_rate']:.2%} Equity={best['res']['final_equity']:.2f}")


if __name__ == '__main__':
    main()
