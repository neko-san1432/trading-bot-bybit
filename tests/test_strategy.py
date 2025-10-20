import pandas as pd
import numpy as np
from backtest.strategy import ScalperConfig, ScalperStrategy


def make_synthetic():
    # create 200 bars of synthetic price action with a quick RSI dip then bounce
    n = 200
    timestamps = pd.date_range('2025-01-01', periods=n, freq='T')
    price = 10000 + (pd.Series(range(n)).apply(lambda x: np.sin(x / 3) * 5))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': price.shift(1).fillna(price.iloc[0]),
        'high': price + 1,
        'low': price - 1,
        'close': price,
        'volume': 100 + (pd.Series(range(n)) % 10) * 10,
    })
    return df


def test_run():
    df = make_synthetic()
    cfg = ScalperConfig()
    strat = ScalperStrategy(cfg)
    res = strat.run_backtest(df)
    assert 'total_trades' in res


if __name__ == '__main__':
    test_run()
    print('ok')
