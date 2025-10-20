import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import ta


@dataclass
class ScalperConfig:
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    ema_fast: int = 9
    ema_slow: int = 20
    vol_ma: int = 20
    take_profit: float = 0.0075  # 0.75% default
    stop_loss: float = 0.003    # 0.3% default
    max_holding_bars: int = 5   # 1-5 minutes target
    risk_per_trade: float = 0.01
    min_volume: float = 0.0
    leverage: float = 10.0      # Default 10x leverage for futures
    max_leverage: float = 20.0  # Maximum allowed leverage


class ScalperStrategy:
    def __init__(self, config: ScalperConfig):
        self.cfg = config

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # expect df with columns: timestamp, open, high, low, close, volume
        df = df.copy()
        # indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.cfg.rsi_period).rsi()
        df['ema_fast'] = df['close'].ewm(span=self.cfg.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.cfg.ema_slow, adjust=False).mean()
        df['vol_ma'] = df['volume'].rolling(self.cfg.vol_ma, min_periods=1).mean()
        # signals
        df['rsi_long'] = (df['rsi'] < self.cfg.rsi_oversold) & (df['rsi'].shift(1) >= self.cfg.rsi_oversold)
        df['rsi_short'] = (df['rsi'] > self.cfg.rsi_overbought) & (df['rsi'].shift(1) <= self.cfg.rsi_overbought)
        df['ema_cross_long'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_short'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        df['vol_spike'] = df['volume'] > (df['vol_ma'] * 1.2)

        # simple support/resistance bounce/break detection: lookback pivot
        lookback = 5
        df['pivot_low'] = df['low'] == df['low'].rolling(lookback, center=False).min()
        df['pivot_high'] = df['high'] == df['high'].rolling(lookback, center=False).max()

        return df

    def entry_signal(self, row: pd.Series) -> Optional[Tuple[str, list]]:
        # returns ('long'|'short', list_of_signals)
        long_signals = []
        short_signals = []

        if row['rsi_long']:
            long_signals.append('rsi')
        if row['ema_cross_long']:
            long_signals.append('ema')
        if row['vol_spike']:
            long_signals.append('vol')
        if row['pivot_low']:
            long_signals.append('pivot')

        if row['rsi_short']:
            short_signals.append('rsi')
        if row['ema_cross_short']:
            short_signals.append('ema')
        if row['vol_spike']:
            short_signals.append('vol')
        if row['pivot_high']:
            short_signals.append('pivot')

        if len(long_signals) >= 2:
            return 'long', long_signals
        if len(short_signals) >= 2:
            return 'short', short_signals
        return None

    def run_backtest(self, df: pd.DataFrame, verbose: bool = False) -> dict:
        df = self.prepare(df)
        equity = 10000.0
        trades = []

        for i in range(len(df)):
            row = df.iloc[i]
            sig = self.entry_signal(row)
            if sig is None:
                continue

            direction, signals = sig
            entry_price = row['close']
            tp = entry_price * (1 + self.cfg.take_profit) if direction == 'long' else entry_price * (1 - self.cfg.take_profit)
            sl = entry_price * (1 - self.cfg.stop_loss) if direction == 'long' else entry_price * (1 + self.cfg.stop_loss)

            # Position sizing using leverage-based approach
            # Calculate notional value based on available equity and leverage
            available_margin = equity * 0.95  # Use 95% of equity as available margin
            notional_value = available_margin * self.cfg.leverage
            
            # Calculate position size (quantity) based on notional value
            qty = notional_value / entry_price
            
            # Calculate actual leverage used (for tracking)
            margin_used = notional_value / self.cfg.leverage
            actual_leverage = notional_value / margin_used if margin_used > 0 else 0
            
            # Calculate risk amount for PnL calculation
            risk_amount = equity * self.cfg.risk_per_trade

            # simulate outcome over next max_holding_bars bars
            exit_price = None
            exit_reason = None
            for j in range(i + 1, min(i + 1 + self.cfg.max_holding_bars, len(df))):
                future = df.iloc[j]
                # check hit high/low
                if direction == 'long':
                    # TP
                    if future['high'] >= tp:
                        exit_price = tp
                        exit_reason = 'tp'
                        break
                    # SL
                    if future['low'] <= sl:
                        exit_price = sl
                        exit_reason = 'sl'
                        break
                else:
                    if future['low'] <= tp:
                        exit_price = tp
                        exit_reason = 'tp'
                        break
                    if future['high'] >= sl:
                        exit_price = sl
                        exit_reason = 'sl'
                        break

            # if neither hit, exit at close of last bar
            if exit_price is None:
                future = df.iloc[min(i + self.cfg.max_holding_bars, len(df) - 1)]
                exit_price = future['close']
                exit_reason = 'timeout'

            pnl = (exit_price - entry_price) * qty if direction == 'long' else (entry_price - exit_price) * qty
            pnl_pct = pnl / equity
            equity += pnl

            trades.append({
                'index': i,
                'time': row['timestamp'],
                'direction': direction,
                'signals': signals,
                'entry': entry_price,
                'exit': exit_price,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'equity': equity,
                'quantity': qty,
                'notional_value': notional_value,
                'margin_used': margin_used,
                'leverage_used': actual_leverage,
                'leverage_ratio': f"{actual_leverage:.1f}x",
            })

            if verbose:
                print(trades[-1])

        wins = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        win_rate = wins / total if total > 0 else 0.0

        return {
            'trades': trades,
            'total_trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'final_equity': equity,
        }
