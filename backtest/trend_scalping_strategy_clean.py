#!/usr/bin/env python3
"""
Clean Trend Scalping Strategy - Production Version
No debugging, no verbose output, optimized for live trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import ta
try:
    import cudf  # type: ignore
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

@dataclass
class TrendScalpingConfig:
    """Configuration for trend scalping strategy"""
    # EMAs
    ema_fast: int = 5
    ema_medium: int = 13
    ema_slow: int = 21
    ema_trend: int = 50
    
    # ADX
    adx_period: int = 14
    adx_threshold: float = 15.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volume
    volume_period: int = 20
    volume_spike_threshold: float = 1.1
    
    # ATR
    atr_period: int = 14
    stop_atr_multiplier: float = 1.5
    
    # Risk Management
    risk_per_trade: float = 0.01  # 1% per trade
    risk_reward_ratio: float = 2.0
    max_positions: int = 1
    leverage: float = 25.0
    
    # Account
    account_balance: float = 10000.0

class TrendScalpingAnalyzer:
    """Technical analysis for trend scalping"""
    
    def __init__(self, config: TrendScalpingConfig):
        self.config = config
        self.gpu = GPU_AVAILABLE
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators"""
        if self.gpu:
            try:
                return self._add_indicators_gpu(df)
            except Exception:
                # Fallback to CPU path
                self.gpu = False
        df = self._add_emas(df)
        df = self._add_adx(df)
        df = self._add_macd(df)
        df = self._add_volume_indicators(df)
        df = self._add_atr(df)
        df = self._add_trend_indicators(df)
        df = self._add_entry_signals(df)
        return df

    def _add_indicators_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated indicator computation using cuDF/cuPy with CPU fallback."""
        # Convert to cuDF
        gdf = cudf.DataFrame.from_pandas(df.reset_index(drop=True))
        
        # EMAs via cuPy-powered ewm
        for name, window in (
            ("ema_fast", self.config.ema_fast),
            ("ema_medium", self.config.ema_medium),
            ("ema_slow", self.config.ema_slow),
            ("ema_trend", self.config.ema_trend),
        ):
            gdf[name] = gdf["close"].ewm(span=window, adjust=False).mean()
        
        # ADX (approximate): compute TR/DM in GPU, then smoothed averages
        high = gdf["high"]
        low = gdf["low"]
        close = gdf["close"]
        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        tr = cp.maximum((high - low).values, cp.maximum(cp.abs(high - prev_close).values, cp.abs(low - prev_close).values))
        plus_dm = cp.maximum((high - prev_high).values, 0)
        minus_dm = cp.maximum((prev_low - low).values, 0)
        # Wilder smoothing via cumulative technique
        period = int(self.config.adx_period)
        def wilder_smooth(x: cp.ndarray, n: int) -> cp.ndarray:
            out = cp.empty_like(x)
            out[:n] = cp.nan
            if x.size >= n:
                out[n-1] = cp.nansum(x[:n])
                for i in range(n, x.size):
                    out[i] = out[i-1] - (out[i-1] / n) + x[i]
            return out
        trn = wilder_smooth(tr, period)
        plus_dmn = wilder_smooth(plus_dm, period)
        minus_dmn = wilder_smooth(minus_dm, period)
        plus_di = 100 * (plus_dmn / trn)
        minus_di = 100 * (minus_dmn / trn)
        dx = 100 * cp.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_vals = wilder_smooth(dx, period) / period
        gdf["adx"] = cudf.Series(cp.asnumpy(adx_vals))
        gdf["adx_pos"] = cudf.Series(cp.asnumpy(plus_di))
        gdf["adx_neg"] = cudf.Series(cp.asnumpy(minus_di))
        
        # MACD via cuDF rolling EMA
        macd_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        macd_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
        macd_line = macd_fast - macd_slow
        macd_signal = macd_line.ewm(span=self.config.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        gdf["macd"] = macd_line
        gdf["macd_signal"] = macd_signal
        gdf["macd_histogram"] = macd_hist
        
        # Volume indicators
        vol_sma = gdf["volume"].rolling(window=self.config.volume_period).mean()
        gdf["vol_sma"] = vol_sma
        gdf["vol_ratio"] = gdf["volume"] / gdf["vol_sma"]
        gdf["vol_spike"] = gdf["vol_ratio"] > self.config.volume_spike_threshold
        
        # ATR (WTR average of TR)
        atr_vals = wilder_smooth(tr, self.config.atr_period) / self.config.atr_period
        gdf["atr"] = cudf.Series(cp.asnumpy(atr_vals))
        
        # Trend and entry signals (vectorized on GPU where possible)
        gdf["trend_direction"] = "sideways"
        mask_up = (gdf["close"] > gdf["ema_trend"]) & (gdf["adx"] > self.config.adx_threshold)
        mask_down = (gdf["close"] < gdf["ema_trend"]) & (gdf["adx"] > self.config.adx_threshold)
        gdf.loc[mask_up, "trend_direction"] = "up"
        gdf.loc[mask_down, "trend_direction"] = "down"
        
        gdf["pullback_long"] = (
            (gdf["close"] > gdf["ema_fast"]) &
            (gdf["ema_fast"] > gdf["ema_medium"]) &
            (gdf["ema_medium"] > gdf["ema_slow"]) &
            (gdf["close"] < gdf["ema_slow"] * 1.02)
        )
        gdf["pullback_short"] = (
            (gdf["close"] < gdf["ema_fast"]) &
            (gdf["ema_fast"] < gdf["ema_medium"]) &
            (gdf["ema_medium"] < gdf["ema_slow"]) &
            (gdf["close"] > gdf["ema_slow"] * 0.98)
        )
        gdf["breakout_long"] = (
            (gdf["close"] > gdf["ema_fast"]) &
            (gdf["ema_fast"] > gdf["ema_medium"]) &
            (gdf["close"] > gdf["close"].shift(1))
        )
        gdf["breakout_short"] = (
            (gdf["close"] < gdf["ema_fast"]) &
            (gdf["ema_fast"] < gdf["ema_medium"]) &
            (gdf["close"] < gdf["close"].shift(1))
        )
        gdf["momentum_long"] = (
            (gdf["close"] > gdf["ema_trend"]) &
            (gdf["macd"] > gdf["macd_signal"]) &
            (gdf["macd_histogram"] > 0)
        )
        gdf["momentum_short"] = (
            (gdf["close"] < gdf["ema_trend"]) &
            (gdf["macd"] < gdf["macd_signal"]) &
            (gdf["macd_histogram"] < 0)
        )
        
        # Back to pandas
        out = gdf.to_pandas()
        return out
    
    def _add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA indicators"""
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_fast).ema_indicator()
        df['ema_medium'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_medium).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_slow).ema_indicator()
        df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_trend).ema_indicator()
        return df
    
    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX indicator"""
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.config.adx_period)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        macd_indicator = ta.trend.MACD(df['close'], window_slow=self.config.macd_slow, 
                                      window_fast=self.config.macd_fast, window_sign=self.config.macd_signal)
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        df['vol_sma'] = df['volume'].rolling(window=self.config.volume_period).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma']
        df['vol_spike'] = df['vol_ratio'] > self.config.volume_spike_threshold
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR indicator"""
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.config.atr_period).average_true_range()
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend direction indicators"""
        # Simplified trend detection
        df['trend_direction'] = 'sideways'
        df.loc[(df['close'] > df['ema_trend']) & (df['adx'] > self.config.adx_threshold), 'trend_direction'] = 'up'
        df.loc[(df['close'] < df['ema_trend']) & (df['adx'] > self.config.adx_threshold), 'trend_direction'] = 'down'
        return df
    
    def _add_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add entry signal indicators"""
        # Pullback signals
        df['pullback_long'] = (
            (df['close'] > df['ema_fast']) &
            (df['ema_fast'] > df['ema_medium']) &
            (df['ema_medium'] > df['ema_slow']) &
            (df['close'] < df['ema_slow'] * 1.02)  # Slight pullback
        )
        
        df['pullback_short'] = (
            (df['close'] < df['ema_fast']) &
            (df['ema_fast'] < df['ema_medium']) &
            (df['ema_medium'] < df['ema_slow']) &
            (df['close'] > df['ema_slow'] * 0.98)  # Slight pullback
        )
        
        # Breakout signals
        df['breakout_long'] = (
            (df['close'] > df['ema_fast']) &
            (df['ema_fast'] > df['ema_medium']) &
            (df['close'] > df['close'].shift(1))  # Upward momentum
        )
        
        df['breakout_short'] = (
            (df['close'] < df['ema_fast']) &
            (df['ema_fast'] < df['ema_medium']) &
            (df['close'] < df['close'].shift(1))  # Downward momentum
        )
        
        # Momentum signals
        df['momentum_long'] = (
            (df['close'] > df['ema_trend']) &
            (df['macd'] > df['macd_signal']) &
            (df['macd_histogram'] > 0)
        )
        
        df['momentum_short'] = (
            (df['close'] < df['ema_trend']) &
            (df['macd'] < df['macd_signal']) &
            (df['macd_histogram'] < 0)
        )
        
        return df

class TrendScalpingStrategy:
    """Clean trend scalping strategy"""
    
    def __init__(self, config: TrendScalpingConfig):
        self.config = config
        self.analyzer = TrendScalpingAnalyzer(config)
    
    def get_entry_signal(self, symbol: str, row: pd.Series) -> Optional[Dict[str, any]]:
        """Get trend scalping entry signal"""
        # Check if we have a valid trend
        if row['trend_direction'] == 'sideways':
            return None
        
        # Check for entry signals
        entry_signals = []
        signal_strength = 0
        
        # Pullback entries (highest priority)
        if row['pullback_long'] and row['trend_direction'] == 'up':
            entry_signals.append('pullback_long')
            signal_strength += 3
        elif row['pullback_short'] and row['trend_direction'] == 'down':
            entry_signals.append('pullback_short')
            signal_strength += 3
        
        # Breakout entries (medium priority)
        if row['breakout_long'] and row['trend_direction'] == 'up':
            entry_signals.append('breakout_long')
            signal_strength += 2
        elif row['breakout_short'] and row['trend_direction'] == 'down':
            entry_signals.append('breakout_short')
            signal_strength += 2
        
        # Momentum entries (lowest priority)
        if row['momentum_long'] and row['trend_direction'] == 'up':
            entry_signals.append('momentum_long')
            signal_strength += 1
        elif row['momentum_short'] and row['trend_direction'] == 'down':
            entry_signals.append('momentum_short')
            signal_strength += 1
        
        # Check volume confirmation
        if not row['vol_spike']:
            return None
        
        if not entry_signals:
            return None
        
        # Calculate dynamic targets based on leverage
        leverage = self.config.leverage
        target_range = self.calculate_dynamic_target(row['close'], leverage)
        
        # Determine entry type and direction
        entry_type = entry_signals[0]  # Use highest priority signal
        is_long = 'long' in entry_type
        
        # Calculate stop loss and take profit
        atr = row['atr']
        stop_distance = atr * self.config.stop_atr_multiplier
        
        if is_long:
            stop_price = row['close'] - stop_distance
            take_profit = row['close'] + (stop_distance * self.config.risk_reward_ratio)
        else:
            stop_price = row['close'] + stop_distance
            take_profit = row['close'] - (stop_distance * self.config.risk_reward_ratio)
        
        # Calculate position size (more conservative)
        risk_amount = self.config.account_balance * self.config.risk_per_trade
        position_size = risk_amount / stop_distance
        
        # Apply maximum position size limit (5% of account per trade)
        max_position_value = self.config.account_balance * 0.05
        max_position_size = max_position_value / row['close']
        position_size = min(position_size, max_position_size)
        
        return {
            'symbol': symbol,
            'entry_type': entry_type,
            'side': 'long' if is_long else 'short',
            'entry_price': row['close'],
            'stop_price': stop_price,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'signal_strength': signal_strength,
            'leverage': leverage,
            'target_range': target_range,
            'atr': atr,
            'adx': row['adx'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'macd_histogram': row['macd_histogram'],
            'vol_ratio': row['vol_ratio']
        }
    
    def calculate_dynamic_target(self, current_price: float, leverage: float) -> Dict[str, float]:
        """Calculate dynamic profit targets based on leverage (minimum 20% profit)"""
        if leverage >= 50:
            move_pct = 0.008  # 0.8% average move (20% profit at 25x leverage)
        elif leverage >= 25:
            move_pct = 0.008  # 0.8% average move (20% profit at 25x leverage)
        elif leverage >= 20:
            move_pct = 0.01   # 1% average move (20% profit at 20x leverage)
        elif leverage >= 12.5:
            move_pct = 0.016  # 1.6% average move (20% profit at 12.5x leverage)
        else:
            move_pct = 0.01   # 1% default (20% profit at 20x leverage)
        
        target_min = current_price * (1 + move_pct * 0.5)
        target_max = current_price * (1 + move_pct * 1.5)
        
        return {
            'min': target_min,
            'max': target_max,
            'move_pct': move_pct * 100
        }
    
    def analyze_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze data and add indicators"""
        return self.analyzer.add_indicators(df)
