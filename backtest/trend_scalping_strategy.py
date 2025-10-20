import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import ta
from datetime import datetime, timedelta
import requests
import json


@dataclass
class TrendScalpingConfig:
    """Configuration for trend scalping strategy"""
    # Trend detection
    adx_period: int = 14
    adx_threshold: float = 15.0  # Lowered from 25 to 15 for more opportunities
    ema_fast: int = 5
    ema_medium: int = 13
    ema_slow: int = 21
    ema_trend: int = 50
    ema_long: int = 200
    
    # Entry signal preferences
    pullback_weight: float = 0.4
    breakout_weight: float = 0.3
    momentum_weight: float = 0.3
    
    # Dynamic profit targets
    base_take_profit: float = 0.015  # 1.5% base target
    min_take_profit: float = 0.005   # 0.5% minimum
    max_take_profit: float = 0.05    # 5% maximum
    volatility_multiplier: float = 0.5  # How much volatility affects target
    trend_multiplier: float = 0.3    # How much trend strength affects target
    
    # ATR-based stops
    atr_period: int = 14
    atr_multiplier: float = 2.0      # 2x ATR for stop loss
    min_atr_multiplier: float = 1.5  # Minimum ATR multiplier
    max_atr_multiplier: float = 2.5  # Maximum ATR multiplier
    trailing_stop_trigger: float = 0.5  # Move to breakeven at 50% of target
    
    # Risk management
    risk_per_trade: float = 0.003    # 0.3% risk per trade
    leverage: float = 30.0           # Base leverage
    max_leverage: float = 50.0       # Maximum leverage
    min_leverage: float = 20.0       # Minimum leverage
    
    # Volatility filters - DISABLED for continuous trading
    min_volatility: float = 0.0      # No minimum volatility filter
    max_volatility: float = 1.0      # No maximum volatility filter
    
    # Trade management
    max_holding_bars: int = 10       # 10 minutes maximum hold
    min_holding_bars: int = 1        # 1 minute minimum hold
    
    # Volume confirmation
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.1  # Lowered from 1.5 to 1.1 for more opportunities


class TrendScalpingAnalyzer:
    """Technical analysis module for trend scalping"""
    
    def __init__(self, config: TrendScalpingConfig):
        self.config = config
    
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for trend scalping"""
        df = df.copy()
        
        # Multi-timeframe EMAs
        df['ema_5'] = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean()
        df['ema_13'] = df['close'].ewm(span=self.config.ema_medium, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=self.config.ema_trend, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=self.config.ema_long, adjust=False).mean()
        
        # ADX and Directional Movement
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.config.adx_period)
        df['adx'] = adx_indicator.adx()
        df['plus_di'] = adx_indicator.adx_pos()
        df['minus_di'] = adx_indicator.adx_neg()
        
        # MACD for momentum
        macd_indicator = ta.trend.MACD(df['close'])
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        
        # ATR for stop loss calculation
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.config.atr_period)
        df['atr'] = atr_indicator.average_true_range()
        
        # Volume analysis
        df['vol_ma'] = df['volume'].rolling(self.config.volume_ma_period, min_periods=1).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        df['vol_spike'] = df['vol_ratio'] > self.config.volume_spike_threshold
        
        # Price action analysis
        df = self._add_price_action_indicators(df)
        
        # Trend detection
        df = self._add_trend_indicators(df)
        
        # Entry signals
        df = self._add_entry_signals(df)
        
        return df
    
    def _add_price_action_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action indicators for trend detection"""
        # Higher highs and lower lows detection
        lookback = 10
        df['higher_high'] = df['high'] > df['high'].rolling(lookback).max().shift(1)
        df['lower_low'] = df['low'] < df['low'].rolling(lookback).min().shift(1)
        df['higher_low'] = df['low'] > df['low'].rolling(lookback).min().shift(1)
        df['lower_high'] = df['high'] < df['high'].rolling(lookback).max().shift(1)
        
        # Support and resistance levels
        pivot_lookback = 5
        df['pivot_high'] = df['high'] == df['high'].rolling(pivot_lookback, center=True).max()
        df['pivot_low'] = df['low'] == df['low'].rolling(pivot_lookback, center=True).min()
        
        # Price relative to EMAs
        df['price_above_ema21'] = df['close'] > df['ema_21']
        df['price_above_ema50'] = df['close'] > df['ema_50']
        df['price_below_ema21'] = df['close'] < df['ema_21']
        df['price_below_ema50'] = df['close'] < df['ema_50']
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend detection indicators"""
        # EMA alignment for trend direction
        df['ema_uptrend'] = (df['ema_5'] > df['ema_13']) & (df['ema_13'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
        df['ema_downtrend'] = (df['ema_5'] < df['ema_13']) & (df['ema_13'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
        
        # Strong trend (all EMAs aligned)
        df['strong_uptrend'] = df['ema_uptrend'] & (df['ema_50'] > df['ema_200'])
        df['strong_downtrend'] = df['ema_downtrend'] & (df['ema_50'] < df['ema_200'])
        
        # Trend strength based on ADX
        df['strong_trend'] = df['adx'] > self.config.adx_threshold
        df['very_strong_trend'] = df['adx'] > (self.config.adx_threshold * 1.5)
        
        # Price action trend confirmation
        df['price_action_uptrend'] = df['higher_high'] & df['higher_low']
        df['price_action_downtrend'] = df['lower_high'] & df['lower_low']
        
        # Overall trend classification - Very lenient
        df['trend_direction'] = 'sideways'
        # Uptrend: Just need price above EMA50 and some trend strength
        df.loc[(df['close'] > df['ema_50']) & (df['adx'] > 15), 'trend_direction'] = 'up'
        # Downtrend: Just need price below EMA50 and some trend strength  
        df.loc[(df['close'] < df['ema_50']) & (df['adx'] > 15), 'trend_direction'] = 'down'
        
        return df
    
    def _add_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add entry signal detection"""
        # Pullback entries
        df['pullback_long'] = (
            (df['trend_direction'] == 'up') &
            (df['close'] <= df['ema_21']) &
            (df['close'] > df['ema_50']) &
            (df['vol_spike']) &
            (df['macd_histogram'] > df['macd_histogram'].shift(1))  # MACD improving
        )
        
        df['pullback_short'] = (
            (df['trend_direction'] == 'down') &
            (df['close'] >= df['ema_21']) &
            (df['close'] < df['ema_50']) &
            (df['vol_spike']) &
            (df['macd_histogram'] < df['macd_histogram'].shift(1))  # MACD deteriorating
        )
        
        # Breakout entries
        df['breakout_long'] = (
            (df['trend_direction'] == 'up') &
            (df['close'] > df['high'].shift(1)) &  # New high
            (df['vol_spike']) &
            (df['macd'] > df['macd_signal']) &
            (df['plus_di'] > df['minus_di'])  # +DI > -DI
        )
        
        df['breakout_short'] = (
            (df['trend_direction'] == 'down') &
            (df['close'] < df['low'].shift(1)) &  # New low
            (df['vol_spike']) &
            (df['macd'] < df['macd_signal']) &
            (df['minus_di'] > df['plus_di'])  # -DI > +DI
        )
        
        # Momentum continuation entries - Very lenient
        df['momentum_long'] = (
            (df['trend_direction'] == 'up') &
            (df['close'] > df['ema_50'])  # Just need price above EMA50
        )
        
        df['momentum_short'] = (
            (df['trend_direction'] == 'down') &
            (df['close'] < df['ema_50'])  # Just need price below EMA50
        )
        
        return df
    
    def calculate_dynamic_target(self, row: pd.Series, leverage: float = None) -> float:
        """Calculate dynamic profit target based on leverage"""
        if leverage is None:
            leverage = self.config.leverage
        
        # Leverage-based profit targets
        if leverage >= 50:
            # 50x leverage: 1-2% target
            base_target = 0.015  # 1.5%
            min_target = 0.01    # 1%
            max_target = 0.02    # 2%
        elif leverage >= 25:
            # 25x leverage: 2-4% target
            base_target = 0.03   # 3%
            min_target = 0.02    # 2%
            max_target = 0.04    # 4%
        elif leverage >= 20:
            # 20x leverage: 2.5-5% target
            base_target = 0.0375 # 3.75%
            min_target = 0.025   # 2.5%
            max_target = 0.05    # 5%
        elif leverage >= 12.5:
            # 12.5x leverage: 6-12% target
            base_target = 0.09   # 9%
            min_target = 0.06    # 6%
            max_target = 0.12    # 12%
        else:
            # Lower leverage: conservative targets
            base_target = 0.02   # 2%
            min_target = 0.01    # 1%
            max_target = 0.05    # 5%
        
        # Trend strength adjustment (smaller impact)
        adx = row['adx']
        if adx > self.config.adx_threshold:
            trend_multiplier = 1 + ((adx - self.config.adx_threshold) / self.config.adx_threshold) * 0.1  # Reduced impact
        else:
            trend_multiplier = 0.9  # Slight reduction for weak trends
        
        # Calculate final target
        target = base_target * trend_multiplier
        
        # Clamp to leverage-specific range
        return max(min_target, min(max_target, target))
    
    def calculate_atr_stop(self, entry_price: float, atr: float, direction: str) -> float:
        """Calculate ATR-based stop loss"""
        # Dynamic ATR multiplier based on market conditions
        atr_multiplier = self.config.atr_multiplier
        
        # Adjust multiplier based on ATR value (wider stops in volatile markets)
        if atr > entry_price * 0.02:  # Very volatile
            atr_multiplier = min(self.config.max_atr_multiplier, atr_multiplier * 1.2)
        elif atr < entry_price * 0.005:  # Low volatility
            atr_multiplier = max(self.config.min_atr_multiplier, atr_multiplier * 0.8)
        
        if direction == 'long':
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)


class TrendScalpingStrategy:
    """Main trend scalping strategy class"""
    
    def __init__(self, config: TrendScalpingConfig, pair_discovery=None):
        self.config = config
        self.analyzer = TrendScalpingAnalyzer(config)
        self.pair_discovery = pair_discovery
    
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all trend scalping indicators"""
        return self.analyzer.prepare_indicators(df)
    
    def get_entry_signal(self, symbol: str, row: pd.Series, verbose: bool = False) -> Optional[Dict[str, any]]:
        """Get trend scalping entry signal"""
        if verbose:
            print(f"\n🔍 EVALUATING {symbol} at price ${row['close']:.2f}")
        
        # Check if we have a valid trend
        if row['trend_direction'] == 'sideways':
            if verbose:
                print(f"   ❌ Trend: SIDEWAYS (ADX: {row['adx']:.1f})")
            return None
        else:
            if verbose:
                print(f"   ✅ Trend: {row['trend_direction'].upper()} (ADX: {row['adx']:.1f})")
        
        # No volatility filtering - trade any pair
        volatility = None
        
        # Check for entry signals
        entry_signals = []
        signal_strength = 0
        
        if verbose:
            print(f"   📊 Signal Analysis:")
            print(f"      Pullback Long: {row['pullback_long']} | Short: {row['pullback_short']}")
            print(f"      Breakout Long: {row['breakout_long']} | Short: {row['breakout_short']}")
            print(f"      Momentum Long: {row['momentum_long']} | Short: {row['momentum_short']}")
            print(f"      Volume Spike: {row['vol_spike']} (Ratio: {row['vol_ratio']:.2f})")
            print(f"      MACD: {row['macd']:.4f} | Signal: {row['macd_signal']:.4f} | Hist: {row['macd_histogram']:.4f}")
        
        # Pullback entries (highest priority)
        if row['pullback_long'] and row['trend_direction'] == 'up':
            entry_signals.append('pullback_long')
            signal_strength += 3
            if verbose:
                print(f"      ✅ PULLBACK LONG signal detected!")
        elif row['pullback_short'] and row['trend_direction'] == 'down':
            entry_signals.append('pullback_short')
            signal_strength += 3
            if verbose:
                print(f"      ✅ PULLBACK SHORT signal detected!")
        
        # Breakout entries (medium priority)
        if row['breakout_long'] and row['trend_direction'] == 'up':
            entry_signals.append('breakout_long')
            signal_strength += 2
            if verbose:
                print(f"      ✅ BREAKOUT LONG signal detected!")
        elif row['breakout_short'] and row['trend_direction'] == 'down':
            entry_signals.append('breakout_short')
            signal_strength += 2
            if verbose:
                print(f"      ✅ BREAKOUT SHORT signal detected!")
        
        # Momentum entries (lower priority)
        if row['momentum_long'] and row['trend_direction'] == 'up':
            entry_signals.append('momentum_long')
            signal_strength += 1
            if verbose:
                print(f"      ✅ MOMENTUM LONG signal detected!")
        elif row['momentum_short'] and row['trend_direction'] == 'down':
            entry_signals.append('momentum_short')
            signal_strength += 1
            if verbose:
                print(f"      ✅ MOMENTUM SHORT signal detected!")
        
        if not entry_signals:
            if verbose:
                print(f"   ❌ No entry signals found")
            return None
        
        # Determine direction
        direction = 'long' if any('long' in signal for signal in entry_signals) else 'short'
        
        # Calculate leverage based on trend strength
        leverage = self._calculate_dynamic_leverage(row)
        
        # Calculate dynamic target based on leverage
        target_pct = self.analyzer.calculate_dynamic_target(row, leverage)
        stop_price = self.analyzer.calculate_atr_stop(row['close'], row['atr'], direction)
        
        if verbose:
            print(f"   🎯 Signal Details:")
            print(f"      Direction: {direction.upper()}")
            print(f"      Signal Strength: {signal_strength}")
            print(f"      Leverage: {leverage:.1f}x")
            print(f"      Target: {target_pct:.1%}")
            print(f"      Stop: ${stop_price:.2f}")
            print(f"      ATR: {row['atr']:.4f}")
        
        return {
            'direction': direction,
            'signal_strength': signal_strength,
            'entry_signals': entry_signals,
            'trend_direction': row['trend_direction'],
            'adx': row['adx'],
            'atr': row['atr'],
            'volatility': volatility,
            'target_pct': target_pct,
            'stop_price': stop_price,
            'leverage': leverage,
            'entry_price': row['close'],
            'timestamp': row.name if hasattr(row, 'name') else datetime.now()
        }
    
    def _calculate_dynamic_leverage(self, row: pd.Series) -> float:
        """Calculate dynamic leverage based on trend strength and volatility"""
        base_leverage = self.config.leverage
        
        # Adjust based on trend strength
        if row['very_strong_trend']:
            leverage_multiplier = 1.2
        elif row['strong_trend']:
            leverage_multiplier = 1.0
        else:
            leverage_multiplier = 0.8
        
        # Adjust based on ADX
        adx_factor = min(1.5, max(0.7, row['adx'] / self.config.adx_threshold))
        
        # Calculate final leverage
        leverage = base_leverage * leverage_multiplier * adx_factor
        
        return max(self.config.min_leverage, min(self.config.max_leverage, leverage))
    
    def calculate_position_size(self, signal: Dict[str, any], account_balance: float) -> Dict[str, float]:
        """Calculate position size for trend scalping"""
        entry_price = signal['entry_price']
        stop_price = signal['stop_price']
        leverage = signal['leverage']
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_price)
        stop_pct = stop_distance / entry_price
        
        # Risk amount
        risk_amount = account_balance * self.config.risk_per_trade
        
        # Position size calculation
        position_size = risk_amount / stop_distance
        
        # Apply leverage
        available_margin = account_balance * 0.95
        notional_value = position_size * entry_price
        required_margin = notional_value / leverage
        
        # Adjust position size if needed
        if required_margin > available_margin:
            position_size = (available_margin * leverage) / entry_price
            notional_value = position_size * entry_price
            required_margin = notional_value / leverage
        
        # Calculate take profit price
        target_pct = signal['target_pct']
        if signal['direction'] == 'long':
            take_profit_price = entry_price * (1 + target_pct)
        else:
            take_profit_price = entry_price * (1 - target_pct)
        
        return {
            'position_size': position_size,
            'notional_value': notional_value,
            'margin_required': required_margin,
            'leverage_used': leverage,
            'stop_loss_price': stop_price,
            'take_profit_price': take_profit_price,
            'target_pct': target_pct,
            'stop_pct': stop_pct,
            'risk_reward_ratio': target_pct / stop_pct if stop_pct > 0 else 0
        }
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = False) -> Dict[str, any]:
        """Run backtest with trend scalping strategy"""
        df = self.prepare(df)
        equity = 10000.0
        trades = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            signal = self.get_entry_signal('BTCUSDT', row)  # Default symbol for backtest
            
            if signal is None:
                continue
            
            # Calculate position size
            position_info = self.calculate_position_size(signal, equity)
            
            # Simulate trade execution
            entry_price = signal['entry_price']
            direction = signal['direction']
            tp_price = position_info['take_profit_price']
            sl_price = position_info['stop_loss_price']
            
            # Simulate outcome over holding period
            exit_price = None
            exit_reason = None
            hold_time = 0
            
            for j in range(i + 1, min(i + 1 + self.config.max_holding_bars, len(df))):
                future = df.iloc[j]
                hold_time += 1
                
                # Check for trend reversal (early exit)
                if future['trend_direction'] != signal['trend_direction']:
                    exit_price = future['close']
                    exit_reason = 'trend_reversal'
                    break
                
                # Check TP/SL
                if direction == 'long':
                    if future['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                        break
                    if future['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                        break
                else:
                    if future['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'take_profit'
                        break
                    if future['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'stop_loss'
                        break
                
                # Check minimum hold time
                if hold_time >= self.config.min_holding_bars:
                    # Check for trailing stop (move to breakeven)
                    if exit_reason is None:
                        profit_pct = (future['close'] - entry_price) / entry_price if direction == 'long' else (entry_price - future['close']) / entry_price
                        target_pct = signal['target_pct']
                        
                        if profit_pct >= target_pct * self.config.trailing_stop_trigger:
                            # Move stop to breakeven
                            sl_price = entry_price
                            if direction == 'long' and future['low'] <= sl_price:
                                exit_price = sl_price
                                exit_reason = 'trailing_stop'
                                break
                            elif direction == 'short' and future['high'] >= sl_price:
                                exit_price = sl_price
                                exit_reason = 'trailing_stop'
                                break
            
            # Timeout exit
            if exit_price is None:
                future = df.iloc[min(i + self.config.max_holding_bars, len(df) - 1)]
                exit_price = future['close']
                exit_reason = 'timeout'
            
            # Calculate PnL
            if direction == 'long':
                pnl = (exit_price - entry_price) * position_info['position_size']
            else:
                pnl = (entry_price - exit_price) * position_info['position_size']
            
            pnl_pct = pnl / equity
            equity += pnl
            
            trade = {
                'index': i,
                'time': row['timestamp'] if 'timestamp' in row else i,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'hold_time': hold_time,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'equity': equity,
                'position_size': position_info['position_size'],
                'notional_value': position_info['notional_value'],
                'leverage_used': position_info['leverage_used'],
                'signal_strength': signal['signal_strength'],
                'entry_signals': signal['entry_signals'],
                'trend_direction': signal['trend_direction'],
                'adx': signal['adx'],
                'target_pct': signal['target_pct'],
                'stop_pct': position_info['stop_pct'],
                'risk_reward_ratio': position_info['risk_reward_ratio']
            }
            
            trades.append(trade)
            
            if verbose:
                print(f"Trade {len(trades)}: {direction} @ {entry_price:.2f} -> {exit_price:.2f} | PnL: {pnl:.2f} | Hold: {hold_time}min | Signals: {signal['entry_signals']}")
        
        # Calculate performance metrics
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total = len(trades)
        win_rate = wins / total if total > 0 else 0.0
        
        # Calculate additional metrics
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if (total - wins) > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * (total - wins))) if avg_loss != 0 else float('inf')
        
        # Scalping-specific metrics
        avg_hold_time = np.mean([t['hold_time'] for t in trades]) if trades else 0
        max_consecutive_losses = self._calculate_max_consecutive_losses(trades)
        avg_risk_reward = np.mean([t['risk_reward_ratio'] for t in trades]) if trades else 0
        
        return {
            'trades': trades,
            'total_trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'final_equity': equity,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self._calculate_max_drawdown([t['equity'] for t in trades]),
            'sharpe_ratio': self._calculate_sharpe_ratio([t['pnl_pct'] for t in trades]),
            'avg_hold_time': avg_hold_time,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_risk_reward': avg_risk_reward
        }
    
    def _calculate_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return * np.sqrt(252)  # Annualized
