#!/usr/bin/env python3
"""
Aggressive Trading Strategy - High Frequency, High Risk, High Reward
Multiple strategies for maximum profit potential
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import ta
from datetime import datetime, timedelta

@dataclass
class AggressiveConfig:
    """Configuration for aggressive trading strategies"""
    
    # === CORE SETTINGS ===
    account_balance: float = 10000.0
    max_positions: int = 5  # Increased from 1 to 5
    leverage: float = 50.0  # Increased from 25 to 50
    
    # === RISK MANAGEMENT (More Aggressive) ===
    risk_per_trade: float = 0.03  # 3% per trade (increased from 1%)
    max_total_risk: float = 0.20  # 20% total account risk
    risk_reward_ratio: float = 1.5  # Lower RR for more frequent wins
    
    # === MULTI-TIMEFRAME ANALYSIS ===
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])  # Multiple timeframes
    primary_timeframe: str = "1m"
    
    # === TECHNICAL INDICATORS (Faster) ===
    # EMAs - Faster for more signals
    ema_ultra_fast: int = 3
    ema_fast: int = 8
    ema_medium: int = 21
    ema_slow: int = 55
    
    # MACD - Faster settings
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5
    
    # RSI - More sensitive
    rsi_period: int = 9
    rsi_oversold: float = 25
    rsi_overbought: float = 75
    
    # ADX - Lower threshold for more signals
    adx_period: int = 10
    adx_threshold: float = 10.0
    
    # Volume - More sensitive
    volume_period: int = 10
    volume_spike_threshold: float = 0.3  # Lower threshold
    
    # ATR - Tighter stops
    atr_period: int = 10
    stop_atr_multiplier: float = 1.0  # Tighter stops
    
    # === MOMENTUM SETTINGS ===
    momentum_period: int = 5
    momentum_threshold: float = 0.5
    
    # === BREAKOUT SETTINGS ===
    breakout_lookback: int = 20
    breakout_threshold: float = 0.02  # 2% breakout
    
    # === SCALPING SETTINGS ===
    min_profit_target: float = 0.005  # 0.5% minimum profit
    max_hold_time: int = 300  # 5 minutes max hold
    quick_exit_threshold: float = 0.002  # 0.2% quick exit
    
    # === VOLATILITY FILTERS ===
    min_volatility: float = 0.01  # 1% minimum daily volatility
    max_volatility: float = 0.10  # 10% maximum daily volatility
    
    # === NEWS/MOMENTUM FILTERS ===
    enable_news_filter: bool = True
    enable_social_sentiment: bool = True
    enable_volume_breakout: bool = True
    
    # === ADVANCED FEATURES ===
    enable_grid_trading: bool = True
    enable_martingale: bool = False  # Dangerous but can be enabled
    enable_dca: bool = True  # Dollar Cost Averaging
    enable_pyramiding: bool = True  # Add to winning positions
    
    # === TIMING SETTINGS ===
    trading_hours_start: int = 0  # 24h format
    trading_hours_end: int = 23
    avoid_news_times: bool = True
    news_avoidance_minutes: int = 30

class AggressiveStrategy:
    """Aggressive multi-strategy trading system"""
    
    def __init__(self, config: AggressiveConfig):
        self.config = config
        self.active_positions = {}
        self.position_history = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_daily_trades = 100
        
    def analyze_multi_timeframe(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze multiple timeframes for confluence"""
        signals = {}
        
        for timeframe, df in data_dict.items():
            if len(df) < 50:  # Need minimum data
                continue
                
            # Add indicators for this timeframe
            df = self._add_indicators(df, timeframe)
            
            # Get signals for this timeframe
            signals[timeframe] = self._get_timeframe_signals(symbol, df, timeframe)
        
        # Combine signals from all timeframes
        return self._combine_timeframe_signals(signals)
    
    def _add_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add technical indicators optimized for timeframe"""
        # EMAs
        df['ema_ultra_fast'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_ultra_fast).ema_indicator()
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_fast).ema_indicator()
        df['ema_medium'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_medium).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=self.config.ema_slow).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=self.config.macd_slow, 
                            window_fast=self.config.macd_fast, window_sign=self.config.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.config.rsi_period).rsi()
        
        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.config.adx_period).adx()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=self.config.volume_period).volume_sma()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['vol_spike'] = df['volume_ratio'] > (1 + self.config.volume_spike_threshold)
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.config.atr_period).average_true_range()
        
        # Momentum
        df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=self.config.momentum_period).roc()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 
                                                window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], window=14).williams_r()
        
        # CCI
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        return df
    
    def _get_timeframe_signals(self, symbol: str, df: pd.DataFrame, timeframe: str) -> Dict:
        """Get trading signals for specific timeframe"""
        if len(df) < 2:
            return {'signals': [], 'strength': 0}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        strength = 0
        
        # === TREND SIGNALS ===
        # Strong uptrend
        if (latest['ema_ultra_fast'] > latest['ema_fast'] > latest['ema_medium'] > latest['ema_slow'] and
            latest['close'] > latest['ema_ultra_fast']):
            signals.append('strong_uptrend')
            strength += 4
            
        # Strong downtrend
        elif (latest['ema_ultra_fast'] < latest['ema_fast'] < latest['ema_medium'] < latest['ema_slow'] and
              latest['close'] < latest['ema_ultra_fast']):
            signals.append('strong_downtrend')
            strength += 4
        
        # === MOMENTUM SIGNALS ===
        # RSI oversold bounce
        if latest['rsi'] < self.config.rsi_oversold and latest['rsi'] > prev['rsi']:
            signals.append('rsi_oversold_bounce')
            strength += 3
            
        # RSI overbought rejection
        elif latest['rsi'] > self.config.rsi_overbought and latest['rsi'] < prev['rsi']:
            signals.append('rsi_overbought_rejection')
            strength += 3
        
        # === MACD SIGNALS ===
        # MACD bullish crossover
        if (latest['macd'] > latest['macd_signal'] and 
            prev['macd'] <= prev['macd_signal'] and 
            latest['macd_histogram'] > 0):
            signals.append('macd_bullish_cross')
            strength += 2
            
        # MACD bearish crossover
        elif (latest['macd'] < latest['macd_signal'] and 
              prev['macd'] >= prev['macd_signal'] and 
              latest['macd_histogram'] < 0):
            signals.append('macd_bearish_cross')
            strength += 2
        
        # === BREAKOUT SIGNALS ===
        # Volume breakout
        if latest['vol_spike'] and latest['close'] > latest['bb_upper']:
            signals.append('volume_breakout_up')
            strength += 3
            
        elif latest['vol_spike'] and latest['close'] < latest['bb_lower']:
            signals.append('volume_breakout_down')
            strength += 3
        
        # === REVERSAL SIGNALS ===
        # Stochastic oversold
        if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
            signals.append('stoch_oversold')
            strength += 2
            
        # Stochastic overbought
        elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
            signals.append('stoch_overbought')
            strength += 2
        
        # === VOLATILITY SIGNALS ===
        # High volatility expansion
        if latest['bb_width'] > latest['bb_width'].rolling(20).mean().iloc[-1] * 1.5:
            signals.append('volatility_expansion')
            strength += 1
        
        return {
            'signals': signals,
            'strength': strength,
            'timeframe': timeframe,
            'price': latest['close'],
            'volume': latest['volume'],
            'volatility': latest['bb_width']
        }
    
    def _combine_timeframe_signals(self, signals: Dict) -> Dict:
        """Combine signals from multiple timeframes"""
        if not signals:
            return {'action': 'hold', 'strength': 0}
        
        # Calculate weighted strength based on timeframe priority
        timeframe_weights = {
            '1m': 1.0,
            '5m': 0.8,
            '15m': 0.6,
            '1h': 0.4
        }
        
        total_strength = 0
        all_signals = []
        bullish_signals = 0
        bearish_signals = 0
        
        for timeframe, data in signals.items():
            weight = timeframe_weights.get(timeframe, 0.5)
            weighted_strength = data['strength'] * weight
            total_strength += weighted_strength
            all_signals.extend(data['signals'])
            
            # Count bullish vs bearish signals
            for signal in data['signals']:
                if any(word in signal.lower() for word in ['up', 'bull', 'long', 'oversold']):
                    bullish_signals += 1
                elif any(word in signal.lower() for word in ['down', 'bear', 'short', 'overbought']):
                    bearish_signals += 1
        
        # Determine action based on signal confluence
        if total_strength >= 8 and bullish_signals > bearish_signals:
            action = 'long'
        elif total_strength >= 8 and bearish_signals > bullish_signals:
            action = 'short'
        elif total_strength >= 5:
            action = 'strong_hold'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'strength': total_strength,
            'signals': all_signals,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'timeframe_data': signals
        }
    
    def get_aggressive_entry_signal(self, symbol: str, analysis: Dict) -> Optional[Dict]:
        """Get aggressive entry signal with multiple strategies"""
        if analysis['action'] in ['hold', 'strong_hold']:
            return None
        
        # Check if we can open new position
        if len(self.active_positions) >= self.config.max_positions:
            return None
        
        # Check daily trade limit
        if self.trades_today >= self.config.max_daily_trades:
            return None
        
        # Get current price (would be passed from main system)
        current_price = analysis['timeframe_data'].get('1m', {}).get('price', 0)
        if current_price <= 0:
            return None
        
        # Calculate position size based on multiple factors
        position_size = self._calculate_aggressive_position_size(symbol, current_price, analysis)
        
        # Calculate dynamic stop loss and take profit
        stop_loss, take_profit = self._calculate_dynamic_levels(symbol, current_price, analysis)
        
        # Determine signal type
        signal_type = 'aggressive_' + analysis['action']
        
        return {
            'symbol': symbol,
            'side': analysis['action'],
            'entry_price': current_price,
            'stop_price': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'signal_type': signal_type,
            'strength': analysis['strength'],
            'signals': analysis['signals'],
            'risk_amount': position_size * abs(current_price - stop_loss),
            'leverage': self.config.leverage,
            'strategy': 'aggressive_multi_timeframe'
        }
    
    def _calculate_aggressive_position_size(self, symbol: str, price: float, analysis: Dict) -> float:
        """Calculate aggressive position size based on signal strength and volatility"""
        # Base risk amount
        base_risk = self.config.account_balance * self.config.risk_per_trade
        
        # Adjust based on signal strength
        strength_multiplier = min(2.0, analysis['strength'] / 5.0)  # Up to 2x for strong signals
        
        # Adjust based on volatility (higher volatility = smaller position)
        volatility = analysis['timeframe_data'].get('1m', {}).get('volatility', 0.02)
        volatility_multiplier = max(0.5, 1.0 - (volatility - 0.02) * 10)  # Reduce size for high volatility
        
        # Final risk amount
        adjusted_risk = base_risk * strength_multiplier * volatility_multiplier
        
        # Calculate position size
        stop_distance = abs(price * 0.01)  # 1% stop for aggressive trading
        position_size = adjusted_risk / stop_distance
        
        # Apply maximum position size (10% of account)
        max_position_value = self.config.account_balance * 0.10
        max_position_size = max_position_value / price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def _calculate_dynamic_levels(self, symbol: str, price: float, analysis: Dict) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        # Base stop distance (1% for aggressive trading)
        base_stop_distance = price * 0.01
        
        # Adjust based on volatility
        volatility = analysis['timeframe_data'].get('1m', {}).get('volatility', 0.02)
        volatility_multiplier = max(0.5, min(2.0, volatility * 50))  # 0.5x to 2x based on volatility
        
        stop_distance = base_stop_distance * volatility_multiplier
        
        # Calculate stop loss
        if analysis['action'] == 'long':
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * self.config.risk_reward_ratio)
        else:
            stop_loss = price + stop_distance
            take_profit = price - (stop_distance * self.config.risk_reward_ratio)
        
        return stop_loss, take_profit
    
    def should_exit_position(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """Check if position should be exited aggressively"""
        if symbol not in self.active_positions:
            return {'action': 'hold', 'reason': 'position_not_found'}
        
        # Time-based exit
        if hasattr(position, 'entry_time'):
            hold_time = datetime.now() - position['entry_time']
            if hold_time.total_seconds() > self.config.max_hold_time:
                return {'action': 'exit', 'reason': 'max_hold_time_reached'}
        
        # Quick profit exit
        if position['side'] == 'long':
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            profit_pct = (position['entry_price'] - current_price) / position['entry_price']
        
        if profit_pct >= self.config.quick_exit_threshold:
            return {'action': 'exit', 'reason': 'quick_profit_target'}
        
        # Stop loss hit
        if position['side'] == 'long' and current_price <= position['stop_price']:
            return {'action': 'exit', 'reason': 'stop_loss_hit'}
        elif position['side'] == 'short' and current_price >= position['stop_price']:
            return {'action': 'exit', 'reason': 'stop_loss_hit'}
        
        # Take profit hit
        if position['side'] == 'long' and current_price >= position['take_profit']:
            return {'action': 'exit', 'reason': 'take_profit_hit'}
        elif position['side'] == 'short' and current_price <= position['take_profit']:
            return {'action': 'exit', 'reason': 'take_profit_hit'}
        
        return {'action': 'hold', 'reason': 'no_exit_conditions'}

# Example usage and configuration
def create_aggressive_config() -> AggressiveConfig:
    """Create aggressive trading configuration"""
    return AggressiveConfig(
        account_balance=10000.0,
        max_positions=5,
        leverage=50.0,
        risk_per_trade=0.03,  # 3% per trade
        max_total_risk=0.20,  # 20% total risk
        risk_reward_ratio=1.5,
        timeframes=["1m", "5m", "15m"],
        ema_ultra_fast=3,
        ema_fast=8,
        ema_medium=21,
        ema_slow=55,
        rsi_period=9,
        rsi_oversold=25,
        rsi_overbought=75,
        adx_threshold=10.0,
        volume_spike_threshold=0.3,
        stop_atr_multiplier=1.0,
        min_profit_target=0.005,
        max_hold_time=300,
        quick_exit_threshold=0.002,
        enable_grid_trading=True,
        enable_dca=True,
        enable_pyramiding=True
    )

if __name__ == "__main__":
    # Example usage
    config = create_aggressive_config()
    strategy = AggressiveStrategy(config)
    print("🚀 Aggressive Trading Strategy Initialized!")
    print(f"Max Positions: {config.max_positions}")
    print(f"Leverage: {config.leverage}x")
    print(f"Risk per Trade: {config.risk_per_trade*100}%")
    print(f"Max Total Risk: {config.max_total_risk*100}%")
