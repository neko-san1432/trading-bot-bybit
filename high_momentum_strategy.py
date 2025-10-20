#!/usr/bin/env python3
"""
High Momentum Strategy - Trade 50-100% gainers with high volatility
Specialized for small balances and high-risk, high-reward trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import ta
from datetime import datetime, timedelta

@dataclass
class HighMomentumConfig:
    """Configuration for high momentum trading"""
    
    # Core settings
    account_balance: float = 1.7  # Start with $1.7
    target_balance: float = 200.0  # Target $200
    max_positions: int = 3  # Fewer positions for small balance
    leverage: float = 100.0  # Maximum leverage
    
    # Risk management for small balance
    risk_per_trade: float = 0.15  # 15% per trade (aggressive for small balance)
    max_total_risk: float = 0.50  # 50% total risk
    risk_reward_ratio: float = 3.0  # Higher RR for momentum
    
    # Momentum filters
    min_24h_gain: float = 0.50  # 50% minimum 24h gain
    max_24h_gain: float = 2.00  # 200% maximum 24h gain
    min_volume_spike: float = 2.0  # 2x volume spike
    min_volatility: float = 0.05  # 5% minimum volatility
    
    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m"])
    primary_timeframe: str = "1m"
    
    # Technical indicators (fast for momentum)
    ema_ultra_fast: int = 2
    ema_fast: int = 5
    ema_medium: int = 13
    ema_slow: int = 21
    
    # MACD
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 3
    
    # RSI
    rsi_period: int = 7
    rsi_oversold: float = 20
    rsi_overbought: float = 80
    
    # ADX
    adx_period: int = 10
    adx_threshold: float = 15.0
    
    # Volume
    volume_period: int = 10
    volume_spike_threshold: float = 1.5
    
    # ATR
    atr_period: int = 10
    stop_atr_multiplier: float = 0.8  # Tighter stops for momentum
    
    # Momentum settings
    momentum_period: int = 5
    momentum_threshold: float = 1.0
    
    # Exit settings
    min_profit_target: float = 0.02  # 2% minimum profit
    max_hold_time: int = 600  # 10 minutes max
    quick_exit_threshold: float = 0.01  # 1% quick exit
    
    # Position sizing for small balance
    min_position_value: float = 5.0  # $5 minimum position
    max_position_value: float = 50.0  # $50 maximum position
    position_size_multiplier: float = 1.5  # Multiply by signal strength

class HighMomentumStrategy:
    """High momentum trading strategy for volatile gainers"""
    
    def __init__(self, config: HighMomentumConfig):
        self.config = config
        self.active_positions = {}
        self.position_history = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_daily_trades = 50
        
    def scan_momentum_coins(self, all_symbols: List[str]) -> List[Dict]:
        """Scan for high momentum coins with 50-100% gains"""
        momentum_coins = []
        
        for symbol in all_symbols:
            try:
                # Get 24h ticker data
                ticker_data = self._get_24h_ticker(symbol)
                if not ticker_data:
                    continue
                
                # Check 24h gain
                price_change_24h = float(ticker_data.get('price24hPcnt', 0))
                volume_24h = float(ticker_data.get('volume24h', 0))
                
                # Filter for high momentum
                if (self.config.min_24h_gain <= price_change_24h <= self.config.max_24h_gain and
                    volume_24h > 1000000):  # Minimum volume
                    
                    momentum_coins.append({
                        'symbol': symbol,
                        '24h_gain': price_change_24h,
                        'volume_24h': volume_24h,
                        'current_price': float(ticker_data.get('lastPrice', 0)),
                        'high_24h': float(ticker_data.get('high24h', 0)),
                        'low_24h': float(ticker_data.get('low24h', 0))
                    })
                    
            except Exception as e:
                continue
        
        # Sort by 24h gain (highest first)
        momentum_coins.sort(key=lambda x: x['24h_gain'], reverse=True)
        
        return momentum_coins[:10]  # Top 10 momentum coins
    
    def _get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker data for symbol"""
        # This would be implemented with actual API call
        # For now, return mock data
        return {
            'price24hPcnt': np.random.uniform(0.3, 1.5),  # 30-150% gain
            'volume24h': np.random.uniform(1000000, 10000000),
            'lastPrice': np.random.uniform(0.1, 100),
            'high24h': np.random.uniform(0.1, 100),
            'low24h': np.random.uniform(0.1, 100)
        }
    
    def analyze_momentum_coin(self, symbol: str, coin_data: Dict) -> Optional[Dict]:
        """Analyze high momentum coin for entry"""
        try:
            # Get multi-timeframe data
            timeframes = ['1m', '5m', '15m']
            data_dict = {}
            
            for tf in timeframes:
                klines = self._get_klines(symbol, tf, 100)
                if klines:
                    df = pd.DataFrame(klines)
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
                    data_dict[tf] = self._add_indicators(df, tf)
            
            if not data_dict:
                return None
            
            # Analyze momentum
            analysis = self._analyze_momentum(symbol, data_dict, coin_data)
            
            if analysis['action'] in ['long', 'short']:
                # Calculate position size for small balance
                position_size = self._calculate_momentum_position_size(symbol, analysis)
                
                if position_size > 0:
                    return {
                        'symbol': symbol,
                        'side': analysis['action'],
                        'entry_price': analysis['entry_price'],
                        'stop_price': analysis['stop_price'],
                        'take_profit': analysis['take_profit'],
                        'position_size': position_size,
                        'signal_strength': analysis['strength'],
                        '24h_gain': coin_data['24h_gain'],
                        'risk_amount': position_size * abs(analysis['entry_price'] - analysis['stop_price']),
                        'leverage': self.config.leverage,
                        'strategy': 'high_momentum'
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _add_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add technical indicators for momentum analysis"""
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
        
        # Volume
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
        df['bb_lower'] = bb.bollinger_lower()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _analyze_momentum(self, symbol: str, data_dict: Dict, coin_data: Dict) -> Dict:
        """Analyze momentum for entry signal"""
        # Get 1m data for primary analysis
        df_1m = data_dict.get('1m')
        if df_1m is None or len(df_1m) < 2:
            return {'action': 'hold', 'strength': 0}
        
        latest = df_1m.iloc[-1]
        prev = df_1m.iloc[-2]
        
        signals = []
        strength = 0
        
        # Strong momentum signals
        if latest['close'] > latest['ema_ultra_fast'] > latest['ema_fast'] > latest['ema_medium']:
            signals.append('strong_uptrend')
            strength += 5
        
        # MACD momentum
        if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
            signals.append('macd_momentum')
            strength += 3
        
        # RSI momentum (not overbought)
        if 30 < latest['rsi'] < 70 and latest['rsi'] > prev['rsi']:
            signals.append('rsi_momentum')
            strength += 2
        
        # Volume confirmation
        if latest['vol_spike']:
            signals.append('volume_breakout')
            strength += 3
        
        # ADX trend strength
        if latest['adx'] > self.config.adx_threshold:
            signals.append('trend_strength')
            strength += 2
        
        # Price near 24h high (momentum continuation)
        if coin_data['current_price'] > coin_data['high_24h'] * 0.95:
            signals.append('near_24h_high')
            strength += 4
        
        # Determine action
        if strength >= 8 and len(signals) >= 3:
            return {
                'action': 'long',
                'strength': strength,
                'signals': signals,
                'entry_price': latest['close'],
                'stop_price': latest['close'] * 0.98,  # 2% stop
                'take_profit': latest['close'] * 1.06,  # 6% target
                'confidence': min(1.0, strength / 10.0)
            }
        else:
            return {'action': 'hold', 'strength': strength, 'signals': signals}
    
    def _calculate_momentum_position_size(self, symbol: str, analysis: Dict) -> float:
        """Calculate position size for momentum trading with small balance"""
        try:
            entry_price = analysis['entry_price']
            stop_price = analysis['stop_price']
            signal_strength = analysis['strength']
            confidence = analysis.get('confidence', 0.5)
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_price)
            stop_distance_pct = stop_distance / entry_price
            
            # Base position size calculation
            # Use 15% of balance per trade (aggressive for small balance)
            base_risk_amount = self.config.account_balance * self.config.risk_per_trade
            
            # Adjust for signal strength and confidence
            strength_multiplier = min(2.0, signal_strength / 5.0)
            confidence_multiplier = min(1.5, confidence * 2.0)
            
            # Final risk amount
            risk_amount = base_risk_amount * strength_multiplier * confidence_multiplier
            
            # Calculate position size
            position_size = risk_amount / stop_distance
            
            # Apply position value limits
            position_value = position_size * entry_price
            
            if position_value < self.config.min_position_value:
                # Too small, skip
                return 0
            elif position_value > self.config.max_position_value:
                # Too large, cap it
                position_size = self.config.max_position_value / entry_price
            
            # Apply leverage
            leveraged_position_size = position_size * self.config.leverage
            
            # Final check - ensure we have enough margin
            required_margin = position_value / self.config.leverage
            if required_margin > self.config.account_balance * 0.8:  # Use 80% of balance max
                return 0
            
            return leveraged_position_size
            
        except Exception as e:
            return 0
    
    def _get_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Get kline data for symbol"""
        # This would be implemented with actual API call
        # For now, return mock data
        return []
    
    def get_position_size_calculation(self, symbol: str, entry_price: float, stop_price: float, 
                                    signal_strength: float = 5.0) -> Dict:
        """Calculate detailed position size for debugging"""
        try:
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_price)
            stop_distance_pct = stop_distance / entry_price
            
            # Base risk amount
            base_risk_amount = self.config.account_balance * self.config.risk_per_trade
            
            # Adjustments
            strength_multiplier = min(2.0, signal_strength / 5.0)
            risk_amount = base_risk_amount * strength_multiplier
            
            # Position size
            position_size = risk_amount / stop_distance
            position_value = position_size * entry_price
            
            # Leverage
            leveraged_position_size = position_size * self.config.leverage
            leveraged_position_value = leveraged_position_size * entry_price
            
            # Required margin
            required_margin = leveraged_position_value / self.config.leverage
            
            # Available balance check
            available_balance = self.config.account_balance * 0.8  # 80% of balance
            
            return {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'stop_distance': stop_distance,
                'stop_distance_pct': stop_distance_pct,
                'base_risk_amount': base_risk_amount,
                'strength_multiplier': strength_multiplier,
                'adjusted_risk_amount': risk_amount,
                'position_size': position_size,
                'position_value': position_value,
                'leveraged_position_size': leveraged_position_size,
                'leveraged_position_value': leveraged_position_value,
                'required_margin': required_margin,
                'available_balance': available_balance,
                'can_trade': required_margin <= available_balance,
                'leverage': self.config.leverage,
                'account_balance': self.config.account_balance
            }
            
        except Exception as e:
            return {'error': str(e)}

def create_micro_balance_config() -> HighMomentumConfig:
    """Create configuration optimized for small balance ($1.7 -> $200)"""
    return HighMomentumConfig(
        account_balance=1.7,
        target_balance=200.0,
        max_positions=2,  # Only 2 positions for small balance
        leverage=100.0,
        risk_per_trade=0.20,  # 20% per trade (aggressive)
        max_total_risk=0.60,  # 60% total risk
        risk_reward_ratio=3.0,  # 3:1 RR
        min_24h_gain=0.50,  # 50% minimum gain
        max_24h_gain=2.00,  # 200% maximum gain
        min_volume_spike=2.0,
        min_volatility=0.05,
        timeframes=["1m", "5m"],
        ema_ultra_fast=2,
        ema_fast=5,
        ema_medium=13,
        ema_slow=21,
        macd_fast=5,
        macd_slow=13,
        macd_signal=3,
        rsi_period=7,
        rsi_oversold=20,
        rsi_overbought=80,
        adx_period=10,
        adx_threshold=15.0,
        volume_period=10,
        volume_spike_threshold=1.5,
        atr_period=10,
        stop_atr_multiplier=0.8,
        momentum_period=5,
        momentum_threshold=1.0,
        min_profit_target=0.02,
        max_hold_time=600,
        quick_exit_threshold=0.01,
        min_position_value=3.0,  # $3 minimum
        max_position_value=25.0,  # $25 maximum
        position_size_multiplier=1.5
    )

if __name__ == "__main__":
    # Demo position size calculation
    config = create_micro_balance_config()
    strategy = HighMomentumStrategy(config)
    
    print("🚀 High Momentum Strategy - Position Size Calculator")
    print("=" * 60)
    print(f"💰 Account Balance: ${config.account_balance}")
    print(f"🎯 Target Balance: ${config.target_balance}")
    print(f"📈 Leverage: {config.leverage}x")
    print(f"🎯 Risk per Trade: {config.risk_per_trade*100}%")
    print("=" * 60)
    
    # Example calculations
    examples = [
        ("BTCUSDT", 50000, 49000, 8.0),
        ("ETHUSDT", 3000, 2940, 6.0),
        ("DOGEUSDT", 0.08, 0.078, 7.0),
        ("XRPUSDT", 0.5, 0.49, 5.0)
    ]
    
    for symbol, entry, stop, strength in examples:
        calc = strategy.get_position_size_calculation(symbol, entry, stop, strength)
        if 'error' not in calc:
            print(f"\n📊 {symbol}:")
            print(f"   Entry: ${calc['entry_price']:.4f}")
            print(f"   Stop: ${calc['stop_price']:.4f}")
            print(f"   Stop Distance: {calc['stop_distance_pct']:.2%}")
            print(f"   Position Size: {calc['leveraged_position_size']:.6f}")
            print(f"   Position Value: ${calc['leveraged_position_value']:.2f}")
            print(f"   Required Margin: ${calc['required_margin']:.2f}")
            print(f"   Available: ${calc['available_balance']:.2f}")
            print(f"   Can Trade: {'✅' if calc['can_trade'] else '❌'}")
        else:
            print(f"❌ Error calculating {symbol}: {calc['error']}")
