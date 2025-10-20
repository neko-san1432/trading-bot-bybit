#!/usr/bin/env python3
"""
Aggressive Trading Configurations - Different strategies for different risk levels
"""

from aggressive_strategy import AggressiveConfig

def create_scalping_config() -> AggressiveConfig:
    """Ultra-aggressive scalping configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=10,  # More positions
        leverage=100.0,    # Maximum leverage
        
        # Risk management
        risk_per_trade=0.05,      # 5% per trade
        max_total_risk=0.30,      # 30% total risk
        risk_reward_ratio=1.2,    # Lower RR for more wins
        
        # Timeframes
        timeframes=["1m", "3m", "5m"],
        primary_timeframe="1m",
        
        # Ultra-fast indicators
        ema_ultra_fast=2,
        ema_fast=5,
        ema_medium=13,
        ema_slow=34,
        
        # Fast MACD
        macd_fast=5,
        macd_slow=13,
        macd_signal=3,
        
        # Sensitive RSI
        rsi_period=5,
        rsi_oversold=20,
        rsi_overbought=80,
        
        # Lower thresholds
        adx_threshold=8.0,
        volume_spike_threshold=0.2,
        stop_atr_multiplier=0.8,
        
        # Quick exits
        min_profit_target=0.003,  # 0.3%
        max_hold_time=180,        # 3 minutes
        quick_exit_threshold=0.001,  # 0.1%
        
        # Features
        enable_grid_trading=True,
        enable_dca=True,
        enable_pyramiding=True,
        enable_martingale=False,  # Too risky for scalping
    )

def create_momentum_config() -> AggressiveConfig:
    """Momentum-based aggressive configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=5,
        leverage=75.0,
        
        # Risk management
        risk_per_trade=0.04,      # 4% per trade
        max_total_risk=0.25,      # 25% total risk
        risk_reward_ratio=1.8,    # Higher RR for momentum
        
        # Timeframes
        timeframes=["1m", "5m", "15m"],
        primary_timeframe="5m",
        
        # Momentum-focused indicators
        ema_ultra_fast=3,
        ema_fast=8,
        ema_medium=21,
        ema_slow=55,
        
        # MACD for momentum
        macd_fast=8,
        macd_slow=21,
        macd_signal=5,
        
        # RSI for momentum
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        
        # ADX for trend strength
        adx_threshold=15.0,
        volume_spike_threshold=0.4,
        stop_atr_multiplier=1.2,
        
        # Momentum exits
        min_profit_target=0.008,  # 0.8%
        max_hold_time=600,        # 10 minutes
        quick_exit_threshold=0.003,  # 0.3%
        
        # Features
        enable_grid_trading=False,
        enable_dca=False,
        enable_pyramiding=True,
        enable_martingale=False,
    )

def create_breakout_config() -> AggressiveConfig:
    """Breakout-based aggressive configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=3,
        leverage=50.0,
        
        # Risk management
        risk_per_trade=0.06,      # 6% per trade
        max_total_risk=0.20,      # 20% total risk
        risk_reward_ratio=2.0,    # Higher RR for breakouts
        
        # Timeframes
        timeframes=["5m", "15m", "1h"],
        primary_timeframe="15m",
        
        # Breakout indicators
        ema_ultra_fast=5,
        ema_fast=13,
        ema_medium=34,
        ema_slow=89,
        
        # MACD for confirmation
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        
        # RSI for overbought/oversold
        rsi_period=14,
        rsi_oversold=25,
        rsi_overbought=75,
        
        # ADX for trend strength
        adx_threshold=20.0,
        volume_spike_threshold=0.5,
        stop_atr_multiplier=1.5,
        
        # Breakout exits
        min_profit_target=0.015,  # 1.5%
        max_hold_time=1800,       # 30 minutes
        quick_exit_threshold=0.005,  # 0.5%
        
        # Features
        enable_grid_trading=False,
        enable_dca=True,
        enable_pyramiding=True,
        enable_martingale=False,
    )

def create_news_trading_config() -> AggressiveConfig:
    """News-based aggressive configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=8,
        leverage=60.0,
        
        # Risk management
        risk_per_trade=0.04,      # 4% per trade
        max_total_risk=0.35,      # 35% total risk
        risk_reward_ratio=1.5,    # Lower RR for news
        
        # Timeframes
        timeframes=["1m", "3m", "5m"],
        primary_timeframe="1m",
        
        # Fast indicators for news
        ema_ultra_fast=2,
        ema_fast=5,
        ema_medium=13,
        ema_slow=21,
        
        # Fast MACD
        macd_fast=5,
        macd_slow=13,
        macd_signal=3,
        
        # Sensitive RSI
        rsi_period=7,
        rsi_oversold=15,
        rsi_overbought=85,
        
        # Lower thresholds for news
        adx_threshold=5.0,
        volume_spike_threshold=0.1,  # Very sensitive to volume
        stop_atr_multiplier=0.5,     # Tight stops for news
        
        # Quick news exits
        min_profit_target=0.002,  # 0.2%
        max_hold_time=120,        # 2 minutes
        quick_exit_threshold=0.001,  # 0.1%
        
        # Features
        enable_grid_trading=True,
        enable_dca=False,
        enable_pyramiding=False,
        enable_martingale=False,
        
        # News-specific
        enable_news_filter=True,
        enable_social_sentiment=True,
        enable_volume_breakout=True,
        avoid_news_times=True,
        news_avoidance_minutes=15,
    )

def create_grid_trading_config() -> AggressiveConfig:
    """Grid trading aggressive configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=20,  # Many small positions
        leverage=25.0,     # Lower leverage for grid
        
        # Risk management
        risk_per_trade=0.01,      # 1% per trade
        max_total_risk=0.40,      # 40% total risk
        risk_reward_ratio=1.0,    # 1:1 for grid
        
        # Timeframes
        timeframes=["1m", "5m"],
        primary_timeframe="1m",
        
        # Grid indicators
        ema_ultra_fast=3,
        ema_fast=8,
        ema_medium=21,
        ema_slow=55,
        
        # MACD
        macd_fast=8,
        macd_slow=21,
        macd_signal=5,
        
        # RSI
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        
        # ADX
        adx_threshold=10.0,
        volume_spike_threshold=0.3,
        stop_atr_multiplier=1.0,
        
        # Grid exits
        min_profit_target=0.005,  # 0.5%
        max_hold_time=3600,       # 1 hour
        quick_exit_threshold=0.002,  # 0.2%
        
        # Features
        enable_grid_trading=True,
        enable_dca=True,
        enable_pyramiding=False,
        enable_martingale=True,   # Enable for grid
    )

def create_volatility_config() -> AggressiveConfig:
    """High volatility aggressive configuration"""
    return AggressiveConfig(
        # Core settings
        account_balance=10000.0,
        max_positions=4,
        leverage=80.0,
        
        # Risk management
        risk_per_trade=0.05,      # 5% per trade
        max_total_risk=0.25,      # 25% total risk
        risk_reward_ratio=1.5,    # Lower RR for volatility
        
        # Timeframes
        timeframes=["1m", "5m", "15m"],
        primary_timeframe="5m",
        
        # Volatility indicators
        ema_ultra_fast=3,
        ema_fast=8,
        ema_medium=21,
        ema_slow=55,
        
        # MACD
        macd_fast=8,
        macd_slow=21,
        macd_signal=5,
        
        # RSI
        rsi_period=14,
        rsi_oversold=20,
        rsi_overbought=80,
        
        # ADX
        adx_threshold=12.0,
        volume_spike_threshold=0.4,
        stop_atr_multiplier=1.2,
        
        # Volatility exits
        min_profit_target=0.010,  # 1.0%
        max_hold_time=900,        # 15 minutes
        quick_exit_threshold=0.003,  # 0.3%
        
        # Volatility filters
        min_volatility=0.02,      # 2% minimum
        max_volatility=0.15,      # 15% maximum
        
        # Features
        enable_grid_trading=False,
        enable_dca=True,
        enable_pyramiding=True,
        enable_martingale=False,
    )

# Configuration presets
AGGRESSIVE_CONFIGS = {
    'scalping': create_scalping_config,
    'momentum': create_momentum_config,
    'breakout': create_breakout_config,
    'news': create_news_trading_config,
    'grid': create_grid_trading_config,
    'volatility': create_volatility_config,
}

def get_aggressive_config(strategy_name: str) -> AggressiveConfig:
    """Get aggressive configuration by strategy name"""
    if strategy_name in AGGRESSIVE_CONFIGS:
        return AGGRESSIVE_CONFIGS[strategy_name]()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(AGGRESSIVE_CONFIGS.keys())}")

def list_available_strategies():
    """List all available aggressive strategies"""
    print("🚀 Available Aggressive Trading Strategies:")
    print("=" * 50)
    
    strategies = {
        'scalping': 'Ultra-fast scalping with 1m timeframes',
        'momentum': 'Momentum-based trading with trend following',
        'breakout': 'Breakout trading with higher timeframes',
        'news': 'News-based trading with volume spikes',
        'grid': 'Grid trading with multiple small positions',
        'volatility': 'High volatility trading with ATR-based stops'
    }
    
    for name, description in strategies.items():
        print(f"📈 {name.upper()}: {description}")
    
    print("=" * 50)

if __name__ == "__main__":
    list_available_strategies()
