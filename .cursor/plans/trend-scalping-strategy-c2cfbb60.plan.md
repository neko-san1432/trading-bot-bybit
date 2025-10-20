<!-- c2cfbb60-3856-498e-a6d4-bb34f7cbd07e 95c64de1-853d-4e20-b72a-e281af5de86c -->
# Trend Scalping Strategy Implementation

## Overview

Transform the existing 50% profit target strategy into a trend scalping strategy that:

- Uses multiple trend confirmation methods (EMA, price action, higher highs/lows)
- Enters on pullbacks, breakouts, and momentum continuation
- Dynamic profit targets based on volatility and trend strength (0.5-5%)
- ATR-based dynamic stops
- Very aggressive leverage (20-50x)
- Keeps dynamic pair discovery and volatility analysis

## Implementation Plan

### 1. Create New Trend Scalping Strategy Module

**File**: `backtest/trend_scalping_strategy.py`

Create a new strategy class `TrendScalpingStrategy` with:

- **Trend Detection System**:
  - Multiple EMA periods (5, 13, 21, 50, 200) for multi-timeframe trend
  - Price action analysis (higher highs/lows for uptrend, lower highs/lows for downtrend)
  - ADX (Average Directional Index) for trend strength measurement
  - MACD for momentum confirmation

- **Entry Signal Types**:
  - Pullback entries: Price retraces to key EMA within strong trend
  - Breakout entries: Price breaks micro-resistance/support in trend direction
  - Momentum continuation: Strong candles after brief consolidation
  - Priority scoring for best entry type based on trend strength

- **Dynamic Profit Targets**:
  - Base target: 1-2% for most trades
  - Scale based on: volatility (higher vol = higher target), trend strength (stronger trend = higher target)
  - Range: 0.5% (weak trend/low vol) to 5% (strong trend/high vol)
  - ATR multiplier for target calculation

- **ATR-Based Stops**:
  - Stop loss: 1.5-2.5x ATR below entry (long) or above entry (short)
  - Trailing stop: Move stop to breakeven when profit reaches 50% of target
  - Dynamic adjustment: Tighter stops in choppy markets, wider in trending markets

### 2. Update Technical Analysis Module

**File**: `backtest/trend_scalping_strategy.py` (new indicators)

Add new indicators:

- ADX (Average Directional Index) for trend strength
- +DI/-DI for directional movement
- MACD (12, 26, 9) for momentum
- ATR (14-period) for stop loss calculation
- Multi-timeframe EMAs (5, 13, 21, 50, 200)
- Higher high/lower low detection for price action analysis

### 3. Create Trend Scalping Configuration

**File**: `backtest/trend_scalping_strategy.py`

Define `TrendScalpingConfig` with:

- Trend confirmation settings (ADX threshold > 25, EMA alignment)
- Entry type preferences (pullback weight, breakout weight, momentum weight)
- Dynamic TP/SL settings (base TP 1-2%, ATR multiplier 1.5-2.5x)
- Leverage settings (20-50x based on trend strength)
- Volatility filters (min 2%, max 15%)
- Risk per trade (0.3-0.5%)

### 4. Update Live Trader for Trend Scalping

**File**: `live/trend_scalping_trader.py` (new file)

Create `TrendScalpingTrader` that:

- Inherits from `EnhancedDynamicTrader`
- Uses `TrendScalpingStrategy` instead of `EnhancedCryptoStrategy`
- Keeps pair discovery and volatility analysis
- Implements trend scalping entry logic
- Monitors trend changes and exits positions when trend reverses
- Very fast execution (1-5 minute holds)

### 5. Create Backtest Runner for Trend Scalping

**File**: `backtest/run_trend_scalping_backtest.py` (new file)

Create backtest runner with:

- Trend scalping parameter optimization
- Performance metrics specific to scalping (win rate, avg hold time, max consecutive losses)
- Comparison mode to test different entry types
- Volatility-based performance analysis

### 6. Update Examples and Documentation

**Files**:

- `examples/trend_scalping_demo.py` (new)
- `TREND_SCALPING_GUIDE.md` (new)
- Update `README.md` to include trend scalping option

Add:

- Demo showing trend detection and entry signals
- Examples of pullback, breakout, and momentum entries
- Performance comparison vs 50% target strategy
- Usage instructions and configuration examples

## Key Technical Details

### Trend Confirmation Logic

```python
def detect_trend(df):
    # Multi-EMA alignment
    ema_uptrend = (df['ema_5'] > df['ema_13'] > df['ema_21'] > df['ema_50'])
    
    # ADX for strength
    strong_trend = df['adx'] > 25
    
    # Price action
    higher_highs = df['high'] > df['high'].shift(5).rolling(10).max()
    higher_lows = df['low'] > df['low'].shift(5).rolling(10).min()
    
    # Combine signals
    uptrend = ema_uptrend & strong_trend & higher_highs & higher_lows
    return 'up' if uptrend else ('down' if opposite else 'sideways')
```

### Entry Signal Priority

1. Strong trend + pullback to EMA21 + volume spike = HIGH priority
2. Strong trend + breakout + momentum = MEDIUM priority  
3. Moderate trend + momentum continuation = LOW priority

### Dynamic Profit Target Calculation

```python
def calculate_target(atr, volatility, trend_strength, adx):
    base_target = 0.015  # 1.5%
    volatility_multiplier = volatility / 0.05  # Normalize to 5%
    trend_multiplier = (adx - 25) / 25  # Scale from ADX 25-50
    
    target = base_target * (1 + volatility_multiplier * 0.5) * (1 + trend_multiplier * 0.3)
    return min(max(target, 0.005), 0.05)  # Clamp between 0.5% and 5%
```

### ATR-Based Stop Loss

```python
def calculate_stop(entry_price, atr, direction):
    atr_multiplier = 2.0  # Standard 2x ATR
    if direction == 'long':
        stop = entry_price - (atr * atr_multiplier)
    else:
        stop = entry_price + (atr * atr_multiplier)
    return stop
```

## Migration Path

1. Keep existing `enhanced_strategy.py` and `enhanced_dynamic_trader.py` intact
2. Create new trend scalping modules alongside existing ones
3. User can choose which strategy to use via command line flag
4. Both strategies share pair discovery and volatility analysis infrastructure

## Expected Performance

- Win rate: 50-65% (higher than 50% target strategy due to trend following)
- Average profit: 1-2% per winning trade
- Average loss: 0.5-1% per losing trade (ATR-based stops)
- Risk-reward: 2:1 to 3:1 ratio
- Trade frequency: Higher (more opportunities with lower targets)
- Holding time: 1-10 minutes (true scalping)

### To-dos

- [ ] Create trend_scalping_strategy.py with trend detection, entry signals, and dynamic TP/SL
- [ ] Add ADX, MACD, multi-timeframe EMAs, and price action analysis indicators
- [ ] Create trend_scalping_trader.py with fast execution and trend monitoring
- [ ] Create run_trend_scalping_backtest.py with scalping-specific metrics
- [ ] Create trend_scalping_demo.py showing trend detection and entry examples
- [ ] Create TREND_SCALPING_GUIDE.md and update README.md