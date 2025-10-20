# Enhanced Crypto Trading Strategy

This project implements a comprehensive crypto trading strategy that combines **technical analysis**, **sentiment analysis**, and **fundamental analysis** to identify high-probability trading opportunities. The strategy supports multiple trading styles (scalping, day trading, swing trading) and provides both backtesting and live trading capabilities.

## 🚀 Features

### Multi-Analysis Approach
- **Technical Analysis**: EMA crossovers, RSI divergence, volume analysis, market structure
- **Sentiment Analysis**: News sentiment, social media sentiment, technical momentum
- **Fundamental Analysis**: Tokenomics events, macro events, regulatory news
- **Volatility Analysis**: Real-time volatility measurement and adaptive position sizing

### Dynamic Pair Discovery
- **API Integration**: Automatic discovery of trading pairs from Bybit API
- **Real-time Updates**: Automatic monitoring for new listings and delistings
- **Pair Scoring**: Intelligent scoring based on volatility, volume, and market conditions
- **Style-specific Selection**: Different pair recommendations for scalping, day, and swing trading

### Aggressive Profit Targets
- **50% Profit Target**: Optimized for high-reward trades with 25:1 risk-reward ratio
- **High Leverage**: Up to 50x leverage for maximum profit potential
- **Volatility-aware Sizing**: Dynamic position sizing based on real-time volatility
- **Tight Risk Management**: 0.5% risk per trade with 2% stop losses

### Trading Styles
- **Trend Scalping**: 1-10 minute trades with 0.5-5% targets and 50-65% win rate
- **Aggressive Scalping**: 1-5 minute trades targeting 50%+ profits with high leverage
- **Day Trading**: 5-60 minute trades with moderate leverage and volatility
- **Swing Trading**: 1-24 hour trades with lower leverage and trend following

### Advanced Risk Management
- Volatility-based position sizing and leverage adjustment
- Dynamic stop loss and take profit based on market conditions
- Maximum concurrent positions and daily trade limits
- Real-time pair evaluation and automatic selection

### Connection Health Monitoring
- **Comprehensive Ping Tests**: DNS resolution, HTTP connectivity, API endpoints
- **Real-time Monitoring**: Continuous connection health checks during trading
- **Automatic Retry Logic**: Exponential backoff retry for failed operations
- **Connection Recovery**: Automatic detection and recovery from connection issues
- **Trading Safety**: Prevents trading during connection problems

## 📁 Project Structure

```
├── backtest/
│   ├── strategy.py              # Original scalping strategy
│   ├── enhanced_strategy.py     # Enhanced strategy with 50% target
│   ├── trend_scalping_strategy.py # NEW: Trend scalping strategy (50-65% win rate)
│   ├── pair_discovery.py        # Dynamic pair discovery system
│   ├── run_backtest.py          # Original backtest runner
│   ├── run_enhanced_backtest.py # Enhanced backtest runner
│   └── run_trend_scalping_backtest.py # NEW: Trend scalping backtest runner
├── live/
│   ├── trader.py                # Original live trader
│   ├── enhanced_trader.py       # Enhanced live trader
│   ├── enhanced_dynamic_trader.py # Dynamic pair discovery trader
│   └── trend_scalping_trader.py # NEW: Trend scalping live trader
├── examples/
│   ├── strategy_demo.py         # Basic strategy demonstration
│   ├── trend_scalping_demo.py   # NEW: Trend scalping demonstration
│   └── ping_demo.py            # Ping test functionality demo
├── ping_utils.py               # Comprehensive ping test utility
├── test_ping.py               # Standalone ping test script
│   └── dynamic_strategy_demo.py # Dynamic features demonstration
├── tests/
│   ├── test_strategy.py         # Original strategy tests
│   └── test_enhanced_strategy.py # Enhanced strategy tests
└── ENHANCED_STRATEGY_GUIDE.md   # Comprehensive strategy guide
```

## 🛠️ Requirements

- Python 3.9+
- See `requirements.txt` for packages

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Strategy Demos

```bash
# Basic strategy demo
python examples/strategy_demo.py

# Dynamic features demo (pair discovery, volatility analysis, 50% targets)
python examples/dynamic_strategy_demo.py
```

### 3. Backtest with Your Data

```bash
# Basic backtest
python backtest/run_enhanced_backtest.py --file your_data.csv --symbol BTCUSDT --style swing

# Parameter optimization
python backtest/run_enhanced_backtest.py --file your_data.csv --symbol BTCUSDT --style swing --optimize

# Save results
python backtest/run_enhanced_backtest.py --file your_data.csv --symbol BTCUSDT --style swing --save-results
```

### 4. Live Trading

```bash
# Enhanced trader with fixed pairs
python live/enhanced_trader.py --symbols BTCUSDT ETHUSDT --style swing --testnet

# Dynamic trader with automatic pair discovery (RECOMMENDED)
python live/enhanced_dynamic_trader.py --style scalping --testnet

# Trend scalping trader (NEW - Higher win rate, smaller targets)
python live/trend_scalping_trader.py --testnet
```

### 5. Connection Health Testing

```bash
# Test Bybit connectivity
python test_ping.py

# Run ping test demo
python examples/ping_demo.py

# Monitor connection health
python -c "from ping_utils import BybitPingTester; BybitPingTester().monitor_connection_health(60)"
```

## 📊 Strategy Components

### Trend Scalping Strategy (NEW)
- **Multi-Timeframe EMAs**: 5/13/21/50/200 for trend confirmation
- **ADX**: 14-period for trend strength measurement
- **MACD**: 12/26/9 for momentum confirmation
- **ATR-Based Stops**: Dynamic stop loss calculation
- **Entry Signals**: Pullback, breakout, and momentum entries
- **Dynamic Targets**: 0.5-5% based on volatility and trend strength

### Technical Analysis
- **Moving Averages**: EMA 9/20/50 for trend identification
- **RSI**: 14-period with divergence detection
- **Volume Analysis**: Volume spikes and confirmation
- **Market Structure**: Support/resistance and breakouts

### Sentiment Analysis
- **News Sentiment**: Crypto news and announcements
- **Social Sentiment**: Twitter, Reddit, Telegram sentiment
- **Technical Momentum**: Indicator alignment scoring

### Fundamental Analysis
- **Tokenomics**: Burns, unlocks, staking rewards
- **Macro Events**: Regulatory news, exchange listings
- **Risk Filters**: Avoid trading during negative events

## 🎯 Entry Criteria

### Long Entry
1. **Technical**: At least 2 signals (trend breakout, RSI divergence, support bounce)
2. **Sentiment**: Combined score > 0.6
3. **Fundamental**: Score > 0.3, no major negative events

### Short Entry
1. **Technical**: At least 2 signals (trend breakdown, RSI divergence, resistance rejection)
2. **Sentiment**: Combined score < -0.6
3. **Fundamental**: Score < -0.3, negative catalysts present

## 🔍 Connection Health Monitoring

### Ping Test Features
- **DNS Resolution**: Tests domain name resolution
- **HTTP Connectivity**: Basic HTTP connection to Bybit servers
- **API Endpoints**: Tests server time, market data, trading, and authenticated endpoints
- **Latency Monitoring**: Tracks response times with configurable thresholds
- **Connection Recovery**: Automatic retry logic with exponential backoff
- **Trading Safety**: Prevents trading during connection issues

### Ping Test Thresholds
- **Good**: < 200ms latency
- **Warning**: 200-500ms latency
- **Critical**: 500-1000ms latency
- **Error**: > 1000ms or connection failure

### Usage Examples
```python
from ping_utils import BybitPingTester

# Initialize tester
tester = BybitPingTester(use_testnet=True)

# Quick ping test
if tester.quick_ping_test():
    print("Connection OK")

# Comprehensive test
results = tester.comprehensive_ping_test()

# Monitor connection
tester.monitor_connection_health(duration_minutes=60)
```

## 📈 Performance Metrics

- **Win Rate**: Target > 60%
- **Profit Factor**: Target > 1.5
- **Sharpe Ratio**: Target > 1.0
- **Max Drawdown**: Target < 10%
- **Risk-Reward**: Target > 1.5:1

## 🔧 Configuration

The strategy is highly configurable. You can modify:

- Technical indicator parameters
- Sentiment thresholds
- Fundamental event weights
- Risk management rules
- Entry/exit criteria
- Position sizing algorithms

See `ENHANCED_STRATEGY_GUIDE.md` for detailed configuration options.

## 🧪 Testing

Run the test suite:

```bash
python tests/test_enhanced_strategy.py
```

## 📚 Documentation

- **Strategy Guide**: `ENHANCED_STRATEGY_GUIDE.md` - Comprehensive strategy documentation
- **Examples**: `examples/strategy_demo.py` - Strategy demonstration
- **API Reference**: Code comments and docstrings

## ⚠️ Disclaimer

This strategy is for educational purposes only. Cryptocurrency trading involves significant risk and may not be suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the strategy.

## 📄 License

This project is open source and available under the MIT License.
