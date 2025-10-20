# Enhanced Crypto Trading Strategy Guide

## Overview

This enhanced crypto trading strategy combines **technical analysis**, **sentiment analysis**, and **fundamental analysis** to identify high-probability trading opportunities. The strategy is designed to be flexible and adaptable to different trading styles: scalping, day trading, and swing trading.

## Strategy Components

### 1. Technical Analysis

#### Moving Averages
- **EMA Fast/Slow**: 9/20 for trend identification
- **EMA Trend**: 50 for overall market direction
- **Crossover Signals**: Fast EMA crossing above/below slow EMA

#### RSI (Relative Strength Index)
- **Period**: 14
- **Oversold**: < 30 (potential buy signal)
- **Overbought**: > 70 (potential sell signal)
- **Divergence Detection**: Price vs RSI divergence for reversal signals

#### Volume Analysis
- **Volume MA**: 20-period moving average
- **Volume Spike**: Volume > 1.5x average (confirmation signal)
- **Breakout Confirmation**: Price breakout with volume spike

#### Market Structure
- **Pivot Points**: 5-period lookback for support/resistance
- **Breakout Detection**: Price breaking above resistance or below support
- **Support/Resistance Levels**: Dynamic levels based on recent highs/lows

### 2. Sentiment Analysis

#### News Sentiment
- **Sources**: Crypto news websites, press releases, announcements
- **Weight**: 40% of total sentiment score
- **Lookback**: 24 hours
- **Thresholds**: 
  - Positive: > 0.6
  - Negative: < -0.6
  - Neutral: -0.2 to 0.2

#### Social Media Sentiment
- **Sources**: Twitter, Reddit, Telegram, Discord
- **Weight**: 30% of total sentiment score
- **Lookback**: 6 hours
- **Influencer Tracking**: Key crypto influencers and their mentions

#### Technical Momentum
- **Weight**: 30% of total sentiment score
- **Calculation**: Based on technical indicator alignment

### 3. Fundamental Analysis

#### Tokenomics Events
- **Token Burns**: Supply reduction events
- **Token Unlocks**: Supply increase events (negative)
- **Staking Rewards**: Changes in staking mechanisms
- **Supply Changes**: Any modifications to token supply

#### Macro Events
- **Regulatory News**: Government announcements, regulations
- **Exchange Listings**: New exchange listings (positive)
- **Partnerships**: Strategic partnerships and collaborations
- **Influencer Mentions**: Key figures discussing the asset

#### Risk Filters
- **Avoid High Unlock Periods**: Skip trading during major token unlocks
- **Avoid Regulatory Uncertainty**: Skip during regulatory FUD
- **Avoid High Volatility**: Skip during extreme market volatility

## Trading Styles Configuration

### Scalping (1-5 minutes)
```python
config = EnhancedStrategyConfig(
    trading_style="scalping",
    technical=TechnicalConfig(
        ema_fast=5,
        ema_slow=13,
        take_profit=0.005,  # 0.5%
        stop_loss=0.003,    # 0.3%
        max_holding_bars=3,
        risk_per_trade=0.005,  # 0.5%
        leverage=10.0
    )
)
```

### Day Trading (5-60 minutes)
```python
config = EnhancedStrategyConfig(
    trading_style="day",
    technical=TechnicalConfig(
        ema_fast=9,
        ema_slow=20,
        take_profit=0.01,   # 1%
        stop_loss=0.007,    # 0.7%
        max_holding_bars=10,
        risk_per_trade=0.01,  # 1%
        leverage=7.0
    )
)
```

### Swing Trading (1-24 hours)
```python
config = EnhancedStrategyConfig(
    trading_style="swing",
    technical=TechnicalConfig(
        ema_fast=9,
        ema_slow=20,
        take_profit=0.02,   # 2%
        stop_loss=0.015,    # 1.5%
        max_holding_bars=20,
        risk_per_trade=0.015,  # 1.5%
        leverage=5.0
    )
)
```

## Entry Criteria

### Long Entry Requirements
1. **Technical Signals** (at least 2):
   - Trend breakout (price > resistance + volume spike)
   - RSI bullish divergence
   - Support bounce with volume confirmation
   - EMA crossover (fast > slow)

2. **Sentiment Filter**:
   - Combined sentiment score > 0.6
   - News sentiment > 0.6
   - Social sentiment > 0.6

3. **Fundamental Filter**:
   - Fundamental score > 0.3
   - No major negative events (unlocks, regulatory FUD)
   - Positive catalysts present

### Short Entry Requirements
1. **Technical Signals** (at least 2):
   - Trend breakdown (price < support + volume spike)
   - RSI bearish divergence
   - Resistance rejection with volume
   - EMA crossover (fast < slow)

2. **Sentiment Filter**:
   - Combined sentiment score < -0.6
   - News sentiment < -0.6
   - Social sentiment < -0.6

3. **Fundamental Filter**:
   - Fundamental score < -0.3
   - Negative catalysts present
   - No major positive events

## Risk Management

### Position Sizing
- **Fixed Risk**: 1% of account balance per trade
- **Leverage**: 3-10x depending on trading style
- **Maximum Positions**: 3 concurrent positions
- **Daily Trade Limit**: 10 trades per day
- **Maximum Daily Loss**: 5% of account balance

### Stop Loss & Take Profit
- **Risk-Reward Ratio**: Minimum 1.5:1
- **Stop Loss**: 0.3% - 1.5% depending on style
- **Take Profit**: 0.5% - 2% depending on style
- **Trailing Stops**: Not implemented (can be added)

### Risk Controls
- **Rate Limiting**: 5 minutes between trades per symbol
- **Volatility Filter**: Skip trading during extreme volatility
- **News Filter**: Avoid trading during major FUD events
- **Liquidity Check**: Ensure sufficient volume before entry

## Usage Examples

### Backtesting
```bash
# Basic backtest
python backtest/run_enhanced_backtest.py --file data/BTCUSDT_1m.csv --symbol BTCUSDT --style swing

# Parameter optimization
python backtest/run_enhanced_backtest.py --file data/BTCUSDT_1m.csv --symbol BTCUSDT --style swing --optimize

# Save results
python backtest/run_enhanced_backtest.py --file data/BTCUSDT_1m.csv --symbol BTCUSDT --style swing --save-results
```

### Live Trading
```bash
# Swing trading on mainnet
python live/enhanced_trader.py --symbols BTCUSDT ETHUSDT --style swing --max-positions 3

# Day trading on testnet
python live/enhanced_trader.py --symbols BTCUSDT ETHUSDT ADAUSDT --style day --testnet

# Scalping with custom parameters
python live/enhanced_trader.py --symbols BTCUSDT --style scalping --max-daily-trades 20 --max-positions 2
```

## Strategy Checklist

### Pre-Trade Analysis
- [ ] Check market conditions (trend, volatility)
- [ ] Analyze sentiment (news, social media)
- [ ] Review fundamental factors (tokenomics, events)
- [ ] Confirm technical setup (at least 2 signals)
- [ ] Verify risk management parameters
- [ ] Check position limits and daily limits

### Entry Execution
- [ ] Confirm all entry criteria met
- [ ] Calculate position size based on risk
- [ ] Set stop loss and take profit levels
- [ ] Place order with proper leverage
- [ ] Log trade details and reasoning

### Trade Management
- [ ] Monitor position for early exit signals
- [ ] Update stop loss if trend strengthens
- [ ] Take partial profits at key levels
- [ ] Exit if fundamental conditions change
- [ ] Log exit reason and PnL

### Post-Trade Analysis
- [ ] Review trade performance
- [ ] Analyze what went right/wrong
- [ ] Update strategy parameters if needed
- [ ] Maintain trade log for learning
- [ ] Check daily PnL and limits

## Performance Metrics

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Win/Loss**: Average profit/loss per trade
- **Risk-Reward Ratio**: Average win / Average loss

### Target Performance
- **Win Rate**: > 60%
- **Profit Factor**: > 1.5
- **Sharpe Ratio**: > 1.0
- **Maximum Drawdown**: < 10%
- **Risk-Reward**: > 1.5:1

## Example Trade Setups

### Long Setup Example
```
Symbol: BTCUSDT
Timeframe: 1m
Entry Price: $45,250

Technical Signals:
- EMA crossover (9 > 20)
- RSI bullish divergence
- Volume spike (2.1x average)
- Support bounce at $45,200

Sentiment:
- News: +0.7 (positive)
- Social: +0.8 (very positive)
- Combined: +0.75

Fundamental:
- Exchange listing announcement
- No major unlocks
- Score: +0.6

Position:
- Size: 0.1 BTC
- Leverage: 5x
- Stop Loss: $44,700 (-1.2%)
- Take Profit: $46,000 (+1.7%)
- Risk-Reward: 1.4:1
```

### Short Setup Example
```
Symbol: ETHUSDT
Timeframe: 1m
Entry Price: $3,150

Technical Signals:
- Resistance rejection at $3,160
- RSI bearish divergence
- Volume spike (1.8x average)
- EMA crossover (9 < 20)

Sentiment:
- News: -0.6 (negative)
- Social: -0.7 (very negative)
- Combined: -0.65

Fundamental:
- Regulatory uncertainty
- Large token unlock next week
- Score: -0.5

Position:
- Size: 0.5 ETH
- Leverage: 7x
- Stop Loss: $3,200 (+1.6%)
- Take Profit: $3,100 (-1.6%)
- Risk-Reward: 1.0:1
```

## Troubleshooting

### Common Issues
1. **No trades executed**: Check if all criteria are being met
2. **High drawdown**: Reduce position size or improve entry criteria
3. **Low win rate**: Review technical indicators and sentiment thresholds
4. **API errors**: Check API credentials and rate limits
5. **Data issues**: Ensure CSV format is correct for backtesting

### Debug Mode
Enable verbose logging to see detailed analysis:
```bash
python backtest/run_enhanced_backtest.py --file data.csv --verbose
```

## Future Enhancements

### Planned Features
- [ ] Machine learning sentiment analysis
- [ ] Real-time news API integration
- [ ] Social media sentiment tracking
- [ ] Advanced order types (trailing stops, OCO)
- [ ] Portfolio optimization
- [ ] Multi-timeframe analysis
- [ ] Correlation analysis between assets
- [ ] Dynamic position sizing based on volatility

### Customization
The strategy is highly customizable. You can modify:
- Technical indicator parameters
- Sentiment thresholds
- Fundamental event weights
- Risk management rules
- Entry/exit criteria
- Position sizing algorithms

## Disclaimer

This strategy is for educational purposes only. Cryptocurrency trading involves significant risk and may not be suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Test with small amounts first
4. Use testnet for live trading experiments
5. Keep detailed logs for analysis
