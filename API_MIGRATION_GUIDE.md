# 🔄 API Migration Guide - Testnet to Mainnet

## Overview
Successfully migrated all trading bots from testnet to mainnet API with support for both real trading and demo trading.

## 🌐 API Endpoints

### Before (Testnet)
- **Testnet API**: `https://api-testnet.bybit.com`
- **Default**: Always used testnet
- **Parameter**: `testnet=True`

### After (Mainnet + Demo)
- **Mainnet API**: `https://api.bybit.com` (default)
- **Demo API**: `https://api-demo.bybit.com` (uses demo=True parameter)
- **Parameter**: `demo=False` (mainnet) or `demo=True` (demo)

## 🔧 Changes Made

### 1. **Momentum Trader** (`live/momentum_trader.py`)
```python
# Before
def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
    self.testnet = testnet
    self.client = HTTP(testnet=self.testnet, ...)

# After
def __init__(self, api_key: str = None, api_secret: str = None, demo: bool = False):
    self.demo = demo
    if demo:
        self.client = HTTP(demo=True, ...)  # Demo uses demo=True
    else:
        self.client = HTTP(testnet=False, ...)  # Mainnet uses testnet=False
```

### 2. **Aggressive Trader** (`live/aggressive_trader.py`)
```python
# Same changes as momentum trader
# Now defaults to mainnet instead of testnet
```

### 3. **Position Calculator** (`position_calculator.py`)
```python
# Before
client = HTTP(testnet=True, api_key=os.getenv('BYBIT_API_KEY_DEMO'), ...)

# After
# Try mainnet first, fallback to demo
api_key = os.getenv('BYBIT_API_KEY_REAL')
if not api_key:
    api_key = os.getenv('BYBIT_API_KEY_DEMO')
client = HTTP(testnet=False, api_key=api_key, ...)
```

### 4. **Runner Scripts**
```bash
# Before
python run_momentum_bot.py --testnet
python run_aggressive_bot.py --strategy scalping --testnet

# After
python run_momentum_bot.py                    # Mainnet (default)
python run_momentum_bot.py --demo            # Demo API
python run_aggressive_bot.py --strategy scalping  # Mainnet (default)
python run_aggressive_bot.py --strategy scalping --demo  # Demo API
```

## 🚀 Usage Examples

### Mainnet Trading (Real Money)
```bash
# Momentum trading on mainnet
python run_momentum_bot.py --balance 1.7 --target 200

# Aggressive trading on mainnet
python run_aggressive_bot.py --strategy scalping --balance 1000

# Position calculator (tries mainnet first)
python run_momentum_bot.py --calculator
```

### Demo Trading (Paper Money)
```bash
# Momentum trading on demo
python run_momentum_bot.py --demo --balance 1000 --target 2000

# Aggressive trading on demo
python run_aggressive_bot.py --strategy momentum --demo --balance 5000

# Position calculator with demo
python run_momentum_bot.py --calculator --demo
```

## 🔑 Environment Variables

### Required for Mainnet
```bash
export BYBIT_API_KEY_REAL="your_mainnet_api_key"
export BYBIT_API_SECRET_REAL="your_mainnet_api_secret"
```

### Required for Demo
```bash
export BYBIT_API_KEY_DEMO="your_demo_api_key"
export BYBIT_API_SECRET_DEMO="your_demo_api_secret"
```

### Both (Recommended)
```bash
# Set both for maximum flexibility
export BYBIT_API_KEY_REAL="your_mainnet_api_key"
export BYBIT_API_SECRET_REAL="your_mainnet_api_secret"
export BYBIT_API_KEY_DEMO="your_demo_api_key"
export BYBIT_API_SECRET_DEMO="your_demo_api_secret"
```

## ⚠️ Important Changes

### 1. **Default Behavior**
- **Before**: Defaulted to testnet (safe)
- **After**: Defaults to mainnet (real money)
- **Impact**: More dangerous, but more realistic

### 2. **API Endpoints**
- **Testnet**: `https://api-testnet.bybit.com` (removed)
- **Mainnet**: `https://api.bybit.com` (default)
- **Demo**: `https://api-demo.bybit.com` (uses demo=True parameter)

### 3. **Parameter Changes**
- **Before**: `testnet=True/False`
- **After**: `demo=True/False`
- **Logic**: `demo=False` = mainnet, `demo=True` = demo

### 4. **Safety Considerations**
- **Mainnet**: Uses real money - be careful!
- **Demo**: Uses paper money - safe for testing
- **Position Calculator**: Tries mainnet first, falls back to demo

## 🛡️ Safety Features

### 1. **API Key Validation**
- Checks for mainnet keys first
- Falls back to demo keys if mainnet not available
- Clear error messages if no keys found

### 2. **Endpoint Verification**
- Shows which API endpoint is being used
- Displays in startup messages
- Easy to verify before trading

### 3. **Balance Warnings**
- Warns for small balances on mainnet
- Suggests demo mode for testing
- Clear confirmation prompts

## 📊 Migration Checklist

- [x] Updated Momentum Trader
- [x] Updated Aggressive Trader  
- [x] Updated Position Calculator
- [x] Updated Runner Scripts
- [x] Updated Documentation
- [x] Added Demo API Support
- [x] Changed Default to Mainnet
- [x] Added Safety Warnings
- [x] Tested API Endpoints

## 🚀 Quick Start

### 1. Set Environment Variables
```bash
# For mainnet trading
export BYBIT_API_KEY_REAL="your_key"
export BYBIT_API_SECRET_REAL="your_secret"

# For demo trading
export BYBIT_API_KEY_DEMO="your_demo_key"
export BYBIT_API_SECRET_DEMO="your_demo_secret"
```

### 2. Test with Demo First
```bash
# Test momentum strategy on demo
python run_momentum_bot.py --demo --balance 1000

# Test aggressive strategy on demo
python run_aggressive_bot.py --strategy scalping --demo
```

### 3. Switch to Mainnet
```bash
# Real money trading (be careful!)
python run_momentum_bot.py --balance 1.7 --target 200
python run_aggressive_bot.py --strategy momentum --balance 1000
```

## ⚠️ Warnings

### Mainnet Trading
- **Real Money**: Uses actual funds
- **High Risk**: Can lose all money
- **No Undo**: Trades are permanent
- **Monitor Closely**: Watch positions constantly

### Demo Trading
- **Paper Money**: No real risk
- **Good for Testing**: Test strategies safely
- **Realistic Data**: Uses real market data
- **No Real P&L**: Profits/losses are virtual

## 🎯 Recommendations

1. **Start with Demo**: Test strategies on demo first
2. **Small Amounts**: Start with small amounts on mainnet
3. **Monitor Closely**: Watch all trades carefully
4. **Set Limits**: Use stop losses and position limits
5. **Keep Logs**: Monitor all trading activity

---

**Migration Complete!** All bots now use mainnet by default with demo support. 🚀
