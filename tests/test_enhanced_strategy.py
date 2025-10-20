import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.enhanced_strategy import EnhancedStrategyConfig, EnhancedCryptoStrategy


def create_test_data(n_bars: int = 200) -> pd.DataFrame:
    """Create synthetic test data"""
    np.random.seed(42)
    
    # Generate price series
    base_price = 45000
    returns = np.random.normal(0, 0.002, n_bars)
    prices = [base_price]
    
    for i in range(1, n_bars):
        price = prices[-1] * (1 + returns[i])
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.001))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = prices[i-1] if i > 0 else close
        
        volume = 1000 * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
        timestamp = datetime.now() - timedelta(minutes=n_bars-i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_strategy_initialization():
    """Test strategy initialization"""
    config = EnhancedStrategyConfig(trading_style="swing")
    strategy = EnhancedCryptoStrategy(config)
    
    assert strategy is not None
    assert strategy.config.trading_style == "swing"
    assert strategy.sentiment_analyzer is not None
    assert strategy.fundamental_analyzer is not None
    assert strategy.technical_analyzer is not None


def test_data_preparation():
    """Test data preparation with indicators"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    df = create_test_data(100)
    prepared_df = strategy.prepare(df)
    
    # Check that indicators are calculated
    assert 'ema_fast' in prepared_df.columns
    assert 'ema_slow' in prepared_df.columns
    assert 'rsi' in prepared_df.columns
    assert 'vol_ma' in prepared_df.columns
    assert 'pivot_low' in prepared_df.columns
    assert 'pivot_high' in prepared_df.columns
    
    # Check that values are reasonable
    assert not prepared_df['rsi'].isna().all()
    assert prepared_df['ema_fast'].iloc[-1] > 0
    assert prepared_df['ema_slow'].iloc[-1] > 0


def test_sentiment_analysis():
    """Test sentiment analysis"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    # Test sentiment analysis
    sentiment = strategy.analyze_sentiment("BTCUSDT", 0.5)
    
    assert 'news' in sentiment
    assert 'social' in sentiment
    assert 'technical' in sentiment
    assert 'combined' in sentiment
    
    # Values should be between -1 and 1
    assert -1 <= sentiment['combined'] <= 1


def test_fundamental_analysis():
    """Test fundamental analysis"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    # Test fundamental analysis
    fundamental = strategy.analyze_fundamental("BTCUSDT")
    
    assert 'score' in fundamental
    assert 'tokenomics' in fundamental
    assert 'macro' in fundamental
    assert 'positive_factors' in fundamental
    assert 'negative_factors' in fundamental
    
    # Score should be between -1 and 1
    assert -1 <= fundamental['score'] <= 1


def test_entry_signal():
    """Test entry signal generation"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    df = create_test_data(100)
    prepared_df = strategy.prepare(df)
    
    # Test with last row
    last_row = prepared_df.iloc[-1]
    signal = strategy.get_entry_signal("BTCUSDT", last_row)
    
    # Signal might be None (no trade) or a valid signal
    if signal is not None:
        assert 'direction' in signal
        assert 'signal_strength' in signal
        assert 'technical_signals' in signal
        assert 'sentiment_analysis' in signal
        assert 'fundamental_analysis' in signal
        assert 'combined_score' in signal
        assert signal['direction'] in ['long', 'short']


def test_position_sizing():
    """Test position sizing calculation"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    # Create mock signal
    mock_signal = {
        'direction': 'long',
        'entry_price': 45000,
        'signal_strength': 4,
        'combined_score': 0.7
    }
    
    # Test position sizing
    position_info = strategy.calculate_position_size(mock_signal, 10000)
    
    assert 'position_size' in position_info
    assert 'notional_value' in position_info
    assert 'margin_required' in position_info
    assert 'leverage_used' in position_info
    assert 'stop_loss_price' in position_info
    assert 'take_profit_price' in position_info
    
    # Position size should be positive
    assert position_info['position_size'] > 0
    assert position_info['notional_value'] > 0


def test_backtest_execution():
    """Test backtest execution"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    df = create_test_data(200)
    results = strategy.run_backtest(df, verbose=False)
    
    # Check that results contain expected keys
    expected_keys = [
        'trades', 'total_trades', 'wins', 'win_rate', 'final_equity',
        'total_pnl', 'avg_win', 'avg_loss', 'profit_factor',
        'max_drawdown', 'sharpe_ratio'
    ]
    
    for key in expected_keys:
        assert key in results
    
    # Check that values are reasonable
    assert results['total_trades'] >= 0
    assert 0 <= results['win_rate'] <= 1
    assert results['final_equity'] > 0
    assert results['max_drawdown'] >= 0


def test_different_trading_styles():
    """Test different trading style configurations"""
    styles = ["scalping", "day", "swing"]
    
    for style in styles:
        config = EnhancedStrategyConfig(trading_style=style)
        strategy = EnhancedCryptoStrategy(config)
        
        assert strategy.config.trading_style == style
        
        # Test that strategy can be created and run
        df = create_test_data(100)
        results = strategy.run_backtest(df, verbose=False)
        
        assert results is not None
        assert 'total_trades' in results


def test_risk_management():
    """Test risk management features"""
    config = EnhancedStrategyConfig()
    strategy = EnhancedCryptoStrategy(config)
    
    # Test with different account balances
    balances = [1000, 5000, 10000]
    
    for balance in balances:
        mock_signal = {
            'direction': 'long',
            'entry_price': 45000,
            'signal_strength': 4,
            'combined_score': 0.7
        }
        
        position_info = strategy.calculate_position_size(mock_signal, balance)
        
        # Position size should scale with account balance
        assert position_info['position_size'] > 0
        assert position_info['margin_required'] <= balance * 0.95  # Should not exceed available margin


def run_all_tests():
    """Run all tests"""
    print("Running Enhanced Strategy Tests...")
    print("=" * 40)
    
    tests = [
        test_strategy_initialization,
        test_data_preparation,
        test_sentiment_analysis,
        test_fundamental_analysis,
        test_entry_signal,
        test_position_sizing,
        test_backtest_execution,
        test_different_trading_styles,
        test_risk_management
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("=" * 40)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {len(tests)}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
