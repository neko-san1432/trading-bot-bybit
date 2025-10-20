import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.enhanced_strategy import EnhancedStrategyConfig, EnhancedCryptoStrategy
from ping_utils import BybitPingTester

# Load environment variables
load_dotenv()


class EnhancedLiveTrader:
    """Enhanced live trader with sentiment and fundamental analysis"""
    
    def __init__(self, 
                 api_key: str = None,
                 api_secret: str = None,
                 symbols: List[str] = None,
                 trading_style: str = "swing",
                 max_positions: int = 3,
                 max_daily_trades: int = 10,
                 max_daily_loss: float = 0.05,
                 use_testnet: bool = False):
        
        # Determine environment and get appropriate credentials
        self.use_testnet = use_testnet
        
        if self.use_testnet:
            self.api_key = api_key or os.getenv('BYBIT_API_KEY_DEMO') or os.getenv('BYBIT_API_KEY')
            self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET_DEMO') or os.getenv('BYBIT_API_SECRET')
            env_name = "TESTNET (Backtesting)"
        else:
            self.api_key = api_key or os.getenv('BYBIT_API_KEY')
            self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
            env_name = "MAINNET (Live Trading)"

        if not self.api_key or not self.api_secret:
            raise ValueError(f"API key and secret required for {env_name}")

        # Trading configuration
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
        self.trading_style = trading_style
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss

        # Initialize trading client
        self.session = HTTP(
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        print(f"🚀 Initialized Enhanced Crypto Trader on {env_name}")
        print(f"📊 Trading Style: {trading_style.upper()}")
        print(f"💰 Symbols: {', '.join(self.symbols)}")
        
        # Initialize ping tester for connectivity monitoring
        self.ping_tester = BybitPingTester(
            api_key=self.api_key,
            api_secret=self.api_secret,
            use_testnet=self.use_testnet
        )
        
        # Test API connection with comprehensive ping test
        self._test_api_connection()
        
        # Initialize strategy
        self.config = self._create_strategy_config()
        self.strategy = EnhancedCryptoStrategy(self.config)
        
        # Trading state
        self.active_positions: Dict[str, dict] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_balance = self._get_balance()
        self.last_reset = datetime.now()
        self.trade_log = []
        
        # Risk management
        self.last_trade_time = {}
        self.min_trade_interval = 300  # 5 minutes minimum between trades per symbol
        
        # Connection monitoring
        self.last_ping_time = datetime.now()
        self.ping_interval = 300  # 5 minutes between ping tests
        self.connection_failures = 0
        self.max_connection_failures = 3
        
        print(f"✅ Strategy initialized with {len(self.symbols)} symbols")
    
    def _create_strategy_config(self) -> EnhancedStrategyConfig:
        """Create strategy configuration based on trading style"""
        if self.trading_style == "scalping":
            return EnhancedStrategyConfig(
                trading_style="scalping",
                technical=type('TechnicalConfig', (), {
                    'ema_fast': 5,
                    'ema_slow': 13,
                    'ema_trend': 21,
                    'rsi_period': 14,
                    'rsi_oversold': 25,
                    'rsi_overbought': 75,
                    'vol_ma': 10,
                    'vol_spike_threshold': 2.0,
                    'lookback_pivot': 3,
                    'take_profit': 0.005,  # 0.5%
                    'stop_loss': 0.003,    # 0.3%
                    'max_holding_bars': 3,
                    'risk_per_trade': 0.005,  # 0.5%
                    'leverage': 10.0
                })(),
                max_positions=self.max_positions,
                max_daily_trades=self.max_daily_trades,
                max_daily_loss=self.max_daily_loss
            )
        elif self.trading_style == "day":
            return EnhancedStrategyConfig(
                trading_style="day",
                technical=type('TechnicalConfig', (), {
                    'ema_fast': 9,
                    'ema_slow': 20,
                    'ema_trend': 50,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'vol_ma': 20,
                    'vol_spike_threshold': 1.5,
                    'lookback_pivot': 5,
                    'take_profit': 0.01,   # 1%
                    'stop_loss': 0.007,    # 0.7%
                    'max_holding_bars': 10,
                    'risk_per_trade': 0.01,  # 1%
                    'leverage': 7.0
                })(),
                max_positions=self.max_positions,
                max_daily_trades=self.max_daily_trades,
                max_daily_loss=self.max_daily_loss
            )
        else:  # swing
            return EnhancedStrategyConfig(
                trading_style="swing",
                technical=type('TechnicalConfig', (), {
                    'ema_fast': 9,
                    'ema_slow': 20,
                    'ema_trend': 50,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'vol_ma': 20,
                    'vol_spike_threshold': 1.5,
                    'lookback_pivot': 5,
                    'take_profit': 0.02,   # 2%
                    'stop_loss': 0.015,    # 1.5%
                    'max_holding_bars': 20,
                    'risk_per_trade': 0.015,  # 1.5%
                    'leverage': 5.0
                })(),
                max_positions=self.max_positions,
                max_daily_trades=self.max_daily_trades,
                max_daily_loss=self.max_daily_loss
            )
    
    def _test_api_connection(self):
        """Test API connection with comprehensive ping test"""
        print("🔍 Running comprehensive connectivity test...")
        
        # Run comprehensive ping test
        ping_results = self.ping_tester.comprehensive_ping_test(include_auth=True)
        
        if ping_results['overall_status'] == 'critical':
            print("❌ Critical connectivity issues detected. Trading disabled.")
            raise ConnectionError("Critical connectivity issues - cannot proceed with trading")
        elif ping_results['overall_status'] == 'warning':
            print("⚠️  Warning: High latency detected. Trading may be affected.")
        else:
            print("✅ All connectivity tests passed!")
        
        # Test balance endpoint
        balance = self._get_balance()
        if balance > 0:
            print(f"✅ Authentication successful! Balance: {balance:.2f} USDT")
        else:
            print("❌ Authentication failed - check your API credentials")
    
    def _get_balance(self) -> float:
        """Get current USDT balance"""
        try:
            wallet = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if wallet.get('retCode') == 0:
                return float(wallet['result']['list'][0]['totalWalletBalance'])
            else:
                print(f"❌ Error getting balance: {wallet.get('retMsg')}")
                return 0.0
        except Exception as e:
            print(f"❌ Exception getting balance: {e}")
            return 0.0
    
    def _get_klines(self, symbol: str, interval: str = "1", limit: int = 100) -> Optional[pd.DataFrame]:
        """Get recent klines for analysis"""
        try:
            klines = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if klines.get('retCode') != 0:
                print(f"❌ Error fetching klines for {symbol}: {klines.get('retMsg')}")
                return None
                
            df = pd.DataFrame(klines['result']['list'])
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            
            return df.iloc[::-1]  # Reverse to get chronological order
            
        except Exception as e:
            print(f"❌ Error fetching klines for {symbol}: {e}")
            return None
    
    def _place_order(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float) -> bool:
        """Place a new order with TP/SL with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check connection health before placing order
                if not self._check_connection_health():
                    print("❌ Connection health check failed, skipping order")
                    return False
                
                # Set leverage
                self.session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(self.config.technical.leverage),
                    sellLeverage=str(self.config.technical.leverage)
                )
                
                # Place the main order
                order = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(qty),
                    isLeverage=1,
                    stopLoss=str(stop_loss),
                    takeProfit=str(take_profit),
                    timeInForce="GoodTillCancel",
                    reduce_only=False,
                    close_on_trigger=False
                )
                
                if order.get('retCode') == 0:
                    print(f"✅ Order placed successfully for {symbol}")
                    return True
                else:
                    print(f"❌ Order failed for {symbol}: {order.get('retMsg')}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return False
                
            except Exception as e:
                print(f"❌ Error placing order for {symbol}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return False
        
        return False
    
    def _check_position_updates(self):
        """Update status of open positions"""
        try:
            positions = self.session.get_positions(
                category="linear",
                settleCoin="USDT",
                limit=50
            )
            
            current_symbols = set()
            for pos in positions['result']['list']:
                symbol = pos['symbol']
                current_symbols.add(symbol)
                
                if float(pos['size']) == 0:  # Position closed
                    if symbol in self.active_positions:
                        realized_pnl = float(pos['cumRealisedPnl'])
                        self.daily_pnl += realized_pnl
                        
                        # Log trade
                        trade_info = self.active_positions[symbol].copy()
                        trade_info.update({
                            'exit_time': datetime.now(),
                            'realized_pnl': realized_pnl,
                            'exit_reason': 'tp_sl_hit'
                        })
                        self.trade_log.append(trade_info)
                        
                        print(f"📊 Position closed for {symbol} - PnL: {realized_pnl:.2f} USDT")
                        del self.active_positions[symbol]
                
                else:  # Active position
                    position_size = float(pos['size'])
                    entry_price = float(pos['avgPrice'])
                    notional_value = position_size * entry_price
                    margin_used = float(pos['positionBalance']) if 'positionBalance' in pos else 0
                    leverage_used = notional_value / margin_used if margin_used > 0 else 0
                    
                    self.active_positions[symbol] = {
                        'size': position_size,
                        'entry_price': entry_price,
                        'unrealized_pnl': float(pos['unrealisedPnl']),
                        'notional_value': notional_value,
                        'margin_used': margin_used,
                        'leverage_used': leverage_used,
                        'entry_time': datetime.now()
                    }
            
            # Remove closed positions
            for symbol in list(self.active_positions.keys()):
                if symbol not in current_symbols:
                    del self.active_positions[symbol]
                    
        except Exception as e:
            print(f"❌ Error checking positions: {e}")
    
    def _reset_daily_stats(self):
        """Reset daily trading statistics"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = now
            self.start_balance = self._get_balance()
            print(f"📅 Daily stats reset. New balance: {self.start_balance:.2f} USDT")
    
    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits to trade"""
        # Check max positions
        if len(self.active_positions) >= self.max_positions:
            return False
            
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
            
        # Check max loss
        if self.daily_pnl < -(self.start_balance * self.max_daily_loss):
            return False
            
        return True
    
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade a specific symbol (rate limiting)"""
        now = datetime.now()
        if symbol in self.last_trade_time:
            time_since_last = (now - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < self.min_trade_interval:
                return False
        return True
    
    def _check_connection_health(self) -> bool:
        """Check connection health and perform ping test if needed"""
        now = datetime.now()
        time_since_last_ping = (now - self.last_ping_time).total_seconds()
        
        # Perform ping test if interval has passed
        if time_since_last_ping >= self.ping_interval:
            print("🔍 Performing connection health check...")
            
            # Quick ping test
            if not self.ping_tester.quick_ping_test():
                self.connection_failures += 1
                print(f"⚠️  Connection health check failed ({self.connection_failures}/{self.max_connection_failures})")
                
                if self.connection_failures >= self.max_connection_failures:
                    print("❌ Too many connection failures. Stopping trading.")
                    return False
            else:
                # Reset failure counter on successful ping
                if self.connection_failures > 0:
                    print("✅ Connection restored!")
                    self.connection_failures = 0
            
            self.last_ping_time = now
        
        return True
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol for trading opportunities"""
        # Get recent data
        df = self._get_klines(symbol)
        if df is None or len(df) < 50:
            return None
        
        # Prepare data with indicators
        df = self.strategy.prepare(df)
        
        # Get entry signal
        last_row = df.iloc[-1]
        signal = self.strategy.get_entry_signal(symbol, last_row)
        
        if signal is None:
            return None
        
        # Calculate position size
        balance = self._get_balance()
        position_info = self.strategy.calculate_position_size(signal, balance)
        
        return {
            'signal': signal,
            'position_info': position_info,
            'current_price': last_row['close'],
            'data': last_row
        }
    
    def _execute_trade(self, symbol: str, analysis: Dict) -> bool:
        """Execute a trade based on analysis"""
        signal = analysis['signal']
        position_info = analysis['position_info']
        current_price = analysis['current_price']
        
        # Calculate order parameters
        direction = signal['direction']
        side = "Buy" if direction == 'long' else "Sell"
        qty = position_info['position_size']
        stop_loss = position_info['stop_loss_price']
        take_profit = position_info['take_profit_price']
        
        # Place order
        success = self._place_order(symbol, side, qty, stop_loss, take_profit)
        
        if success:
            # Update tracking
            self.daily_trades += 1
            self.last_trade_time[symbol] = datetime.now()
            
            # Log trade
            trade_info = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'quantity': qty,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'leverage': position_info['leverage_used'],
                'notional_value': position_info['notional_value'],
                'signal_strength': signal['signal_strength'],
                'combined_score': signal['combined_score'],
                'technical_signals': signal['technical_signals'],
                'sentiment_score': signal['sentiment_analysis']['combined'],
                'fundamental_score': signal['fundamental_analysis']['score'],
                'entry_time': datetime.now()
            }
            
            self.trade_log.append(trade_info)
            
            print(f"🎯 TRADE EXECUTED: {direction.upper()} {symbol}")
            print(f"   💰 Price: ${current_price:.2f} | Size: {qty:.4f}")
            print(f"   📊 TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")
            print(f"   🔥 Leverage: {position_info['leverage_used']:.1f}x | Notional: ${position_info['notional_value']:.2f}")
            print(f"   📈 Signal Strength: {signal['signal_strength']} | Score: {signal['combined_score']:.3f}")
            print(f"   🎪 Signals: {', '.join(signal['technical_signals'])}")
            print(f"   💭 Sentiment: {signal['sentiment_analysis']['combined']:.3f} | Fundamental: {signal['fundamental_analysis']['score']:.3f}")
            print("-" * 60)
        
        return success
    
    def _print_status(self):
        """Print current trading status"""
        balance = self._get_balance()
        print(f"\n📊 TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Balance: ${balance:.2f} USDT | Daily PnL: ${self.daily_pnl:.2f}")
        print(f"📈 Active Positions: {len(self.active_positions)}/{self.max_positions}")
        print(f"🔄 Daily Trades: {self.daily_trades}/{self.max_daily_trades}")
        
        if self.active_positions:
            print("\n📋 ACTIVE POSITIONS:")
            for symbol, pos in self.active_positions.items():
                print(f"   {symbol}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} | PnL: ${pos['unrealized_pnl']:.2f}")
    
    def run(self):
        """Main trading loop"""
        print(f"\n🚀 Starting Enhanced Crypto Trading Bot")
        print(f"📊 Strategy: {self.trading_style.upper()}")
        print(f"💰 Symbols: {', '.join(self.symbols)}")
        print(f"⚙️  Max Positions: {self.max_positions} | Max Daily Trades: {self.max_daily_trades}")
        print("=" * 60)
        
        try:
            while True:
                # Check connection health first
                if not self._check_connection_health():
                    print("❌ Connection health check failed. Stopping trading.")
                    break
                
                # Reset daily stats if needed
                self._reset_daily_stats()
                
                # Update position status
                self._check_position_updates()
                
                # Check risk limits
                if not self._check_risk_limits():
                    print("⚠️  Risk limits reached, waiting...")
                    time.sleep(60)
                    continue
                
                # Analyze each symbol
                for symbol in self.symbols:
                    # Skip if we already have a position
                    if symbol in self.active_positions:
                        continue
                    
                    # Check rate limiting
                    if not self._can_trade_symbol(symbol):
                        continue
                    
                    # Analyze symbol
                    analysis = self._analyze_symbol(symbol)
                    if analysis is None:
                        continue
                    
                    # Execute trade if signal is strong enough
                    signal = analysis['signal']
                    if signal['signal_strength'] >= 3 and signal['combined_score'] > 0.5:
                        self._execute_trade(symbol, analysis)
                
                # Print status every 5 minutes
                if datetime.now().minute % 5 == 0:
                    self._print_status()
                
                # Small delay between scans
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Trading bot stopped by user")
            self._print_final_summary()
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final trading summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL TRADING SUMMARY")
        print("=" * 60)
        
        balance = self._get_balance()
        total_pnl = balance - self.start_balance
        
        print(f"💰 Starting Balance: ${self.start_balance:.2f}")
        print(f"💰 Final Balance: ${balance:.2f}")
        print(f"📈 Total PnL: ${total_pnl:.2f} ({total_pnl/self.start_balance:.2%})")
        print(f"🔄 Total Trades: {len(self.trade_log)}")
        
        if self.trade_log:
            wins = sum(1 for t in self.trade_log if t.get('realized_pnl', 0) > 0)
            win_rate = wins / len(self.trade_log)
            print(f"🎯 Win Rate: {win_rate:.1%}")
            
            # Save trade log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_log_{self.trading_style}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
            
            print(f"📁 Trade log saved to {filename}")


def main():
    """Main function to run the enhanced trader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Crypto Trading Bot')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Trading symbols')
    parser.add_argument('--style', choices=['scalping', 'day', 'swing'], default='swing', help='Trading style')
    parser.add_argument('--max-positions', type=int, default=3, help='Maximum concurrent positions')
    parser.add_argument('--max-daily-trades', type=int, default=10, help='Maximum daily trades')
    parser.add_argument('--max-daily-loss', type=float, default=0.05, help='Maximum daily loss (as decimal)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = EnhancedLiveTrader(
        symbols=args.symbols,
        trading_style=args.style,
        max_positions=args.max_positions,
        max_daily_trades=args.max_daily_trades,
        max_daily_loss=args.max_daily_loss,
        use_testnet=args.testnet
    )
    
    trader.run()


if __name__ == '__main__':
    main()
