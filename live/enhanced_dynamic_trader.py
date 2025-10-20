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
from backtest.pair_discovery import PairDiscovery

# Load environment variables
load_dotenv()


class EnhancedDynamicTrader:
    """Enhanced live trader with dynamic pair discovery and volatility analysis"""
    
    def __init__(self, 
                 api_key: str = None,
                 api_secret: str = None,
                 trading_style: str = "scalping",
                 max_positions: int = 5,
                 max_daily_trades: int = 20,
                 max_daily_loss: float = 0.03,  # 3% max daily loss for aggressive strategy
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
        self.trading_style = trading_style
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss

        # Initialize API client
        self.session = HTTP(
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        # Initialize pair discovery
        self.pair_discovery = PairDiscovery(
            api_key=self.api_key,
            api_secret=self.api_secret,
            use_testnet=self.use_testnet
        )

        print(f"🚀 Enhanced Dynamic Crypto Trader initialized on {env_name}")
        print(f"📊 Trading Style: {trading_style.upper()}")
        print(f"🎯 Target: >=50% profit per trade")
        
        # Test API connection
        self._test_api_connection()
        
        # Initialize strategy with pair discovery
        self.config = self._create_strategy_config()
        self.strategy = EnhancedCryptoStrategy(self.config, self.pair_discovery)
        
        # Trading state
        self.active_positions: Dict[str, dict] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_balance = self._get_balance()
        self.last_reset = datetime.now()
        self.trade_log = []
        
        # Dynamic pair management
        self.tracked_pairs: List[str] = []
        self.pair_scores: Dict[str, float] = {}
        self.last_pair_update = None
        self.pair_update_interval = 300  # 5 minutes
        
        # Risk management
        self.last_trade_time = {}
        self.min_trade_interval = 60  # 1 minute minimum between trades per symbol
        
        print(f"✅ Strategy initialized with dynamic pair discovery")
    
    def _create_strategy_config(self) -> EnhancedStrategyConfig:
        """Create strategy configuration optimized for 50% profit target"""
        return EnhancedStrategyConfig(
            trading_style=self.trading_style,
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
                'take_profit': 0.50,  # 50% target
                'stop_loss': 0.02,    # 2% stop
                'max_holding_bars': 5,
                'risk_per_trade': 0.005,  # 0.5% risk
                'leverage': 25.0,     # High leverage
                'max_leverage': 50.0,
                'volatility_multiplier': 1.0,
                'min_volatility_threshold': 0.03,  # 3% minimum
                'max_volatility_threshold': 0.15   # 15% maximum
            })(),
            max_positions=self.max_positions,
            max_daily_trades=self.max_daily_trades,
            max_daily_loss=self.max_daily_loss
        )
    
    def _test_api_connection(self):
        """Test API connection"""
        print("🔍 Testing API connection...")
        
        try:
            server_time = self.session.get_server_time()
            if server_time.get('retCode') == 0:
                print("✅ API connection successful!")
            else:
                print(f"❌ API connection failed: {server_time.get('retMsg')}")
                return
        except Exception as e:
            print(f"❌ API connection failed with exception: {e}")
            return
        
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
                return None
                
            df = pd.DataFrame(klines['result']['list'])
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            
            return df.iloc[::-1]  # Reverse to chronological order
            
        except Exception as e:
            print(f"❌ Error fetching klines for {symbol}: {e}")
            return None
    
    def _update_tracked_pairs(self):
        """Update the list of tracked pairs based on current market conditions"""
        if (self.last_pair_update is None or 
            (datetime.now() - self.last_pair_update).seconds > self.pair_update_interval):
            
            print("🔄 Updating tracked pairs...")
            
            # Get pair recommendations for current trading style
            recommendations = self.pair_discovery.get_pair_recommendations(self.trading_style)
            
            # Update tracked pairs
            new_pairs = []
            for rec in recommendations:
                symbol = rec['symbol']
                new_pairs.append(symbol)
                self.pair_scores[symbol] = rec['volatility_score']
            
            # Remove pairs that are no longer recommended
            self.tracked_pairs = new_pairs
            
            # Update pair discovery status
            self.pair_discovery.update_pair_status()
            
            self.last_pair_update = datetime.now()
            print(f"✅ Updated tracked pairs: {len(self.tracked_pairs)} pairs")
            print(f"📊 Top pairs: {self.tracked_pairs[:10]}")
    
    def _place_order(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float) -> bool:
        """Place a new order with TP/SL"""
        try:
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
                return False
            
        except Exception as e:
            print(f"❌ Error placing order for {symbol}: {e}")
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
                'volatility_score': signal.get('volatility_metrics', {}).get('volatility_score', 0),
                'volatility_adjusted': position_info.get('volatility_adjusted', False),
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
            if signal.get('volatility_metrics'):
                print(f"   📊 Volatility: {signal['volatility_metrics'].volatility_score:.3f} | Avg Move: {signal['volatility_metrics'].avg_move_1m:.3f}")
            print("-" * 60)
        
        return success
    
    def _print_status(self):
        """Print current trading status"""
        balance = self._get_balance()
        print(f"\n📊 DYNAMIC TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Balance: ${balance:.2f} USDT | Daily PnL: ${self.daily_pnl:.2f}")
        print(f"📈 Active Positions: {len(self.active_positions)}/{self.max_positions}")
        print(f"🔄 Daily Trades: {self.daily_trades}/{self.max_daily_trades}")
        print(f"🎯 Tracked Pairs: {len(self.tracked_pairs)}")
        
        if self.active_positions:
            print("\n📋 ACTIVE POSITIONS:")
            for symbol, pos in self.active_positions.items():
                print(f"   {symbol}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} | PnL: ${pos['unrealized_pnl']:.2f}")
        
        if self.tracked_pairs:
            print(f"\n🎯 TOP TRACKED PAIRS: {self.tracked_pairs[:10]}")
    
    def run(self):
        """Main trading loop with dynamic pair discovery"""
        print(f"\n🚀 Starting Enhanced Dynamic Crypto Trading Bot")
        print(f"📊 Strategy: {self.trading_style.upper()}")
        print(f"🎯 Target: >=50% profit per trade")
        print(f"⚙️  Max Positions: {self.max_positions} | Max Daily Trades: {self.max_daily_trades}")
        print("=" * 60)
        
        try:
            while True:
                # Reset daily stats if needed
                self._reset_daily_stats()
                
                # Update tracked pairs
                self._update_tracked_pairs()
                
                # Update position status
                self._check_position_updates()
                
                # Check risk limits
                if not self._check_risk_limits():
                    print("⚠️  Risk limits reached, waiting...")
                    time.sleep(60)
                    continue
                
                # Analyze tracked pairs
                for symbol in self.tracked_pairs:
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
                    
                    # Execute trade if signal is strong enough (stricter for 50% target)
                    signal = analysis['signal']
                    if signal['signal_strength'] >= 5 and signal['combined_score'] > 0.8:
                        self._execute_trade(symbol, analysis)
                
                # Print status every 2 minutes
                if datetime.now().minute % 2 == 0:
                    self._print_status()
                
                # Small delay between scans
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Trading bot stopped by user")
            self._print_final_summary()
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final trading summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL DYNAMIC TRADING SUMMARY")
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
            
            # Calculate average profit per winning trade
            winning_trades = [t for t in self.trade_log if t.get('realized_pnl', 0) > 0]
            if winning_trades:
                avg_win = np.mean([t['realized_pnl'] for t in winning_trades])
                print(f"💰 Average Win: ${avg_win:.2f}")
            
            # Save trade log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dynamic_trade_log_{self.trading_style}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
            
            print(f"📁 Trade log saved to {filename}")


def main():
    """Main function to run the dynamic trader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Dynamic Crypto Trading Bot')
    parser.add_argument('--style', choices=['scalping', 'day', 'swing'], default='scalping', help='Trading style')
    parser.add_argument('--max-positions', type=int, default=5, help='Maximum concurrent positions')
    parser.add_argument('--max-daily-trades', type=int, default=20, help='Maximum daily trades')
    parser.add_argument('--max-daily-loss', type=float, default=0.03, help='Maximum daily loss (as decimal)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = EnhancedDynamicTrader(
        trading_style=args.style,
        max_positions=args.max_positions,
        max_daily_trades=args.max_daily_trades,
        max_daily_loss=args.max_daily_loss,
        use_testnet=args.testnet
    )
    
    trader.run()


if __name__ == '__main__':
    main()
