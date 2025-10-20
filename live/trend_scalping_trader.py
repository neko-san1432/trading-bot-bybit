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
from backtest.trend_scalping_strategy import TrendScalpingConfig, TrendScalpingStrategy
from backtest.pair_discovery import PairDiscovery

# Load environment variables
load_dotenv()


class TrendScalpingTrader:
    """Live trader for trend scalping strategy with dynamic pair discovery"""
    
    def __init__(self, 
                 api_key: str = None,
                 api_secret: str = None,
                 max_positions: int = 8,
                 max_daily_trades: int = 50,
                 max_daily_loss: float = 0.05,  # 5% max daily loss
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

        # Trading configuration - ONE PAIR AT A TIME
        self.max_positions = 1  # Force single position trading
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

        print(f"🚀 Trend Scalping Trader initialized on {env_name}")
        print(f"📊 Strategy: TREND SCALPING")
        print(f"🎯 Target: 0.5-5% profit per trade with ATR-based stops")
        
        # Test API connection
        self._test_api_connection()
        
        # Initialize strategy with pair discovery
        self.config = self._create_strategy_config()
        self.strategy = TrendScalpingStrategy(self.config, self.pair_discovery)
        
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
        self.min_trade_interval = 5   # 5 seconds minimum between trades (very aggressive)
        self.current_trading_pair = None  # Track current pair
        
        print(f"✅ Strategy initialized with dynamic pair discovery")
    
    def _create_strategy_config(self) -> TrendScalpingConfig:
        """Create trend scalping configuration"""
        return TrendScalpingConfig(
            # Trend detection
            adx_period=14,
            adx_threshold=15.0,  # Lowered for more opportunities
            ema_fast=5,
            ema_medium=13,
            ema_slow=21,
            ema_trend=50,
            ema_long=200,
            
            # Entry signal preferences
            pullback_weight=0.4,
            breakout_weight=0.3,
            momentum_weight=0.3,
            
            # Dynamic profit targets
            base_take_profit=0.015,  # 1.5%
            min_take_profit=0.005,   # 0.5%
            max_take_profit=0.05,    # 5%
            volatility_multiplier=0.5,
            trend_multiplier=0.3,
            
            # ATR-based stops
            atr_period=14,
            atr_multiplier=2.0,
            min_atr_multiplier=1.5,
            max_atr_multiplier=2.5,
            trailing_stop_trigger=0.5,
            
            # Risk management
            risk_per_trade=0.003,    # 0.3%
            leverage=30.0,           # Base leverage
            max_leverage=50.0,
            min_leverage=20.0,
            
            # Volatility filters - DISABLED for continuous trading
            min_volatility=0.0,      # No minimum volatility
            max_volatility=1.0,      # No maximum volatility
            
            # Trade management
            max_holding_bars=10,     # 10 minutes
            min_holding_bars=1,      # 1 minute
            
            # Volume confirmation
            volume_ma_period=20,
            volume_spike_threshold=1.1  # Lowered for more opportunities
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
            
            print("🔄 Updating tracked pairs for trend scalping...")
            
            # Get pair recommendations for scalping
            recommendations = self.pair_discovery.get_pair_recommendations("scalping")
            
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
    
    def _place_order(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, leverage: float) -> bool:
        """Place a new order with TP/SL"""
        try:
            # Set leverage
            self.session.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
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
    
    def _get_fallback_pairs(self) -> List[str]:
        """Get fallback trading pairs if pair discovery fails - EXPANDED LIST"""
        return [
            # Major cryptocurrencies
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
            'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT',
            'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'ALGOUSDT',
            'VETUSDT', 'ICPUSDT', 'THETAUSDT', 'EOSUSDT', 'AAVEUSDT',
            'MKRUSDT', 'COMPUSDT', 'YFIUSDT', 'SNXUSDT', 'SUSHIUSDT',
            # Additional popular pairs
            'NEARUSDT', 'FTMUSDT', 'ONEUSDT', 'HBARUSDT', 'EGLDUSDT',
            'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'GALAUSDT', 'ENJUSDT',
            'CHZUSDT', 'FLOWUSDT', 'ROSEUSDT', 'IOTAUSDT', 'ZILUSDT',
            'BATUSDT', 'ZECUSDT', 'DASHUSDT', 'NEOUSDT', 'QTUMUSDT',
            'ONTUSDT', 'ICXUSDT', 'NANOUSDT', 'DGBUSDT', 'SCUSDT',
            'ZENUSDT', 'RVNUSDT', 'DCRUSDT', 'LSKUSDT', 'WAVESUSDT',
            'OMGUSDT', 'KNCUSDT', 'REPUSDT', 'ZRXUSDT', 'STORJUSDT',
            'CVCUSDT', 'GNTUSDT', 'FUNUSDT', 'REQUSDT', 'MCOUSDT',
            'DENTUSDT', 'MTHUSDT', 'ADXUSDT', 'BNTUSDT', 'LRCUSDT',
            'RDNUSDT', 'RLCUSDT', 'TNTUSDT', 'FUELUSDT', 'CLOAKUSDT',
            'GUPUSDT', 'LUNUSDT', 'BNGUSDT', 'REPUSDT', 'DGDUSDT',
            'WINGSUSDT', 'KMDUSDT', 'SALTUSDT', 'XZCUSDT', 'QSPUSDT',
            'MDAUSDT', 'MTLUSDT', 'FCTUSDT', 'IOCUSDT', 'CFIUSDT',
            'DNTUSDT', 'WTCUSDT', 'LENDUSDT', 'QASHUSDT', 'MDAUSDT',
            'MTLUSDT', 'FCTUSDT', 'IOCUSDT', 'CFIUSDT', 'DNTUSDT',
            'WTCUSDT', 'LENDUSDT', 'QASHUSDT', 'MDAUSDT', 'MTLUSDT'
        ]
    
    def _analyze_symbol(self, symbol: str, verbose: bool = False) -> Optional[Dict]:
        """Analyze a symbol for trend scalping opportunities"""
        # Get recent data
        df = self._get_klines(symbol)
        if df is None or len(df) < 100:  # Need more data for trend analysis
            if verbose:
                print(f"   ❌ {symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
            return None
        
        # Prepare data with indicators
        df = self.strategy.prepare(df)
        
        # Get entry signal with verbose logging
        last_row = df.iloc[-1]
        signal = self.strategy.get_entry_signal(symbol, last_row, verbose=verbose)
        
        if signal is None:
            if verbose:
                print(f"   ❌ {symbol}: No signal found")
            return None
        
        # Calculate position size
        balance = self._get_balance()
        position_info = self.strategy.calculate_position_size(signal, balance)
        
        if verbose:
            print(f"   ✅ {symbol}: Signal found! Strength: {signal['signal_strength']}")
        
        return {
            'signal': signal,
            'position_info': position_info,
            'current_price': last_row['close'],
            'data': last_row
        }
    
    def _execute_trade(self, symbol: str, analysis: Dict) -> bool:
        """Execute a trend scalping trade"""
        signal = analysis['signal']
        position_info = analysis['position_info']
        current_price = analysis['current_price']
        
        # Calculate order parameters
        direction = signal['direction']
        side = "Buy" if direction == 'long' else "Sell"
        qty = position_info['position_size']
        stop_loss = position_info['stop_loss_price']
        take_profit = position_info['take_profit_price']
        leverage = signal['leverage']
        
        # Place order
        success = self._place_order(symbol, side, qty, stop_loss, take_profit, leverage)
        
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
                'leverage': leverage,
                'notional_value': position_info['notional_value'],
                'signal_strength': signal['signal_strength'],
                'entry_signals': signal['entry_signals'],
                'trend_direction': signal['trend_direction'],
                'adx': signal['adx'],
                'atr': signal['atr'],
                'volatility': signal.get('volatility', 0),
                'target_pct': signal['target_pct'],
                'stop_pct': position_info['stop_pct'],
                'risk_reward_ratio': position_info['risk_reward_ratio'],
                'entry_time': datetime.now()
            }
            
            self.trade_log.append(trade_info)
            
            print(f"🎯 TREND SCALP TRADE: {direction.upper()} {symbol}")
            print(f"   💰 Price: ${current_price:.2f} | Size: {qty:.4f}")
            print(f"   📊 TP: ${take_profit:.2f} ({signal['target_pct']:.1%}) | SL: ${stop_loss:.2f} ({position_info['stop_pct']:.1%})")
            print(f"   🔥 Leverage: {leverage:.1f}x | Notional: ${position_info['notional_value']:.2f}")
            print(f"   📈 Signal Strength: {signal['signal_strength']} | Trend: {signal['trend_direction']}")
            print(f"   🎪 Entry Signals: {', '.join(signal['entry_signals'])}")
            print(f"   📊 ADX: {signal['adx']:.1f} | ATR: {signal['atr']:.4f} | R:R: {position_info['risk_reward_ratio']:.1f}")
            
            # Show leverage-based target info
            if leverage >= 50:
                target_range = "1-2%"
            elif leverage >= 25:
                target_range = "2-4%"
            elif leverage >= 20:
                target_range = "2.5-5%"
            elif leverage >= 12.5:
                target_range = "6-12%"
            else:
                target_range = "1-5%"
            print(f"   🎯 Leverage Target Range: {target_range}")
            print("-" * 60)
        
        return success
    
    def _print_status(self):
        """Print current trading status"""
        balance = self._get_balance()
        print(f"\n📊 TREND SCALPING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Balance: ${balance:.2f} USDT | Daily PnL: ${self.daily_pnl:.2f}")
        print(f"📈 Active Positions: {len(self.active_positions)}/{self.max_positions} (ONE PAIR AT A TIME)")
        print(f"🔄 Daily Trades: {self.daily_trades}/{self.max_daily_trades}")
        print(f"🎯 Tracked Pairs: {len(self.tracked_pairs)}")
        if self.current_trading_pair:
            print(f"🎪 Current Pair: {self.current_trading_pair}")
        
        if self.active_positions:
            print("\n📋 ACTIVE POSITIONS:")
            for symbol, pos in self.active_positions.items():
                print(f"   {symbol}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} | PnL: ${pos['unrealized_pnl']:.2f}")
        
        if self.tracked_pairs:
            print(f"\n🎯 TOP TRACKED PAIRS: {self.tracked_pairs[:10]}")
    
    def run(self):
        """Main trading loop for trend scalping"""
        print(f"\n🚀 Starting Trend Scalping Bot")
        print(f"📊 Strategy: TREND SCALPING")
        print(f"🎯 Target: 0.5-5% profit per trade with ATR-based stops")
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
                
                # Focus on one pair at a time for continuous trading
                if not self.active_positions:
                    # No active position - find the best pair to trade
                    print(f"\n🔍 SCANNING FOR OPPORTUNITIES - {len(self.tracked_pairs)} pairs to check...")
                    
                    best_pair = None
                    best_signal = None
                    best_analysis = None
                    
                    # Use all tracked pairs, or fallback to common pairs
                    pairs_to_check = self.tracked_pairs if self.tracked_pairs else self._get_fallback_pairs()
                    total_pairs = len(pairs_to_check)
                    
                    for i, symbol in enumerate(pairs_to_check):
                        print(f"\n[{i+1}/{total_pairs}] Checking {symbol}...")
                        
                        # Check rate limiting
                        if not self._can_trade_symbol(symbol):
                            print(f"   ⏰ Rate limited - waiting...")
                            continue
                        
                        # Analyze symbol with verbose logging
                        analysis = self._analyze_symbol(symbol, verbose=True)
                        if analysis is None:
                            continue
                        
                        signal = analysis['signal']
                        # Take any signal (very aggressive)
                        if signal['signal_strength'] >= 1:
                            if best_signal is None or signal['signal_strength'] > best_signal['signal_strength']:
                                best_pair = symbol
                                best_signal = signal
                                best_analysis = analysis
                                print(f"   🏆 NEW BEST SIGNAL! Strength: {signal['signal_strength']}")
                        else:
                            print(f"   ⚠️  Signal too weak: {signal['signal_strength']} < 1")
                    
                    # Execute trade on best pair found
                    if best_pair and best_analysis:
                        print(f"\n🎯 EXECUTING TRADE ON {best_pair}!")
                        self.current_trading_pair = best_pair
                        self._execute_trade(best_pair, best_analysis)
                    else:
                        print(f"\n❌ No suitable signals found in {total_pairs} pairs")
                        print(f"   Will retry in 5 seconds...")
                else:
                    # We have a position - monitor it and prepare for next trade
                    current_symbol = list(self.active_positions.keys())[0]
                    self.current_trading_pair = current_symbol
                
                # Print status every 1 minute
                if datetime.now().minute % 1 == 0:
                    self._print_status()
                
                # Small delay between scans (scalping needs fast execution)
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Trend scalping bot stopped by user")
            self._print_final_summary()
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final trading summary"""
        print("\n" + "=" * 60)
        print("📊 FINAL TREND SCALPING SUMMARY")
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
            
            # Calculate average metrics
            avg_hold_time = np.mean([(t.get('exit_time', datetime.now()) - t['entry_time']).total_seconds() / 60 
                                   for t in self.trade_log if 'exit_time' in t]) if self.trade_log else 0
            avg_risk_reward = np.mean([t.get('risk_reward_ratio', 0) for t in self.trade_log]) if self.trade_log else 0
            
            print(f"⏱️  Average Hold Time: {avg_hold_time:.1f} minutes")
            print(f"📊 Average Risk-Reward: {avg_risk_reward:.1f}")
            
            # Save trade log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_scalping_log_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
            
            print(f"📁 Trade log saved to {filename}")


def main():
    """Main function to run the trend scalping trader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trend Scalping Trading Bot')
    parser.add_argument('--max-positions', type=int, default=8, help='Maximum concurrent positions')
    parser.add_argument('--max-daily-trades', type=int, default=50, help='Maximum daily trades')
    parser.add_argument('--max-daily-loss', type=float, default=0.05, help='Maximum daily loss (as decimal)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = TrendScalpingTrader(
        max_positions=args.max_positions,
        max_daily_trades=args.max_daily_trades,
        max_daily_loss=args.max_daily_loss,
        use_testnet=args.testnet
    )
    
    trader.run()


if __name__ == '__main__':
    main()
