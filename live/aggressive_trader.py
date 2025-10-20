#!/usr/bin/env python3
"""
Aggressive Live Trader - High Frequency, High Risk, High Reward
Integrates with existing system but with aggressive settings
"""

import os
import time
import math
import logging
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pybit.unified_trading import HTTP
from backtest.trend_scalping_strategy_clean import TrendScalpingStrategy, TrendScalpingConfig
from backtest.pair_discovery import PairDiscovery
from aggressive_strategy import AggressiveStrategy, AggressiveConfig, create_aggressive_config

# Import GPU utilities
try:
    from gpu_utils import initialize_gpu_for_trading, monitor_gpu_usage, cleanup_gpu_memory
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Test CUDA safely and disable if there are errors
def test_cuda_safely():
    """Test CUDA and disable if there are errors"""
    try:
        import cupy as cp
        # Test basic operation
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        return True
    except Exception:
        return False

# Disable CUDA if it's not working (keep a module-level constant flag)
GPU_UTILS_AVAILABLE = GPU_UTILS_AVAILABLE and test_cuda_safely()

class AggressiveTrader:
    """Aggressive high-frequency trader with multiple strategies"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, demo: bool = False):
        # Environment setup
        self.demo = demo
        env_name = "DEMO" if demo else "MAINNET"
        
        # API credentials
        if api_key and api_secret:
            self.api_key = api_key
            self.api_secret = api_secret
        else:
            if demo:
                self.api_key = os.getenv('BYBIT_API_KEY_DEMO')
                self.api_secret = os.getenv('BYBIT_API_SECRET_DEMO')
            else:
                self.api_key = os.getenv('BYBIT_API_KEY_REAL')
                self.api_secret = os.getenv('BYBIT_API_SECRET_REAL')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(f"API key and secret required for {env_name}")
        
        # Initialize API client with correct endpoints
        if demo:
            # Use demo API endpoint - https://api-demo.bybit.com
            self.client = HTTP(
                demo=True,  # Demo uses demo=True
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        else:
            # Use mainnet API endpoint
            self.client = HTTP(
                testnet=False,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        
        # Aggressive configuration
        self.config = create_aggressive_config()
        self.strategy = AggressiveStrategy(self.config)
        
        # Trading state
        self.active_positions = {}
        self.position_history = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_daily_trades = 200  # Increased for aggressive trading
        
        # UI settings
        self.quiet = os.getenv('BOT_QUIET', '0') in ('1', 'true', 'yes')  # Default to verbose for aggressive
        self.ui = os.getenv('BOT_UI', '1') in ('1', 'true', 'yes')
        
        # Logging setup
        self.log_file = os.getenv('BOT_LOG_FILE', 'aggressive_log.txt')
        self.logger = self._setup_logging()
        
        # GPU setup
        self.gpu_utils_available = GPU_UTILS_AVAILABLE
        if self.gpu_utils_available:
            try:
                if initialize_gpu_for_trading():
                    self.logger.info("✅ GPU initialized for aggressive trading")
                else:
                    self.logger.warning("⚠️ GPU initialization failed - using CPU")
            except Exception as e:
                self.logger.error(f"⚠️ GPU initialization error: {e}")
                self.gpu_utils_available = False
        
        # Pair discovery
        self.pair_discovery = PairDiscovery()
        
        # Trading parameters
        self.scan_interval_sec = 1  # 1 second scan interval for aggressive trading
        self.scan_target_count = 50  # Scan more pairs
        self.max_concurrent_trades = 5
        
        # Performance tracking
        self.start_time = time.time()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Risk management
        self.max_drawdown = 0.15  # 15% max drawdown
        self.daily_loss_limit = 0.10  # 10% daily loss limit
        self.emergency_stop = False
        
        print(f"Aggressive Trader Initialized - {env_name}")
        print(f"API Endpoint: {'https://api-demo.bybit.com (Demo)' if demo else 'https://api.bybit.com (Mainnet)'}")
        print(f"Account Balance: ${self.config.account_balance:,.2f}")
        print(f"Max Positions: {self.config.max_positions}")
        print(f"Leverage: {self.config.leverage}x")
        print(f"Risk per Trade: {self.config.risk_per_trade*100}%")
        print(f"Max Total Risk: {self.config.max_total_risk*100}%")
    
    def _setup_logging(self):
        """Setup logging for aggressive trading"""
        logger = logging.getLogger('aggressive_trader')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for aggressive trading (more verbose)
        if not self.quiet:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger
    
    def run(self):
        """Main aggressive trading loop"""
        self.logger.info("🚀 Starting aggressive trading bot...")
        
        # Check initial balance
        self._check_balance()
        
        while not self.emergency_stop:
            try:
                loop_start = time.time()
                
                # Check daily limits
                if self._check_daily_limits():
                    self.logger.warning("⚠️ Daily limits reached - stopping trading")
                    break
                
                # Monitor existing positions
                self._monitor_positions()
                
                # Get trading opportunities
                opportunities = self._scan_opportunities()
                
                # Execute trades
                if opportunities:
                    self._execute_aggressive_trades(opportunities)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for scan interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.scan_interval_sec - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Trading stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {e}")
                time.sleep(5)  # Wait before retrying
        
        # Cleanup
        self._cleanup()
    
    def _check_balance(self):
        """Check account balance and risk limits"""
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] == 0:
                balance = float(response['result']['list'][0]['totalWalletBalance'])
                self.config.account_balance = balance
                self.logger.info(f"💰 Account Balance: ${balance:,.2f}")
                
                # Check if balance is sufficient for aggressive trading
                min_balance = 1000  # Minimum $1000 for aggressive trading
                if balance < min_balance:
                    self.logger.warning(f"⚠️ Low balance: ${balance:,.2f} < ${min_balance:,.2f}")
                    self.emergency_stop = True
            else:
                self.logger.error(f"❌ Failed to get balance: {response['retMsg']}")
                self.emergency_stop = True
        except Exception as e:
            self.logger.error(f"❌ Balance check error: {e}")
            self.emergency_stop = True
    
    def _check_daily_limits(self) -> bool:
        """Check if daily limits are reached"""
        # Check daily trade limit
        if self.trades_today >= self.max_daily_trades:
            return True
        
        # Check daily loss limit
        if self.daily_pnl < -self.config.account_balance * self.daily_loss_limit:
            return True
        
        # Check max drawdown
        if self.total_pnl < -self.config.account_balance * self.max_drawdown:
            return True
        
        return False
    
    def _scan_opportunities(self) -> List[Dict]:
        """Scan for aggressive trading opportunities"""
        try:
            # Get trading pairs
            pairs = self.pair_discovery.get_trending_pairs(limit=50)
            
            opportunities = []
            
            # Scan pairs in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_pair = {
                    executor.submit(self._analyze_pair, pair): pair 
                    for pair in pairs[:self.scan_target_count]
                }
                
                for future in as_completed(future_to_pair):
                    try:
                        opportunity = future.result()
                        if opportunity:
                            opportunities.append(opportunity)
                    except Exception as e:
                        self.logger.error(f"❌ Analysis error: {e}")
            
            # Sort by signal strength
            opportunities.sort(key=lambda x: x.get('strength', 0), reverse=True)
            
            # Limit to max concurrent trades
            return opportunities[:self.max_concurrent_trades]
            
        except Exception as e:
            self.logger.error(f"❌ Opportunity scan error: {e}")
            return []
    
    def _analyze_pair(self, symbol: str) -> Optional[Dict]:
        """Analyze a single pair for trading opportunities"""
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
                    data_dict[tf] = df
            
            if not data_dict:
                return None
            
            # Analyze with aggressive strategy
            analysis = self.strategy.analyze_multi_timeframe(symbol, data_dict)
            
            # Get entry signal
            signal = self.strategy.get_aggressive_entry_signal(symbol, analysis)
            
            if signal:
                signal['timestamp'] = datetime.now()
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Pair analysis error for {symbol}: {e}")
            return None
    
    def _get_klines(self, symbol: str, interval: str, limit: int) -> List:
        """Get kline data for symbol"""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            if response['retCode'] == 0:
                return response['result']['list']
            return []
        except Exception as e:
            self.logger.error(f"❌ Kline error for {symbol}: {e}")
            return []
    
    def _execute_aggressive_trades(self, opportunities: List[Dict]):
        """Execute aggressive trades"""
        for opportunity in opportunities:
            try:
                # Check if we can open new position
                if len(self.active_positions) >= self.config.max_positions:
                    break
                
                # Execute trade
                success = self._place_aggressive_order(opportunity)
                
                if success:
                    self.trades_today += 1
                    self.total_trades += 1
                    
                    # Log trade
                    self.logger.info(
                        f"🚀 AGGRESSIVE TRADE [{datetime.now().strftime('%H:%M:%S')}] - "
                        f"{opportunity['symbol']} {opportunity['side'].upper()} "
                        f"{opportunity['position_size']:.6f} @ ${opportunity['entry_price']:.4f} "
                        f"(Strength: {opportunity['strength']:.1f})"
                    )
                    
                    # Track position
                    self.active_positions[opportunity['symbol']] = {
                        'side': opportunity['side'],
                        'entry_price': opportunity['entry_price'],
                        'stop_price': opportunity['stop_price'],
                        'take_profit': opportunity['take_profit'],
                        'qty': opportunity['position_size'],
                        'timestamp': opportunity['timestamp'],
                        'signal_type': opportunity['signal_type'],
                        'strength': opportunity['strength']
                    }
                
            except Exception as e:
                self.logger.error(f"❌ Trade execution error: {e}")
    
    def _place_aggressive_order(self, signal: Dict) -> bool:
        """Place aggressive order with tight stops"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            qty = signal['position_size']
            price = signal['entry_price']
            
            # Set leverage
            self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(int(self.config.leverage)),
                sellLeverage=str(int(self.config.leverage))
            )
            
            # Place market order
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side="Buy" if side == "long" else "Sell",
                orderType="Market",
                qty=str(qty),
                timeInForce="IOC"
            )
            
            if order['retCode'] == 0:
                # Set stop loss and take profit
                self._set_aggressive_stops(symbol, signal)
                return True
            else:
                self.logger.error(f"❌ Order failed: {order['retMsg']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Order placement error: {e}")
            return False
    
    def _set_aggressive_stops(self, symbol: str, signal: Dict):
        """Set aggressive stop loss and take profit"""
        try:
            side = "Sell" if signal['side'] == "long" else "Buy"
            
            # Stop loss
            self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Stop",
                qty=str(signal['position_size']),
                stopPrice=str(round(signal['stop_price'], 2)),
                timeInForce="GTC"
            )
            
            # Take profit
            self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(signal['position_size']),
                price=str(round(signal['take_profit'], 2)),
                timeInForce="GTC"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Stop setting error: {e}")
    
    def _monitor_positions(self):
        """Monitor existing positions for exits"""
        for symbol, position in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                # Check exit conditions
                exit_decision = self.strategy.should_exit_position(symbol, position, current_price)
                
                if exit_decision['action'] == 'exit':
                    self._close_position(symbol, exit_decision['reason'])
                    
            except Exception as e:
                self.logger.error(f"❌ Position monitoring error for {symbol}: {e}")
    
    def _close_position(self, symbol: str, reason: str):
        """Close position aggressively"""
        try:
            if symbol not in self.active_positions:
                return
            
            position = self.active_positions[symbol]
            side = "Sell" if position['side'] == "long" else "Buy"
            
            # Market close
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(position['qty']),
                timeInForce="IOC"
            )
            
            if order['retCode'] == 0:
                # Calculate P&L
                current_price = self._get_current_price(symbol)
                if current_price:
                    if position['side'] == 'long':
                        pnl = (current_price - position['entry_price']) * position['qty']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['qty']
                    
                    self.total_pnl += pnl
                    self.daily_pnl += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                # Log exit
                self.logger.info(
                    f"🔚 POSITION CLOSED [{datetime.now().strftime('%H:%M:%S')}] - "
                    f"{symbol} {position['side'].upper()} - {reason}"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Position close error: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            response = self.client.get_tickers(category="linear", symbol=symbol)
            if response['retCode'] == 0 and response['result']['list']:
                return float(response['result']['list'][0]['lastPrice'])
            return None
        except Exception as e:
            self.logger.error(f"❌ Price fetch error for {symbol}: {e}")
            return None
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_pnl = self.total_pnl / self.total_trades
            
            # Log performance every 100 trades
            if self.total_trades % 100 == 0:
                self.logger.info(
                    f"📊 PERFORMANCE - Trades: {self.total_trades}, "
                    f"Win Rate: {win_rate:.1f}%, P&L: ${self.total_pnl:.2f}, "
                    f"Daily P&L: ${self.daily_pnl:.2f}"
                )
    
    def _cleanup(self):
        """Cleanup and final report"""
        self.logger.info("🛑 Aggressive trading session ended")
        self.logger.info(f"📊 FINAL STATS - Total Trades: {self.total_trades}")
        self.logger.info(f"💰 Total P&L: ${self.total_pnl:.2f}")
        self.logger.info(f"📈 Win Rate: {(self.winning_trades/self.total_trades*100):.1f}%" if self.total_trades > 0 else "N/A")

def main():
    """Main function for aggressive trading"""
    try:
        # Default to mainnet (real money)
        trader = AggressiveTrader(demo=False)
        trader.run()
    except KeyboardInterrupt:
        print("\n🛑 Aggressive trading stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
