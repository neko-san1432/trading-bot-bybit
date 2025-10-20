import os
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.strategy import ScalperConfig, ScalperStrategy
from ping_utils import BybitPingTester

# Load environment variables from .env file
load_dotenv()

class LiveTrader:
    def __init__(self, 
                 api_key: str = None,
                 api_secret: str = None,
                 symbols: List[str] = None,
                 max_positions: int = 3,
                 max_daily_trades: int = 20,
                 max_loss_percent: float = 5.0,
                 use_testnet: bool = False):
        
        # Determine environment and get appropriate credentials
        self.use_testnet = use_testnet
        
        if self.use_testnet:
            # Testnet for backtesting
            self.api_key = api_key or os.getenv('BYBIT_API_KEY_DEMO') or os.getenv('BYBIT_API_KEY')
            self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET_DEMO') or os.getenv('BYBIT_API_SECRET')
            env_name = "TESTNET (Backtesting)"
        else:
            # Mainnet for live trading
            self.api_key = api_key or os.getenv('BYBIT_API_KEY')
            self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
            env_name = "MAINNET (Live Trading)"

        if not self.api_key or not self.api_secret:
            raise ValueError(f"API key and secret required for {env_name}. Set in constructor or environment variables")

        # Trading pairs to scan
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']

        # Risk limits
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.max_loss_percent = max_loss_percent

        # Initialize trading client - using V5 Unified Trading Account API
        self.session = HTTP(
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )

        print(f"Initialized Bybit client on {env_name}")
        
        # Initialize ping tester for connectivity monitoring
        self.ping_tester = BybitPingTester(
            api_key=self.api_key,
            api_secret=self.api_secret,
            use_testnet=self.use_testnet
        )
        
        # Test API connection with comprehensive ping test
        self._test_api_connection()
        
        # Initialize strategy
        self.config = ScalperConfig()
        self.strategy = ScalperStrategy(self.config)
        
        # Track positions and daily stats
        self.active_positions: Dict[str, dict] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_balance = self._get_balance()
        self.last_reset = datetime.now()
        
        # Connection monitoring
        self.last_ping_time = datetime.now()
        self.ping_interval = 300  # 5 minutes between ping tests
        self.connection_failures = 0
        self.max_connection_failures = 3
    
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
        
        # Test balance endpoint (requires authentication)
        print("🔍 Testing authenticated API call...")
        balance = self._get_balance()
        if balance > 0:
            print(f"✅ Authentication successful! Balance: {balance:.2f} USDT")
        else:
            print("❌ Authentication failed - check your API credentials")
    
    def _get_balance(self) -> float:
        """Get current USDT balance"""
        try:
            # V5 API wallet balance endpoint
            wallet = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if wallet.get('retCode') == 0:  # Success
                return float(wallet['result']['list'][0]['totalWalletBalance'])
            else:
                error_code = wallet.get('retCode')
                error_msg = wallet.get('retMsg')
                print(f"❌ API Error getting balance:")
                print(f"   Code: {error_code}")
                print(f"   Message: {error_msg}")
                
                if error_code == 10003:  # Invalid API key
                    print("   🔑 This is an authentication error - check your API credentials")
                elif error_code == 10004:  # Invalid signature
                    print("   🔐 This is a signature error - check your API secret")
                elif error_code == 10005:  # Permission denied
                    print("   🚫 This is a permission error - check your API key permissions")
                elif error_code == 10006:  # IP not in whitelist
                    print("   🌐 This is an IP restriction error - check your IP whitelist")
                
                return 0.0
        except Exception as e:
            print(f"❌ Exception getting balance: {e}")
            return 0.0

    def _get_klines(self, symbol: str) -> pd.DataFrame:
        """Get recent 1m klines for analysis"""
        try:
            # V5 API klines endpoint
            klines = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval="1",
                limit=100  # Last 100 minutes
            )
            
            if klines.get('retCode') != 0:
                print(f"Error fetching klines: {klines.get('retMsg')}")
                return None
                
            df = pd.DataFrame(klines['result']['list'])
            # V5 API column mapping
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            
            return df.iloc[::-1]  # Reverse to get chronological order
            
        except Exception as e:
            print(f"Error fetching klines for {symbol}: {e}")
            return None

    def _place_order(self, symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, leverage: float = None) -> bool:
        """Place a new order with TP/SL with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check connection health before placing order
                if not self._check_connection_health():
                    print("❌ Connection health check failed, skipping order")
                    return False
                
                # Use configured leverage if not specified
                if leverage is None:
                    leverage = self.config.leverage
                
                # Set leverage for the position
                self.session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
                
                # Place the main order with TP/SL using V5 API
                order = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(qty),
                    isLeverage=1,  # Use leverage
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

    def check_position_updates(self):
        """Update status of open positions"""
        try:
            # V5 API position query
            positions = self.session.get_positions(
                category="linear",
                settleCoin="USDT",
                limit=50
            )
            
            # Update active positions
            current_symbols = set()
            for pos in positions['result']['list']:
                symbol = pos['symbol']
                current_symbols.add(symbol)
                
                if float(pos['size']) == 0:  # Position closed
                    if symbol in self.active_positions:
                        # Calculate PnL
                        realized_pnl = float(pos['cumRealisedPnl'])
                        self.daily_pnl += realized_pnl
                        print(f"Position closed for {symbol} - PnL: {realized_pnl:.2f} USDT")
                        del self.active_positions[symbol]
                
                else:  # Active position
                    # Calculate leverage information
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
                        'leverage_ratio': f"{leverage_used:.1f}x"
                    }
            
            # Remove closed positions
            for symbol in list(self.active_positions.keys()):
                if symbol not in current_symbols:
                    del self.active_positions[symbol]
                    
        except Exception as e:
            print(f"Error checking positions: {e}")

    def reset_daily_stats(self):
        """Reset daily trading statistics"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = now
            self.start_balance = self._get_balance()
            print(f"Daily stats reset. New balance: {self.start_balance:.2f} USDT")

    def check_risk_limits(self) -> bool:
        """Check if we're within risk limits to trade"""
        # Check max positions
        if len(self.active_positions) >= self.max_positions:
            return False
            
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
            
        # Check max loss
        if self.daily_pnl < -(self.start_balance * self.max_loss_percent / 100):
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

    def scan_and_trade(self):
        """Main loop to scan for opportunities and trade"""
        while True:
            try:
                # Check connection health first
                if not self._check_connection_health():
                    print("❌ Connection health check failed. Stopping trading.")
                    break
                
                # Reset daily stats if needed
                self.reset_daily_stats()
                
                # Update position status
                self.check_position_updates()
                
                # Check risk limits
                if not self.check_risk_limits():
                    print("Risk limits reached, waiting...")
                    time.sleep(60)
                    continue
                
                # Scan each symbol
                for symbol in self.symbols:
                    # Skip if we already have a position
                    if symbol in self.active_positions:
                        continue
                        
                    # Get recent klines
                    df = self._get_klines(symbol)
                    if df is None or len(df) < 50:  # Need enough data for indicators
                        continue
                    
                    # Prepare data with indicators
                    df = self.strategy.prepare(df)
                    
                    # Check for entry signal
                    last_row = df.iloc[-1]
                    signal = self.strategy.entry_signal(last_row)
                    
                    if signal:
                        direction, reasons = signal
                        
                        # Current market price
                        price = float(last_row['close'])
                        
                        # Calculate TP/SL levels
                        if direction == 'long':
                            tp = price * (1 + self.config.take_profit)
                            sl = price * (1 - self.config.stop_loss)
                        else:
                            tp = price * (1 - self.config.take_profit)
                            sl = price * (1 + self.config.stop_loss)
                        
                        # Calculate position size using leverage-based approach
                        balance = self._get_balance()
                        available_margin = balance * 0.95  # Use 95% of balance as available margin
                        notional_value = available_margin * self.config.leverage
                        qty = notional_value / price
                        
                        # Calculate leverage metrics for tracking
                        margin_used = notional_value / self.config.leverage
                        actual_leverage = notional_value / margin_used if margin_used > 0 else 0
                        
                        # Place the orders
                        if self._place_order(
                            symbol=symbol,
                            side="Buy" if direction == 'long' else "Sell",
                            qty=qty,
                            take_profit=tp,
                            stop_loss=sl,
                            leverage=self.config.leverage
                        ):
                            print(f"Opened {direction} position on {symbol} - Signals: {reasons}")
                            print(f"  Position: {qty:.4f} @ {price:.2f} | Leverage: {actual_leverage:.1f}x | Notional: ${notional_value:.2f}")
                            self.daily_trades += 1
                            self.active_positions[symbol] = {
                                'direction': direction,
                                'entry_price': price,
                                'size': qty,
                                'notional_value': notional_value,
                                'margin_used': margin_used,
                                'leverage_used': actual_leverage,
                                'leverage_ratio': f"{actual_leverage:.1f}x"
                            }
                
                # Small delay between scans
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(30)  # Longer delay on error