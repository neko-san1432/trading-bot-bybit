#!/usr/bin/env python3
"""
Clean Trend Scalping Trader - Production Version
No debugging, optimized for live trading
"""

import os
import time
import math
import logging
import sys
import pandas as pd
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pybit.unified_trading import HTTP
from backtest.trend_scalping_strategy_clean import TrendScalpingStrategy, TrendScalpingConfig
from backtest.pair_discovery import PairDiscovery

class TrendScalpingTraderClean:
    """Clean trend scalping trader for live trading"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        # Environment setup
        self.testnet = testnet
        env_name = "TESTNET" if testnet else "MAINNET"
        
        # API credentials and env selection
        # Priority: explicit args > env vars
        # Env vars:
        #  - BYBIT_TESTNET: "1"/"true" => use demo creds; else real creds
        #  - BYBIT_API_KEY_DEMO / BYBIT_API_SECRET_DEMO
        #  - BYBIT_API_KEY_REAL / BYBIT_API_SECRET_REAL
        env_testnet = os.getenv('BYBIT_TESTNET', '').lower() in ('1', 'true', 'yes')
        if api_key and api_secret:
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet
        else:
            # If BYBIT_TESTNET indicates demo, prefer demo creds
            if env_testnet or testnet:
                self.testnet = True
                self.api_key = os.getenv('BYBIT_API_KEY_DEMO')
                self.api_secret = os.getenv('BYBIT_API_SECRET_DEMO')
            else:
                self.testnet = False
                self.api_key = os.getenv('BYBIT_API_KEY_REAL')
                self.api_secret = os.getenv('BYBIT_API_SECRET_REAL')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(f"API key and secret required for {env_name}")
        
        # Initialize API client
        self.client = HTTP(
            testnet=self.testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        # Quiet mode and simple console UI
        self.quiet = os.getenv('BOT_QUIET', '1') in ('1', 'true', 'yes')
        self.ui = os.getenv('BOT_UI', '1') in ('1', 'true', 'yes')
        # UI row cap: 'all' to render every scanned pair, or integer limit
        ui_max_rows_env = os.getenv('BOT_UI_MAX_ROWS', '0')
        try:
            self.ui_max_rows = int(ui_max_rows_env)
        except ValueError:
            self.ui_max_rows = 0  # treat non-integer as all
        # Logging setup
        self.log_scans = os.getenv('BOT_LOG_SCANS', '1') in ('1', 'true', 'yes')
        self.log_file = os.getenv('BOT_LOG_FILE', 'log.txt')
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s %(message)s',
                handlers=[logging.FileHandler(self.log_file, encoding='utf-8')]
            )
            self.logger = logging.getLogger('scanner')
        except Exception:
            self.logger = None
        # Spinner setup (animated Searching... when UI table is off)
        self.spinner_enabled = os.getenv('BOT_SPINNER', '1') in ('1', 'true', 'yes')
        self._spinner_step = 0
        
        # Initialize strategy
        self.config = self._create_strategy_config()
        self.strategy = TrendScalpingStrategy(self.config)
        
        # Initialize pair discovery
        self.pair_discovery = PairDiscovery()
        
        # Trading state
        self.active_positions = {}
        self.last_trade_time = {}
        self.min_trade_interval = 5  # 5 seconds between trades
        self.symbol_leverage_cache = {}
        # Parallelism for faster scanning (tune to avoid rate limits)
        try:
            self.max_workers = max(1, int(os.getenv('SCANNER_MAX_WORKERS', '5')))
        except Exception:
            self.max_workers = 5
        # Scan sizing and cadence
        try:
            self.scan_target_count = max(1, int(os.getenv('SCAN_TARGET_COUNT', '500')))
        except Exception:
            self.scan_target_count = 500
        try:
            self.scan_interval_sec = max(1, int(os.getenv('SCAN_INTERVAL_SEC', '6')))
        except Exception:
            self.scan_interval_sec = 6
        
        if not self.quiet:
            print(f"🚀 Trend Scalping Trader initialized ({env_name})")
    
    def _create_strategy_config(self) -> TrendScalpingConfig:
        """Create strategy configuration"""
        return TrendScalpingConfig(
            # Relaxed entry conditions
            adx_threshold=15.0,
            volume_spike_threshold=1.1,
            
            # Risk management
            risk_per_trade=0.01,
            risk_reward_ratio=1.2,  # Lowered to 20% minimum profit
            max_positions=1,
            leverage=25.0,
            
            # Account
            account_balance=10000.0
        )
    
    def run(self):
        """Main trading loop"""
        if not self.quiet:
            print("🔄 Starting trend scalping bot...")
        
        while True:
            try:
                loop_start = time.time()
                # Get trading pairs
                pairs = self._get_trading_pairs()
                
                # Batch fetch klines for the first N symbols to reduce API overhead
                symbols_to_scan = [s for s in pairs if self._can_trade(s)]
                if not symbols_to_scan:
                    if self.spinner_enabled and (not self.ui) and (not self.quiet):
                        self._render_spinner_tick()
                    time.sleep(0.5)
                    continue

                # Limit to target count
                symbols_batch = symbols_to_scan[:self.scan_target_count]
                # Compute per-second limit to hit interval (best-effort)
                per_sec_limit = max(1, math.ceil(len(symbols_batch) / float(self.scan_interval_sec)))
                klines_map = self._get_klines_batch(symbols_batch, interval="1", limit=200, per_second_limit=per_sec_limit)

                # Render UI header only if UI is enabled
                if self.ui:
                    self._render_scan_table_header()

                # Analyze in parallel using fetched klines
                trade_executed = False
                # Determine how many rows to render (0 = all)
                if self.ui_max_rows and self.ui_max_rows > 0:
                    symbols_to_render = set(symbols_batch[:self.ui_max_rows])
                else:
                    symbols_to_render = set(symbols_batch)
                def analyze_with_df(sym: str) -> Optional[Dict]:
                    df = klines_map.get(sym)
                    if df is None or len(df) < 100:
                        if self.ui and (sym in symbols_to_render):
                            self._render_scan_row(sym, status="no-data")
                        elif self.spinner_enabled and (not self.ui) and (not self.quiet):
                            self._render_spinner_tick()
                        return None
                    # Exclude symbols with 100x and set leverage per symbol
                    if not self._is_allowed_symbol(sym):
                        if self.ui and (sym in symbols_to_render):
                            self._render_scan_row(sym, status="skip(100x)")
                        elif self.spinner_enabled and (not self.ui) and (not self.quiet):
                            self._render_spinner_tick()
                        return None
                    max_lev = self._get_symbol_max_leverage(sym)
                    if max_lev is not None:
                        try:
                            self.strategy.config.leverage = float(max_lev)
                        except Exception:
                            pass
                    df2 = self.strategy.analyze_data(df)
                    signal = self.strategy.get_entry_signal(sym, df2.iloc[-1])
                    if self.ui and (sym in symbols_to_render):
                        trend = df2.iloc[-1].get('trend_direction', 'n/a')
                        adx = df2.iloc[-1].get('adx', float('nan'))
                        volr = df2.iloc[-1].get('vol_ratio', float('nan'))
                        status = "signal" if signal else "scan"
                        self._render_scan_row(sym, trend=trend, adx=adx, vol_ratio=volr, status=status)
                    elif self.spinner_enabled and (not self.ui) and (not self.quiet):
                        self._render_spinner_tick()
                    # Always log scans to file if enabled
                    if self.log_scans:
                        trend = df2.iloc[-1].get('trend_direction', 'n/a')
                        adx = float(df2.iloc[-1].get('adx', float('nan')))
                        volr = float(df2.iloc[-1].get('vol_ratio', float('nan')))
                        self._log_scan_row(sym, trend, adx, volr, 'signal' if signal else 'scan')
                    return signal

                # Analyze ALL symbols in the batch; render only first N
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_symbol = {executor.submit(analyze_with_df, s): s for s in symbols_batch}
                    for future in as_completed(future_to_symbol):
                        if trade_executed:
                            break
                        try:
                            signal = future.result()
                            if signal:
                                # Ensure leverage set before order
                                max_lev = self._get_symbol_max_leverage(signal['symbol'])
                                if max_lev is not None:
                                    try:
                                        self.client.set_leverage(
                                            category="linear",
                                            symbol=signal['symbol'],
                                            buyLeverage=str(int(float(max_lev))),
                                            sellLeverage=str(int(float(max_lev)))
                                        )
                                    except Exception:
                                        pass
                                self._execute_trade(signal)
                                trade_executed = True
                                break
                        except Exception:
                            continue
                
                # Wait to maintain target interval (consider time spent in this loop)
                elapsed = time.time() - loop_start
                remaining = self.scan_interval_sec - elapsed
                if remaining > 0:
                    end_wait = time.time() + remaining
                    while time.time() < end_wait:
                        if self.spinner_enabled and (not self.ui) and (not self.quiet):
                            self._render_spinner_tick()
                        time.sleep(0.25)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(5)
    
    def _get_trading_pairs(self) -> List[str]:
        """Get list of trading pairs"""
        try:
            # Try to get pairs from discovery system
            recommendations = self.pair_discovery.get_pair_recommendations("scalping")
            if recommendations:
                # Build symbol list, exclude 100x leverage pairs, sort by name descending
                symbols = [rec['symbol'] for rec in recommendations]
                symbols = [s for s in symbols if self._is_allowed_symbol(s)]
                symbols = sorted(symbols, reverse=True)
                return symbols
        except Exception as e:
            pass
        
        # Fallback to hardcoded pairs
        # Fallback symbols filtered and sorted by name descending
        return sorted([s for s in self._get_fallback_pairs() if self._is_allowed_symbol(s)], reverse=True)
    
    def _get_fallback_pairs(self) -> List[str]:
        """Get fallback trading pairs"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT',
            'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT',
            'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'ALGOUSDT',
            'VETUSDT', 'ICPUSDT', 'THETAUSDT', 'EOSUSDT', 'AAVEUSDT',
            'MKRUSDT', 'COMPUSDT', 'YFIUSDT', 'SNXUSDT', 'SUSHIUSDT',
            'NEARUSDT', 'FTMUSDT', 'ONEUSDT', 'HBARUSDT', 'EGLDUSDT',
            'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'GALAUSDT', 'ENJUSDT',
            'CHZUSDT', 'FLOWUSDT', 'ROSEUSDT', 'IOTAUSDT', 'ZILUSDT',
            'BATUSDT', 'ZECUSDT', 'DASHUSDT', 'NEOUSDT', 'QTUMUSDT'
        ]
    
    def _can_trade(self, symbol: str) -> bool:
        """Check if we can trade this symbol"""
        # Check if we already have a position
        if len(self.active_positions) >= self.config.max_positions:
            return False
        
        # Check minimum time between trades
        if symbol in self.last_trade_time:
            time_since_last = (time.time() - self.last_trade_time[symbol])
            if time_since_last < self.min_trade_interval:
                return False
        
        return True
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol for trading opportunities"""
        try:
            # Exclude symbols with exactly 100x leverage
            if not self._is_allowed_symbol(symbol):
                return None
            
            # Get recent data
            df = self._get_klines(symbol)
            if df is None or len(df) < 100:
                return None
            
            # Set strategy leverage to symbol's max leverage (per-trade)
            max_lev = self._get_symbol_max_leverage(symbol)
            if max_lev is not None:
                try:
                    self.strategy.config.leverage = float(max_lev)
                except Exception:
                    pass
            
            # Add indicators
            df = self.strategy.analyze_data(df)
            
            # Get latest signal
            latest_row = df.iloc[-1]
            signal = self.strategy.get_entry_signal(symbol, latest_row)
            
            return signal
            
        except Exception as e:
            return None
    
    def _get_klines(self, symbol: str, interval: str = "1", limit: int = 200) -> Optional[pd.DataFrame]:
        """Get kline data from Bybit"""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                return None
            
            data = response['result']['list']
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            return None

    def _get_klines_batch(self, symbols: List[str], interval: str = "1", limit: int = 200, per_second_limit: int = 5) -> Dict[str, Optional[pd.DataFrame]]:
        """Batch-fetch klines with simple rate limiting. Returns symbol->DataFrame."""
        results: Dict[str, Optional[pd.DataFrame]] = {}
        start_idx = 0
        while start_idx < len(symbols):
            batch = symbols[start_idx:start_idx + per_second_limit]
            for symbol in batch:
                try:
                    response = self.client.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        limit=limit
                    )
                    if response['retCode'] != 0:
                        results[symbol] = None
                        continue
                    data = response['result']['list']
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        df[col] = pd.to_numeric(df[col])
                    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    results[symbol] = df
                except Exception:
                    results[symbol] = None
            # Sleep one second between batches to respect per-second rate limit
            time.sleep(1)
            start_idx += per_second_limit
        return results
    
    def _check_balance(self, symbol: str, qty: float, price: float) -> bool:
        """Check if account has sufficient balance for the trade"""
        try:
            # Get account balance
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] != 0:
                return False
            
            # Bybit unified format: result.list[0].coin[]
            usdt_balance = 0.0
            acct_list = response.get('result', {}).get('list', [])
            if acct_list:
                coins = acct_list[0].get('coin', [])
                for c in coins:
                    if c.get('coin') == 'USDT':
                        # Prefer availableBalance; fallback to walletBalance
                        usdt_balance = float(c.get('availableBalance') or c.get('walletBalance') or 0)
                        break
            
            # Calculate required margin (using leverage)
            required_margin = (qty * price) / self.config.leverage
            
            # Check if we have enough balance (with 20% buffer)
            required_with_buffer = required_margin * 1.2
            
            if usdt_balance < required_with_buffer:
                print(f"❌ Insufficient balance: Need ${required_with_buffer:.2f}, Have ${usdt_balance:.2f}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Balance check error: {e}")
            return False
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            response = self.client.get_tickers(category="linear", symbol=symbol)
            if response['retCode'] != 0:
                return None
            
            price = float(response['result']['list'][0]['lastPrice'])
            return price
            
        except Exception as e:
            return None

    def _is_allowed_symbol(self, symbol: str) -> bool:
        """Exclude symbols whose max leverage is exactly 100x"""
        try:
            max_lev = self._get_symbol_max_leverage(symbol)
            if max_lev is None:
                return True
            return float(max_lev) != 100.0
        except Exception:
            return True

    def _get_symbol_max_leverage(self, symbol: str) -> Optional[float]:
        """Get and cache symbol's maximum leverage from instruments info"""
        try:
            if symbol in self.symbol_leverage_cache:
                return self.symbol_leverage_cache[symbol]
            resp = self.client.get_instruments_info(category="linear", symbol=symbol)
            if resp.get('retCode') != 0:
                return None
            items = resp.get('result', {}).get('list', [])
            if not items:
                return None
            lev_filter = items[0].get('leverageFilter') or {}
            max_lev = lev_filter.get('maxLeverage')
            if max_lev is None:
                return None
            max_lev_val = float(max_lev)
            self.symbol_leverage_cache[symbol] = max_lev_val
            return max_lev_val
        except Exception:
            return None

    # ---------- Console UI helpers ----------
    def _clear_screen(self):
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            pass

    def _render_scan_table_header(self):
        if self.ui:
            self._clear_screen()
            print("Symbol        Trend    ADX    VolRatio    Status")
            print("------------------------------------------------")

    def _render_scan_row(self, symbol: str, trend: str = "", adx: float = float('nan'), vol_ratio: float = float('nan'), status: str = ""):
        if self.ui:
            try:
                adx_str = f"{adx:5.1f}" if isinstance(adx, (int, float)) else "  n/a"
                vol_str = f"{vol_ratio:8.2f}" if isinstance(vol_ratio, (int, float)) else "    n/a"
                print(f"{symbol:<12} {trend[:6]:<6} {adx_str} {vol_str}    {status}")
            except Exception:
                pass

    def _log_scan_row(self, symbol: str, trend: str, adx: float, vol_ratio: float, status: str):
        try:
            if self.logger:
                self.logger.info(f"scan symbol={symbol} trend={trend} adx={adx:.2f} vol_ratio={vol_ratio:.2f} status={status}")
        except Exception:
            pass

    def _render_spinner_tick(self):
        try:
            dots = (self._spinner_step % 3) + 1
            sys.stdout.write("\rSearching" + "." * dots + " " * (3 - dots))
            sys.stdout.flush()
            self._spinner_step += 1
        except Exception:
            pass
    
    def _adjust_position_size(self, symbol: str, qty: float, price: float) -> Optional[float]:
        """Adjust position size based on available balance"""
        try:
            # Get account balance
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] != 0:
                return None
            
            # Bybit unified format: result.list[0].coin[]
            usdt_balance = 0.0
            acct_list = response.get('result', {}).get('list', [])
            if acct_list:
                coins = acct_list[0].get('coin', [])
                for c in coins:
                    if c.get('coin') == 'USDT':
                        usdt_balance = float(c.get('availableBalance') or c.get('walletBalance') or 0)
                        break
            
            # Calculate maximum position size based on available balance
            # Use 80% of available balance for safety
            max_usable_balance = usdt_balance * 0.8
            max_position_value = max_usable_balance * self.config.leverage
            max_qty = max_position_value / price
            
            # Get minimum order quantity and minimum notional ($5) check
            formatted_qty = self._format_quantity(symbol, qty)
            if formatted_qty is None:
                return None
            min_qty = float(formatted_qty)
            
            # Return adjusted quantity
            # Enforce Bybit minimum notional (approx $5); use current price
            price = self._get_current_price(symbol) or price
            min_notional = 5.0
            min_qty_by_notional = (min_notional / max(price, 1e-9))
            min_qty = max(min_qty, min_qty_by_notional)

            if qty <= max_qty and qty >= min_qty:
                return qty
            elif max_qty >= min_qty:
                return max_qty
            else:
                return None  # Insufficient balance even for minimum order
                
        except Exception as e:
            return None
    
    def _format_quantity(self, symbol: str, qty: float) -> Optional[str]:
        """Format quantity according to symbol requirements"""
        try:
            # Get symbol info to determine precision and minimum qty
            response = self.client.get_instruments_info(category="linear", symbol=symbol)
            if response['retCode'] != 0:
                return None
            
            symbol_info = response['result']['list'][0]
            lot_size_filter = symbol_info['lotSizeFilter']
            
            # Get minimum order quantity and step size
            min_qty = float(lot_size_filter['minOrderQty'])
            qty_step = float(lot_size_filter['qtyStep'])
            
            # Round to step size
            rounded_qty = round(qty / qty_step) * qty_step
            
            # Check minimum quantity
            if rounded_qty < min_qty:
                rounded_qty = min_qty
            
            # Format to appropriate decimal places
            decimal_places = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
            formatted_qty = f"{rounded_qty:.{decimal_places}f}"
            
            return formatted_qty
            
        except Exception as e:
            # Fallback to simple rounding for common pairs
            if 'BTC' in symbol:
                return f"{round(qty, 6)}"
            elif 'ETH' in symbol:
                return f"{round(qty, 5)}"
            else:
                return f"{round(qty, 4)}"
    
    def _execute_trade(self, signal: Dict):
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            qty = signal['position_size']
            price = signal['entry_price']
            
            # Check account balance and adjust position size if needed
            adjusted_qty = self._adjust_position_size(symbol, qty, price)
            if adjusted_qty is None:
                if not self.quiet:
                    print(f"❌ Insufficient balance for {symbol}")
                return
            elif adjusted_qty != qty:
                if not self.quiet:
                    print(f"⚠️ Reduced position size from {qty:.6f} to {adjusted_qty:.6f} due to balance")
                qty = adjusted_qty
            
            # Ensure leverage is set to symbol's maximum before placing order
            max_lev = self._get_symbol_max_leverage(symbol)
            if max_lev is not None:
                try:
                    self.client.set_leverage(
                        category="linear",
                        symbol=symbol,
                        buyLeverage=str(int(float(max_lev))),
                        sellLeverage=str(int(float(max_lev)))
                    )
                except Exception:
                    pass

            # Format quantity according to symbol requirements
            formatted_qty = self._format_quantity(symbol, qty)
            if formatted_qty is None:
                if not self.quiet:
                    print(f"❌ Invalid quantity for {symbol}: {qty}")
                return
            
            # Get current market price for validation
            current_price = self._get_current_price(symbol)
            if current_price is None:
                if not self.quiet:
                    print(f"❌ Could not get current price for {symbol}")
                return
            
            # Validate price difference (shouldn't be more than 5% off)
            price_diff = abs(price - current_price) / current_price
            if price_diff > 0.05:  # 5% difference
                if not self.quiet:
                    print(f"⚠️ Price difference too large: Signal ${price:.2f} vs Market ${current_price:.2f}")
                price = current_price  # Use current market price
            
            # Place order
            order = self.client.place_order(
                category="linear",
                symbol=symbol,
                side="Buy" if side == "long" else "Sell",
                orderType="Market",
                qty=formatted_qty,
                timeInForce="IOC"
            )
            
            if order['retCode'] == 0:
                if self.ui:
                    self._render_trade_summary(symbol, side, price, qty, signal)
                
                # Track position
                self.active_positions[symbol] = {
                    'side': side,
                    'entry_price': price,
                    'stop_price': signal['stop_price'],
                    'take_profit': signal['take_profit'],
                    'qty': qty,
                    'timestamp': time.time()
                }
                
                self.last_trade_time[symbol] = time.time()
                
                # Set stop loss and take profit
                self._set_stop_loss(symbol, signal)
                self._set_take_profit(symbol, signal)
                
            else:
                if not self.quiet:
                    print(f"❌ Order failed: {order['retMsg']}")
                
        except Exception as e:
            if not self.quiet:
                print(f"❌ Trade execution error: {e}")
    
    def _set_stop_loss(self, symbol: str, signal: Dict):
        """Set stop loss order"""
        try:
            side = "Sell" if signal['side'] == "long" else "Buy"
            formatted_qty = self._format_quantity(symbol, signal['position_size'])
            if formatted_qty is None:
                return
                
            self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Stop",
                qty=formatted_qty,
                stopPrice=str(round(signal['stop_price'], 2)),
                timeInForce="GTC"
            )
        except Exception as e:
            pass
    
    def _set_take_profit(self, symbol: str, signal: Dict):
        """Set take profit order"""
        try:
            side = "Sell" if signal['side'] == "long" else "Buy"
            formatted_qty = self._format_quantity(symbol, signal['position_size'])
            if formatted_qty is None:
                return
                
            self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=formatted_qty,
                price=str(round(signal['take_profit'], 2)),
                timeInForce="GTC"
            )
        except Exception as e:
            pass

    def _render_trade_summary(self, symbol: str, side: str, entry_price: float, qty: float, signal: Dict):
        """Clear screen and render a concise trade summary box."""
        self._clear_screen()
        current_price = self._get_current_price(symbol) or entry_price
        # Approx liquidation calc (simplified): assume linear USDT perpetual, isolated maximum leverage
        lev = self._get_symbol_max_leverage(symbol) or self.strategy.config.leverage
        # Very rough estimate (varies by exchange settings):
        # long: liq ~ entry * (1 - 1/lev), short: entry * (1 + 1/lev)
        if side == 'long':
            liq = entry_price * max(0.0, (1 - 1/float(lev)))
        else:
            liq = entry_price * (1 + 1/float(lev))
        # Margin used
        margin = (qty * entry_price) / float(lev)
        print("TRADE EXECUTED")
        print("==============")
        print(f"Symbol        : {symbol}")
        print(f"Side          : {side.upper()}")
        print(f"Entry Price   : ${entry_price:.2f}")
        print(f"Current Price : ${current_price:.2f}")
        print(f"Position Size : {qty:.6f} (~${qty*entry_price:.2f} notional)")
        print(f"Margin Used   : ${margin:.2f} @ {lev:.0f}x")
        print(f"Take Profit   : ${signal['take_profit']:.2f}")
        print(f"Stop Loss     : ${signal['stop_price']:.2f}")
        print(f"Est. Liq Price: ${liq:.2f} (approx)")
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            response = self.client.get_positions(category="linear")
            if response['retCode'] == 0:
                return response['result']['list']
            return []
        except Exception as e:
            return []
    
    def close_position(self, symbol: str):
        """Close a position"""
        try:
            if symbol in self.active_positions:
                position = self.active_positions[symbol]
                side = "Sell" if position['side'] == "long" else "Buy"
                
                self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(position['qty']),
                    timeInForce="IOC"
                )
                
                del self.active_positions[symbol]
                if not self.quiet:
                    print(f"✅ Closed {symbol} position")
                
        except Exception as e:
            print(f"❌ Error closing position: {e}")

def main():
    """Main function"""
    print("🚀 Starting Clean Trend Scalping Bot")
    print("=" * 50)
    
    # Initialize trader
    trader = TrendScalpingTraderClean(testnet=True)
    
    # Start trading
    trader.run()

if __name__ == "__main__":
    main()
