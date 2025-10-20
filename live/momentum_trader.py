#!/usr/bin/env python3
"""
High Momentum Trader - Trade 50-100% gainers with small balance
Specialized for turning $1.7 into $200+ quickly
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
from high_momentum_strategy import HighMomentumStrategy, HighMomentumConfig, create_micro_balance_config

class MomentumTrader:
    """High momentum trader for volatile gainers"""
    
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
        
        # Momentum configuration
        self.config = create_micro_balance_config()
        self.strategy = HighMomentumStrategy(self.config)
        
        # Trading state
        self.active_positions = {}
        self.position_history = []
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_daily_trades = 100
        
        # Performance tracking
        self.start_balance = self.config.account_balance
        self.target_balance = self.config.target_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # UI settings
        self.quiet = os.getenv('BOT_QUIET', '0') in ('1', 'true', 'yes')
        self.ui = os.getenv('BOT_UI', '1') in ('1', 'true', 'yes')
        
        # Logging setup
        self.log_file = os.getenv('BOT_LOG_FILE', 'momentum_log.txt')
        self.logger = self._setup_logging()
        
        # Trading parameters
        self.scan_interval_sec = 2  # 2 second scan for momentum
        self.max_concurrent_trades = 2  # Only 2 positions for small balance
        
        print(f"High Momentum Trader - {env_name}")
        print(f"API Endpoint: {'https://api-demo.bybit.com (Demo)' if demo else 'https://api.bybit.com (Mainnet)'}")
        print(f"Starting Balance: ${self.start_balance}")
        print(f"Target Balance: ${self.target_balance}")
        print(f"Leverage: {self.config.leverage}x")
        print(f"Risk per Trade: {self.config.risk_per_trade*100}%")
        print(f"Max Positions: {self.config.max_positions}")
        print(f"Min 24h Gain: {self.config.min_24h_gain*100}%")
        print(f"Max 24h Gain: {self.config.max_24h_gain*100}%")
    
    def _setup_logging(self):
        """Setup logging for momentum trading"""
        logger = logging.getLogger('momentum_trader')
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
        
        # Console handler
        if not self.quiet:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.propagate = False
        return logger
    
    def run(self):
        """Main momentum trading loop"""
        self.logger.info("🚀 Starting high momentum trading...")
        
        # Check initial balance
        self._check_balance()
        
        while True:
            try:
                loop_start = time.time()
                
                # Check if target reached
                if self.config.account_balance >= self.target_balance:
                    self.logger.info(f"🎉 TARGET REACHED! ${self.config.account_balance:.2f} >= ${self.target_balance}")
                    break
                
                # Check if balance too low
                if self.config.account_balance < 0.5:
                    self.logger.error("💀 Balance too low - stopping trading")
                    break
                
                # Monitor existing positions
                self._monitor_positions()
                
                # Scan for momentum opportunities
                opportunities = self._scan_momentum_opportunities()
                
                # Execute trades
                if opportunities:
                    self._execute_momentum_trades(opportunities)
                
                # Update performance
                self._update_performance()
                
                # Sleep
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.scan_interval_sec - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Trading stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Trading loop error: {e}")
                time.sleep(5)
        
        # Final report
        self._final_report()
    
    def _check_balance(self):
        """Check account balance"""
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] == 0:
                balance = float(response['result']['list'][0]['totalWalletBalance'])
                self.config.account_balance = balance
                self.logger.info(f"💰 Current Balance: ${balance:.2f}")
                
                if balance < 0.5:
                    self.logger.error("💀 Balance too low for trading")
                    return False
            else:
                self.logger.error(f"❌ Failed to get balance: {response['retMsg']}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Balance check error: {e}")
            return False
        
        return True
    
    def _scan_momentum_opportunities(self) -> List[Dict]:
        """Scan for high momentum trading opportunities"""
        try:
            # Get all symbols
            symbols = self._get_all_symbols()
            if not symbols:
                return []
            
            # Scan for momentum coins
            momentum_coins = self.strategy.scan_momentum_coins(symbols)
            
            if not momentum_coins:
                return []
            
            self.logger.info(f"📊 Found {len(momentum_coins)} momentum coins")
            
            # Analyze top momentum coins
            opportunities = []
            for coin in momentum_coins[:5]:  # Top 5 only
                try:
                    opportunity = self.strategy.analyze_momentum_coin(coin['symbol'], coin)
                    if opportunity:
                        opportunities.append(opportunity)
                except Exception as e:
                    self.logger.error(f"❌ Analysis error for {coin['symbol']}: {e}")
            
            # Sort by signal strength
            opportunities.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
            
            return opportunities[:self.max_concurrent_trades]
            
        except Exception as e:
            self.logger.error(f"❌ Opportunity scan error: {e}")
            return []
    
    def _get_all_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        try:
            response = self.client.get_instruments_info(category="linear")
            if response['retCode'] == 0:
                symbols = []
                for instrument in response['result']['list']:
                    if instrument['status'] == 'Trading':
                        symbols.append(instrument['symbol'])
                return symbols
            return []
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols: {e}")
            return []
    
    def _execute_momentum_trades(self, opportunities: List[Dict]):
        """Execute momentum trades"""
        for opportunity in opportunities:
            try:
                # Check if we can open new position
                if len(self.active_positions) >= self.config.max_positions:
                    break
                
                # Execute trade
                success = self._place_momentum_order(opportunity)
                
                if success:
                    self.trades_today += 1
                    self.total_trades += 1
                    
                    # Log trade
                    self.logger.info(
                        f"🚀 MOMENTUM TRADE [{datetime.now().strftime('%H:%M:%S')}] - "
                        f"{opportunity['symbol']} {opportunity['side'].upper()} "
                        f"{opportunity['position_size']:.6f} @ ${opportunity['entry_price']:.4f} "
                        f"(24h: {opportunity['24h_gain']*100:.1f}%, Strength: {opportunity['signal_strength']:.1f})"
                    )
                    
                    # Track position
                    self.active_positions[opportunity['symbol']] = {
                        'side': opportunity['side'],
                        'entry_price': opportunity['entry_price'],
                        'stop_price': opportunity['stop_price'],
                        'take_profit': opportunity['take_profit'],
                        'qty': opportunity['position_size'],
                        'timestamp': datetime.now(),
                        'signal_strength': opportunity['signal_strength'],
                        '24h_gain': opportunity['24h_gain'],
                        'risk_amount': opportunity['risk_amount']
                    }
                
            except Exception as e:
                self.logger.error(f"❌ Trade execution error: {e}")
    
    def _place_momentum_order(self, signal: Dict) -> bool:
        """Place momentum order with detailed position sizing"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            qty = signal['position_size']
            price = signal['entry_price']
            
            # Debug position sizing
            calc = self.strategy.get_position_size_calculation(
                symbol, price, signal['stop_price'], signal['signal_strength']
            )
            
            self.logger.info(f"📊 Position Size Calculation for {symbol}:")
            self.logger.info(f"   Entry: ${calc['entry_price']:.4f}")
            self.logger.info(f"   Stop: ${calc['stop_price']:.4f}")
            self.logger.info(f"   Position Size: {calc['leveraged_position_size']:.6f}")
            self.logger.info(f"   Position Value: ${calc['leveraged_position_value']:.2f}")
            self.logger.info(f"   Required Margin: ${calc['required_margin']:.2f}")
            self.logger.info(f"   Available: ${calc['available_balance']:.2f}")
            self.logger.info(f"   Can Trade: {'✅' if calc['can_trade'] else '❌'}")
            
            if not calc['can_trade']:
                self.logger.warning(f"❌ Insufficient margin for {symbol}")
                return False
            
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
                self._set_momentum_stops(symbol, signal)
                return True
            else:
                self.logger.error(f"❌ Order failed: {order['retMsg']}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Order placement error: {e}")
            return False
    
    def _set_momentum_stops(self, symbol: str, signal: Dict):
        """Set stop loss and take profit for momentum trade"""
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
                should_exit = self._should_exit_position(symbol, position, current_price)
                
                if should_exit['exit']:
                    self._close_position(symbol, should_exit['reason'])
                    
            except Exception as e:
                self.logger.error(f"❌ Position monitoring error for {symbol}: {e}")
    
    def _should_exit_position(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """Check if position should be exited"""
        # Time-based exit
        hold_time = datetime.now() - position['timestamp']
        if hold_time.total_seconds() > self.config.max_hold_time:
            return {'exit': True, 'reason': 'max_hold_time'}
        
        # Profit target
        if position['side'] == 'long':
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            profit_pct = (position['entry_price'] - current_price) / position['entry_price']
        
        if profit_pct >= self.config.min_profit_target:
            return {'exit': True, 'reason': 'profit_target'}
        
        # Quick exit threshold
        if profit_pct >= self.config.quick_exit_threshold and profit_pct < self.config.min_profit_target:
            return {'exit': True, 'reason': 'quick_exit'}
        
        # Stop loss
        if position['side'] == 'long' and current_price <= position['stop_price']:
            return {'exit': True, 'reason': 'stop_loss'}
        elif position['side'] == 'short' and current_price >= position['stop_price']:
            return {'exit': True, 'reason': 'stop_loss'}
        
        # Take profit
        if position['side'] == 'long' and current_price >= position['take_profit']:
            return {'exit': True, 'reason': 'take_profit'}
        elif position['side'] == 'short' and current_price <= position['take_profit']:
            return {'exit': True, 'reason': 'take_profit'}
        
        return {'exit': False, 'reason': 'hold'}
    
    def _close_position(self, symbol: str, reason: str):
        """Close position"""
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
                    self.config.account_balance += pnl
                    
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
    
    def _update_performance(self):
        """Update performance metrics"""
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            current_balance = self.config.account_balance
            total_return = ((current_balance - self.start_balance) / self.start_balance) * 100
            
            # Log performance every 10 trades
            if self.total_trades % 10 == 0:
                self.logger.info(
                    f"📊 PERFORMANCE - Balance: ${current_balance:.2f} "
                    f"({total_return:+.1f}%), Trades: {self.total_trades}, "
                    f"Win Rate: {win_rate:.1f}%, P&L: ${self.total_pnl:.2f}"
                )
    
    def _final_report(self):
        """Final performance report"""
        current_balance = self.config.account_balance
        total_return = ((current_balance - self.start_balance) / self.start_balance) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("📊 FINAL PERFORMANCE REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"💰 Starting Balance: ${self.start_balance:.2f}")
        self.logger.info(f"💰 Final Balance: ${current_balance:.2f}")
        self.logger.info(f"📈 Total Return: {total_return:+.1f}%")
        self.logger.info(f"📊 Total Trades: {self.total_trades}")
        self.logger.info(f"✅ Winning Trades: {self.winning_trades}")
        self.logger.info(f"❌ Losing Trades: {self.losing_trades}")
        self.logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
        self.logger.info(f"💵 Total P&L: ${self.total_pnl:.2f}")
        self.logger.info("=" * 60)
        
        if current_balance >= self.target_balance:
            self.logger.info("🎉 TARGET ACHIEVED! 🎉")
        else:
            self.logger.info("📈 Keep trading to reach target!")

def main():
    """Main function for momentum trading"""
    try:
        # Default to mainnet (real money)
        trader = MomentumTrader(demo=False)
        trader.run()
    except KeyboardInterrupt:
        print("\n🛑 Momentum trading stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
