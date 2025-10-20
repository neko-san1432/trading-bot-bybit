import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json
from dataclasses import dataclass
from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PairInfo:
    """Information about a trading pair"""
    symbol: str
    base_asset: str
    quote_asset: str
    status: str  # 'trading', 'delisting', 'new'
    volatility_score: float
    volume_24h: float
    price_change_24h: float
    market_cap_rank: Optional[int]
    last_updated: datetime
    trading_fees: Dict[str, float]
    min_trade_size: float
    max_trade_size: float
    price_precision: int
    qty_precision: int


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics"""
    symbol: str
    timeframe: str
    avg_true_range: float
    volatility_percentile: float
    avg_move_1m: float
    avg_move_5m: float
    avg_move_15m: float
    volatility_score: float
    trend_strength: float
    volume_profile: Dict[str, float]
    last_updated: datetime


class PairDiscovery:
    """Dynamic pair discovery and monitoring system"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = False):
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.use_testnet = use_testnet
        
        # Initialize API client
        self.session = HTTP(
            testnet=self.use_testnet,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        
        # Pair tracking
        self.tracked_pairs: Dict[str, PairInfo] = {}
        self.volatility_cache: Dict[str, VolatilityMetrics] = {}
        self.last_discovery_update = None
        self.discovery_interval = 300  # 5 minutes
        
        # Volatility analysis settings
        self.volatility_lookback = 1000  # bars for volatility calculation
        self.min_volume_24h = 1000000  # $1M minimum daily volume
        self.min_volatility = 0.01  # 1% minimum volatility
        self.max_volatility = 0.20  # 20% maximum volatility (too risky)
        
        # print suppressed in production
    
    def get_available_pairs(self) -> List[Dict]:
        """Get all available trading pairs from Bybit API"""
        try:
            # Get instruments for linear futures
            instruments = self.session.get_instruments_info(
                category="linear",
                status="Trading"
            )
            
            if instruments.get('retCode') != 0:
                # suppressed verbose error
                return []
            
            pairs = []
            for instrument in instruments['result']['list']:
                if instrument['status'] == 'Trading':
                    pairs.append({
                        'symbol': instrument['symbol'],
                        'baseAsset': instrument['baseCoin'],
                        'quoteAsset': instrument['quoteCoin'],
                        'status': instrument['status'],
                        'minOrderQty': float(instrument['lotSizeFilter']['minOrderQty']),
                        'maxOrderQty': float(instrument['lotSizeFilter']['maxOrderQty']),
                        'pricePrecision': int(instrument['priceFilter']['tickSize'].count('0')),
                        'qtyPrecision': int(instrument['lotSizeFilter']['qtyStep'].count('0')),
                        'tradingFees': {
                            'maker': 0.0001,  # Default fees
                            'taker': 0.0006
                        }
                    })
            
            # print suppressed in production
            return pairs
            
        except Exception:
            # suppressed verbose error
            return []
    
    def get_pair_ticker_info(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get ticker information for multiple pairs"""
        try:
            tickers = self.session.get_tickers(
                category="linear",
                symbol=",".join(symbols)
            )
            
            if tickers.get('retCode') != 0:
                # suppressed verbose error
                return {}
            
            ticker_data = {}
            for ticker in tickers['result']['list']:
                ticker_data[ticker['symbol']] = {
                    'lastPrice': float(ticker['lastPrice']),
                    'volume24h': float(ticker['volume24h']),
                    'priceChange24h': float(ticker['price24hPcnt']) * 100,
                    'high24h': float(ticker['high24h']),
                    'low24h': float(ticker['low24h'])
                }
            
            return ticker_data
            
        except Exception:
            # suppressed verbose error
            return {}
    
    def calculate_volatility_metrics(self, symbol: str, timeframe: str = "1") -> Optional[VolatilityMetrics]:
        """Calculate comprehensive volatility metrics for a pair"""
        try:
            # Get historical data
            klines = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,
                limit=self.volatility_lookback
            )
            
            if klines.get('retCode') != 0:
                return None
            
            df = pd.DataFrame(klines['result']['list'])
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
            df = df.iloc[::-1]  # Reverse to chronological order
            
            if len(df) < 100:  # Need sufficient data
                return None
            
            # Calculate True Range and ATR
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            
            # Calculate price changes for different timeframes
            df['price_change_1m'] = df['close'].pct_change(1)
            df['price_change_5m'] = df['close'].pct_change(5)
            df['price_change_15m'] = df['close'].pct_change(15)
            
            # Calculate volatility metrics
            avg_true_range = df['atr'].iloc[-1]
            volatility_1m = df['price_change_1m'].std() * np.sqrt(1440)  # Annualized
            volatility_5m = df['price_change_5m'].std() * np.sqrt(288)   # Annualized
            volatility_15m = df['price_change_15m'].std() * np.sqrt(96)  # Annualized
            
            # Calculate average moves
            avg_move_1m = abs(df['price_change_1m']).mean()
            avg_move_5m = abs(df['price_change_5m']).mean()
            avg_move_15m = abs(df['price_change_15m']).mean()
            
            # Calculate volatility percentile (compared to all pairs)
            volatility_percentile = self._calculate_volatility_percentile(volatility_1m)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(df)
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(df)
            
            # Overall volatility score (0-1, higher = more volatile)
            volatility_score = min(volatility_1m / 0.5, 1.0)  # Cap at 50% annual volatility
            
            return VolatilityMetrics(
                symbol=symbol,
                timeframe=timeframe,
                avg_true_range=avg_true_range,
                volatility_percentile=volatility_percentile,
                avg_move_1m=avg_move_1m,
                avg_move_5m=avg_move_5m,
                avg_move_15m=avg_move_15m,
                volatility_score=volatility_score,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ Error calculating volatility for {symbol}: {e}")
            return None
    
    def _calculate_volatility_percentile(self, volatility: float) -> float:
        """Calculate volatility percentile (placeholder - would need historical data)"""
        # In a real implementation, this would compare against historical volatility data
        # For now, return a mock percentile
        return min(volatility * 2, 1.0)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using multiple timeframes"""
        # Calculate EMA slopes
        ema_9 = df['close'].ewm(span=9).mean()
        ema_21 = df['close'].ewm(span=21).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        # Calculate slopes
        slope_9 = (ema_9.iloc[-1] - ema_9.iloc[-10]) / 10
        slope_21 = (ema_21.iloc[-1] - ema_21.iloc[-10]) / 10
        slope_50 = (ema_50.iloc[-1] - ema_50.iloc[-10]) / 10
        
        # Average slope as trend strength
        avg_slope = (slope_9 + slope_21 + slope_50) / 3
        return min(abs(avg_slope) * 1000, 1.0)  # Normalize to 0-1
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile metrics"""
        recent_volume = df['volume'].tail(50)
        avg_volume = recent_volume.mean()
        volume_std = recent_volume.std()
        
        return {
            'avg_volume': avg_volume,
            'volume_std': volume_std,
            'volume_trend': (recent_volume.iloc[-1] - recent_volume.iloc[0]) / recent_volume.iloc[0] if recent_volume.iloc[0] > 0 else 0,
            'volume_consistency': 1 - (volume_std / avg_volume) if avg_volume > 0 else 0
        }
    
    def score_pair_for_trading(self, symbol: str, pair_info: Dict, volatility_metrics: VolatilityMetrics) -> float:
        """Score a pair for trading suitability (0-1, higher = better)"""
        score = 0.0
        
        # Volume score (0-0.3)
        volume_24h = pair_info.get('volume_24h', 0)
        if volume_24h > self.min_volume_24h:
            volume_score = min(volume_24h / (self.min_volume_24h * 10), 1.0)
            score += volume_score * 0.3
        
        # Volatility score (0-0.4)
        volatility_score = volatility_metrics.volatility_score
        if self.min_volatility <= volatility_score <= self.max_volatility:
            # Prefer moderate volatility (not too low, not too high)
            optimal_volatility = 0.05  # 5% annual volatility
            vol_distance = abs(volatility_score - optimal_volatility)
            vol_score = max(0, 1 - (vol_distance / optimal_volatility))
            score += vol_score * 0.4
        
        # Trend strength score (0-0.2)
        trend_score = volatility_metrics.trend_strength
        score += trend_score * 0.2
        
        # Volume consistency score (0-0.1)
        volume_consistency = volatility_metrics.volume_profile.get('volume_consistency', 0)
        score += volume_consistency * 0.1
        
        return min(score, 1.0)
    
    def discover_and_score_pairs(self, max_pairs: int = 50) -> List[Tuple[str, float, PairInfo, VolatilityMetrics]]:
        """Discover and score all available pairs"""
        # print suppressed in production
        
        # Get available pairs
        pairs = self.get_available_pairs()
        if not pairs:
            return []
        
        # Filter by volume and get ticker data
        symbols = [p['symbol'] for p in pairs]
        ticker_data = self.get_pair_ticker_info(symbols)
        
        scored_pairs = []
        
        for pair in pairs:
            symbol = pair['symbol']
            ticker = ticker_data.get(symbol, {})
            
            # Skip if insufficient volume
            if ticker.get('volume24h', 0) < self.min_volume_24h:
                continue
            
            # Calculate volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(symbol)
            if not volatility_metrics:
                continue
            
            # Skip if volatility is outside acceptable range
            if not (self.min_volatility <= volatility_metrics.volatility_score <= self.max_volatility):
                continue
            
            # Create pair info
            pair_info = PairInfo(
                symbol=symbol,
                base_asset=pair['baseAsset'],
                quote_asset=pair['quoteAsset'],
                status='trading',
                volatility_score=volatility_metrics.volatility_score,
                volume_24h=ticker.get('volume24h', 0),
                price_change_24h=ticker.get('priceChange24h', 0),
                market_cap_rank=None,  # Would need additional API call
                last_updated=datetime.now(),
                trading_fees=pair['tradingFees'],
                min_trade_size=pair['minOrderQty'],
                max_trade_size=pair['maxOrderQty'],
                price_precision=pair['pricePrecision'],
                qty_precision=pair['qtyPrecision']
            )
            
            # Score the pair
            score = self.score_pair_for_trading(symbol, ticker, volatility_metrics)
            
            if score > 0.3:  # Minimum score threshold
                scored_pairs.append((symbol, score, pair_info, volatility_metrics))
        
        # Sort by score and return top pairs
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # print suppressed in production
        return scored_pairs[:max_pairs]
    
    def get_top_pairs(self, max_pairs: int = 20) -> List[str]:
        """Get top scoring pairs for trading"""
        if (self.last_discovery_update is None or 
            (datetime.now() - self.last_discovery_update).seconds > self.discovery_interval):
            
            scored_pairs = self.discover_and_score_pairs(max_pairs * 2)  # Get more to filter
            
            # Update tracked pairs
            for symbol, score, pair_info, volatility_metrics in scored_pairs:
                self.tracked_pairs[symbol] = pair_info
                self.volatility_cache[symbol] = volatility_metrics
            
            self.last_discovery_update = datetime.now()
        
        # Return top pairs
        return list(self.tracked_pairs.keys())[:max_pairs]
    
    def get_pair_volatility(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get volatility metrics for a specific pair"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # Calculate if not cached
        volatility_metrics = self.calculate_volatility_metrics(symbol)
        if volatility_metrics:
            self.volatility_cache[symbol] = volatility_metrics
        
        return volatility_metrics
    
    def update_pair_status(self):
        """Check for new listings and delistings"""
        try:
            # Get current trading pairs
            current_pairs = set(self.get_available_pairs())
            current_symbols = {p['symbol'] for p in current_pairs}
            tracked_symbols = set(self.tracked_pairs.keys())
            
            # Find new pairs
            new_pairs = current_symbols - tracked_symbols
            if new_pairs:
                # print suppressed in production
                # Re-run discovery to include new pairs
                self.discover_and_score_pairs()
            
            # Find delisted pairs
            delisted_pairs = tracked_symbols - current_symbols
            if delisted_pairs:
                # print suppressed in production
                # Remove from tracking
                for symbol in delisted_pairs:
                    if symbol in self.tracked_pairs:
                        del self.tracked_pairs[symbol]
                    if symbol in self.volatility_cache:
                        del self.volatility_cache[symbol]
            
        except Exception as e:
            print(f"❌ Error updating pair status: {e}")
    
    def get_pair_recommendations(self, trading_style: str = "scalping") -> List[Dict]:
        """Get pair recommendations based on trading style"""
        recommendations = []
        
        # Get ALL available pairs - no limit
        top_pairs = self.get_top_pairs(1000)  # Increased to 1000 to get all pairs
        
        for symbol in top_pairs:
            if symbol not in self.volatility_cache:
                continue
            
            volatility = self.volatility_cache[symbol]
            pair_info = self.tracked_pairs[symbol]
            
            # Include ALL pairs - no filtering for maximum coverage
            if trading_style == "scalping":
                # Include every pair that has volatility data
                recommendations.append({
                    'symbol': symbol,
                    'reason': 'Available for scalping',
                    'volatility_score': volatility.volatility_score,
                    'avg_move_1m': volatility.avg_move_1m,
                    'volume_24h': pair_info.volume_24h
                })
            
            elif trading_style == "day":
                # Prefer moderate volatility
                if (0.02 <= volatility.volatility_score <= 0.08 and
                    volatility.avg_move_5m > 0.005):
                    recommendations.append({
                        'symbol': symbol,
                        'reason': 'Moderate volatility, suitable for day trading',
                        'volatility_score': volatility.volatility_score,
                        'avg_move_5m': volatility.avg_move_5m,
                        'volume_24h': pair_info.volume_24h
                    })
            
            elif trading_style == "swing":
                # Prefer lower volatility, strong trends
                if (0.01 <= volatility.volatility_score <= 0.05 and
                    volatility.trend_strength > 0.3):
                    recommendations.append({
                        'symbol': symbol,
                        'reason': 'Low volatility, strong trend, suitable for swing trading',
                        'volatility_score': volatility.volatility_score,
                        'trend_strength': volatility.trend_strength,
                        'volume_24h': pair_info.volume_24h
                    })
        
        return recommendations[:10]  # Return top 10 recommendations


def main():
    """Test the pair discovery system"""
    discovery = PairDiscovery(use_testnet=True)
    
    print("🔍 Testing Pair Discovery System")
    print("=" * 50)
    
    # Get top pairs
    top_pairs = discovery.get_top_pairs(10)
    print(f"\nTop 10 pairs: {top_pairs}")
    
    # Get recommendations for different styles
    for style in ["scalping", "day", "swing"]:
        print(f"\n{style.upper()} recommendations:")
        recommendations = discovery.get_pair_recommendations(style)
        for rec in recommendations[:5]:
            print(f"  {rec['symbol']}: {rec['reason']} (Vol: {rec['volatility_score']:.3f})")
    
    # Test volatility calculation
    if top_pairs:
        symbol = top_pairs[0]
        volatility = discovery.get_pair_volatility(symbol)
        if volatility:
            print(f"\nVolatility metrics for {symbol}:")
            print(f"  ATR: {volatility.avg_true_range:.4f}")
            print(f"  Avg Move 1m: {volatility.avg_move_1m:.4f}")
            print(f"  Volatility Score: {volatility.volatility_score:.3f}")
            print(f"  Trend Strength: {volatility.trend_strength:.3f}")


if __name__ == "__main__":
    main()
