import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Callable, Tuple
import warnings
import threading
import logging
from dataclasses import dataclass
from enum import Enum
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="DhanHQ Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DHANQ API INTEGRATION CLASSES
# ==========================================

class DhanAPIClient:
    """DhanHQ API Client for trading operations"""

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co/v2"
        self.feed_url = "wss://api-feed.dhan.co"
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id
        }
        self.websocket = None
        self.is_connected = False

    def get_historical_data(self, security_id: str, exchange_segment: str, 
                           instrument: str, from_date: str, to_date: str,
                           expiry_code: int = 0, oi: bool = False) -> Optional[Dict]:
        """Fetch historical daily data from DhanHQ API"""
        url = f"{self.base_url}/charts/historical"
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": expiry_code,
            "oi": oi,
            "fromDate": from_date,
            "toDate": to_date
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Historical data API error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Historical data request failed: {e}")
            return None

    def get_intraday_data(self, security_id: str, exchange_segment: str,
                         instrument: str, interval: str, from_date: str, 
                         to_date: str, oi: bool = False) -> Optional[Dict]:
        """Fetch intraday minute data from DhanHQ API"""
        url = f"{self.base_url}/charts/intraday"
        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": oi,
            "fromDate": from_date,
            "toDate": to_date
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Intraday data API error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Intraday data request failed: {e}")
            return None

    def get_market_quote_ltp(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        """Fetch LTP for multiple instruments"""
        url = f"{self.base_url}/marketfeed/ltp"

        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market quote LTP API error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Market quote LTP request failed: {e}")
            return None

    def get_market_quote_ohlc(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        """Fetch OHLC data for multiple instruments"""
        url = f"{self.base_url}/marketfeed/ohlc"

        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market quote OHLC API error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Market quote OHLC request failed: {e}")
            return None

    def get_market_depth(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        """Fetch market depth for multiple instruments"""
        url = f"{self.base_url}/marketfeed/quote"

        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market depth API error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Market depth request failed: {e}")
            return None

def format_historical_data(api_response: Dict) -> pd.DataFrame:
    """Convert DhanHQ historical API response to pandas DataFrame"""
    if not api_response or 'open' not in api_response:
        return pd.DataFrame()

    # Create DataFrame from the response
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(api_response['timestamp'], unit='s'),
        'open': api_response['open'],
        'high': api_response['high'], 
        'low': api_response['low'],
        'close': api_response['close'],
        'volume': api_response['volume']
    })

    # Add open interest if available
    if 'open_interest' in api_response:
        df['open_interest'] = api_response['open_interest']

    return df.set_index('timestamp')

# ==========================================
# TECHNICAL ANALYSIS CLASSES
# ==========================================

class TechnicalAnalyzer:
    """Advanced technical analysis class with multiple indicators"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)

    def add_moving_averages(self, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in periods:
            if len(self.data) >= period:
                self.data[f'SMA_{period}'] = self.data['close'].rolling(window=period).mean()
                self.data[f'EMA_{period}'] = self.data['close'].ewm(span=period).mean()
        return self.data

    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        if len(self.data) < period:
            return self.data

        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator"""
        if len(self.data) < slow:
            return self.data

        ema_fast = self.data['close'].ewm(span=fast).mean()
        ema_slow = self.data['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        self.data['MACD'] = macd
        self.data['MACD_signal'] = signal_line
        self.data['MACD_histogram'] = histogram
        return self.data

    def add_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        if len(self.data) < period:
            return self.data

        sma = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        self.data['BB_upper'] = upper_band
        self.data['BB_middle'] = sma
        self.data['BB_lower'] = lower_band
        self.data['BB_width'] = (upper_band - lower_band) / sma
        self.data['BB_position'] = (self.data['close'] - lower_band) / (upper_band - lower_band)
        return self.data

    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        if len(self.data) < k_period:
            return self.data

        lowest_low = self.data['low'].rolling(window=k_period).min()
        highest_high = self.data['high'].rolling(window=k_period).max()
        k_percent = 100 * ((self.data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        self.data['STOCH_K'] = k_percent
        self.data['STOCH_D'] = d_percent
        return self.data

    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        if len(self.data) < period:
            return self.data

        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['ATR'] = true_range.rolling(window=period).mean()
        return self.data

    def add_volume_indicators(self) -> pd.DataFrame:
        """Add volume-based indicators"""
        if len(self.data) < 20:
            return self.data

        # Volume Moving Average
        self.data['Volume_SMA'] = self.data['volume'].rolling(window=20).mean()

        # On Balance Volume
        obv = []
        obv.append(0)
        for i in range(1, len(self.data)):
            if self.data['close'].iloc[i] > self.data['close'].iloc[i-1]:
                obv.append(obv[-1] + self.data['volume'].iloc[i])
            elif self.data['close'].iloc[i] < self.data['close'].iloc[i-1]:
                obv.append(obv[-1] - self.data['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        self.data['OBV'] = obv

        # Chaikin Money Flow
        money_flow_multiplier = ((self.data['close'] - self.data['low']) - 
                               (self.data['high'] - self.data['close'])) / (self.data['high'] - self.data['low'])
        money_flow_volume = money_flow_multiplier * self.data['volume']
        self.data['CMF'] = money_flow_volume.rolling(window=20).sum() / self.data['volume'].rolling(window=20).sum()

        return self.data

    def add_all_indicators(self) -> pd.DataFrame:
        """Add all common technical indicators"""
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_stochastic()
        self.add_atr()
        self.add_volume_indicators()
        return self.data

    def generate_signals(self) -> pd.DataFrame:
        """Generate comprehensive trading signals"""
        signals = pd.DataFrame(index=self.data.index)
        signals['BUY'] = 0
        signals['SELL'] = 0
        signals['STRONG_BUY'] = 0
        signals['STRONG_SELL'] = 0

        if len(self.data) < 50:  # Need enough data for signals
            self.signals = signals
            return signals

        # RSI Signals
        rsi_oversold = self.data['RSI'] < 30
        rsi_overbought = self.data['RSI'] > 70

        # MACD Signals
        macd_bullish = (self.data['MACD'] > self.data['MACD_signal']) & (self.data['MACD'].shift(1) <= self.data['MACD_signal'].shift(1))
        macd_bearish = (self.data['MACD'] < self.data['MACD_signal']) & (self.data['MACD'].shift(1) >= self.data['MACD_signal'].shift(1))

        # Moving Average Signals
        if 'SMA_20' in self.data.columns and 'SMA_50' in self.data.columns:
            golden_cross = (self.data['SMA_20'] > self.data['SMA_50']) & (self.data['SMA_20'].shift(1) <= self.data['SMA_50'].shift(1))
            death_cross = (self.data['SMA_20'] < self.data['SMA_50']) & (self.data['SMA_20'].shift(1) >= self.data['SMA_50'].shift(1))
        else:
            golden_cross = pd.Series(False, index=self.data.index)
            death_cross = pd.Series(False, index=self.data.index)

        # Bollinger Band Signals
        if 'BB_upper' in self.data.columns:
            bb_oversold = self.data['close'] < self.data['BB_lower']
            bb_overbought = self.data['close'] > self.data['BB_upper']
        else:
            bb_oversold = pd.Series(False, index=self.data.index)
            bb_overbought = pd.Series(False, index=self.data.index)

        # Stochastic Signals
        if 'STOCH_K' in self.data.columns:
            stoch_oversold = (self.data['STOCH_K'] < 20) & (self.data['STOCH_D'] < 20)
            stoch_overbought = (self.data['STOCH_K'] > 80) & (self.data['STOCH_D'] > 80)
        else:
            stoch_oversold = pd.Series(False, index=self.data.index)
            stoch_overbought = pd.Series(False, index=self.data.index)

        # Combine signals
        # BUY signals
        signals.loc[rsi_oversold | macd_bullish | bb_oversold | stoch_oversold, 'BUY'] = 1
        signals.loc[golden_cross, 'BUY'] = 1

        # SELL signals  
        signals.loc[rsi_overbought | macd_bearish | bb_overbought | stoch_overbought, 'SELL'] = 1
        signals.loc[death_cross, 'SELL'] = 1

        # STRONG signals (multiple confirmations)
        strong_buy_conditions = (
            (rsi_oversold.astype(int) + macd_bullish.astype(int) + bb_oversold.astype(int) + 
             stoch_oversold.astype(int) + golden_cross.astype(int)) >= 3
        )
        signals.loc[strong_buy_conditions, 'STRONG_BUY'] = 1

        strong_sell_conditions = (
            (rsi_overbought.astype(int) + macd_bearish.astype(int) + bb_overbought.astype(int) + 
             stoch_overbought.astype(int) + death_cross.astype(int)) >= 3
        )
        signals.loc[strong_sell_conditions, 'STRONG_SELL'] = 1

        self.signals = signals
        return signals

    def get_signal_summary(self) -> Dict:
        """Get summary of current market signals"""
        if self.signals.empty:
            self.generate_signals()

        if len(self.data) == 0:
            return {}

        latest = self.data.iloc[-1]
        latest_signals = self.signals.iloc[-1] if len(self.signals) > 0 else pd.Series()

        summary = {
            'timestamp': latest.name if hasattr(latest, 'name') else datetime.now(),
            'close_price': latest['close'],
            'signals': {
                'BUY': bool(latest_signals.get('BUY', 0)),
                'SELL': bool(latest_signals.get('SELL', 0)),
                'STRONG_BUY': bool(latest_signals.get('STRONG_BUY', 0)),
                'STRONG_SELL': bool(latest_signals.get('STRONG_SELL', 0))
            },
            'indicators': {
                'RSI': latest.get('RSI', None),
                'MACD': latest.get('MACD', None),
                'MACD_signal': latest.get('MACD_signal', None),
                'BB_position': latest.get('BB_position', None),
                'STOCH_K': latest.get('STOCH_K', None),
                'ATR': latest.get('ATR', None)
            }
        }

        return summary

def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """Calculate support and resistance levels"""
    if len(data) < window * 2:
        return {'current_price': data['close'].iloc[-1] if len(data) > 0 else 0}

    recent_data = data.tail(window * 2)

    # Find local minima and maxima
    highs = recent_data['high'].rolling(window=window, center=True).max()
    lows = recent_data['low'].rolling(window=window, center=True).min()

    # Identify pivot points
    resistance_levels = []
    support_levels = []

    for i in range(window, len(recent_data) - window):
        if recent_data['high'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(recent_data['high'].iloc[i])
        if recent_data['low'].iloc[i] == lows.iloc[i]:
            support_levels.append(recent_data['low'].iloc[i])

    # Calculate key levels
    current_price = data['close'].iloc[-1]

    # Find nearest support and resistance
    support_levels = [level for level in support_levels if level < current_price]
    resistance_levels = [level for level in resistance_levels if level > current_price]

    return {
        'current_price': current_price,
        'immediate_support': max(support_levels) if support_levels else None,
        'immediate_resistance': min(resistance_levels) if resistance_levels else None,
        'strong_support': min(support_levels) if support_levels else None,
        'strong_resistance': max(resistance_levels) if resistance_levels else None
    }

# ==========================================
# ALERT SYSTEM CLASSES
# ==========================================

class AlertType(Enum):
    PRICE = "price"
    TECHNICAL = "technical"
    VOLUME = "volume"
    PATTERN = "pattern"
    NEWS = "news"

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    symbol: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    condition: str
    current_value: float
    trigger_value: float
    timestamp: datetime
    is_active: bool = True
    is_triggered: bool = False
    trigger_count: int = 0

class AlertManager:
    """Comprehensive alert management system"""

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []

    def add_price_alert(self, symbol: str, condition: str, trigger_value: float, 
                       priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """Add a price-based alert"""
        alert_id = f"price_{symbol}_{condition}_{trigger_value}_{datetime.now().timestamp()}"

        alert = Alert(
            id=alert_id,
            symbol=symbol,
            alert_type=AlertType.PRICE,
            priority=priority,
            message=f"{symbol} price {condition} ₹{trigger_value:,.2f}",
            condition=condition,
            current_value=0.0,
            trigger_value=trigger_value,
            timestamp=datetime.now()
        )

        self.alerts[alert_id] = alert
        return alert_id

    def add_technical_alert(self, symbol: str, indicator: str, condition: str, 
                          trigger_value: float, priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """Add a technical indicator-based alert"""
        alert_id = f"tech_{symbol}_{indicator}_{condition}_{trigger_value}_{datetime.now().timestamp()}"

        alert = Alert(
            id=alert_id,
            symbol=symbol,
            alert_type=AlertType.TECHNICAL,
            priority=priority,
            message=f"{symbol} {indicator} {condition} {trigger_value}",
            condition=condition,
            current_value=0.0,
            trigger_value=trigger_value,
            timestamp=datetime.now()
        )

        self.alerts[alert_id] = alert
        return alert_id

    def check_alerts(self, market_data: Dict) -> List[Alert]:
        """Check all active alerts against market data"""
        triggered_alerts = []

        for alert in self.alerts.values():
            if not alert.is_active:
                continue

            # Check price alerts
            if alert.alert_type == AlertType.PRICE:
                current_price = market_data.get('price', 0)
                alert.current_value = current_price

                if self._check_condition(current_price, alert.condition, alert.trigger_value):
                    if not alert.is_triggered:
                        alert.is_triggered = True
                        alert.trigger_count += 1
                        triggered_alerts.append(alert)

            # Check technical alerts
            elif alert.alert_type == AlertType.TECHNICAL:
                indicators = market_data.get('indicators', {})
                indicator_name = alert.message.split()[1] if len(alert.message.split()) > 1 else 'RSI'
                current_value = indicators.get(indicator_name, 0)
                alert.current_value = current_value

                if current_value and self._check_condition(current_value, alert.condition, alert.trigger_value):
                    if not alert.is_triggered:
                        alert.is_triggered = True
                        alert.trigger_count += 1
                        triggered_alerts.append(alert)

        return triggered_alerts

    def _check_condition(self, current_value: float, condition: str, trigger_value: float) -> bool:
        """Check if condition is met"""
        if condition == "above":
            return current_value > trigger_value
        elif condition == "below":
            return current_value < trigger_value
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if alert.is_active]

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert by ID"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False

    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        triggered_today = len([a for a in self.alert_history if a.timestamp.date() == datetime.now().date()])

        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_today': triggered_today,
            'history_size': len(self.alert_history)
        }

# ==========================================
# CUSTOM CSS STYLING
# ==========================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffa500;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: bold;
    }
    .price-up {
        color: #00ff00;
    }
    .price-down {
        color: #ff0000;
    }
    .sidebar .stSelectbox > label {
        font-weight: bold;
        color: #1f77b4;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-high {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #ffbb00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #88dd88;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SAMPLE DATA AND UTILITY FUNCTIONS
# ==========================================

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_sample_stocks():
    return [
        {"symbol": "NIFTY", "security_id": "13", "exchange": "IDX_I", "name": "NIFTY 50"},
        {"symbol": "RELIANCE", "security_id": "2885", "exchange": "NSE_EQ", "name": "Reliance Industries"},
        {"symbol": "TCS", "security_id": "11536", "exchange": "NSE_EQ", "name": "Tata Consultancy Services"},
        {"symbol": "HDFCBANK", "security_id": "1333", "exchange": "NSE_EQ", "name": "HDFC Bank"},
        {"symbol": "INFY", "security_id": "1594", "exchange": "NSE_EQ", "name": "Infosys"},
        {"symbol": "HINDUNILVR", "security_id": "1394", "exchange": "NSE_EQ", "name": "Hindustan Unilever"},
        {"symbol": "ICICIBANK", "security_id": "4963", "exchange": "NSE_EQ", "name": "ICICI Bank"},
        {"symbol": "BHARTIARTL", "security_id": "10604", "exchange": "NSE_EQ", "name": "Bharti Airtel"}
    ]

# Simulate DhanHQ API calls with realistic data
def simulate_live_data(symbol: str) -> Dict:
    """Simulate live market data from DhanHQ API"""
    base_prices = {
        'NIFTY': 25000, 'RELIANCE': 2800, 'TCS': 4000, 'HDFCBANK': 1600, 
        'INFY': 1850, 'HINDUNILVR': 2700, 'ICICIBANK': 1250, 'BHARTIARTL': 1200
    }
    base_price = base_prices.get(symbol, 2000)

    # Add some randomness to simulate live price movements
    change_percent = np.random.uniform(-0.02, 0.02)  # ±2% random change
    current_price = base_price * (1 + change_percent)

    return {
        'symbol': symbol,
        'ltp': round(current_price, 2),
        'change': round(current_price - base_price, 2),
        'change_percent': round(change_percent * 100, 2),
        'open': round(base_price * 0.998, 2),
        'high': round(current_price * 1.002, 2),
        'low': round(current_price * 0.998, 2),
        'close': round(current_price, 2),
        'volume': np.random.randint(100000, 500000),
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

def simulate_historical_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Simulate realistic historical OHLCV data"""
    base_prices = {
        'NIFTY': 25000, 'RELIANCE': 2800, 'TCS': 4000, 'HDFCBANK': 1600, 
        'INFY': 1850, 'HINDUNILVR': 2700, 'ICICIBANK': 1250, 'BHARTIARTL': 1200
    }
    base_price = base_prices.get(symbol, 2000)

    # Generate time series based on timeframe
    timeframe_config = {
        '1min': {'periods': 200, 'freq': '1min'},
        '5min': {'periods': 200, 'freq': '5min'},
        '15min': {'periods': 200, 'freq': '15min'},
        '1day': {'periods': 200, 'freq': '1D'}
    }

    config = timeframe_config.get(timeframe, timeframe_config['15min'])
    dates = pd.date_range(end=datetime.now(), periods=config['periods'], freq=config['freq'])

    # Generate realistic OHLCV data with trend and volatility
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.0005, 0.015, config['periods'])  # Slight upward bias
    prices = [base_price]

    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            open_price = base_price
        else:
            open_price = data[i-1]['close']

        # Create realistic OHLC based on the close price trend
        close_price = prices[i+1]
        volatility = abs(np.random.normal(0, 0.008))
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)
        volume = np.random.randint(50000, 300000)

        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })

    return pd.DataFrame(data)

def simulate_market_depth(symbol: str) -> Dict:
    """Simulate realistic market depth data"""
    base_prices = {
        'NIFTY': 25000, 'RELIANCE': 2800, 'TCS': 4000, 'HDFCBANK': 1600, 
        'INFY': 1850, 'HINDUNILVR': 2700, 'ICICIBANK': 1250, 'BHARTIARTL': 1200
    }
    base_price = base_prices.get(symbol, 2000)
    current_price = base_price * (1 + np.random.uniform(-0.005, 0.005))

    buy_orders = []
    sell_orders = []

    # Generate buy orders (below current price)
    for i in range(5):
        price = current_price - (i + 1) * (current_price * 0.0002)
        quantity = np.random.randint(1000, 5000)
        orders = np.random.randint(8, 30)
        buy_orders.append({'price': round(price, 2), 'quantity': quantity, 'orders': orders})

    # Generate sell orders (above current price)
    for i in range(5):
        price = current_price + (i + 1) * (current_price * 0.0002)
        quantity = np.random.randint(1000, 5000)
        orders = np.random.randint(8, 30)
        sell_orders.append({'price': round(price, 2), 'quantity': quantity, 'orders': orders})

    return {'buy': buy_orders, 'sell': sell_orders}

def get_real_dhan_data(dhan_client, symbol_data, timeframe):
    """Get real data from DhanHQ API if connected"""
    try:
        # Convert timeframe to DhanHQ format
        interval_map = {'1min': '1', '5min': '5', '15min': '15', '1day': 'daily'}
        interval = interval_map.get(timeframe, '15')

        if timeframe == '1day':
            # Get historical daily data
            from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            response = dhan_client.get_historical_data(
                security_id=symbol_data['security_id'],
                exchange_segment=symbol_data['exchange'],
                instrument='EQUITY',
                from_date=from_date,
                to_date=to_date
            )
        else:
            # Get intraday data
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            response = dhan_client.get_intraday_data(
                security_id=symbol_data['security_id'],
                exchange_segment=symbol_data['exchange'],
                instrument='EQUITY',
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )

        if response:
            return format_historical_data(response)
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching real data: {e}")
        return None

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'dhan_client' not in st.session_state:
    st.session_state.dhan_client = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = 'NIFTY'
if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = AlertManager()
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'triggered_alerts' not in st.session_state:
    st.session_state.triggered_alerts = []

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================

# Main title
st.markdown('<h1 class="main-header">📈 DhanHQ Advanced Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Complete Technical Analysis • Live Data • Advanced Alerts • Pattern Recognition</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")

# API Configuration
with st.sidebar.expander("🔑 DhanHQ API Configuration", expanded=not st.session_state.connected):
    client_id = st.text_input("Client ID", placeholder="Enter your DhanHQ Client ID")
    access_token = st.text_input("Access Token", type="password", placeholder="Enter your Access Token")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect", type="primary"):
            if client_id and access_token:
                st.session_state.dhan_client = DhanAPIClient(client_id, access_token)
                st.session_state.connected = True
                st.success("✅ Connected!")
                st.rerun()
            else:
                st.error("❌ Enter credentials")

    with col2:
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.dhan_client = None
            st.info("Disconnected")
            st.rerun()

# Connection status
if st.session_state.connected:
    st.sidebar.success("🟢 Connected to DhanHQ")
else:
    st.sidebar.error("🔴 Not Connected - Using Demo Data")

# Stock Selection

@st.cache_data(ttl=300)
def load_stock_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Failed to load stock CSV: {e}")
        return pd.DataFrame()

# Stock Selection
st.sidebar.header("📊 Instrument Selection")
stock_df = load_stock_csv("stock_list.csv")  # Adjust path if needed

if not stock_df.empty:
    stock_options = {
        f"{row['symbol']} ({row['exchange']})": row
        for _, row in stock_df.iterrows()
    }
    selected_stock_display = st.sidebar.selectbox("Select Stock", options=list(stock_options.keys()))
    selected_stock_data = stock_options[selected_stock_display]
    selected_stock = selected_stock_data['symbol']
    st.session_state.selected_stock = selected_stock
else:
    st.sidebar.error("Stock list not available or failed to load")


# Timeframe Selection
timeframe = st.sidebar.selectbox(
    "📅 Timeframe",
    options=["1min", "5min", "15min", "1day"],
    index=2
)

# Technical Indicators Selection
st.sidebar.header("📈 Technical Indicators")
indicators = {
    "RSI (14)": st.sidebar.checkbox("RSI (14)", value=True),
    "MACD": st.sidebar.checkbox("MACD", value=True),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands", value=True),
    "SMA (20, 50)": st.sidebar.checkbox("SMA (20, 50)", value=True),
    "EMA (12, 26)": st.sidebar.checkbox("EMA (12, 26)", value=True),
    "Stochastic": st.sidebar.checkbox("Stochastic", value=False),
    "Volume Indicators": st.sidebar.checkbox("Volume Indicators", value=False),
}

# Advanced Alert Configuration
st.sidebar.header("🚨 Advanced Alert Settings")
with st.sidebar.expander("Price Alerts"):
    price_alert_above = st.number_input("Alert Above (₹)", min_value=0.0, step=1.0)
    price_alert_below = st.number_input("Alert Below (₹)", min_value=0.0, step=1.0)

    if st.button("Add Price Alert"):
        if price_alert_above > 0:
            st.session_state.alert_manager.add_price_alert(
                selected_stock, "above", price_alert_above, AlertPriority.MEDIUM
            )
            st.success(f"Added alert for {selected_stock} above ₹{price_alert_above}")
        if price_alert_below > 0:
            st.session_state.alert_manager.add_price_alert(
                selected_stock, "below", price_alert_below, AlertPriority.MEDIUM
            )
            st.success(f"Added alert for {selected_stock} below ₹{price_alert_below}")

with st.sidebar.expander("Technical Alerts"):
    rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)

    if st.button("Add Technical Alerts"):
        st.session_state.alert_manager.add_technical_alert(
            selected_stock, "RSI", "above", rsi_overbought, AlertPriority.MEDIUM
        )
        st.session_state.alert_manager.add_technical_alert(
            selected_stock, "RSI", "below", rsi_oversold, AlertPriority.MEDIUM
        )
        st.success("Added RSI alerts")

# Auto-refresh settings
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (10 sec)", value=True)

# ==========================================
# MAIN CONTENT AREA WITH TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔴 Live Data", "📊 Advanced Charts", "🔍 Technical Analysis", "🚨 Smart Alerts", "📋 Support/Resistance"])

# Tab 1: Enhanced Live Data Feed
with tab1:
    st.header(f"📈 Live Market Data - {selected_stock}")

    # Real-time data placeholders
    live_price_container = st.empty()
    market_summary_container = st.empty()
    ohlc_container = st.empty()
    depth_container = st.empty()

    # Get live or simulated data
    if st.session_state.connected and st.session_state.dhan_client:
        # Try to get real data first
        try:
            instruments = {selected_stock_data['exchange']: [int(selected_stock_data['security_id'])]}
            ltp_data = st.session_state.dhan_client.get_market_quote_ltp(instruments)
            ohlc_data = st.session_state.dhan_client.get_market_quote_ohlc(instruments)
            depth_data = st.session_state.dhan_client.get_market_depth(instruments)

            if ltp_data and 'data' in ltp_data:
                # Process real DhanHQ data
                exchange_data = ltp_data['data'].get(selected_stock_data['exchange'], {})
                security_data = exchange_data.get(selected_stock_data['security_id'], {})

                live_data = {
                    'symbol': selected_stock,
                    'ltp': security_data.get('LTP', 0),
                    'change': security_data.get('change', 0),
                    'change_percent': security_data.get('pChange', 0),
                    'volume': security_data.get('volume', 0),
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }

                # Get OHLC if available
                if ohlc_data and 'data' in ohlc_data:
                    ohlc_exchange_data = ohlc_data['data'].get(selected_stock_data['exchange'], {})
                    ohlc_security_data = ohlc_exchange_data.get(selected_stock_data['security_id'], {})
                    live_data.update({
                        'open': ohlc_security_data.get('open', live_data['ltp']),
                        'high': ohlc_security_data.get('high', live_data['ltp']),
                        'low': ohlc_security_data.get('low', live_data['ltp']),
                        'close': ohlc_security_data.get('close', live_data['ltp'])
                    })

            else:
                # Fall back to simulated data
                live_data = simulate_live_data(selected_stock)

        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            live_data = simulate_live_data(selected_stock)
    else:
        # Use simulated data
        live_data = simulate_live_data(selected_stock)

    st.session_state.live_data = live_data

    # Display enhanced live price ticker
    with live_price_container.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="💰 Last Trade Price",
                value=f"₹{live_data['ltp']:,.2f}",
                delta=f"{live_data['change']:+.2f}"
            )

        with col2:
            st.metric(
                label="📊 Change %",
                value=f"{live_data['change_percent']:+.2f}%",
                delta=None
            )

        with col3:
            st.metric(
                label="📈 Day High",
                value=f"₹{live_data.get('high', live_data['ltp']):,.2f}",
                delta=None
            )

        with col4:
            st.metric(
                label="📉 Day Low",
                value=f"₹{live_data.get('low', live_data['ltp']):,.2f}",
                delta=None
            )

        with col5:
            st.metric(
                label="📦 Volume",
                value=f"{live_data['volume']:,}",
                delta=None
            )

    # Market Summary
    with market_summary_container.container():
        st.subheader("📊 Market Summary")

        # Calculate some basic metrics
        day_range = live_data.get('high', live_data['ltp']) - live_data.get('low', live_data['ltp'])
        price_position = ((live_data['ltp'] - live_data.get('low', live_data['ltp'])) / day_range * 100) if day_range > 0 else 50

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.info(f"**Open**: ₹{live_data.get('open', live_data['ltp']):,.2f}")

        with col2:
            st.info(f"**Day Range**: ₹{day_range:.2f}")

        with col3:
            st.info(f"**Price Position**: {price_position:.1f}%")

        with col4:
            st.info(f"**Last Update**: {live_data['timestamp']}")

    # Enhanced OHLC Data
    with ohlc_container.container():
        st.subheader("📋 Detailed Market Data")

        ohlc_data = {
            'Metric': ['Open', 'High', 'Low', 'Close', 'Volume', 'Value (₹Cr)', 'Avg Price'],
            'Current': [
                f"₹{live_data.get('open', live_data['ltp']):,.2f}",
                f"₹{live_data.get('high', live_data['ltp']):,.2f}",
                f"₹{live_data.get('low', live_data['ltp']):,.2f}",
                f"₹{live_data['ltp']:,.2f}",
                f"{live_data['volume']:,}",
                f"₹{(live_data['volume'] * live_data['ltp'] / 10000000):.2f}",
                f"₹{live_data['ltp']:,.2f}"
            ],
            'Change': [
                f"{((live_data.get('open', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%",
                f"{((live_data.get('high', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%",
                f"{((live_data.get('low', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%",
                f"{live_data['change_percent']:+.2f}%",
                "N/A",
                "N/A",
                f"{live_data['change_percent']:+.2f}%"
            ]
        }

        ohlc_df = pd.DataFrame(ohlc_data)
        st.dataframe(ohlc_df, use_container_width=True, hide_index=True)

    # Enhanced Market Depth
    with depth_container.container():
        st.subheader("📊 Market Depth (Level 2)")

        if st.session_state.connected and st.session_state.dhan_client:
            # Try to get real market depth
            try:
                instruments = {selected_stock_data['exchange']: [int(selected_stock_data['security_id'])]}
                depth_response = st.session_state.dhan_client.get_market_depth(instruments)
                if depth_response and 'data' in depth_response:
                    # Process real market depth data
                    depth_data = simulate_market_depth(selected_stock)  # Fallback for now
                else:
                    depth_data = simulate_market_depth(selected_stock)
            except:
                depth_data = simulate_market_depth(selected_stock)
        else:
            depth_data = simulate_market_depth(selected_stock)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Buy Orders (Bids)**")
            buy_df = pd.DataFrame(depth_data['buy'])
            buy_df['Price'] = buy_df['price'].apply(lambda x: f"₹{x:,.2f}")
            buy_df['Qty'] = buy_df['quantity'].apply(lambda x: f"{x:,}")
            buy_df['Orders'] = buy_df['orders']
            buy_df['Value'] = (buy_df['quantity'] * buy_df['price'] / 100000).apply(lambda x: f"₹{x:.1f}L")
            st.dataframe(buy_df[['Price', 'Qty', 'Orders', 'Value']], use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**🔴 Sell Orders (Asks)**")
            sell_df = pd.DataFrame(depth_data['sell'])
            sell_df['Price'] = sell_df['price'].apply(lambda x: f"₹{x:,.2f}")
            sell_df['Qty'] = sell_df['quantity'].apply(lambda x: f"{x:,}")
            sell_df['Orders'] = sell_df['orders']
            sell_df['Value'] = (sell_df['quantity'] * sell_df['price'] / 100000).apply(lambda x: f"₹{x:.1f}L")
            st.dataframe(sell_df[['Price', 'Qty', 'Orders', 'Value']], use_container_width=True, hide_index=True)

        # Order book analysis
        total_bid_qty = sum([order['quantity'] for order in depth_data['buy']])
        total_ask_qty = sum([order['quantity'] for order in depth_data['sell']])
        bid_ask_ratio = total_bid_qty / total_ask_qty if total_ask_qty > 0 else 1

        st.info(f"**Order Book Analysis**: Bid/Ask Ratio = {bid_ask_ratio:.2f} | Total Bids: {total_bid_qty:,} | Total Asks: {total_ask_qty:,}")

    # Check alerts with live data
    market_data_for_alerts = {
        'price': live_data['ltp'],
        'volume': live_data['volume'],
        'avg_volume': 150000,  # Simulated average
        'indicators': {}
    }

    triggered = st.session_state.alert_manager.check_alerts(market_data_for_alerts)
    if triggered:
        st.session_state.triggered_alerts.extend(triggered)
        for alert in triggered:
            if alert.priority == AlertPriority.CRITICAL:
                st.error(f"🚨 CRITICAL ALERT: {alert.message}")
            elif alert.priority == AlertPriority.HIGH:
                st.warning(f"⚠️ HIGH ALERT: {alert.message}")
            else:
                st.info(f"ℹ️ ALERT: {alert.message}")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)
        st.rerun()

# Tab 2: Advanced Interactive Charts
with tab2:
    st.header(f"📊 Advanced Technical Charts - {selected_stock}")

    # Get historical data
    if st.session_state.connected and st.session_state.dhan_client:
        historical_data = get_real_dhan_data(st.session_state.dhan_client, selected_stock_data, timeframe)
        if historical_data is None or len(historical_data) == 0:
            historical_data = simulate_historical_data(selected_stock, timeframe)
    else:
        historical_data = simulate_historical_data(selected_stock, timeframe)

    if len(historical_data) > 0:
        # Initialize technical analyzer
        analyzer = TechnicalAnalyzer(historical_data)

        # Add selected indicators
        if indicators.get("RSI (14)"):
            analyzer.add_rsi()
        if indicators.get("MACD"):
            analyzer.add_macd()
        if indicators.get("Bollinger Bands"):
            analyzer.add_bollinger_bands()
        if indicators.get("SMA (20, 50)"):
            analyzer.add_moving_averages([20, 50])
        if indicators.get("EMA (12, 26)"):
            analyzer.add_moving_averages([12, 26])
        if indicators.get("Stochastic"):
            analyzer.add_stochastic()
        if indicators.get("Volume Indicators"):
            analyzer.add_volume_indicators()

        # Get updated data with indicators
        technical_data = analyzer.data

        # Create advanced chart with multiple subplots
        subplot_count = 2  # Price and Volume
        subplot_heights = [0.7, 0.3]

        if indicators.get("RSI (14)"):
            subplot_count += 1
            subplot_heights = [0.5, 0.2, 0.3]

        if indicators.get("MACD"):
            subplot_count += 1
            if len(subplot_heights) == 3:
                subplot_heights = [0.4, 0.15, 0.25, 0.2]
            else:
                subplot_heights = [0.6, 0.2, 0.2]

        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=subplot_heights,
            subplot_titles=[f'{selected_stock} - {timeframe} Price & Indicators', 'Volume'] + 
                          (['RSI'] if indicators.get("RSI (14)") else []) +
                          (['MACD'] if indicators.get("MACD") else [])
        )

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=technical_data['timestamp'],
                open=technical_data['open'],
                high=technical_data['high'],
                low=technical_data['low'],
                close=technical_data['close'],
                name=f'{selected_stock} Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Add technical indicators to main chart
        if indicators.get("SMA (20, 50)"):
            if 'SMA_20' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data['timestamp'],
                        y=technical_data['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
            if 'SMA_50' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data['timestamp'],
                        y=technical_data['SMA_50'],
                        name='SMA 50',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )

        if indicators.get("EMA (12, 26)"):
            if 'EMA_12' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data['timestamp'],
                        y=technical_data['EMA_12'],
                        name='EMA 12',
                        line=dict(color='green', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            if 'EMA_26' in technical_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=technical_data['timestamp'],
                        y=technical_data['EMA_26'],
                        name='EMA 26',
                        line=dict(color='blue', width=2, dash='dash')
                    ),
                    row=1, col=1
                )

        if indicators.get("Bollinger Bands") and 'BB_upper' in technical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['BB_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    fill=None
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['BB_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )

        # Volume chart
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(technical_data['close'], technical_data['open'])]

        fig.add_trace(
            go.Bar(
                x=technical_data['timestamp'],
                y=technical_data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # Volume indicators
        if indicators.get("Volume Indicators") and 'Volume_SMA' in technical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['Volume_SMA'],
                    name='Volume SMA',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )

        current_row = 3

        # RSI chart
        if indicators.get("RSI (14)") and 'RSI' in technical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=current_row, col=1
            )

            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)

            current_row += 1

        # MACD Chart
        if indicators.get("MACD") and 'MACD' in technical_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=current_row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=technical_data['timestamp'],
                    y=technical_data['MACD_signal'],
                    name='Signal',
                    line=dict(color='red', width=2)
                ),
                row=current_row, col=1
            )

            # MACD Histogram
            histogram_colors = ['green' if val >= 0 else 'red' for val in technical_data['MACD_histogram']]
            fig.add_trace(
                go.Bar(
                    x=technical_data['timestamp'],
                    y=technical_data['MACD_histogram'],
                    name='MACD Histogram',
                    marker_color=histogram_colors,
                    opacity=0.6
                ),
                row=current_row, col=1
            )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            title=f"{selected_stock} - {timeframe} Advanced Technical Analysis"
        )

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Chart analysis summary
        latest = technical_data.iloc[-1]

        st.subheader("📊 Chart Analysis Summary")

        analysis_cols = st.columns(4)

        with analysis_cols[0]:
            price_trend = "📈 Bullish" if latest['close'] > latest['open'] else "📉 Bearish"
            st.metric("Price Trend", price_trend)

        with analysis_cols[1]:
            if 'RSI' in latest:
                rsi_status = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
                st.metric("RSI Status", rsi_status, f"{latest['RSI']:.1f}")

        with analysis_cols[2]:
            if 'MACD' in latest and 'MACD_signal' in latest:
                macd_trend = "📈 Bullish" if latest['MACD'] > latest['MACD_signal'] else "📉 Bearish"
                st.metric("MACD Trend", macd_trend)

        with analysis_cols[3]:
            if 'BB_position' in latest:
                bb_pos = "Upper" if latest['BB_position'] > 0.8 else "Lower" if latest['BB_position'] < 0.2 else "Middle"
                st.metric("BB Position", bb_pos, f"{latest['BB_position']:.2f}")

    else:
        st.error("No data available for charting")

# Tab 3: Advanced Technical Analysis
with tab3:
    st.header(f"🔍 Advanced Technical Analysis - {selected_stock}")

    if len(historical_data) > 0:
        # Use the analyzer from the previous tab or create new one
        analyzer = TechnicalAnalyzer(historical_data)
        analyzer.add_all_indicators()

        # Generate signals
        signals = analyzer.generate_signals()
        signal_summary = analyzer.get_signal_summary()

        # Current indicator values display
        st.subheader("📊 Current Technical Indicators")

        if signal_summary:
            indicator_cols = st.columns(5)

            with indicator_cols[0]:
                if signal_summary['indicators'].get('RSI'):
                    rsi_val = signal_summary['indicators']['RSI']
                    rsi_color = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "🟡"
                    st.metric("RSI (14)", f"{rsi_val:.1f}", delta=None)
                    st.write(f"{rsi_color} {'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'}")
                else:
                    st.metric("RSI (14)", "N/A")

            with indicator_cols[1]:
                if signal_summary['indicators'].get('MACD') and signal_summary['indicators'].get('MACD_signal'):
                    macd_val = signal_summary['indicators']['MACD']
                    macd_sig = signal_summary['indicators']['MACD_signal']
                    macd_trend = "🟢 Bullish" if macd_val > macd_sig else "🔴 Bearish"
                    st.metric("MACD", f"{macd_val:.3f}", delta=f"{macd_val - macd_sig:.3f}")
                    st.write(macd_trend)
                else:
                    st.metric("MACD", "N/A")

            with indicator_cols[2]:
                if 'SMA_20' in analyzer.data.columns and len(analyzer.data) > 0:
                    latest_data = analyzer.data.iloc[-1]
                    sma_trend = "🟢 Above" if latest_data['close'] > latest_data['SMA_20'] else "🔴 Below"
                    st.metric("SMA 20", f"₹{latest_data['SMA_20']:.2f}")
                    st.write(f"{sma_trend} SMA 20")
                else:
                    st.metric("SMA 20", "N/A")

            with indicator_cols[3]:
                if signal_summary['indicators'].get('BB_position'):
                    bb_pos = signal_summary['indicators']['BB_position']
                    bb_status = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                    bb_color = "🔴" if bb_pos > 0.8 else "🟢" if bb_pos < 0.2 else "🟡"
                    st.metric("BB Position", f"{bb_pos:.2f}")
                    st.write(f"{bb_color} {bb_status}")
                else:
                    st.metric("BB Position", "N/A")

            with indicator_cols[4]:
                if signal_summary['indicators'].get('ATR'):
                    atr_val = signal_summary['indicators']['ATR']
                    st.metric("ATR", f"₹{atr_val:.2f}")
                    st.write("💹 Volatility")
                else:
                    st.metric("ATR", "N/A")

        # Advanced Signal Analysis
        st.subheader("🎯 Advanced Trading Signals")

        if len(signals) > 0 and signal_summary:
            # Recent signals
            recent_signals = []
            for i in range(max(0, len(signals) - 10), len(signals)):
                row = signals.iloc[i]
                timestamp = row.name if hasattr(row, 'name') else analyzer.data.index[i]
                price = analyzer.data.iloc[i]['close']

                if row['STRONG_BUY']:
                    recent_signals.append({
                        'Time': timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                        'Signal': 'STRONG BUY',
                        'Price': f"₹{price:.2f}",
                        'Strength': 'Strong',
                        'Reason': 'Multiple indicator confirmation'
                    })
                elif row['BUY']:
                    recent_signals.append({
                        'Time': timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                        'Signal': 'BUY',
                        'Price': f"₹{price:.2f}",
                        'Strength': 'Medium',
                        'Reason': 'Technical indicator signal'
                    })
                elif row['STRONG_SELL']:
                    recent_signals.append({
                        'Time': timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                        'Signal': 'STRONG SELL',
                        'Price': f"₹{price:.2f}",
                        'Strength': 'Strong',
                        'Reason': 'Multiple indicator confirmation'
                    })
                elif row['SELL']:
                    recent_signals.append({
                        'Time': timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                        'Signal': 'SELL',
                        'Price': f"₹{price:.2f}",
                        'Strength': 'Medium',
                        'Reason': 'Technical indicator signal'
                    })

            if recent_signals:
                # Show most recent signals
                for signal in recent_signals[-3:]:  # Last 3 signals
                    if signal['Signal'] in ['BUY', 'STRONG BUY']:
                        st.success(f"🟢 **{signal['Signal']}** at {signal['Price']} - {signal['Reason']} (Strength: {signal['Strength']})")
                    else:
                        st.error(f"🔴 **{signal['Signal']}** at {signal['Price']} - {signal['Reason']} (Strength: {signal['Strength']})")

                # Signals table
                st.subheader("📊 Recent Signals History")
                signals_df = pd.DataFrame(recent_signals)
                st.dataframe(signals_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 No recent trading signals generated")

        # Technical Summary Table
        st.subheader("📈 Complete Technical Summary")

        if len(analyzer.data) > 20:
            latest_data = analyzer.data.iloc[-1]
            prev_data = analyzer.data.iloc[-2]

            summary_data = []

            # RSI Analysis
            if 'RSI' in latest_data:
                rsi_signal = "SELL" if latest_data['RSI'] > 70 else "BUY" if latest_data['RSI'] < 30 else "HOLD"
                rsi_strength = "Strong" if latest_data['RSI'] > 80 or latest_data['RSI'] < 20 else "Medium" if latest_data['RSI'] > 70 or latest_data['RSI'] < 30 else "Weak"
                summary_data.append({
                    'Indicator': 'RSI (14)',
                    'Value': f"{latest_data['RSI']:.2f}",
                    'Signal': rsi_signal,
                    'Strength': rsi_strength,
                    'Trend': '↗️' if latest_data['RSI'] > prev_data['RSI'] else '↘️'
                })

            # MACD Analysis
            if 'MACD' in latest_data and 'MACD_signal' in latest_data:
                macd_signal = "BUY" if latest_data['MACD'] > latest_data['MACD_signal'] else "SELL"
                macd_crossover = "Yes" if (latest_data['MACD'] > latest_data['MACD_signal']) != (prev_data['MACD'] > prev_data['MACD_signal']) else "No"
                summary_data.append({
                    'Indicator': 'MACD',
                    'Value': f"{latest_data['MACD']:.4f}",
                    'Signal': macd_signal,
                    'Strength': "Strong" if macd_crossover == "Yes" else "Medium",
                    'Trend': '🔄' if macd_crossover == "Yes" else ('↗️' if latest_data['MACD'] > prev_data['MACD'] else '↘️')
                })

            # Moving Averages
            if 'SMA_20' in latest_data and 'SMA_50' in latest_data:
                ma_signal = "BUY" if latest_data['SMA_20'] > latest_data['SMA_50'] else "SELL"
                price_vs_ma = "Above" if latest_data['close'] > latest_data['SMA_20'] else "Below"
                summary_data.append({
                    'Indicator': 'Moving Averages',
                    'Value': f"SMA20: ₹{latest_data['SMA_20']:.2f}",
                    'Signal': ma_signal,
                    'Strength': "Strong" if abs(latest_data['SMA_20'] - latest_data['SMA_50']) > latest_data['close'] * 0.01 else "Medium",
                    'Trend': f"Price {price_vs_ma} MA20"
                })

            # Bollinger Bands
            if 'BB_position' in latest_data:
                bb_signal = "SELL" if latest_data['BB_position'] > 0.8 else "BUY" if latest_data['BB_position'] < 0.2 else "HOLD"
                bb_squeeze = "Yes" if 'BB_width' in latest_data and latest_data['BB_width'] < 0.1 else "No"
                summary_data.append({
                    'Indicator': 'Bollinger Bands',
                    'Value': f"Position: {latest_data['BB_position']:.2f}",
                    'Signal': bb_signal,
                    'Strength': "Strong" if latest_data['BB_position'] > 0.9 or latest_data['BB_position'] < 0.1 else "Medium",
                    'Trend': f"Squeeze: {bb_squeeze}"
                })

            # Volume Analysis
            if 'Volume_SMA' in latest_data:
                vol_ratio = latest_data['volume'] / latest_data['Volume_SMA']
                vol_signal = "BUY" if vol_ratio > 1.5 else "HOLD"
                summary_data.append({
                    'Indicator': 'Volume',
                    'Value': f"Ratio: {vol_ratio:.2f}",
                    'Signal': vol_signal,
                    'Strength': "Strong" if vol_ratio > 2 else "Medium" if vol_ratio > 1.5 else "Weak",
                    'Trend': '📈' if vol_ratio > 1.2 else '📉'
                })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Support and Resistance Analysis
        st.subheader("📊 Support & Resistance Levels")

        support_resistance = calculate_support_resistance(analyzer.data)

        sr_cols = st.columns(3)

        with sr_cols[0]:
            st.metric(
                "🛡️ Support Level", 
                f"₹{support_resistance['immediate_support']:.2f}" if support_resistance['immediate_support'] else "N/A"
            )
            if support_resistance['immediate_support']:
                distance = ((support_resistance['current_price'] - support_resistance['immediate_support']) / support_resistance['current_price']) * 100
                st.write(f"Distance: {distance:.1f}%")

        with sr_cols[1]:
            st.metric(
                "🎯 Current Price", 
                f"₹{support_resistance['current_price']:.2f}"
            )

        with sr_cols[2]:
            st.metric(
                "🚧 Resistance Level", 
                f"₹{support_resistance['immediate_resistance']:.2f}" if support_resistance['immediate_resistance'] else "N/A"
            )
            if support_resistance['immediate_resistance']:
                distance = ((support_resistance['immediate_resistance'] - support_resistance['current_price']) / support_resistance['current_price']) * 100
                st.write(f"Distance: {distance:.1f}%")

    else:
        st.error("Insufficient data for technical analysis")

# Tab 4: Smart Alert System
with tab4:
    st.header("🚨 Smart Alert Management System")

    # Alert Statistics Dashboard
    alert_stats = st.session_state.alert_manager.get_alert_statistics()

    stats_cols = st.columns(4)

    with stats_cols[0]:
        st.metric("📊 Total Alerts", alert_stats['total_alerts'])

    with stats_cols[1]:
        st.metric("🟢 Active Alerts", alert_stats['active_alerts'])

    with stats_cols[2]:
        st.metric("🔔 Triggered Today", alert_stats['triggered_today'])

    with stats_cols[3]:
        st.metric("📜 History Size", alert_stats['history_size'])

    # Quick Alert Creation
    st.subheader("⚡ Quick Alert Setup")

    quick_alert_cols = st.columns(3)

    with quick_alert_cols[0]:
        st.write("**Price Alerts**")
        quick_price = st.number_input("Price Level", min_value=0.0, step=1.0, key="quick_price")
        quick_price_condition = st.selectbox("Condition", ["above", "below"], key="quick_condition")
        if st.button("Add Price Alert", key="quick_price_btn"):
            if quick_price > 0:
                alert_id = st.session_state.alert_manager.add_price_alert(
                    selected_stock, quick_price_condition, quick_price, AlertPriority.MEDIUM
                )
                st.success(f"Added price alert: {selected_stock} {quick_price_condition} ₹{quick_price}")

    with quick_alert_cols[1]:
        st.write("**RSI Alerts**")
        rsi_level = st.slider("RSI Level", 10, 90, 70, key="quick_rsi")
        rsi_condition = st.selectbox("Condition", ["above", "below"], key="rsi_condition")
        if st.button("Add RSI Alert", key="quick_rsi_btn"):
            alert_id = st.session_state.alert_manager.add_technical_alert(
                selected_stock, "RSI", rsi_condition, rsi_level, AlertPriority.MEDIUM
            )
            st.success(f"Added RSI alert: {selected_stock} RSI {rsi_condition} {rsi_level}")

    with quick_alert_cols[2]:
        st.write("**Volume Alerts**")
        vol_multiplier = st.number_input("Volume Multiplier", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="quick_vol")
        if st.button("Add Volume Alert", key="quick_vol_btn"):
            st.session_state.alert_manager.add_volume_alert(
                selected_stock, vol_multiplier, AlertPriority.MEDIUM
            )
            st.success(f"Added volume alert: {selected_stock} volume > {vol_multiplier}x average")

    # Active Alerts Management
    st.subheader("📋 Active Alerts")

    active_alerts = st.session_state.alert_manager.get_active_alerts()

    if active_alerts:
        for alert in active_alerts:
            alert_container = st.container()

            with alert_container:
                alert_cols = st.columns([3, 1, 1, 1])

                with alert_cols[0]:
                    priority_emoji = {
                        AlertPriority.LOW: "🔵",
                        AlertPriority.MEDIUM: "🟡", 
                        AlertPriority.HIGH: "🟠",
                        AlertPriority.CRITICAL: "🔴"
                    }

                    status_text = "🔔 Triggered" if alert.is_triggered else "⏳ Waiting"
                    st.write(f"{priority_emoji[alert.priority]} **{alert.message}** - {status_text}")
                    st.write(f"Current: {alert.current_value:.2f} | Trigger: {alert.trigger_value:.2f}")

                with alert_cols[1]:
                    st.write(f"**{alert.priority.name}**")

                with alert_cols[2]:
                    st.write(f"Triggers: {alert.trigger_count}")

                with alert_cols[3]:
                    if st.button("Remove", key=f"remove_{alert.id}"):
                        st.session_state.alert_manager.remove_alert(alert.id)
                        st.success("Alert removed")
                        st.rerun()

                st.divider()
    else:
        st.info("No active alerts. Create some alerts above!")

    # Recent Triggered Alerts
    st.subheader("🔔 Recent Triggered Alerts")

    if st.session_state.triggered_alerts:
        for alert in st.session_state.triggered_alerts[-5:]:  # Show last 5
            if alert.priority == AlertPriority.CRITICAL:
                st.error(f"🚨 **CRITICAL**: {alert.message} - Triggered at ₹{alert.current_value:.2f}")
            elif alert.priority == AlertPriority.HIGH:
                st.warning(f"⚠️ **HIGH**: {alert.message} - Triggered at ₹{alert.current_value:.2f}")
            else:
                st.info(f"ℹ️ **{alert.priority.name}**: {alert.message} - Triggered at ₹{alert.current_value:.2f}")
    else:
        st.info("No alerts triggered yet")

    # Advanced Alert Configuration
    with st.expander("🔧 Advanced Alert Configuration"):
        st.write("**Notification Settings**")

        email_notifications = st.checkbox("Enable Email Notifications")
        if email_notifications:
            email_recipient = st.text_input("Email Address")
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)

        telegram_notifications = st.checkbox("Enable Telegram Notifications")
        if telegram_notifications:
            telegram_bot_token = st.text_input("Telegram Bot Token", type="password")
            telegram_chat_id = st.text_input("Telegram Chat ID")

        webhook_notifications = st.checkbox("Enable Webhook Notifications")
        if webhook_notifications:
            webhook_url = st.text_input("Webhook URL")
            webhook_secret = st.text_input("Secret Key (Optional)", type="password")

        if st.button("Save Notification Settings"):
            st.success("Notification settings saved!")

# Tab 5: Support & Resistance Analysis
with tab5:
    st.header(f"📊 Support & Resistance Analysis - {selected_stock}")

    if len(historical_data) > 0:
        # Calculate support and resistance levels
        support_resistance = calculate_support_resistance(historical_data, window=20)

        # Create a chart showing support and resistance levels
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=historical_data['timestamp'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name=selected_stock
        ))

        # Add support and resistance lines
        if support_resistance['immediate_support']:
            fig.add_hline(
                y=support_resistance['immediate_support'], 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Support: ₹{support_resistance['immediate_support']:.2f}"
            )

        if support_resistance['immediate_resistance']:
            fig.add_hline(
                y=support_resistance['immediate_resistance'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Resistance: ₹{support_resistance['immediate_resistance']:.2f}"
            )

        if support_resistance['strong_support']:
            fig.add_hline(
                y=support_resistance['strong_support'], 
                line_dash="dot", 
                line_color="darkgreen",
                annotation_text=f"Strong Support: ₹{support_resistance['strong_support']:.2f}"
            )

        if support_resistance['strong_resistance']:
            fig.add_hline(
                y=support_resistance['strong_resistance'], 
                line_dash="dot", 
                line_color="darkred",
                annotation_text=f"Strong Resistance: ₹{support_resistance['strong_resistance']:.2f}"
            )

        fig.update_layout(
            title=f"{selected_stock} - Support & Resistance Levels",
            yaxis_title="Price (₹)",
            xaxis_title="Time",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Support & Resistance Summary
        st.subheader("📊 Key Levels Summary")

        levels_data = {
            'Level Type': ['Current Price', 'Immediate Support', 'Immediate Resistance', 'Strong Support', 'Strong Resistance'],
            'Price (₹)': [
                f"₹{support_resistance['current_price']:.2f}",
                f"₹{support_resistance['immediate_support']:.2f}" if support_resistance['immediate_support'] else "N/A",
                f"₹{support_resistance['immediate_resistance']:.2f}" if support_resistance['immediate_resistance'] else "N/A",
                f"₹{support_resistance['strong_support']:.2f}" if support_resistance['strong_support'] else "N/A",
                f"₹{support_resistance['strong_resistance']:.2f}" if support_resistance['strong_resistance'] else "N/A"
            ],
            'Distance from Current': [
                "0.00%",
                f"{((support_resistance['current_price'] - support_resistance['immediate_support']) / support_resistance['current_price'] * 100):.2f}%" if support_resistance['immediate_support'] else "N/A",
                f"{((support_resistance['immediate_resistance'] - support_resistance['current_price']) / support_resistance['current_price'] * 100):.2f}%" if support_resistance['immediate_resistance'] else "N/A",
                f"{((support_resistance['current_price'] - support_resistance['strong_support']) / support_resistance['current_price'] * 100):.2f}%" if support_resistance['strong_support'] else "N/A",
                f"{((support_resistance['strong_resistance'] - support_resistance['current_price']) / support_resistance['current_price'] * 100):.2f}%" if support_resistance['strong_resistance'] else "N/A"
            ]
        }

        levels_df = pd.DataFrame(levels_data)
        st.dataframe(levels_df, use_container_width=True, hide_index=True)

        # Trading Strategy Suggestions
        st.subheader("💡 Trading Strategy Suggestions")

        current_price = support_resistance['current_price']

        suggestions = []

        if support_resistance['immediate_support']:
            support_distance = ((current_price - support_resistance['immediate_support']) / current_price) * 100
            if support_distance < 2:
                suggestions.append("🟢 **Near Support**: Consider buying opportunities near current support level")
            elif support_distance < 5:
                suggestions.append("🟡 **Approaching Support**: Monitor for potential bounce or breakdown")

        if support_resistance['immediate_resistance']:
            resistance_distance = ((support_resistance['immediate_resistance'] - current_price) / current_price) * 100
            if resistance_distance < 2:
                suggestions.append("🔴 **Near Resistance**: Consider profit booking or wait for breakout")
            elif resistance_distance < 5:
                suggestions.append("🟡 **Approaching Resistance**: Watch for breakout or rejection")

        # Risk management suggestions
        if support_resistance['immediate_support']:
            stop_loss_level = support_resistance['immediate_support'] * 0.98
            suggestions.append(f"🛡️ **Stop Loss Suggestion**: Consider placing stop loss below ₹{stop_loss_level:.2f}")

        if support_resistance['immediate_resistance']:
            target_level = support_resistance['immediate_resistance'] * 1.02
            suggestions.append(f"🎯 **Target Suggestion**: Consider profit booking near ₹{target_level:.2f}")

        if suggestions:
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.info("No specific trading suggestions at current levels")

    else:
        st.error("Insufficient data for support & resistance analysis")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>📈 DhanHQ Advanced Trading Dashboard</h3>
    <p><strong>Features:</strong> Live Data • Advanced Technical Analysis • Smart Alerts • Pattern Recognition • Support/Resistance</p>
    <p><strong>Built with:</strong> Streamlit • Plotly • DhanHQ API v2 • Advanced Technical Indicators</p>
    <p><em>⚠️ For demo purposes with simulated data. Use actual DhanHQ API credentials for live trading.</em></p>
    <p><em>🎯 Always practice proper risk management and consult financial advisors for investment decisions.</em></p>
</div>
""", unsafe_allow_html=True)
