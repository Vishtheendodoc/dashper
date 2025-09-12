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
# Removed problematic email imports
# import smtplib
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart
# import websocket (also removed as it may cause issues)
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

    def add_volume_alert(self, symbol: str, multiplier: float, 
                        priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        """Add a volume-based alert"""
        alert_id = f"volume_{symbol}_{multiplier}_{datetime.now().timestamp()}"

        alert = Alert(
            id=alert_id,
            symbol=symbol,
            alert_type=AlertType.VOLUME,
            priority=priority,
            message=f"{symbol} volume > {multiplier}x average",
            condition="above",
            current_value=0.0,
            trigger_value=multiplier,
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

            # Check volume alerts
            elif alert.alert_type == AlertType.VOLUME:
                current_volume = market_data.get('volume', 0)
                avg_volume = market_data.get('avg_volume', 100000)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                alert.current_value = volume_ratio

                if self._check_condition(volume_ratio, alert.condition, alert.trigger_value):
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
        {"symbol": "NIFTY", "security_id": "13", "exchange": "NSE_EQ", "name": "NIFTY 50"},
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
        'INFY': 1
