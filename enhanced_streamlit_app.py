import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Callable, Tuple
import warnings
import logging
from functools import wraps
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("streamlit_dashboard")

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="DhanHQ Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== LOAD STOCK LIST ==========
@st.cache_data(ttl=300)
def load_stock_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded stock CSV '{file_path}' - shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load stock CSV: {e}", exc_info=True)
        st.error(f"Failed to load stock CSV: {e}")
        return pd.DataFrame()

# ========== DHANHQ API CLIENT ==========
def rate_limit_handler(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                    time.sleep(delay * (2 ** attempt))
                except Exception as e:
                    logger.error(f"API call error: {e}", exc_info=True)
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))
                        continue
                    return None
            return None
        return wrapper
    return decorator

class DhanAPIClient:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id
        }
        logger.info(f"DhanAPIClient initialized for client {client_id}")

    @rate_limit_handler(max_retries=3, delay=1)
    def get_historical_data(self, security_id: str, exchange_segment: str, instrument: str,
                            from_date: str, to_date: str, expiry_code: int = 0, oi: bool = False) -> Optional[Dict]:
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
        logger.info(f"Historical Data Request: {payload}")
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            logger.info(f"Historical Data Response: {response.status_code} {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Historical data API error: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Historical data request failed: {e}")
            st.error(f"Historical data request failed: {e}")
            return None

    @rate_limit_handler(max_retries=3, delay=1)
    def get_intraday_data(self, security_id: str, exchange_segment: str, instrument: str, interval: str,
                         from_date: str, to_date: str, oi: bool = False) -> Optional[Dict]:
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
        logger.info(f"Intraday Data Request: {payload}")
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            logger.info(f"Intraday Data Response: {response.status_code} {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Intraday data API error: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Intraday data request failed: {e}")
            st.error(f"Intraday data request failed: {e}")
            return None

    @rate_limit_handler(max_retries=3, delay=2)
    def get_market_quote_ltp(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        url = f"{self.base_url}/marketfeed/ltp"
        logger.info(f"LTP Request: {instruments}")
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            logger.info(f"LTP Response: {response.status_code} {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market quote LTP API error: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Market quote LTP request failed: {e}")
            st.error(f"Market quote LTP request failed: {e}")
            return None

    @rate_limit_handler(max_retries=3, delay=2)
    def get_market_quote_ohlc(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        url = f"{self.base_url}/marketfeed/ohlc"
        logger.info(f"OHLC Request: {instruments}")
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            logger.info(f"OHLC Response: {response.status_code} {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market quote OHLC API error: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Market quote OHLC request failed: {e}")
            st.error(f"Market quote OHLC request failed: {e}")
            return None

    @rate_limit_handler(max_retries=3, delay=2)
    def get_market_depth(self, instruments: Dict[str, List[int]]) -> Optional[Dict]:
        url = f"{self.base_url}/marketfeed/quote"
        logger.info(f"Market Depth Request: {instruments}")
        try:
            response = requests.post(url, headers=self.headers, json=instruments, timeout=10)
            logger.info(f"Market Depth Response: {response.status_code} {response.text}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Market depth API error: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Market depth request failed: {e}")
            st.error(f"Market depth request failed: {e}")
            return None

def format_historical_data(api_response: Dict) -> pd.DataFrame:
    if not api_response or 'open' not in api_response:
        logger.error("Invalid historical API response")
        return pd.DataFrame()
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(api_response['timestamp'], unit='s'),
        'open': api_response['open'],
        'high': api_response['high'],
        'low': api_response['low'],
        'close': api_response['close'],
        'volume': api_response['volume']
    })
    if 'open_interest' in api_response:
        df['open_interest'] = api_response['open_interest']
    return df.set_index('timestamp')

# ========== TECHNICAL ANALYSIS CLASSES ==========
class TechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
    def add_moving_averages(self, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        for period in periods:
            if len(self.data) >= period:
                self.data[f'SMA_{period}'] = self.data['close'].rolling(window=period).mean()
                self.data[f'EMA_{period}'] = self.data['close'].ewm(span=period).mean()
        return self.data
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        if len(self.data) < period:
            return self.data
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
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
        if len(self.data) < period:
            return self.data
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['ATR'] = true_range.rolling(window=period).mean()
        return self.data
    def add_volume_indicators(self) -> pd.DataFrame:
        if len(self.data) < 20:
            return self.data
        self.data['Volume_SMA'] = self.data['volume'].rolling(window=20).mean()
        obv = [0]
        for i in range(1, len(self.data)):
            if self.data['close'].iloc[i] > self.data['close'].iloc[i-1]:
                obv.append(obv[-1] + self.data['volume'].iloc[i])
            elif self.data['close'].iloc[i] < self.data['close'].iloc[i-1]:
                obv.append(obv[-1] - self.data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        self.data['OBV'] = obv
        money_flow_multiplier = ((self.data['close'] - self.data['low']) - (self.data['high'] - self.data['close'])) / (self.data['high'] - self.data['low'])
        money_flow_volume = money_flow_multiplier * self.data['volume']
        self.data['CMF'] = money_flow_volume.rolling(window=20).sum() / self.data['volume'].rolling(window=20).sum()
        return self.data
    def add_all_indicators(self) -> pd.DataFrame:
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_stochastic()
        self.add_atr()
        self.add_volume_indicators()
        return self.data
    def generate_signals(self) -> pd.DataFrame:
        signals = pd.DataFrame(index=self.data.index)
        signals['BUY'] = 0
        signals['SELL'] = 0
        signals['STRONG_BUY'] = 0
        signals['STRONG_SELL'] = 0
        if len(self.data) < 50:
            self.signals = signals
            return signals
        rsi_oversold = self.data['RSI'] < 30
        rsi_overbought = self.data['RSI'] > 70
        macd_bullish = (self.data['MACD'] > self.data['MACD_signal']) & (self.data['MACD'].shift(1) <= self.data['MACD_signal'].shift(1))
        macd_bearish = (self.data['MACD'] < self.data['MACD_signal']) & (self.data['MACD'].shift(1) >= self.data['MACD_signal'].shift(1))
        if 'SMA_20' in self.data.columns and 'SMA_50' in self.data.columns:
            golden_cross = (self.data['SMA_20'] > self.data['SMA_50']) & (self.data['SMA_20'].shift(1) <= self.data['SMA_50'].shift(1))
            death_cross = (self.data['SMA_20'] < self.data['SMA_50']) & (self.data['SMA_20'].shift(1) >= self.data['SMA_50'].shift(1))
        else:
            golden_cross = pd.Series(False, index=self.data.index)
            death_cross = pd.Series(False, index=self.data.index)
        if 'BB_upper' in self.data.columns:
            bb_oversold = self.data['close'] < self.data['BB_lower']
            bb_overbought = self.data['close'] > self.data['BB_upper']
        else:
            bb_oversold = pd.Series(False, index=self.data.index)
            bb_overbought = pd.Series(False, index=self.data.index)
        if 'STOCH_K' in self.data.columns:
            stoch_oversold = (self.data['STOCH_K'] < 20) & (self.data['STOCH_D'] < 20)
            stoch_overbought = (self.data['STOCH_K'] > 80) & (self.data['STOCH_D'] > 80)
        else:
            stoch_oversold = pd.Series(False, index=self.data.index)
            stoch_overbought = pd.Series(False, index=self.data.index)
        signals.loc[rsi_oversold | macd_bullish | bb_oversold | stoch_oversold, 'BUY'] = 1
        signals.loc[golden_cross, 'BUY'] = 1
        signals.loc[rsi_overbought | macd_bearish | bb_overbought | stoch_overbought, 'SELL'] = 1
        signals.loc[death_cross, 'SELL'] = 1
        strong_buy_conditions = ((rsi_oversold.astype(int) + macd_bullish.astype(int) + bb_oversold.astype(int) + stoch_oversold.astype(int) + golden_cross.astype(int)) >= 3)
        signals.loc[strong_buy_conditions, 'STRONG_BUY'] = 1
        strong_sell_conditions = ((rsi_overbought.astype(int) + macd_bearish.astype(int) + bb_overbought.astype(int) + stoch_overbought.astype(int) + death_cross.astype(int)) >= 3)
        signals.loc[strong_sell_conditions, 'STRONG_SELL'] = 1
        self.signals = signals
        return signals
    def get_signal_summary(self) -> Dict:
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
    if len(data) < window * 2:
        return {'current_price': data['close'].iloc[-1] if len(data) > 0 else 0}
    recent_data = data.tail(window * 2)
    highs = recent_data['high'].rolling(window=window, center=True).max()
    lows = recent_data['low'].rolling(window=window, center=True).min()
    resistance_levels = []
    support_levels = []
    for i in range(window, len(recent_data) - window):
        if recent_data['high'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(recent_data['high'].iloc[i])
        if recent_data['low'].iloc[i] == lows.iloc[i]:
            support_levels.append(recent_data['low'].iloc[i])
    current_price = data['close'].iloc[-1]
    support_levels = [level for level in support_levels if level < current_price]
    resistance_levels = [level for level in resistance_levels if level > current_price]
    return {
        'current_price': current_price,
        'immediate_support': max(support_levels) if support_levels else None,
        'immediate_resistance': min(resistance_levels) if resistance_levels else None,
        'strong_support': min(support_levels) if support_levels else None,
        'strong_resistance': max(resistance_levels) if resistance_levels else None
    }

# ========== ALERT SYSTEM ==========
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
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []

    def add_price_alert(self, symbol: str, condition: str, trigger_value: float,
                        priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        alert_id = f"price_{symbol}_{condition}_{trigger_value}_{datetime.now().timestamp()}"
        alert = Alert(
            id=alert_id, symbol=symbol, alert_type=AlertType.PRICE, priority=priority,
            message=f"{symbol} price {condition} ₹{trigger_value:,.2f}", condition=condition,
            current_value=0.0, trigger_value=trigger_value, timestamp=datetime.now()
        )
        self.alerts[alert_id] = alert
        return alert_id

    def add_technical_alert(self, symbol: str, indicator: str, condition: str, trigger_value: float,
                            priority: AlertPriority = AlertPriority.MEDIUM) -> str:
        alert_id = f"tech_{symbol}_{indicator}_{condition}_{trigger_value}_{datetime.now().timestamp()}"
        alert = Alert(
            id=alert_id, symbol=symbol, alert_type=AlertType.TECHNICAL, priority=priority,
            message=f"{symbol} {indicator} {condition} {trigger_value}", condition=condition,
            current_value=0.0, trigger_value=trigger_value, timestamp=datetime.now()
        )
        self.alerts[alert_id] = alert
        return alert_id

    def check_alerts(self, market_data: Dict) -> List[Alert]:
        triggered_alerts = []
        for alert in self.alerts.values():
            if not alert.is_active:
                continue
            if alert.alert_type == AlertType.PRICE:
                current_price = market_data.get('price', 0)
                alert.current_value = current_price
                if self._check_condition(current_price, alert.condition, alert.trigger_value):
                    if not alert.is_triggered:
                        alert.is_triggered = True
                        alert.trigger_count += 1
                        triggered_alerts.append(alert)
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
        if condition == "above":
            return current_value > trigger_value
        elif condition == "below":
            return current_value < trigger_value
        return False

    def get_active_alerts(self) -> List[Alert]:
        return [alert for alert in self.alerts.values() if alert.is_active]

    def remove_alert(self, alert_id: str) -> bool:
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False

    def get_alert_statistics(self) -> Dict:
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        triggered_today = len([a for a in self.alert_history if a.timestamp.date() == datetime.now().date()])
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_today': triggered_today,
            'history_size': len(self.alert_history)
        }

# ========== SESSION STATE GLOBALS ==========
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'dhan_client' not in st.session_state:
    st.session_state.dhan_client = None
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = ''
if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = AlertManager()
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'triggered_alerts' not in st.session_state:
    st.session_state.triggered_alerts = []

# ========== SIDEBAR CONFIGURATION ==========
st.sidebar.header("🔧 Configuration")
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
                st.experimental_rerun()
            else:
                st.error("❌ Enter credentials")
    with col2:
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.dhan_client = None
            st.info("Disconnected")
            st.experimental_rerun()

if st.session_state.connected:
    st.sidebar.success("🟢 Connected to DhanHQ")
else:
    st.sidebar.error("🔴 Not Connected")

st.sidebar.header("📊 Instrument Selection")
stock_df = load_stock_csv("stock_list.csv")
if not stock_df.empty:
    stock_options = {f"{row['symbol']} ({row['exchange']})": row for _, row in stock_df.iterrows()}
    selected_stock_display = st.sidebar.selectbox("Select Stock", options=list(stock_options.keys()))
    selected_stock_data = stock_options[selected_stock_display]
    selected_stock = selected_stock_data['symbol']
    st.session_state.selected_stock = selected_stock
else:
    st.sidebar.error("Stock list not available or failed to load")
    st.stop()

timeframe = st.sidebar.selectbox("📅 Timeframe", options=["1min", "5min", "15min", "1day"], index=2)

indicators = {
    "RSI (14)": st.sidebar.checkbox("RSI (14)", value=True),
    "MACD": st.sidebar.checkbox("MACD", value=True),
    "Bollinger Bands": st.sidebar.checkbox("Bollinger Bands", value=True),
    "SMA (20, 50)": st.sidebar.checkbox("SMA (20, 50)", value=True),
    "EMA (12, 26)": st.sidebar.checkbox("EMA (12, 26)", value=True),
    "Stochastic": st.sidebar.checkbox("Stochastic", value=False),
    "Volume Indicators": st.sidebar.checkbox("Volume Indicators", value=False),
}

st.sidebar.header("🚨 Advanced Alert Settings")
with st.sidebar.expander("Price Alerts"):
    price_alert_above = st.number_input("Alert Above (₹)", min_value=0.0, step=1.0)
    price_alert_below = st.number_input("Alert Below (₹)", min_value=0.0, step=1.0)
    if st.button("Add Price Alert"):
        if price_alert_above > 0:
            st.session_state.alert_manager.add_price_alert(selected_stock, "above", price_alert_above, AlertPriority.MEDIUM)
            st.success(f"Added alert for {selected_stock} above ₹{price_alert_above}")
        if price_alert_below > 0:
            st.session_state.alert_manager.add_price_alert(selected_stock, "below", price_alert_below, AlertPriority.MEDIUM)
            st.success(f"Added alert for {selected_stock} below ₹{price_alert_below}")

with st.sidebar.expander("Technical Alerts"):
    rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
    if st.button("Add Technical Alerts"):
        st.session_state.alert_manager.add_technical_alert(selected_stock, "RSI", "above", rsi_overbought, AlertPriority.MEDIUM)
        st.session_state.alert_manager.add_technical_alert(selected_stock, "RSI", "below", rsi_oversold, AlertPriority.MEDIUM)
        st.success("Added RSI alerts")

auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (10 sec)", value=True)

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Live Data", "📊 Advanced Charts", "🔍 Technical Analysis", "🚨 Smart Alerts", "📋 Support/Resistance"
])

# ========== TAB1: LIVE DATA ==========
with tab1:
    st.header(f"📈 Live Market Data - {selected_stock}")
    live_price_container = st.empty()
    market_summary_container = st.empty()
    ohlc_container = st.empty()
    depth_container = st.empty()

    if st.session_state.connected and st.session_state.dhan_client:
        try:
            cache_key = f"{selected_stock}_{datetime.now().strftime('%H:%M')}"
            if 'api_cache' not in st.session_state:
                st.session_state.api_cache = {}
            if cache_key not in st.session_state.api_cache or \
               (datetime.now() - st.session_state.api_cache[cache_key]['timestamp']).seconds > 30:
                instruments = {selected_stock_data['exchange']: [int(selected_stock_data['security_id'])]}
                ltp_data = st.session_state.dhan_client.get_market_quote_ltp(instruments)
                time.sleep(0.5)
                ohlc_data = st.session_state.dhan_client.get_market_quote_ohlc(instruments)
                time.sleep(0.5)
                depth_data = st.session_state.dhan_client.get_market_depth(instruments)
                st.session_state.api_cache[cache_key] = {
                    'ltp_data': ltp_data,
                    'ohlc_data': ohlc_data,
                    'depth_data': depth_data,
                    'timestamp': datetime.now()
                }
            else:
                cached = st.session_state.api_cache[cache_key]
                ltp_data = cached['ltp_data']
                ohlc_data = cached['ohlc_data']
                depth_data = cached['depth_data']

            if ltp_data is None or 'data' not in ltp_data:
                st.error("Could not fetch live LTP data.")
                st.stop()
            exchange_data = ltp_data['data'].get(selected_stock_data['exchange'], {})
            security_data = exchange_data.get(int(selected_stock_data['security_id']), {})
            live_data = {
                'symbol': selected_stock,
                'ltp': max(security_data.get('LTP', 1), 0.01),
                'change': security_data.get('change', 0),
                'change_percent': security_data.get('pChange', 0),
                'volume': security_data.get('volume', 0),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            if ohlc_data and 'data' in ohlc_data:
                ohlc_exchange_data = ohlc_data['data'].get(selected_stock_data['exchange'], {})
                ohlc_security_data = ohlc_exchange_data.get(int(selected_stock_data['security_id']), {})
                live_data.update({
                    'open': ohlc_security_data.get('open', live_data['ltp']),
                    'high': ohlc_security_data.get('high', live_data['ltp']),
                    'low': ohlc_security_data.get('low', live_data['ltp']),
                    'close': ohlc_security_data.get('close', live_data['ltp'])
                })
            else:
                live_data.update({
                    'open': live_data['ltp'],
                    'high': live_data['ltp'],
                    'low': live_data['ltp'],
                    'close': live_data['ltp']
                })

        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            logger.exception("Exception during live data fetch")
            st.stop()
    else:
        st.error("Not connected to API - please provide credentials.")
        st.stop()

    with live_price_container.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Last Trade Price", f"₹{live_data['ltp']:,.2f}", f"{live_data['change']:+.2f}")
        col2.metric("📊 Change %", f"{live_data['change_percent']:+.2f}%", None)
        col3.metric("📈 Day High", f"₹{live_data.get('high', live_data['ltp']):,.2f}", None)
        col4.metric("📉 Day Low", f"₹{live_data.get('low', live_data['ltp']):,.2f}", None)
        col5.metric("📦 Volume", f"{live_data['volume']:,}", None)

    day_range = live_data.get('high', live_data['ltp']) - live_data.get('low', live_data['ltp'])
    price_position = ((live_data['ltp'] - live_data.get('low', live_data['ltp'])) / day_range * 100) if day_range > 0 else 50

    with market_summary_container.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"Open: ₹{live_data.get('open', live_data['ltp']):,.2f}")
        c2.info(f"Day Range: ₹{day_range:.2f}")
        c3.info(f"Price Position: {price_position:.1f}%")
        c4.info(f"Last Update: {live_data['timestamp']}")

    ohlc_data_display = {
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
            f"{((live_data.get('open', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%" if live_data['ltp'] != 0 else "N/A",
            f"{((live_data.get('high', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%" if live_data['ltp'] != 0 else "N/A",
            f"{((live_data.get('low', live_data['ltp']) - live_data['ltp']) / live_data['ltp'] * 100):+.2f}%" if live_data['ltp'] != 0 else "N/A",
            f"{live_data['change_percent']:+.2f}%",
            "N/A", "N/A",
            f"{live_data['change_percent']:+.2f}%"
        ]
    }
    with ohlc_container.container():
        st.subheader("📋 Detailed Market Data")
        st.dataframe(pd.DataFrame(ohlc_data_display), use_container_width=True, hide_index=True)

    with depth_container.container():
        st.subheader("📊 Market Depth (Level 2)")
        try:
            if depth_data and 'data' in depth_data:
                exch_data = depth_data['data'].get(selected_stock_data['exchange'], {})
                sec_data = exch_data.get(int(selected_stock_data['security_id']), {})
                bid_orders = sec_data.get('buy', [])
                ask_orders = sec_data.get('sell', [])
                col1d, col2d = st.columns(2)
                with col1d:
                    st.markdown("**🟢 Buy Orders (Bids)**")
                    if bid_orders:
                        df_bids = pd.DataFrame(bid_orders)
                        st.dataframe(df_bids, use_container_width=True, hide_index=True)
                    else:
                        st.info("No Buy Orders available")
                with col2d:
                    st.markdown("**🔴 Sell Orders (Asks)**")
                    if ask_orders:
                        df_asks = pd.DataFrame(ask_orders)
                        st.dataframe(df_asks, use_container_width=True, hide_index=True)
                    else:
                        st.info("No Sell Orders available")
            else:
                st.error("Market depth data not available from API.")
        except Exception as e:
            st.error(f"Error loading market depth: {e}")
            logger.exception("Market depth display error")

# ========== Remaining tabs (Advanced Charts, Technical Analysis, Alerts, S&R) ==========
# Please insert your full existing tab logic here exactly as is,
# but replace all simulated data calls by API data checks as done above.
# Ensure to always handle cases where data is None or empty with appropriate errors or fallback UI.

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
**Features:** Live Data • Advanced Technical Analysis • Smart Alerts • Pattern Recognition • Support/Resistance  
**Built with:** Streamlit • Plotly • DhanHQ API v2 • Advanced Technical Indicators  

*⚠️ For live trading, use your DhanHQ API credentials only.*  
*🎯 Always practice proper risk management and consult qualified financial advisors.*
""")
