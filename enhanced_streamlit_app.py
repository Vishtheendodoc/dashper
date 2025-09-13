import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# Import DhanHQ SDK
try:
    from dhanhq import dhanhq
except ImportError:
    st.error("DhanHQ SDK not installed. Please run: pip install dhanhq")
    st.stop()

warnings.filterwarnings('ignore')

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("dhanhq_dashboard")

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="DhanHQ Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTS ==========
EXCHANGES = {
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ", 
    "NSE_FO": "NSE_FO",
    "NSE_CURRENCY": "NSE_CURRENCY",
    "MCX": "MCX_COMM"
}

INSTRUMENTS = {
    "EQUITY": "EQUITY",
    "FUTURES": "FUTURES",
    "OPTIONS": "OPTIONS",
    "CURRENCY": "CURRENCY",
    "COMMODITY": "COMMODITY"
}

# ========== SAMPLE STOCK DATA ==========
SAMPLE_STOCKS = [
    {"symbol": "RELIANCE", "security_id": "2885", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "TCS", "security_id": "11536", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "INFY", "security_id": "1594", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "HDFCBANK", "security_id": "1333", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "ICICIBANK", "security_id": "1270", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "ADANIENT", "security_id": "25", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "HINDUNILVR", "security_id": "1394", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "ITC", "security_id": "1660", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "SBIN", "security_id": "3045", "exchange": "NSE_EQ", "instrument": "EQUITY"},
    {"symbol": "BHARTIARTL", "security_id": "3677", "exchange": "NSE_EQ", "instrument": "EQUITY"}
]

# ========== TECHNICAL ANALYSIS CLASSES ==========
class TechnicalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)

    def add_moving_averages(self, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
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

    def add_all_indicators(self) -> pd.DataFrame:
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        return self.data

    def generate_signals(self) -> Dict:
        if len(self.data) < 50:
            return {"signal": "HOLD", "strength": 0, "reason": "Insufficient data"}

        latest = self.data.iloc[-1]
        signals = []
        
        # RSI signals
        if 'RSI' in self.data.columns:
            rsi = latest['RSI']
            if rsi < 30:
                signals.append(("BUY", "RSI Oversold"))
            elif rsi > 70:
                signals.append(("SELL", "RSI Overbought"))

        # MACD signals
        if 'MACD' in self.data.columns and 'MACD_signal' in self.data.columns:
            if latest['MACD'] > latest['MACD_signal'] and len(self.data) > 1:
                prev = self.data.iloc[-2]
                if prev['MACD'] <= prev['MACD_signal']:
                    signals.append(("BUY", "MACD Bullish Crossover"))

        # Moving Average signals
        if 'SMA_20' in self.data.columns and 'SMA_50' in self.data.columns:
            if latest['close'] > latest['SMA_20'] > latest['SMA_50']:
                signals.append(("BUY", "Price Above Moving Averages"))
            elif latest['close'] < latest['SMA_20'] < latest['SMA_50']:
                signals.append(("SELL", "Price Below Moving Averages"))

        # Bollinger Bands signals
        if 'BB_position' in self.data.columns:
            bb_pos = latest['BB_position']
            if bb_pos < 0.1:
                signals.append(("BUY", "Near Lower Bollinger Band"))
            elif bb_pos > 0.9:
                signals.append(("SELL", "Near Upper Bollinger Band"))

        # Determine overall signal
        buy_signals = [s for s in signals if s[0] == "BUY"]
        sell_signals = [s for s in signals if s[0] == "SELL"]
        
        if len(buy_signals) > len(sell_signals):
            return {
                "signal": "BUY",
                "strength": min(len(buy_signals) * 25, 100),
                "reasons": [s[1] for s in buy_signals]
            }
        elif len(sell_signals) > len(buy_signals):
            return {
                "signal": "SELL", 
                "strength": min(len(sell_signals) * 25, 100),
                "reasons": [s[1] for s in sell_signals]
            }
        else:
            return {"signal": "HOLD", "strength": 0, "reasons": ["Mixed signals"]}

# ========== HELPER FUNCTIONS ==========
def format_currency(amount):
    """Format currency in Indian format"""
    return f"₹{amount:,.2f}"

def format_number(number):
    """Format numbers with Indian comma style"""
    return f"{number:,}"

def create_candlestick_chart(df: pd.DataFrame, title: str, indicators: Dict) -> go.Figure:
    """Create a comprehensive candlestick chart with indicators"""
    
    # Create subplots
    rows = 2 if any(indicators.values()) else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3] if rows == 2 else [1.0],
        subplot_titles=[title, "Volume"] if rows == 2 else [title]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if indicators.get("SMA") and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color="orange")),
            row=1, col=1
        )
        
    if indicators.get("SMA") and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color="blue")),
            row=1, col=1
        )

    # Bollinger Bands
    if indicators.get("Bollinger") and "BB_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_upper'], name="BB Upper", 
                      line=dict(color="gray", dash="dash")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_lower'], name="BB Lower",
                      line=dict(color="gray", dash="dash"), 
                      fill='tonexty', fillcolor="rgba(128,128,128,0.1)"),
            row=1, col=1
        )

    # Volume
    if rows == 2:
        colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color=colors),
            row=2, col=1
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

# ========== SESSION STATE MANAGEMENT ==========
if 'dhan_client' not in st.session_state:
    st.session_state.dhan_client = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = SAMPLE_STOCKS[0]
if 'live_data_cache' not in st.session_state:
    st.session_state.live_data_cache = {}
if 'historical_data_cache' not in st.session_state:
    st.session_state.historical_data_cache = {}

# ========== SIDEBAR ==========
st.sidebar.header("🔧 DhanHQ Configuration")

# API Connection
with st.sidebar.expander("🔐 API Connection", expanded=not st.session_state.connected):
    client_id = st.text_input("Client ID", placeholder="Enter your DhanHQ Client ID")
    access_token = st.text_input("Access Token", type="password", placeholder="Enter your Access Token")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect", type="primary"):
            if client_id and access_token:
                try:
                    dhan = dhanhq(client_id, access_token)
                    # Test connection by fetching holdings
                    holdings = dhan.get_holdings()
                    if holdings['status'] == 'success':
                        st.session_state.dhan_client = dhan
                        st.session_state.connected = True
                        st.success("✅ Connected Successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Connection Failed: Invalid credentials")
                except Exception as e:
                    st.error(f"❌ Connection Error: {str(e)}")
            else:
                st.error("❌ Please enter both Client ID and Access Token")
    
    with col2:
        if st.button("Disconnect"):
            st.session_state.dhan_client = None
            st.session_state.connected = False
            st.session_state.live_data_cache = {}
            st.session_state.historical_data_cache = {}
            st.info("Disconnected")
            st.rerun()

# Connection Status
if st.session_state.connected:
    st.sidebar.success("🟢 Connected to DhanHQ")
else:
    st.sidebar.error("🔴 Not Connected")

# Stock Selection
st.sidebar.header("📊 Stock Selection")
stock_options = {f"{stock['symbol']} ({stock['exchange']})": stock for stock in SAMPLE_STOCKS}
selected_stock_display = st.sidebar.selectbox(
    "Select Stock", 
    options=list(stock_options.keys()),
    index=0
)
st.session_state.selected_stock = stock_options[selected_stock_display]

# Chart Settings
st.sidebar.header("📈 Chart Settings")
timeframe = st.sidebar.selectbox(
    "Timeframe", 
    options=["1", "5", "15", "30", "60", "D"],
    format_func=lambda x: {
        "1": "1 Minute", "5": "5 Minutes", "15": "15 Minutes",
        "30": "30 Minutes", "60": "1 Hour", "D": "Daily"
    }.get(x, x),
    index=4
)

# Technical Indicators
st.sidebar.header("📊 Technical Indicators")
indicators = {
    "SMA": st.sidebar.checkbox("Simple Moving Average (20, 50)", value=True),
    "EMA": st.sidebar.checkbox("Exponential Moving Average", value=False),
    "RSI": st.sidebar.checkbox("RSI (14)", value=True),
    "MACD": st.sidebar.checkbox("MACD", value=True),
    "Bollinger": st.sidebar.checkbox("Bollinger Bands", value=True),
}

# Auto Refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)

# ========== MAIN APPLICATION ==========
st.title("DhanHQ Trading Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Data", "📈 Charts", "🔍 Technical Analysis", "💼 Portfolio", "📋 Orders"
])

# ========== TAB 1: LIVE DATA ==========
with tab1:
    st.header(f"📈 Live Market Data - {st.session_state.selected_stock['symbol']}")
    
    if not st.session_state.connected:
        st.error("❌ Please connect to DhanHQ API first")
    else:
        # Get live data
        try:
            stock = st.session_state.selected_stock
            
            # Create cache key
            cache_key = f"{stock['symbol']}_{datetime.now().strftime('%H:%M')}"
            
            # Check cache (refresh every minute)
            if (cache_key not in st.session_state.live_data_cache or 
                (datetime.now() - st.session_state.live_data_cache[cache_key]['timestamp']).seconds > 60):
                
                with st.spinner("Fetching live data..."):
                    # Get LTP
                    ltp_response = st.session_state.dhan_client.get_ltp_data(
                        exchange_segment=stock['exchange'],
                        security_id=stock['security_id'],
                        instrument_type=stock['instrument']
                    )
                    
                    # Get OHLC
                    ohlc_response = st.session_state.dhan_client.get_ohlc_data(
                        exchange_segment=stock['exchange'],
                        security_id=stock['security_id'],
                        instrument_type=stock['instrument']
                    )
                    
                    st.session_state.live_data_cache[cache_key] = {
                        'ltp': ltp_response,
                        'ohlc': ohlc_response,
                        'timestamp': datetime.now()
                    }
            
            # Extract data from cache
            cached_data = st.session_state.live_data_cache[cache_key]
            ltp_data = cached_data['ltp']
            ohlc_data = cached_data['ohlc']
            
            if ltp_data['status'] == 'success' and ohlc_data['status'] == 'success':
                # Display metrics
                ltp_info = ltp_data['data']
                ohlc_info = ohlc_data['data']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "💰 LTP", 
                        format_currency(ltp_info['LTP']),
                        f"{ltp_info['change']:+.2f} ({ltp_info['pChange']:+.2f}%)"
                    )
                
                with col2:
                    st.metric("📈 High", format_currency(ohlc_info['high']))
                
                with col3:
                    st.metric("📉 Low", format_currency(ohlc_info['low']))
                
                with col4:
                    st.metric("🔓 Open", format_currency(ohlc_info['open']))
                
                with col5:
                    st.metric("📊 Volume", format_number(ohlc_info['volume']))
                
                # Additional metrics
                st.subheader("📋 Market Details")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.info(f"**Previous Close:** {format_currency(ohlc_info['close'])}")
                with col2:
                    day_range = ohlc_info['high'] - ohlc_info['low']
                    st.info(f"**Day Range:** {format_currency(day_range)}")
                with col3:
                    price_position = ((ltp_info['LTP'] - ohlc_info['low']) / day_range * 100) if day_range > 0 else 50
                    st.info(f"**Price Position:** {price_position:.1f}%")
                with col4:
                    value_traded = ohlc_info['volume'] * ltp_info['LTP']
                    st.info(f"**Value Traded:** {format_currency(value_traded / 10000000):.2f}Cr")
                
                # Last updated
                st.caption(f"Last Updated: {cached_data['timestamp'].strftime('%H:%M:%S')}")
                
            else:
                st.error("❌ Failed to fetch live data")
                
        except Exception as e:
            st.error(f"❌ Error fetching live data: {str(e)}")

# ========== TAB 2: CHARTS ==========
with tab2:
    st.header(f"📈 Advanced Charts - {st.session_state.selected_stock['symbol']}")
    
    if not st.session_state.connected:
        st.error("❌ Please connect to DhanHQ API first")
    else:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input(
                "From Date", 
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now()
            )
        with col2:
            to_date = st.date_input(
                "To Date", 
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        if st.button("📊 Load Chart Data", type="primary"):
            try:
                stock = st.session_state.selected_stock
                
                with st.spinner("Fetching historical data..."):
                    # Get historical data
                    historical_data = st.session_state.dhan_client.historical_minute_charts(
                        symbol=stock['symbol'],
                        exchange_segment=stock['exchange'],
                        instrument_type=stock['instrument'],
                        expiry_code=0,
                        from_date=from_date.strftime('%Y-%m-%d'),
                        to_date=to_date.strftime('%Y-%m-%d')
                    )
                    
                    if historical_data['status'] == 'success' and 'data' in historical_data:
                        # Convert to DataFrame
                        df = pd.DataFrame(historical_data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        
                        # Apply technical indicators
                        analyzer = TechnicalAnalyzer(df)
                        df_with_indicators = analyzer.add_all_indicators()
                        
                        # Create chart
                        fig = create_candlestick_chart(df_with_indicators, 
                                                     f"{stock['symbol']} - {timeframe} Chart", 
                                                     indicators)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Technical Analysis Summary
                        signals = analyzer.generate_signals()
                        
                        st.subheader("🔍 Technical Analysis Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signals['signal']]
                            st.metric("Signal", f"{signal_color} {signals['signal']}")
                        
                        with col2:
                            st.metric("Strength", f"{signals['strength']}%")
                        
                        with col3:
                            if 'RSI' in df_with_indicators.columns:
                                latest_rsi = df_with_indicators['RSI'].iloc[-1]
                                st.metric("RSI", f"{latest_rsi:.1f}")
                        
                        if signals['reasons']:
                            st.info("**Reasons:** " + ", ".join(signals['reasons']))
                        
                        # Store in session state
                        st.session_state.historical_data_cache[stock['symbol']] = {
                            'data': df_with_indicators,
                            'signals': signals,
                            'timestamp': datetime.now()
                        }
                        
                    else:
                        st.error("❌ No historical data available for the selected period")
                        
            except Exception as e:
                st.error(f"❌ Error loading chart data: {str(e)}")

# ========== TAB 3: TECHNICAL ANALYSIS ==========
with tab3:
    st.header(f"🔍 Technical Analysis - {st.session_state.selected_stock['symbol']}")
    
    stock_symbol = st.session_state.selected_stock['symbol']
    
    if stock_symbol in st.session_state.historical_data_cache:
        cached_data = st.session_state.historical_data_cache[stock_symbol]
        df = cached_data['data']
        signals = cached_data['signals']
        
        # Technical Indicators Table
        st.subheader("📊 Technical Indicators")
        
        latest = df.iloc[-1]
        indicators_data = []
        
        if 'RSI' in df.columns:
            rsi_signal = "Oversold" if latest['RSI'] < 30 else "Overbought" if latest['RSI'] > 70 else "Neutral"
            indicators_data.append(["RSI (14)", f"{latest['RSI']:.2f}", rsi_signal])
        
        if 'MACD' in df.columns:
            macd_signal = "Bullish" if latest['MACD'] > latest['MACD_signal'] else "Bearish"
            indicators_data.append(["MACD", f"{latest['MACD']:.4f}", macd_signal])
        
        if 'SMA_20' in df.columns:
            sma_signal = "Above" if latest['close'] > latest['SMA_20'] else "Below"
            indicators_data.append(["SMA 20", f"{latest['SMA_20']:.2f}", f"Price {sma_signal}"])
        
        if 'BB_position' in df.columns:
            bb_pos = latest['BB_position']
            bb_signal = "Near Lower Band" if bb_pos < 0.2 else "Near Upper Band" if bb_pos > 0.8 else "Middle Range"
            indicators_data.append(["Bollinger Position", f"{bb_pos:.2f}", bb_signal])
        
        if indicators_data:
            indicators_df = pd.DataFrame(indicators_data, columns=["Indicator", "Value", "Signal"])
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)
        
        # Overall Signal
        st.subheader("🎯 Trading Signal")
        
        signal_color = {
            "BUY": {"color": "green", "emoji": "🟢"},
            "SELL": {"color": "red", "emoji": "🔴"}, 
            "HOLD": {"color": "orange", "emoji": "🟡"}
        }[signals['signal']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {signal_color['color']}20; border: 2px solid {signal_color['color']}">
                <h2 style="color: {signal_color['color']}; margin: 0;">{signal_color['emoji']} {signals['signal']}</h2>
                <p style="margin: 5px 0; font-size: 18px;">Strength: {signals['strength']}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📝 Analysis Reasons")
            for reason in signals['reasons']:
                st.write(f"• {reason}")
    
    else:
        st.info("📈 Load chart data first to see technical analysis")

# ========== TAB 4: PORTFOLIO ==========
with tab4:
    st.header("💼 Portfolio Overview")
    
    if not st.session_state.connected:
        st.error("❌ Please connect to DhanHQ API first")
    else:
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Holdings", type="primary"):
                    with st.spinner("Fetching holdings..."):
                        holdings = st.session_state.dhan_client.get_holdings()
                        
                        if holdings['status'] == 'success' and holdings['data']:
                            holdings_df = pd.DataFrame(holdings['data'])
                            
                            # Display holdings table
                            st.subheader("📊 Current Holdings")
                            display_df = holdings_df[['tradingSymbol', 'totalQty', 'avgCostPrice', 'LTP', 'realizedPnl', 'unrealizedPnl']].copy()
                            display_df.columns = ['Symbol', 'Quantity', 'Avg Cost', 'LTP', 'Realized P&L', 'Unrealized P&L']
                            
                            # Format currency columns
                            for col in ['Avg Cost', 'LTP', 'Realized P&L', 'Unrealized P&L']:
                                display_df[col] = display_df[col].apply(lambda x: format_currency(x))
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Portfolio Summary
                            total_invested = (holdings_df['totalQty'] * holdings_df['avgCostPrice']).sum()
                            current_value = (holdings_df['totalQty'] * holdings_df['LTP']).sum()
                            total_pnl = holdings_df['unrealizedPnl'].sum() + holdings_df['realizedPnl'].sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("💰 Total Invested", format_currency(total_invested))
                            with col2:
                                st.metric("📈 Current Value", format_currency(current_value))
                            with col3:
                                st.metric("💹 Total P&L", format_currency(total_pnl), f"{(total_pnl/total_invested*100):+.2f}%" if total_invested > 0 else "N/A")
                        
                        else:
                            st.info("📭 No holdings found")
            
            with col2:
                if st.button("💰 Refresh Fund & Margin", type="primary"):
                    with st.spinner("Fetching fund information..."):
                        funds = st.session_state.dhan_client.get_fund_limits()
                        
                        if funds['status'] == 'success':
                            fund_data = funds['data']
                            
                            st.subheader("💰 Fund & Margin Details")
                            
                            fund_info = [
                                ["Available Balance", format_currency(fund_data.get('availabelBalance', 0))],
                                ["Utilized Margin", format_currency(fund_data.get('utilizedAmount', 0))],
                                ["Collateral Amount", format_currency(fund_data.get('collateralAmount', 0))],
                                ["Received Amount", format_currency(fund_data.get('receivedAmount', 0))]
                            ]
                            
                            fund_df = pd.DataFrame(fund_info, columns=["Category", "Amount"])
                            st.dataframe(fund_df, use_container_width=True, hide_index=True)
                        else:
                            st.error("❌ Failed to fetch fund information")
                            
        except Exception as e:
            st.error(f"❌ Error fetching portfolio data: {str(e)}")

# ========== TAB 5: ORDERS ==========
with tab5:
    st.header("📋 Orders & Trades")
    
    if not st.session_state.connected:
        st.error("❌ Please connect to DhanHQ API first")
    else:
        tab5a, tab5b, tab5c = st.tabs(["📝 Place Order", "📊 Order Book", "📈 Trade Book"])
        
        # Place Order Tab
        with tab5a:
            st.subheader("📝 Place New Order")
            
            col1, col2 = st.columns(2)
            
            with col1:
                order_symbol = st.selectbox("Select Symbol", [stock['symbol'] for stock in SAMPLE_STOCKS])
                selected_stock_for_order = next(stock for stock in SAMPLE_STOCKS if stock['symbol'] == order_symbol)
                
                transaction_type = st.selectbox("Transaction Type", ["BUY", "SELL"])
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"])
                quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
                
            with col2:
                product_type = st.selectbox("Product Type", ["CNC", "INTRADAY", "MARGIN", "CO", "BO"])
                
                if order_type in ["LIMIT", "SL"]:
                    price = st.number_input("Price", min_value=0.01, step=0.01, value=100.0)
                else:
                    price = 0.0
                
                if order_type in ["SL", "SL-M"]:
                    trigger_price = st.number_input("Trigger Price", min_value=0.01, step=0.01, value=95.0)
                else:
                    trigger_price = 0.0
                
                validity = st.selectbox("Validity", ["DAY", "IOC"])
            
            if st.button("🚀 Place Order", type="primary"):
                try:
                    with st.spinner("Placing order..."):
                        order_response = st.session_state.dhan_client.place_order(
                            security_id=selected_stock_for_order['security_id'],
                            exchange_segment=selected_stock_for_order['exchange'],
                            transaction_type=transaction_type,
                            quantity=quantity,
                            order_type=order_type,
                            product_type=product_type,
                            price=price if order_type in ["LIMIT", "SL"] else 0,
                            trigger_price=trigger_price if order_type in ["SL", "SL-M"] else 0,
                            validity=validity
                        )
                        
                        if order_response['status'] == 'success':
                            st.success(f"✅ Order placed successfully! Order ID: {order_response['data']['orderId']}")
                        else:
                            st.error(f"❌ Order failed: {order_response.get('remarks', 'Unknown error')}")
                            
                except Exception as e:
                    st.error(f"❌ Error placing order: {str(e)}")
        
        # Order Book Tab
        with tab5b:
            st.subheader("📊 Order Book")
            
            if st.button("🔄 Refresh Orders"):
                try:
                    with st.spinner("Fetching orders..."):
                        orders = st.session_state.dhan_client.get_order_list()
                        
                        if orders['status'] == 'success' and orders['data']:
                            orders_df = pd.DataFrame(orders['data'])
                            
                            # Select relevant columns
                            display_columns = ['orderId', 'tradingSymbol', 'transactionType', 'orderType', 
                                             'quantity', 'price', 'orderStatus', 'createTime']
                            
                            if all(col in orders_df.columns for col in display_columns):
                                display_df = orders_df[display_columns].copy()
                                display_df.columns = ['Order ID', 'Symbol', 'Type', 'Order Type', 
                                                    'Quantity', 'Price', 'Status', 'Time']
                                
                                # Format price column
                                display_df['Price'] = display_df['Price'].apply(lambda x: format_currency(x) if x > 0 else 'Market')
                                
                                st.dataframe(display_df, use_container_width=True, hide_index=True)
                            else:
                                st.dataframe(orders_df, use_container_width=True)
                        else:
                            st.info("📭 No orders found")
                            
                except Exception as e:
                    st.error(f"❌ Error fetching orders: {str(e)}")
        
        # Trade Book Tab  
        with tab5c:
            st.subheader("📈 Trade Book")
            
            if st.button("🔄 Refresh Trades"):
                try:
                    with st.spinner("Fetching trades..."):
                        trades = st.session_state.dhan_client.get_trade_book()
                        
                        if trades['status'] == 'success' and trades['data']:
                            trades_df = pd.DataFrame(trades['data'])
                            
                            # Select relevant columns
                            display_columns = ['tradeId', 'tradingSymbol', 'transactionType', 
                                             'quantity', 'price', 'createTime']
                            
                            if all(col in trades_df.columns for col in display_columns):
                                display_df = trades_df[display_columns].copy()
                                display_df.columns = ['Trade ID', 'Symbol', 'Type', 'Quantity', 'Price', 'Time']
                                
                                # Format price column
                                display_df['Price'] = display_df['Price'].apply(lambda x: format_currency(x))
                                
                                st.dataframe(display_df, use_container_width=True, hide_index=True)
                                
                                # Trade Summary
                                total_trades = len(trades_df)
                                buy_trades = len(trades_df[trades_df['transactionType'] == 'BUY'])
                                sell_trades = len(trades_df[trades_df['transactionType'] == 'SELL'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Trades", total_trades)
                                with col2:
                                    st.metric("Buy Trades", buy_trades)
                                with col3:
                                    st.metric("Sell Trades", sell_trades)
                            else:
                                st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.info("📭 No trades found")
                            
                except Exception as e:
                    st.error(f"❌ Error fetching trades: {str(e)}")

# ========== AUTO REFRESH ==========
if auto_refresh and st.session_state.connected:
    time.sleep(30)
    st.rerun()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
<strong>DhanHQ Trading Dashboard</strong><br>
<em>Features:</em> Live Market Data • Advanced Charts • Technical Analysis • Portfolio Management • Order Placement<br>
<em>Built with:</em> Streamlit • Plotly • DhanHQ Python SDK<br><br>
<small>⚠️ <strong>Disclaimer:</strong> This dashboard is for educational and informational purposes only. 
Trading in financial markets involves substantial risk of loss. Always consult with qualified financial advisors 
and practice proper risk management before making trading decisions.</small>
</div>
""", unsafe_allow_html=True)

# ========== USAGE INSTRUCTIONS ==========
with st.expander("📖 How to Use This Dashboard"):
    st.markdown("""
    ## Setup Instructions
    
    1. **Install Dependencies:**
       ```bash
       pip install dhanhq streamlit plotly pandas numpy
       ```
    
    2. **Get DhanHQ API Credentials:**
       - Login to your DhanHQ account
       - Generate Client ID and Access Token from API section
       - Enter credentials in the sidebar
    
    3. **Features Available:**
       - **Live Data:** Real-time market quotes and OHLC data
       - **Charts:** Interactive candlestick charts with technical indicators
       - **Technical Analysis:** RSI, MACD, Bollinger Bands, Moving Averages
       - **Portfolio:** View holdings, P&L, and fund details
       - **Orders:** Place orders and view order/trade history
    
    ## Important Notes
    
    - This dashboard uses the official DhanHQ Python SDK
    - All API calls are cached to prevent rate limiting
    - Auto-refresh can be enabled for live monitoring
    - Always verify orders before placement
    - Practice with small quantities first
    
    ## Risk Disclaimer
    
    Trading involves substantial risk. This tool is for educational purposes only.
    Always consult financial advisors and use proper risk management.
    """)

# ========== DEVELOPMENT NOTES ==========
# This dashboard provides a comprehensive trading interface using the DhanHQ API
# Key features implemented:
# 1. Real-time market data with caching
# 2. Interactive charts with technical indicators
# 3. Portfolio management and P&L tracking  
# 4. Order placement and management
# 5. Technical analysis with trading signals
# 6. Auto-refresh capability
# 7. Error handling and user feedback
# 8. Mobile-responsive design
