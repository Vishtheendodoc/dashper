import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging
import json

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

# ========== DIRECT API CLIENT ==========
class DhanDirectAPI:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co/v2"
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token
        }
        logger.info(f"DhanDirectAPI initialized for client {client_id}")

    def make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request to DhanHQ API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, timeout=10)
            else:
                response = requests.post(url, headers=self.headers, json=data, timeout=10)
            
            logger.info(f"{method} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {"status": "error", "message": f"HTTP {response.status_code}"}
                
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "Request timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"status": "error", "message": str(e)}

    def test_connection(self) -> dict:
        """Test API connection using user profile"""
        return self.make_request("GET", "profile")

    def get_ltp_data(self, instruments: dict) -> dict:
        """Get LTP data for instruments"""
        return self.make_request("POST", "marketfeed/ltp", instruments)

    def get_ohlc_data(self, instruments: dict) -> dict:
        """Get OHLC data for instruments"""
        return self.make_request("POST", "marketfeed/ohlc", instruments)

    def get_quote_data(self, instruments: dict) -> dict:
        """Get detailed quote data for instruments"""
        return self.make_request("POST", "marketfeed/quote", instruments)

    def get_holdings(self) -> dict:
        """Get user holdings"""
        return self.make_request("GET", "holdings")

    def get_funds(self) -> dict:
        """Get fund limits"""
        return self.make_request("GET", "fundlimit")

    def get_orders(self) -> dict:
        """Get order list"""
        return self.make_request("GET", "orders")

    def get_trades(self) -> dict:
        """Get trade book"""
        return self.make_request("GET", "tradebook")

    def place_order(self, **kwargs) -> dict:
        """Place order"""
        order_data = {
            "dhanClientId": self.client_id,
            "correlationId": f"order_{int(time.time())}",
            "transactionType": kwargs.get("transaction_type", "BUY"),
            "exchangeSegment": kwargs.get("exchange_segment", "NSE_EQ"),
            "productType": kwargs.get("product_type", "CNC"),
            "orderType": kwargs.get("order_type", "MARKET"),
            "validity": kwargs.get("validity", "DAY"),
            "securityId": kwargs.get("security_id"),
            "quantity": kwargs.get("quantity", 1),
            "price": kwargs.get("price", 0),
            "triggerPrice": kwargs.get("trigger_price", 0)
        }
        return self.make_request("POST", "orders", order_data)

# ========== CSV DATA LOADING ==========
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_instruments_csv(file_path: str) -> pd.DataFrame:
    """Load instruments from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded instruments CSV '{file_path}' - shape={df.shape}")
        
        # Rename columns to standard format
        column_mapping = {
            'SEM_EXM_EXCH_ID': 'exchange_id',
            'SEM_SEGMENT': 'segment', 
            'SEM_SMST_SECURITY_ID': 'security_id',
            'SEM_INSTRUMENT_NAME': 'instrument_name',
            'SEM_EXPIRY_CODE': 'expiry_code',
            'SEM_TRADING_SYMBOL': 'trading_symbol',
            'SEM_LOT_UNITS': 'lot_units',
            'SEM_CUSTOM_SYMBOL': 'custom_symbol',
            'SEM_EXPIRY_DATE': 'expiry_date',
            'SEM_STRIKE_PRICE': 'strike_price',
            'SEM_OPTION_TYPE': 'option_type',
            'SEM_TICK_SIZE': 'tick_size',
            'SEM_EXPIRY_FLAG': 'expiry_flag',
            'SEM_EXCH_INSTRUMENT_TYPE': 'exchange_instrument_type',
            'SEM_SERIES': 'series',
            'SM_SYMBOL_NAME': 'symbol_name'
        }
        
        # Rename columns if they exist
        df = df.rename(columns=column_mapping)
        
        # Create exchange segment mapping for DhanHQ API
        exchange_mapping = {
            'NSE': 'NSE_EQ',
            'BSE': 'BSE_EQ',
            'F': 'NSE_FO',  # Futures & Options
            'C': 'NSE_CURRENCY',  # Currency
            'M': 'MCX_COMM'  # Commodity
        }
        
        # Map exchange segments
        if 'segment' in df.columns:
            df['exchange_segment'] = df['segment'].map(exchange_mapping).fillna(df['segment'])
        
        # Create display symbol combining symbol and instrument info
        if 'symbol_name' in df.columns and 'instrument_name' in df.columns:
            df['display_symbol'] = df['symbol_name'] + ' (' + df['instrument_name'] + ')'
        elif 'trading_symbol' in df.columns:
            df['display_symbol'] = df['trading_symbol']
        else:
            df['display_symbol'] = df.get('custom_symbol', 'Unknown')
        
        return df
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        st.error(f"CSV file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}", exc_info=True)
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

def get_instruments_for_selection(df: pd.DataFrame, limit: int = 100) -> List[Dict]:
    """Convert CSV data to instrument list for selection"""
    if df.empty:
        return SAMPLE_STOCKS  # Fallback to hardcoded stocks
    
    instruments = []
    
    # Group by instrument type for better organization
    equity_instruments = df[df['instrument_name'].isin(['EQUITY', 'EQ'])].head(50)
    futures_instruments = df[df['instrument_name'].isin(['FUTURES', 'FUTIDX', 'FUTSTK', 'FUTCUR'])].head(30)
    options_instruments = df[df['instrument_name'].isin(['OPTIONS', 'OPTIDX', 'OPTSTK'])].head(20)
    
    # Process each category
    for category_df in [equity_instruments, futures_instruments, options_instruments]:
        for _, row in category_df.iterrows():
            instrument = {
                'symbol': row.get('symbol_name', row.get('trading_symbol', 'Unknown')),
                'security_id': str(row['security_id']),
                'exchange': row.get('exchange_segment', 'NSE_EQ'),
                'instrument_name': row.get('instrument_name', 'EQUITY'),
                'display_symbol': row.get('display_symbol', row.get('trading_symbol', 'Unknown')),
                'lot_units': row.get('lot_units', 1),
                'tick_size': row.get('tick_size', 0.05),
                'expiry_date': row.get('expiry_date', ''),
                'strike_price': row.get('strike_price', 0)
            }
            instruments.append(instrument)
            
            if len(instruments) >= limit:
                break
        
        if len(instruments) >= limit:
            break
    
    return instruments if instruments else SAMPLE_STOCKS

# ========== SAMPLE STOCK DATA (FALLBACK) ==========
SAMPLE_STOCKS = [
    {"symbol": "RELIANCE", "security_id": "2885", "exchange": "NSE_EQ"},
    {"symbol": "TCS", "security_id": "11536", "exchange": "NSE_EQ"},
    {"symbol": "INFY", "security_id": "1594", "exchange": "NSE_EQ"},
    {"symbol": "HDFCBANK", "security_id": "1333", "exchange": "NSE_EQ"},
    {"symbol": "ICICIBANK", "security_id": "1270", "exchange": "NSE_EQ"},
    {"symbol": "ADANIENT", "security_id": "25", "exchange": "NSE_EQ"},
    {"symbol": "HINDUNILVR", "security_id": "1394", "exchange": "NSE_EQ"},
    {"symbol": "ITC", "security_id": "1660", "exchange": "NSE_EQ"},
    {"symbol": "SBIN", "security_id": "3045", "exchange": "NSE_EQ"},
    {"symbol": "BHARTIARTL", "security_id": "3677", "exchange": "NSE_EQ"}
]

# ========== HELPER FUNCTIONS ==========
def format_currency(amount):
    """Format currency in Indian format"""
    if amount is None or pd.isna(amount):
        return "₹0.00"
    return f"₹{amount:,.2f}"

def format_number(number):
    """Format numbers with Indian comma style"""
    if number is None or pd.isna(number):
        return "0"
    return f"{number:,}"

# ========== SESSION STATE MANAGEMENT ==========
if 'dhan_client' not in st.session_state:
    st.session_state.dhan_client = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = SAMPLE_STOCKS[0]
if 'live_data_cache' not in st.session_state:
    st.session_state.live_data_cache = {}
if 'connection_error' not in st.session_state:
    st.session_state.connection_error = ""

# ========== SIDEBAR ==========
st.sidebar.header("🔧 DhanHQ Configuration")

# API Connection
with st.sidebar.expander("🔐 API Connection", expanded=not st.session_state.connected):
    st.markdown("""
    **Steps to get API access:**
    1. Login to [web.dhan.co](https://web.dhan.co)
    2. Go to My Profile → "DhanHQ Trading APIs"
    3. Click "Request Access" (wait for approval)
    4. Generate Access Token
    5. Use your Dhan Client ID (numeric)
    """)
    
    client_id = st.text_input("Client ID", placeholder="e.g., 1000000123", help="Your numeric Dhan Client ID")
    access_token = st.text_input("Access Token", type="password", placeholder="Enter JWT token", help="JWT token from DhanHQ portal")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect", type="primary"):
            if client_id and access_token:
                with st.spinner("Testing connection..."):
                    try:
                        # Create direct API client
                        dhan_client = DhanDirectAPI(client_id, access_token)
                        
                        # Test connection
                        profile_response = dhan_client.test_connection()
                        
                        if 'status' in profile_response:
                            if profile_response['status'] == 'success':
                                st.session_state.dhan_client = dhan_client
                                st.session_state.connected = True
                                st.session_state.connection_error = ""
                                
                                # Show user info
                                if 'data' in profile_response:
                                    user_data = profile_response['data']
                                    st.success(f"✅ Connected! Client: {user_data.get('dhanClientId', 'N/A')}")
                                    st.info(f"Token valid till: {user_data.get('tokenValidity', 'N/A')}")
                                else:
                                    st.success("✅ Connected Successfully!")
                                st.rerun()
                            else:
                                error_msg = profile_response.get('message', 'Unknown error')
                                st.session_state.connection_error = error_msg
                                st.error(f"❌ Connection Failed: {error_msg}")
                        else:
                            # Handle different response formats
                            if 'errorType' in profile_response:
                                error_msg = profile_response.get('errorMessage', 'Invalid credentials')
                                st.session_state.connection_error = error_msg
                                st.error(f"❌ {error_msg}")
                            elif 'dhanClientId' in profile_response:
                                # Direct success response
                                st.session_state.dhan_client = dhan_client
                                st.session_state.connected = True
                                st.session_state.connection_error = ""
                                st.success(f"✅ Connected! Client: {profile_response.get('dhanClientId')}")
                                st.rerun()
                            else:
                                st.error("❌ Unexpected response format")
                                
                    except Exception as e:
                        error_msg = f"Connection error: {str(e)}"
                        st.session_state.connection_error = error_msg
                        st.error(f"❌ {error_msg}")
            else:
                st.error("❌ Please enter both Client ID and Access Token")
    
    with col2:
        if st.button("Disconnect"):
            st.session_state.dhan_client = None
            st.session_state.connected = False
            st.session_state.live_data_cache = {}
            st.session_state.connection_error = ""
            st.info("Disconnected")
            st.rerun()

# Show connection error if any
if st.session_state.connection_error:
    st.sidebar.error(f"Last Error: {st.session_state.connection_error}")

# Connection Status
if st.session_state.connected:
    st.sidebar.success("🟢 Connected to DhanHQ")
else:
    st.sidebar.error("🔴 Not Connected")

# ========== TROUBLESHOOTING SECTION ==========
if not st.session_state.connected:
    with st.sidebar.expander("🧪 Authentication Checker", expanded=False):
    st.markdown("**Step-by-step credential verification:**")
    
    test_client_id = st.text_input("Test Client ID", placeholder="1000000123")
    test_token = st.text_input("Test Token", type="password", placeholder="eyJ...")
    
    if st.button("🔍 Diagnose Issue"):
        if test_client_id and test_token:
            st.markdown("**Running diagnostics...**")
            
            # Check 1: Token format
            if test_token.startswith('eyJ'):
                st.success("✅ Token format looks correct (JWT)")
            else:
                st.error("❌ Token should start with 'eyJ' (JWT format)")
            
            # Check 2: Client ID format
            if test_client_id.isdigit():
                st.success("✅ Client ID format is numeric")
            else:
                st.error("❌ Client ID should be numeric only")
            
            # Check 3: API call test
            with st.spinner("Testing API call..."):
                headers = {"access-token": test_token}
                try:
                    response = requests.get(
                        "https://api.dhan.co/v2/profile",
                        headers=headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ API call successful!")
                        data = response.json()
                        st.json(data)
                    elif response.status_code == 401:
                        st.error("❌ HTTP 401: Invalid token or API access not approved")
                        st.markdown("""
                        **Possible solutions:**
                        1. Go to web.dhan.co → Profile → "DhanHQ Trading APIs"
                        2. Click "Request Access" if you see it
                        3. Generate a new access token
                        4. Make sure you copied the complete token
                        """)
                    else:
                        st.error(f"❌ HTTP {response.status_code}: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Network error: {str(e)}")
            
        else:
            st.warning("Enter both credentials to run diagnostics")
        st.markdown("""
        **Common Issues:**
        
        1. **API Access Not Approved**
           - Check if you clicked "Request Access"
           - Wait 24-48 hours for approval
        
        2. **Wrong Client ID**
           - Use numeric Client ID (not email)
           - Found in Profile section
        
        3. **Token Expired**
           - Generate new access token
           - Check token validity date
        
        4. **Test Your Credentials**
           - Use browser/Postman to test:
        ```
        GET https://api.dhan.co/v2/profile
        Headers: access-token: YOUR_TOKEN
        ```
        
        **Still having issues?**
        Contact DhanHQ support with your Client ID.
        """)

# Stock Selection
st.sidebar.header("📊 Stock Selection")
stock_options = {f"{stock['symbol']} ({stock['exchange']})": stock for stock in SAMPLE_STOCKS}
selected_stock_display = st.sidebar.selectbox(
    "Select Stock", 
    options=list(stock_options.keys()),
    index=0
)
st.session_state.selected_stock = stock_options[selected_stock_display]

# Auto Refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (60s)", value=False)

# ========== MAIN APPLICATION ==========
st.title("DhanHQ Trading Dashboard (Direct API)")

# ========== LIVE DATA TAB ==========
st.header(f"📈 Live Market Data - {st.session_state.selected_stock['symbol']}")

if not st.session_state.connected:
    st.error("❌ Please connect to DhanHQ API first using the sidebar")
    
    # Show manual test option
    with st.expander("🧪 Manual API Test"):
        st.markdown("""
        Test your credentials manually:
        
        1. Open browser/Postman
        2. Make GET request to: `https://api.dhan.co/v2/profile`
        3. Add header: `access-token: YOUR_ACCESS_TOKEN`
        4. Should return your profile data if credentials are correct
        """)
        
        if st.button("Test Profile API"):
            test_client_id = st.text_input("Test Client ID")
            test_token = st.text_input("Test Access Token", type="password")
            
            if test_client_id and test_token:
                with st.spinner("Testing..."):
                    test_client = DhanDirectAPI(test_client_id, test_token)
                    result = test_client.test_connection()
                    st.json(result)

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
                # Create instruments dict
                instruments = {
                    stock['exchange']: [int(stock['security_id'])]
                }
                
                # Get LTP and OHLC data
                ltp_response = st.session_state.dhan_client.get_ltp_data(instruments)
                time.sleep(0.1)  # Small delay between requests
                ohlc_response = st.session_state.dhan_client.get_ohlc_data(instruments)
                
                st.session_state.live_data_cache[cache_key] = {
                    'ltp': ltp_response,
                    'ohlc': ohlc_response,
                    'timestamp': datetime.now()
                }
        
        # Extract data from cache
        cached_data = st.session_state.live_data_cache[cache_key]
        ltp_data = cached_data['ltp']
        ohlc_data = cached_data['ohlc']
        
        # Display data
        if ltp_data.get('status') == 'success' or 'data' in ltp_data:
            # Extract nested data
            exchange_key = stock['exchange']
            security_key = int(stock['security_id'])
            
            # Get LTP info
            ltp_info = {}
            if 'data' in ltp_data and exchange_key in ltp_data['data']:
                ltp_info = ltp_data['data'][exchange_key].get(str(security_key), {})
            
            # Get OHLC info
            ohlc_info = {}
            if ohlc_data.get('status') == 'success' and 'data' in ohlc_data:
                if exchange_key in ohlc_data['data']:
                    ohlc_info = ohlc_data['data'][exchange_key].get(str(security_key), {})
            
            # Use available data or defaults
            current_ltp = ltp_info.get('LTP', 100.0)
            high_val = ohlc_info.get('high', current_ltp)
            low_val = ohlc_info.get('low', current_ltp)
            open_val = ohlc_info.get('open', current_ltp)
            volume_val = ohlc_info.get('volume', 0)
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                change_val = ltp_info.get('change', 0)
                change_pct = ltp_info.get('pChange', 0)
                st.metric(
                    "💰 LTP", 
                    format_currency(current_ltp),
                    f"{change_val:+.2f} ({change_pct:+.2f}%)"
                )
            
            with col2:
                st.metric("📈 High", format_currency(high_val))
            
            with col3:
                st.metric("📉 Low", format_currency(low_val))
            
            with col4:
                st.metric("🔓 Open", format_currency(open_val))
            
            with col5:
                st.metric("📊 Volume", format_number(volume_val))
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            with col1:
                day_range = high_val - low_val
                st.info(f"**Day Range:** {format_currency(day_range)}")
            
            with col2:
                if day_range > 0:
                    price_pos = ((current_ltp - low_val) / day_range * 100)
                    st.info(f"**Price Position:** {price_pos:.1f}%")
                else:
                    st.info("**Price Position:** N/A")
            
            with col3:
                st.info(f"**Last Updated:** {cached_data['timestamp'].strftime('%H:%M:%S')}")
            
            # Raw data for debugging
            with st.expander("🔍 Debug: Raw API Response"):
                st.json({
                    "LTP_Response": ltp_data,
                    "OHLC_Response": ohlc_data
                })
        
        else:
            st.error("❌ Failed to fetch market data")
            error_msg = ltp_data.get('message', 'Unknown error')
            st.error(f"Error: {error_msg}")
            
            # Show raw response for debugging
            with st.expander("🔍 Debug: Error Response"):
                st.json(ltp_data)
                
    except Exception as e:
        st.error(f"❌ Error fetching live data: {str(e)}")
        logger.exception("Live data fetch error")

# ========== PORTFOLIO SECTION ==========
if st.session_state.connected:
    st.header("💼 Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Get Holdings"):
            with st.spinner("Fetching holdings..."):
                holdings = st.session_state.dhan_client.get_holdings()
                if holdings.get('status') == 'success':
                    if holdings.get('data'):
                        df = pd.DataFrame(holdings['data'])
                        st.dataframe(df)
                    else:
                        st.info("No holdings found")
                else:
                    st.error(f"Error: {holdings.get('message', 'Failed to fetch holdings')}")
    
    with col2:
        if st.button("💰 Get Funds"):
            with st.spinner("Fetching funds..."):
                funds = st.session_state.dhan_client.get_funds()
                if funds.get('status') == 'success':
                    st.json(funds.get('data', {}))
                else:
                    st.error(f"Error: {funds.get('message', 'Failed to fetch funds')}")

# ========== AUTO REFRESH ==========
if auto_refresh and st.session_state.connected:
    time.sleep(60)
    st.rerun()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
**DhanHQ Trading Dashboard (Direct API Implementation)**

This version uses direct HTTP calls to DhanHQ API v2 instead of the Python SDK, which can help resolve authentication issues.

⚠️ **Important:** This is for educational purposes only. Always verify trades and use proper risk management.
""")
