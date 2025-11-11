"""
OKX API Client for Cryptocurrency Spot Data

This module provides functions to fetch cryptocurrency spot data from OKX API
with support for various timeframes, trading pairs, and date ranges.
"""

import os
import time
import json
import hmac
import hashlib
import base64
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Union, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OKXAPIError(Exception):
    """Custom exception for OKX API errors"""
    pass


class OKXClient:
    """
    OKX API client for fetching cryptocurrency spot data
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 passphrase: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OKX API client
        
        Args:
            api_key: OKX API key (if None, will try to load from environment)
            api_secret: OKX API secret (if None, will try to load from environment)
            passphrase: OKX API passphrase (if None, will try to load from environment)
            base_url: OKX API base URL (defaults to production)
        """
        self.api_key = api_key or os.getenv('OKX_API_KEY')
        self.api_secret = api_secret or os.getenv('OKX_API_SECRET')
        self.passphrase = passphrase or os.getenv('OKX_PASSPHRASE')
        self.base_url = base_url or os.getenv('OKX_BASE_URL', 'https://www.okx.com')
        self.timeout = int(os.getenv('OKX_TIMEOUT', '30'))
        
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError(
                "API credentials not provided. Please set OKX_API_KEY, "
                "OKX_API_SECRET, and OKX_PASSPHRASE environment variables "
                "or pass them directly to the constructor."
            )
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """
        Generate OKX API signature
        
        Args:
            timestamp: Current timestamp in ISO format
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body (empty for GET requests)
            
        Returns:
            Base64 encoded signature
        """
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """
        Generate headers for OKX API request
        
        Args:
            method: HTTP method
            request_path: API endpoint path
            body: Request body
            
        Returns:
            Dictionary of headers
        """
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict:
        """
        Make authenticated request to OKX API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            API response data
            
        Raises:
            OKXAPIError: If API request fails
        """
        url = f"{self.base_url}{endpoint}"
        body = ''
        
        if data:
            body = str(data)
        
        headers = self._get_headers(method, endpoint, body)
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('code') != '0':
                raise OKXAPIError(f"API Error {result.get('code')}: {result.get('msg')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise OKXAPIError(f"Request failed: {str(e)}")
        except ValueError as e:
            raise OKXAPIError(f"Invalid JSON response: {str(e)}")


def parse_okx_orderbook_file(
    file_path: Union[str, os.PathLike],
    as_dataframe: bool = True,
    convert_timestamp: bool = True,
    nrows: float | None = None
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Parse a local OKX Level 2 orderbook dump (newline-delimited JSON) into structured data.

    Args:
        file_path: Path to the `.data` file downloaded from OKX.
        as_dataframe: Return a `pandas.DataFrame` when True, otherwise return a list of dictionaries.
        convert_timestamp: When True, append a UTC `timestamp` derived from the millisecond `ts`.

    Returns:
        pandas.DataFrame or list: Orderbook rows with columns/keys:
            ['instrument', 'action', 'side', 'price', 'size', 'count', 'ts', 'timestamp' (optional)]

    Raises:
        FileNotFoundError: If `file_path` does not point to an existing file.
        ValueError: If any line in the file cannot be decoded as JSON.
    """
    path_str = os.fspath(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    records: List[Dict[str, Any]] = []

    with open(path_str, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if nrows is not None and line_number > nrows:
                break

            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_number}: {exc}") from exc

            ts_raw = payload.get("ts")
            try:
                ts_value = int(ts_raw) if ts_raw is not None else None
            except (TypeError, ValueError):
                ts_value = None

            instrument = payload.get("instId")
            action = payload.get("action")

            for side_key in ("bids", "asks"):
                levels = payload.get(side_key) or []
                side = "bid" if side_key == "bids" else "ask"

                for level in levels:
                    if not isinstance(level, (list, tuple)) or len(level) < 2:
                        continue

                    price_raw = level[0]
                    size_raw = level[1]
                    count_raw = level[2] if len(level) > 2 else None

                    try:
                        price_val = float(price_raw)
                    except (TypeError, ValueError):
                        price_val = None

                    try:
                        size_val = float(size_raw)
                    except (TypeError, ValueError):
                        size_val = None

                    try:
                        count_val = int(count_raw) if count_raw is not None else None
                    except (TypeError, ValueError):
                        count_val = None

                    records.append(
                        {
                            "instrument": instrument,
                            "action": action,
                            "side": side,
                            "price": price_val,
                            "size": size_val,
                            "count": count_val,
                            "ts": ts_value,
                        }
                    )

            # If no bids/asks were present, still keep a placeholder record so nothing is lost.
            if not payload.get("bids") and not payload.get("asks"):
                records.append(
                    {
                        "instrument": instrument,
                        "action": action,
                        "side": None,
                        "price": None,
                        "size": None,
                        "count": None,
                        "ts": ts_value,
                    }
                )

    def _attach_timestamp(target_records: List[Dict[str, Any]]) -> None:
        for record in target_records:
            ts_entry = record.get("ts")
            try:
                record["timestamp"] = (
                    pd.to_datetime(ts_entry, unit="ms", utc=True) if ts_entry is not None else pd.NaT
                )
            except (TypeError, ValueError, OverflowError):
                record["timestamp"] = pd.NaT

    if not as_dataframe:
        if convert_timestamp:
            _attach_timestamp(records)
        return records

    df = pd.DataFrame.from_records(records, columns=["instrument", "action", "side", "price", "size", "count", "ts"])

    if df.empty:
        return df

    if convert_timestamp:
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")

    return df


def get_crypto_spot_data(
    symbol: str,
    interval: str = '1m',
    start_time: Optional[Union[str, datetime, int]] = None,
    end_time: Optional[Union[str, datetime, int]] = None,
    limit: int = 100,
    client: Optional[OKXClient] = None
) -> pd.DataFrame:
    """
    Fetch cryptocurrency spot data from OKX API
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USDT', 'ETH-USDT')
        interval: Time interval ('1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M', '3M')
        start_time: Start time (ISO string, datetime object, or Unix timestamp in ms)
        end_time: End time (ISO string, datetime object, or Unix timestamp in ms)
        limit: Maximum number of candles to return (1-300, default 100)
        client: OKXClient instance (if None, will create new instance)
        
    Returns:
        pandas.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_currency', 'volume_currency_pair', 'confirm']
        
    Raises:
        OKXAPIError: If API request fails
        ValueError: If parameters are invalid
    """
    if client is None:
        client = OKXClient()
    
    # Validate interval
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M', '3M']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval '{interval}'. Must be one of: {valid_intervals}")
    
    # Validate limit
    if not 1 <= limit <= 300:
        raise ValueError("Limit must be between 1 and 300")
    
    # Convert time parameters to Unix timestamp in milliseconds
    def to_timestamp_ms(time_input):
        if time_input is None:
            return None
        if isinstance(time_input, int):
            return time_input
        if isinstance(time_input, str):
            dt = datetime.fromisoformat(time_input.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        if isinstance(time_input, datetime):
            return int(time_input.timestamp() * 1000)
        raise ValueError(f"Invalid time format: {type(time_input)}")
    
    start_ts = to_timestamp_ms(start_time)
    end_ts = to_timestamp_ms(end_time)
    
    # Prepare request parameters
    params = {
        'instId': symbol,
        'bar': interval,
        'limit': str(limit)
    }
    
    # OKX API logic: 
    # - 'before': get candles before this timestamp (most recent data)
    # - 'after': get candles after this timestamp (older data)
    # For recent data, we typically only use 'before' parameter
    
    # Always use 'before' if end_time is specified
    if end_ts:
        params['before'] = str(end_ts)
    
    # Only use 'after' for historical data requests (> 1 day)
    # For recent data (last 24 hours), omit 'after' to avoid conflicts
    if start_ts and start_time and end_time:
        time_diff_hours = (end_time - start_time).total_seconds() / 3600
        if time_diff_hours > 24:  # Only for requests > 24 hours
            params['after'] = str(start_ts)
    
    # Make API request
    response = client._make_request('GET', '/api/v5/market/candles', params=params)
    
    # Convert response to DataFrame
    if not response.get('data'):
        return pd.DataFrame()
    
    df = pd.DataFrame(response['data'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'volume_currency', 'volume_currency_pair', 'confirm'
    ])
    
    # Convert data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'volume_currency', 'volume_currency_pair']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['confirm'] = df['confirm'].astype(int)
    
    # Sort by timestamp (oldest first)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def get_available_instruments(client: Optional[OKXClient] = None) -> pd.DataFrame:
    """
    Get list of available trading instruments from OKX
    
    Args:
        client: OKXClient instance (if None, will create new instance)
        
    Returns:
        pandas.DataFrame: Available instruments with their details
    """
    if client is None:
        client = OKXClient()
    
    response = client._make_request('GET', '/api/v5/public/instruments', params={'instType': 'SPOT'})
    
    if not response.get('data'):
        return pd.DataFrame()
    
    df = pd.DataFrame(response['data'])
    
    # Convert numeric columns
    numeric_columns = ['minSz', 'lotSz', 'tickSz', 'minPx', 'maxPx']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_ticker(symbol: str, client: Optional[OKXClient] = None) -> Dict:
    """
    Get current ticker data for a symbol
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USDT')
        client: OKXClient instance (if None, will create new instance)
        
    Returns:
        dict: Ticker data
    """
    if client is None:
        client = OKXClient()
    
    response = client._make_request('GET', '/api/v5/market/ticker', params={'instId': symbol})
    
    if response.get('data'):
        return response['data'][0]
    return {}


# Example usage and utility functions
def get_recent_data(symbol: str, hours: int = 24, interval: str = '1H', 
                   client: Optional[OKXClient] = None) -> pd.DataFrame:
    """
    Convenience function to get recent data for the last N hours
    
    Args:
        symbol: Trading pair symbol
        hours: Number of hours to look back
        interval: Time interval
        client: OKXClient instance
        
    Returns:
        pandas.DataFrame: Recent OHLCV data
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - pd.Timedelta(hours=hours)
    
    return get_crypto_spot_data(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        client=client
    )


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize client
        client = OKXClient()
        
        # Get recent BTC-USDT data
        print("Fetching recent BTC-USDT data...")
        df = get_recent_data('BTC-USDT', hours=24, interval='1H')
        print(f"Retrieved {len(df)} data points")
        print(df.head())
        
        # Get available instruments
        print("\nFetching available instruments...")
        instruments = get_available_instruments()
        print(f"Found {len(instruments)} spot trading pairs")
        print(instruments[['instId', 'baseCcy', 'quoteCcy']].head())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your .env file with OKX API credentials")
