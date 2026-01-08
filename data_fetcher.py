"""
data_fetcher.py - Asset Data Fetching from Alpha Vantage
"""

import os
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

AV_API_KEY = os.getenv("AV_API_KEY")


def fetch_stock_data(symbol: str) -> dict:
    """Fetch daily stock data from Alpha Vantage"""
    if not AV_API_KEY:
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        data = r.json()
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            return {}
        
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4. close"])
        open_price = float(daily_data["1. open"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": symbol,
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change:.2f}%",
        }
    except Exception as e:
        return {}


def fetch_forex_data(from_symbol: str, to_symbol: str) -> dict:
    """Fetch daily forex data from Alpha Vantage"""
    if not AV_API_KEY:
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        data = r.json()
        time_series = data.get("Time Series FX (Daily)", {})
        if not time_series:
            return {}
        
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4. close"])
        open_price = float(daily_data["1. open"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": f"{from_symbol}/{to_symbol}",
            "Current Price": f"${close_price:.4f}",
            "Price Change (Today)": f"{price_change:.2f}%",
        }
    except Exception as e:
        return {}


def fetch_crypto_data(symbol: str, market="USD") -> dict:
    """Fetch daily crypto data from Alpha Vantage"""
    if not AV_API_KEY:
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": symbol,
        "market": market,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        data = r.json()
        time_series = data.get("Time Series (Digital Currency Daily)", {})
        if not time_series:
            return {}
        
        latest_date = sorted(time_series.keys())[-1]
        daily_data = time_series[latest_date]

        close_price = float(daily_data["4a. close (USD)"])
        open_price = float(daily_data["1a. open (USD)"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": f"{symbol}/{market}",
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change:.2f}%",
        }
    except Exception as e:
        return {}


def fetch_indicator_data(symbol: str, function_name: str, interval: str = "daily", time_period: int = 10, series_type: str = "close") -> dict:
    """Fetch technical indicator from Alpha Vantage"""
    if not AV_API_KEY:
        return {}

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function_name,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": series_type,
        "apikey": AV_API_KEY,
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        return r.json()
    except Exception as e:
        return {}


def fetch_all_assets() -> list:
    """Fetch stocks, forex pairs, and cryptos"""
    stocks_to_fetch = ["IBM", "AAPL", "GOOGL"]
    forex_pairs = [("EUR", "USD"), ("GBP", "USD")]
    cryptos = [("BTC", "USD"), ("ETH", "USD")]

    results = []
    
    # Fetch stocks with threading
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, sym): sym for sym in stocks_to_fetch}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    # Fetch forex
    for (f, t) in forex_pairs:
        forex_data = fetch_forex_data(f, t)
        if forex_data:
            results.append(forex_data)

    # Fetch crypto
    for (c, market) in cryptos:
        crypto_data = fetch_crypto_data(c, market)
        if crypto_data:
            results.append(crypto_data)

    return results


def get_top_movers() -> list:
    """Fetch top cryptocurrency movers"""
    try:
        cryptos_to_check = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        movers = []

        for crypto in cryptos_to_check:
            crypto_data = fetch_crypto_data(crypto, "USD")
            if crypto_data:
                price_change_str = crypto_data.get("Price Change (Today)", "0.00%")
                price_change = float(price_change_str.replace("%", ""))

                movers.append({
                    "Symbol": crypto,
                    "Price": crypto_data.get("Current Price", "N/A"),
                    "24h Change": price_change_str,
                    "Change_Value": price_change
                })

        movers.sort(key=lambda x: abs(x.get("Change_Value", 0)), reverse=True)
        
        for mover in movers:
            mover.pop("Change_Value", None)

        return movers[:5]
        
    except Exception as e:
        return []
