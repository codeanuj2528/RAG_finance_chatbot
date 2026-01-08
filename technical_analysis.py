"""
technical_analysis.py - Technical Indicators
"""

import streamlit as st
from data_fetcher import fetch_indicator_data


def parse_technical_indicators(symbol: str) -> dict:
    """
    Fetch technical indicators from Alpha Vantage
    """
    indicators = {}
    
    # RSI
    rsi_data = fetch_indicator_data(symbol, function_name="RSI", interval="daily", time_period=14, series_type="close")
    if "Technical Analysis: RSI" in rsi_data:
        ta_rsi = rsi_data["Technical Analysis: RSI"]
        dates = sorted(ta_rsi.keys())
        if dates:
            latest_date = dates[-1]
            latest_rsi = ta_rsi[latest_date]["RSI"]
            indicators["RSI"] = float(latest_rsi)
            indicators["RSI_Date"] = latest_date

    # MACD
    macd_data = fetch_indicator_data(symbol, function_name="MACD", interval="daily", series_type="close")
    if "Technical Analysis: MACD" in macd_data:
        ta_macd = macd_data["Technical Analysis: MACD"]
        dates = sorted(ta_macd.keys())
        if dates:
            latest_date = dates[-1]
            macd_val = ta_macd[latest_date]["MACD"]
            signal_val = ta_macd[latest_date]["MACD_Signal"]
            indicators["MACD"] = float(macd_val)
            indicators["MACD_Signal"] = float(signal_val)
            indicators["MACD_Date"] = latest_date

    # SMA
    sma_data = fetch_indicator_data(symbol, function_name="SMA", interval="daily", time_period=20, series_type="close")
    if "Technical Analysis: SMA" in sma_data:
        ta_sma = sma_data["Technical Analysis: SMA"]
        dates = sorted(ta_sma.keys())
        if dates:
            latest_date = dates[-1]
            indicators["SMA_20"] = float(ta_sma[latest_date]["SMA"])

    return indicators
