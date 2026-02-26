"""
Market Data Fetcher with Multi-Source Support
Handles data collection from various exchanges with robust error handling
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import ccxt
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

from config.settings import settings, MarketType
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source: str
    
class DataSource(Enum):
    CCXT = "ccxt"
    YFINANCE = "yfinance"
    CSV = "csv"

class MarketDataFetcher:
    """Unified market data fetcher with failover support"""
    
    def __init__(self, market_type: MarketType = MarketType.CRYPTO):
        self.market_type = market_type
        self.exchange = None
        self.retry_count = 0
        self.max_retries = settings.data.max_retries