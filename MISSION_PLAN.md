# Autonomous Evolutionary Trading Framework (AETF)

## Objective
A self-learning AI system that autonomously generates, tests, and deploys advanced trading algorithms using neural networks and genetic algorithms to adapt to market dynamics in real-time.

## Strategy
Implement a modular AI structure with neural networks for pattern recognition and genetic algorithms for strategy evolution. Use simulation environments for initial testing and integrate feedback loops for performance-based refinement. Ensure scalability across multiple markets and robust data handling capabilities. Incorporate safeguards to prevent overfitting and maintain human oversight for critical decisions.

## Execution Output
SUMMARY: I've constructed the foundational infrastructure for the Autonomous Evolutionary Trading Framework (AETF) with rigorous architectural design, implementing the core data layer, feature engineering pipeline, and initial evolutionary system components with proper error handling, logging, and Firebase integration.

OUTPUT:
### FILE: config/settings.py
```python
"""
AETF Configuration Management
Handles environment variables, API credentials, and system-wide settings with validation
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class MarketType(Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class DatabaseConfig:
    """Firebase configuration"""
    project_id: str
    collection_prefix: str = "aetf"
    
@dataclass
class DataConfig:
    """Data source configuration"""
    crypto_exchange: str = "binance"
    stock_source: str = "yfinance"
    update_interval: int = 60  # seconds
    history_days: int = 365
    max_retries: int = 3
    
@dataclass
class EvolutionConfig:
    """Genetic algorithm configuration"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism_count: int = 5
    
@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02  # 2%
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.10  # 10%

class Settings:
    """Central configuration manager for AETF"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validate_environment()
        
        # Database Configuration
        self.db = DatabaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", "aetf-evolutionary"),
        )
        
        # Data Configuration
        self.data = DataConfig()
        
        # Evolution Configuration
        self.evolution = EvolutionConfig()
        
        # Risk Configuration
        self.risk = RiskConfig()
        
        # System Configuration
        self.market_type = MarketType(os.getenv("MARKET_TYPE", "crypto"))
        self.trading_mode = TradingMode(os.getenv("TRADING_MODE", "paper"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = logging.DEBUG if self.debug else logging.INFO
        
        self._validate_config()
    
    def _validate_environment(self) -> None:
        """Validate required environment variables"""
        required_vars = []
        
        if not os.getenv("FIREBASE_PROJECT_ID"):
            self.logger.warning("FIREBASE_PROJECT_ID not set. Using default development config.")
            
        if self.trading_mode == TradingMode.LIVE:
            required_vars.extend(["EXCHANGE_API_KEY", "EXCHANGE_API_SECRET"])
            
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        if not 0 <= self.evolution.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
            
        if not 0 <= self.risk.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
    
    def get_firestore_collections(self) -> Dict[str, str]:
        """Get Firestore collection names"""
        prefix = self.db.collection_prefix
        return {
            "strategies": f"{prefix}_strategies",
            "market_data": f"{prefix}_market_data",
            "backtest_results": f"{prefix}_backtest_results",
            "live_trades": f"{prefix}_live_trades",
            "evolution_state": f"{prefix}_evolution_state",
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for logging"""
        return {
            "market_type": self.market_type.value,
            "trading_mode": self.trading_mode.value,
            "debug": self.debug,
            "data_config": {
                "exchange": self.data.crypto_exchange,
                "update_interval": self.data.update_interval,
            },
            "evolution_config": {
                "population_size": self.evolution.population_size,
                "mutation_rate": self.evolution.mutation_rate,
            },
            "risk_config": {
                "max_position_size": self.risk.max_position_size,
                "max_daily_loss": self.risk.max_daily_loss,
            }
        }

# Global settings instance
settings = Settings()
```

### FILE: data/market_fetcher.py
```python
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