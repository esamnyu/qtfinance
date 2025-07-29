"""Data loading utilities for fetching market data."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

import pandas as pd
import yfinance as yf
from pandas import DataFrame

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def fetch_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> DataFrame:
        """Fetch market data for given symbols and date range."""
        pass
    
    @abstractmethod
    def fetch_latest(
        self,
        symbols: Union[str, List[str]],
        period: str = "1d"
    ) -> DataFrame:
        """Fetch latest market data."""
        pass


class YFinanceLoader(DataLoader):
    """Yahoo Finance data loader implementation."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize YFinance loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
    def fetch_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        auto_adjust: bool = True,
        prepost: bool = False,
        threads: bool = True
    ) -> DataFrame:
        """Fetch historical market data from Yahoo Finance.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            auto_adjust: Adjust OHLC for splits and dividends
            prepost: Include pre and post market data
            threads: Use multi-threading for multiple symbols
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        try:
            logger.info(f"Fetching data for {symbols} from {start_date} to {end_date}")
            
            data = yf.download(
                tickers=symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=auto_adjust,
                prepost=prepost,
                threads=threads,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbols}")
                return pd.DataFrame()
                
            # Handle single symbol case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
                
            logger.info(f"Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
            
    def fetch_latest(
        self,
        symbols: Union[str, List[str]],
        period: str = "1d",
        interval: str = "1d"
    ) -> DataFrame:
        """Fetch latest market data.
        
        Args:
            symbols: Single symbol or list of symbols
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval
            
        Returns:
            DataFrame with latest OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        try:
            logger.info(f"Fetching latest {period} data for {symbols}")
            
            data = yf.download(
                tickers=symbols,
                period=period,
                interval=interval,
                auto_adjust=True,
                threads=True,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbols}")
                return pd.DataFrame()
                
            # Handle single symbol case
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([data.columns, symbols])
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            raise
            
    def fetch_info(self, symbol: str) -> Dict:
        """Fetch ticker information.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
            
    def fetch_sp500_constituents(self) -> List[str]:
        """Fetch current S&P 500 constituents.
        
        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            # Fetch from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols
            symbols = [s.replace('.', '-') for s in symbols]  # Handle BRK.B -> BRK-B
            
            logger.info(f"Fetched {len(symbols)} S&P 500 constituents")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 constituents: {e}")
            # Return some default symbols as fallback
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 
                   'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'NVDA', 'HD']


class MultiSourceLoader(DataLoader):
    """Data loader that can fetch from multiple sources."""
    
    def __init__(self, primary_loader: DataLoader, fallback_loaders: List[DataLoader] = None):
        """Initialize multi-source loader.
        
        Args:
            primary_loader: Primary data source
            fallback_loaders: List of fallback data sources
        """
        self.primary_loader = primary_loader
        self.fallback_loaders = fallback_loaders or []
        
    def fetch_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> DataFrame:
        """Fetch data with automatic fallback to alternative sources."""
        try:
            return self.primary_loader.fetch_data(symbols, start_date, end_date, **kwargs)
        except Exception as e:
            logger.warning(f"Primary loader failed: {e}")
            
            for loader in self.fallback_loaders:
                try:
                    logger.info(f"Trying fallback loader: {type(loader).__name__}")
                    return loader.fetch_data(symbols, start_date, end_date, **kwargs)
                except Exception as fallback_error:
                    logger.warning(f"Fallback loader failed: {fallback_error}")
                    
            raise RuntimeError("All data loaders failed")
            
    def fetch_latest(
        self,
        symbols: Union[str, List[str]],
        period: str = "1d"
    ) -> DataFrame:
        """Fetch latest data with automatic fallback."""
        try:
            return self.primary_loader.fetch_latest(symbols, period)
        except Exception as e:
            logger.warning(f"Primary loader failed: {e}")
            
            for loader in self.fallback_loaders:
                try:
                    return loader.fetch_latest(symbols, period)
                except Exception:
                    continue
                    
            raise RuntimeError("All data loaders failed")