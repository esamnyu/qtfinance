"""Feature engineering for regime detection and trading strategies."""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for market regime detection."""
    
    def __init__(self, lookback_periods: Optional[Dict[str, int]] = None):
        """Initialize feature engineer.
        
        Args:
            lookback_periods: Dictionary of lookback periods for various features
        """
        self.lookback_periods = lookback_periods or {
            'returns': 1,
            'volatility_short': 20,
            'volatility_long': 60,
            'volume': 20,
            'correlation': 60,
            'skew': 20,
            'kurtosis': 20
        }
        
    def calculate_returns(self, prices: DataFrame, periods: int = 1) -> DataFrame:
        """Calculate log returns.
        
        Args:
            prices: DataFrame with price data
            periods: Number of periods for returns calculation
            
        Returns:
            DataFrame with log returns
        """
        return np.log(prices / prices.shift(periods))
        
    def calculate_volatility(self, returns: DataFrame, window: int = 20) -> DataFrame:
        """Calculate realized volatility.
        
        Args:
            returns: DataFrame with returns
            window: Rolling window size
            
        Returns:
            DataFrame with annualized volatility
        """
        # Annualize assuming 252 trading days
        return returns.rolling(window=window).std() * np.sqrt(252)
        
    def calculate_price_features(self, ohlcv: DataFrame) -> DataFrame:
        """Calculate price-based features.
        
        Args:
            ohlcv: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=ohlcv.index)
        
        # Price range
        features['price_range'] = (ohlcv['High'] - ohlcv['Low']) / ohlcv['Close']
        
        # Garman-Klass volatility (more efficient than close-to-close)
        features['gk_volatility'] = np.sqrt(
            252 * 0.5 * np.log(ohlcv['High'] / ohlcv['Low'])**2 -
            (2 * np.log(2) - 1) * np.log(ohlcv['Close'] / ohlcv['Open'])**2
        )
        
        # Volume features
        features['volume_ratio'] = ohlcv['Volume'] / ohlcv['Volume'].rolling(
            window=self.lookback_periods['volume']).mean()
        
        # Price position within daily range
        features['price_position'] = (ohlcv['Close'] - ohlcv['Low']) / \
                                   (ohlcv['High'] - ohlcv['Low'])
        
        return features
        
    def calculate_statistical_features(self, returns: DataFrame) -> DataFrame:
        """Calculate statistical moments and features.
        
        Args:
            returns: DataFrame with returns
            
        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=returns.index)
        
        # Rolling skewness
        features['skew'] = returns.rolling(
            window=self.lookback_periods['skew']).skew()
        
        # Rolling kurtosis
        features['kurtosis'] = returns.rolling(
            window=self.lookback_periods['kurtosis']).kurt()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        features['downside_vol'] = downside_returns.rolling(
            window=self.lookback_periods['volatility_short']).std() * np.sqrt(252)
        
        return features
        
    def calculate_regime_features(self, prices: DataFrame, returns: DataFrame) -> DataFrame:
        """Calculate features specifically useful for regime detection.
        
        Args:
            prices: DataFrame with price data
            returns: DataFrame with returns
            
        Returns:
            DataFrame with regime detection features
        """
        features = pd.DataFrame(index=prices.index)
        
        # Volatility ratio (short/long)
        vol_short = self.calculate_volatility(returns, self.lookback_periods['volatility_short'])
        vol_long = self.calculate_volatility(returns, self.lookback_periods['volatility_long'])
        features['vol_ratio'] = vol_short / vol_long
        
        # Rolling maximum drawdown
        rolling_max = prices.rolling(window=self.lookback_periods['volatility_long']).max()
        features['drawdown'] = (prices - rolling_max) / rolling_max
        
        # Trend strength (using moving averages)
        ma_short = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=50).mean()
        features['trend_strength'] = (ma_short - ma_long) / ma_long
        
        # Volatility of volatility
        features['vol_of_vol'] = vol_short.rolling(
            window=self.lookback_periods['volatility_short']).std()
        
        # Correlation structure break
        if isinstance(prices, pd.DataFrame) and len(prices.columns) > 1:
            # Calculate rolling correlation matrix eigenvalues
            corr_eigenvalues = []
            for i in range(self.lookback_periods['correlation'], len(prices)):
                window_data = returns.iloc[i-self.lookback_periods['correlation']:i]
                corr_matrix = window_data.corr()
                eigenvalues = np.linalg.eigvals(corr_matrix)
                corr_eigenvalues.append(np.max(eigenvalues) / np.sum(eigenvalues))
            
            # Pad the beginning with NaN
            padding = [np.nan] * self.lookback_periods['correlation']
            features['correlation_concentration'] = padding + corr_eigenvalues
        
        return features
        
    def calculate_technical_indicators(self, ohlcv: DataFrame) -> DataFrame:
        """Calculate technical indicators useful for regime detection.
        
        Args:
            ohlcv: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=ohlcv.index)
        
        # RSI
        features['rsi'] = ta.momentum.RSIIndicator(
            close=ohlcv['Close'], window=14).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=ohlcv['Close'], window=20, window_dev=2)
        features['bb_width'] = bb.bollinger_wband()
        features['bb_position'] = (ohlcv['Close'] - bb.bollinger_mavg()) / \
                                 (bb.bollinger_hband() - bb.bollinger_lband())
        
        # ATR (Average True Range)
        features['atr'] = ta.volatility.AverageTrueRange(
            high=ohlcv['High'], low=ohlcv['Low'], close=ohlcv['Close']).average_true_range()
        
        # ADX (Average Directional Index)
        features['adx'] = ta.trend.ADXIndicator(
            high=ohlcv['High'], low=ohlcv['Low'], close=ohlcv['Close']).adx()
        
        return features
        
    def engineer_features(
        self,
        ohlcv: DataFrame,
        feature_groups: List[str] = None
    ) -> Tuple[DataFrame, List[str]]:
        """Engineer all features for regime detection.
        
        Args:
            ohlcv: DataFrame with OHLCV data
            feature_groups: List of feature groups to calculate
                          ['price', 'statistical', 'regime', 'technical']
            
        Returns:
            Tuple of (features DataFrame, feature names list)
        """
        if feature_groups is None:
            feature_groups = ['price', 'statistical', 'regime', 'technical']
            
        all_features = []
        
        # Calculate returns first (needed for other features)
        if 'Close' in ohlcv.columns:
            prices = ohlcv['Close']
        else:
            # Multi-index columns (multiple symbols)
            prices = ohlcv['Close']
            
        returns = self.calculate_returns(prices)
        
        # Add returns as a feature
        returns_df = pd.DataFrame({'returns': returns}) if isinstance(returns, Series) else returns
        returns_df.columns = [f'returns_{col}' if col != 'returns' else col 
                             for col in returns_df.columns]
        all_features.append(returns_df)
        
        # Price features
        if 'price' in feature_groups:
            price_features = self.calculate_price_features(ohlcv)
            all_features.append(price_features)
            
        # Statistical features
        if 'statistical' in feature_groups:
            stat_features = self.calculate_statistical_features(returns)
            all_features.append(stat_features)
            
        # Regime-specific features
        if 'regime' in feature_groups:
            regime_features = self.calculate_regime_features(prices, returns)
            all_features.append(regime_features)
            
        # Technical indicators
        if 'technical' in feature_groups:
            tech_features = self.calculate_technical_indicators(ohlcv)
            all_features.append(tech_features)
            
        # Combine all features
        features_df = pd.concat(all_features, axis=1)
        
        # Drop rows with NaN (from rolling calculations)
        features_df = features_df.dropna()
        
        feature_names = features_df.columns.tolist()
        logger.info(f"Engineered {len(feature_names)} features")
        
        return features_df, feature_names
        
    def select_features(
        self,
        features: DataFrame,
        target: Series,
        n_features: int = 10,
        method: str = 'mutual_info'
    ) -> List[str]:
        """Select most important features for regime detection.
        
        Args:
            features: DataFrame with all features
            target: Series with target variable (regime labels)
            n_features: Number of features to select
            method: Feature selection method
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return selected_features