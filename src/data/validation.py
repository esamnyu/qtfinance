"""Data validation utilities for ensuring data quality."""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate market data quality and integrity."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data validator.
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or {
            'max_missing_pct': 0.05,  # Maximum 5% missing data
            'max_zero_volume_pct': 0.10,  # Maximum 10% zero volume days
            'min_price': 0.01,  # Minimum valid price
            'max_price_jump': 0.50,  # Maximum 50% single-day price change
            'outlier_std': 10,  # Number of standard deviations for outlier detection
        }
        
    def validate_ohlcv(self, data: DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data integrity.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns.get_level_values(0)]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues
            
        # Check OHLC relationships
        ohlc_issues = self._check_ohlc_relationships(data)
        issues.extend(ohlc_issues)
        
        # Check for missing data
        missing_issues = self._check_missing_data(data)
        issues.extend(missing_issues)
        
        # Check for data anomalies
        anomaly_issues = self._check_anomalies(data)
        issues.extend(anomaly_issues)
        
        # Check volume data
        volume_issues = self._check_volume_data(data)
        issues.extend(volume_issues)
        
        is_valid = len(issues) == 0
        
        if issues:
            logger.warning(f"Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
                
        return is_valid, issues
        
    def _check_ohlc_relationships(self, data: DataFrame) -> List[str]:
        """Check OHLC price relationships."""
        issues = []
        
        # High should be >= Low
        invalid_hl = data['High'] < data['Low']
        if invalid_hl.any():
            count = invalid_hl.sum()
            issues.append(f"High < Low in {count} rows ({count/len(data)*100:.2f}%)")
            
        # High should be >= Open and Close
        invalid_high = (data['High'] < data['Open']) | (data['High'] < data['Close'])
        if invalid_high.any():
            count = invalid_high.sum()
            issues.append(f"High < Open/Close in {count} rows ({count/len(data)*100:.2f}%)")
            
        # Low should be <= Open and Close
        invalid_low = (data['Low'] > data['Open']) | (data['Low'] > data['Close'])
        if invalid_low.any():
            count = invalid_low.sum()
            issues.append(f"Low > Open/Close in {count} rows ({count/len(data)*100:.2f}%)")
            
        return issues
        
    def _check_missing_data(self, data: DataFrame) -> List[str]:
        """Check for missing data."""
        issues = []
        
        missing_pct = data.isnull().sum() / len(data)
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in missing_pct.index:
                pct = missing_pct[col]
                if pct > self.config['max_missing_pct']:
                    issues.append(f"{col} has {pct*100:.2f}% missing data (threshold: {self.config['max_missing_pct']*100:.2f}%)")
                    
        return issues
        
    def _check_anomalies(self, data: DataFrame) -> List[str]:
        """Check for price anomalies and outliers."""
        issues = []
        
        # Check for extreme price jumps
        close_returns = data['Close'].pct_change()
        extreme_jumps = close_returns.abs() > self.config['max_price_jump']
        if extreme_jumps.any():
            count = extreme_jumps.sum()
            max_jump = close_returns.abs().max()
            issues.append(f"Found {count} extreme price jumps (max: {max_jump*100:.2f}%)")
            
        # Check for statistical outliers using z-score
        z_scores = np.abs(stats.zscore(close_returns.dropna()))
        outliers = z_scores > self.config['outlier_std']
        if outliers.any():
            count = outliers.sum()
            issues.append(f"Found {count} statistical outliers (>{self.config['outlier_std']} std devs)")
            
        # Check for prices below minimum threshold
        low_prices = data['Close'] < self.config['min_price']
        if low_prices.any():
            count = low_prices.sum()
            issues.append(f"Found {count} prices below ${self.config['min_price']}")
            
        return issues
        
    def _check_volume_data(self, data: DataFrame) -> List[str]:
        """Check volume data quality."""
        issues = []
        
        # Check for zero volume days
        zero_volume = data['Volume'] == 0
        zero_volume_pct = zero_volume.sum() / len(data)
        
        if zero_volume_pct > self.config['max_zero_volume_pct']:
            issues.append(f"Zero volume in {zero_volume_pct*100:.2f}% of days (threshold: {self.config['max_zero_volume_pct']*100:.2f}%)")
            
        # Check for negative volume
        negative_volume = data['Volume'] < 0
        if negative_volume.any():
            count = negative_volume.sum()
            issues.append(f"Found {count} negative volume values")
            
        return issues
        
    def clean_data(self, data: DataFrame, method: str = 'forward_fill') -> DataFrame:
        """Clean data by handling missing values and anomalies.
        
        Args:
            data: DataFrame with market data
            method: Method for handling missing data ('forward_fill', 'interpolate', 'drop')
            
        Returns:
            Cleaned DataFrame
        """
        cleaned = data.copy()
        
        # Handle missing data
        if method == 'forward_fill':
            cleaned = cleaned.fillna(method='ffill')
        elif method == 'interpolate':
            cleaned = cleaned.interpolate(method='linear')
        elif method == 'drop':
            cleaned = cleaned.dropna()
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
            
        # Fix OHLC relationships
        cleaned['High'] = cleaned[['High', 'Open', 'Close']].max(axis=1)
        cleaned['Low'] = cleaned[['Low', 'Open', 'Close']].min(axis=1)
        
        # Handle zero/negative volume
        cleaned.loc[cleaned['Volume'] <= 0, 'Volume'] = cleaned['Volume'].median()
        
        logger.info(f"Cleaned data from {len(data)} to {len(cleaned)} rows")
        
        return cleaned
        
    def validate_features(self, features: DataFrame) -> Tuple[bool, List[str]]:
        """Validate engineered features.
        
        Args:
            features: DataFrame with engineered features
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for infinite values
        inf_counts = np.isinf(features).sum()
        inf_features = inf_counts[inf_counts > 0]
        if len(inf_features) > 0:
            for feature, count in inf_features.items():
                issues.append(f"Feature '{feature}' has {count} infinite values")
                
        # Check for constant features (no variance)
        constant_features = features.columns[features.std() == 0].tolist()
        if constant_features:
            issues.append(f"Constant features with no variance: {constant_features}")
            
        # Check feature distributions
        for col in features.columns:
            # Check if feature is mostly NaN
            nan_pct = features[col].isnull().sum() / len(features)
            if nan_pct > 0.5:
                issues.append(f"Feature '{col}' is {nan_pct*100:.2f}% NaN")
                
        is_valid = len(issues) == 0
        
        return is_valid, issues
        
    def validate_regime_transitions(self, regimes: Series) -> Tuple[bool, List[str]]:
        """Validate regime labels and transitions.
        
        Args:
            regimes: Series with regime labels
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for valid regime labels
        unique_regimes = regimes.unique()
        if len(unique_regimes) < 2:
            issues.append(f"Only {len(unique_regimes)} regime(s) detected")
            
        # Check regime durations
        regime_changes = regimes != regimes.shift()
        regime_durations = []
        current_duration = 0
        
        for change in regime_changes:
            if change:
                if current_duration > 0:
                    regime_durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
                
        if current_duration > 0:
            regime_durations.append(current_duration)
            
        if regime_durations:
            avg_duration = np.mean(regime_durations)
            min_duration = np.min(regime_durations)
            
            if avg_duration < 5:
                issues.append(f"Average regime duration too short: {avg_duration:.1f} days")
                
            if min_duration < 2:
                issues.append(f"Found regimes lasting only {min_duration} day(s)")
                
        # Check for regime imbalance
        regime_counts = regimes.value_counts()
        regime_pcts = regime_counts / len(regimes)
        
        for regime, pct in regime_pcts.items():
            if pct < 0.05:
                issues.append(f"Regime {regime} appears in only {pct*100:.2f}% of data")
                
        is_valid = len(issues) == 0
        
        return is_valid, issues