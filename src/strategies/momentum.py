"""Momentum trading strategy for bull market regimes."""

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """Momentum strategy that performs well in trending markets."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize momentum strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Default parameters
        self.lookback = config.get('lookback', 20)
        self.holding_period = config.get('holding_period', 5)
        self.n_stocks = config.get('n_stocks', 10)
        self.rebalance_frequency = config.get('rebalance_frequency', 'weekly')
        self.min_volume = config.get('min_volume', 1000000)
        
    def generate_signals(
        self,
        data: DataFrame,
        regime: Optional[Series] = None
    ) -> DataFrame:
        """Generate momentum signals.
        
        Args:
            data: Market data with OHLCV
            regime: Optional regime labels
            
        Returns:
            DataFrame with momentum signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Handle single stock or multiple stocks
        if 'Close' in data.columns and not isinstance(data['Close'], DataFrame):
            # Single stock
            signals['momentum_signal'] = self._calculate_single_stock_momentum(data)
            signals['momentum_position'] = signals['momentum_signal']
            
        else:
            # Multiple stocks
            close_prices = data['Close'] if 'Close' in data.columns else data
            volume = data['Volume'] if 'Volume' in data.columns else None
            
            # Calculate momentum scores
            momentum_scores = self._calculate_momentum_scores(close_prices)
            
            # Generate signals for top momentum stocks
            signals = self._generate_portfolio_signals(
                momentum_scores,
                close_prices,
                volume
            )
            
        # Apply regime filter if provided
        if regime is not None:
            # Momentum works best in bull regime (2)
            signals = self.apply_regime_filter(signals, regime, allowed_regimes=[2])
            
        self.signals = signals
        return signals
        
    def _calculate_single_stock_momentum(self, data: DataFrame) -> Series:
        """Calculate momentum signal for single stock."""
        close = data['Close']
        
        # Simple momentum: current price vs lookback price
        momentum = close / close.shift(self.lookback) - 1
        
        # Generate signal: 1 if positive momentum, 0 otherwise
        signal = (momentum > 0).astype(int)
        
        # Add moving average filter
        ma_short = close.rolling(window=50).mean()
        ma_long = close.rolling(window=200).mean()
        trend_filter = ma_short > ma_long
        
        # Combine momentum and trend
        signal = signal & trend_filter
        
        return signal.astype(float)
        
    def _calculate_momentum_scores(self, prices: DataFrame) -> DataFrame:
        """Calculate momentum scores for multiple stocks."""
        # Calculate returns over lookback period
        returns = prices / prices.shift(self.lookback) - 1
        
        # Calculate additional momentum metrics
        scores = pd.DataFrame(index=prices.index)
        
        for col in prices.columns:
            # Basic momentum
            basic_momentum = returns[col]
            
            # Volatility-adjusted momentum
            volatility = prices[col].pct_change().rolling(window=self.lookback).std()
            vol_adj_momentum = basic_momentum / volatility
            
            # Price vs moving average
            ma = prices[col].rolling(window=self.lookback).mean()
            ma_ratio = prices[col] / ma - 1
            
            # Combine metrics
            scores[col] = (
                0.5 * basic_momentum.rank(pct=True) +
                0.3 * vol_adj_momentum.rank(pct=True) +
                0.2 * ma_ratio.rank(pct=True)
            )
            
        return scores
        
    def _generate_portfolio_signals(
        self,
        momentum_scores: DataFrame,
        prices: DataFrame,
        volume: Optional[DataFrame] = None
    ) -> DataFrame:
        """Generate portfolio signals based on momentum scores."""
        signals = pd.DataFrame(index=momentum_scores.index)
        
        # Initialize position columns
        for col in momentum_scores.columns:
            signals[f'{col}_signal'] = 0.0
            signals[f'{col}_position'] = 0.0
            
        # Rebalancing logic
        rebalance_dates = self._get_rebalance_dates(momentum_scores.index)
        current_positions = {}
        
        for date in momentum_scores.index:
            # Check if we should rebalance
            if date in rebalance_dates or not current_positions:
                # Get valid stocks (with volume filter if available)
                valid_stocks = self._filter_stocks(momentum_scores.loc[date], volume, date)
                
                if len(valid_stocks) > 0:
                    # Select top N stocks by momentum
                    top_stocks = valid_stocks.nlargest(min(self.n_stocks, len(valid_stocks)))
                    
                    # Clear current positions
                    current_positions = {}
                    
                    # Equal weight allocation
                    position_size = 1.0 / len(top_stocks)
                    
                    for stock in top_stocks.index:
                        current_positions[stock] = position_size
                        signals.loc[date, f'{stock}_signal'] = 1.0
                        
            # Set positions based on current holdings
            for stock, position in current_positions.items():
                signals.loc[date, f'{stock}_position'] = position
                
        return signals
        
    def _filter_stocks(
        self,
        scores: Series,
        volume: Optional[DataFrame],
        date: pd.Timestamp
    ) -> Series:
        """Filter stocks based on criteria."""
        valid = scores.dropna()
        
        # Volume filter
        if volume is not None and self.min_volume > 0:
            # Get average volume over last 20 days
            avg_volume = volume.rolling(window=20).mean()
            
            if date in avg_volume.index:
                volume_filter = avg_volume.loc[date] >= self.min_volume
                valid = valid[volume_filter]
                
        # Remove stocks with extreme movements (potential data errors)
        if len(valid) > 0:
            z_scores = np.abs((valid - valid.mean()) / valid.std())
            valid = valid[z_scores < 3]
            
        return valid
        
    def _get_rebalance_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_frequency == 'daily':
            return index.tolist()
            
        elif self.rebalance_frequency == 'weekly':
            # Rebalance on Mondays
            return index[index.dayofweek == 0].tolist()
            
        elif self.rebalance_frequency == 'monthly':
            # Rebalance on first trading day of month
            return index[index.is_month_start].tolist()
            
        else:
            # Default to weekly
            return index[index.dayofweek == 0].tolist()
            
    def calculate_positions(
        self,
        signals: DataFrame,
        capital: float,
        risk_params: Optional[Dict] = None
    ) -> DataFrame:
        """Calculate position sizes with risk management.
        
        Args:
            signals: Trading signals
            capital: Available capital
            risk_params: Risk management parameters
            
        Returns:
            DataFrame with position sizes in dollars
        """
        positions = pd.DataFrame(index=signals.index)
        
        # Get position columns
        position_cols = [col for col in signals.columns if col.endswith('_position')]
        
        for col in position_cols:
            # Base position from signal
            base_position = signals[col] * capital
            
            # Apply risk scaling if provided
            if risk_params:
                # Position size limit
                max_position = capital * risk_params.get('position_size', 0.02)
                base_position = base_position.clip(upper=max_position)
                
                # Regime scaling
                if 'regime_scaling' in risk_params and 'regime' in signals.columns:
                    regime = signals['regime']
                    scaling = risk_params['regime_scaling']
                    
                    # Apply scaling based on regime
                    for regime_val, scale in scaling.items():
                        mask = regime == regime_val
                        base_position[mask] *= scale
                        
            positions[col.replace('_position', '_size')] = base_position
            
        return positions
        
    def backtest_momentum_factor(
        self,
        data: DataFrame,
        lookback_periods: List[int] = None
    ) -> Dict[int, Dict]:
        """Backtest momentum strategy with different lookback periods.
        
        Args:
            data: Market data
            lookback_periods: List of lookback periods to test
            
        Returns:
            Dictionary with results for each lookback period
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60, 120, 252]
            
        results = {}
        
        for period in lookback_periods:
            # Update lookback
            original_lookback = self.lookback
            self.lookback = period
            
            # Generate signals
            signals = self.generate_signals(data)
            
            # Calculate returns
            if 'Close' in data.columns:
                returns = data['Close'].pct_change()
            else:
                returns = data.pct_change()
                
            # Get first position column
            pos_col = [c for c in signals.columns if c.endswith('_position')][0]
            
            # Calculate performance
            metrics = self.calculate_performance_metrics(
                returns,
                signals[pos_col]
            )
            
            results[period] = metrics
            
            # Restore original lookback
            self.lookback = original_lookback
            
        return results
        
    def calculate_momentum_decay(
        self,
        data: DataFrame,
        holding_periods: List[int] = None
    ) -> DataFrame:
        """Calculate momentum decay over different holding periods.
        
        Args:
            data: Market data
            holding_periods: List of holding periods to analyze
            
        Returns:
            DataFrame with average returns by holding period
        """
        if holding_periods is None:
            holding_periods = list(range(1, 61))  # 1 to 60 days
            
        close_prices = data['Close'] if 'Close' in data.columns else data
        
        # Calculate momentum scores
        momentum_scores = self._calculate_momentum_scores(close_prices)
        
        # Track returns by holding period
        decay_results = []
        
        for period in holding_periods:
            period_returns = []
            
            # For each rebalance date
            for i in range(self.lookback, len(momentum_scores) - period):
                # Get top momentum stocks
                scores = momentum_scores.iloc[i]
                valid_scores = scores.dropna()
                
                if len(valid_scores) >= self.n_stocks:
                    top_stocks = valid_scores.nlargest(self.n_stocks).index
                    
                    # Calculate forward returns
                    entry_prices = close_prices.iloc[i][top_stocks]
                    exit_prices = close_prices.iloc[i + period][top_stocks]
                    
                    returns = (exit_prices / entry_prices - 1).mean()
                    period_returns.append(returns)
                    
            if period_returns:
                avg_return = np.mean(period_returns)
                decay_results.append({
                    'holding_period': period,
                    'avg_return': avg_return,
                    'annualized_return': avg_return * 252 / period,
                    'n_trades': len(period_returns)
                })
                
        return pd.DataFrame(decay_results)