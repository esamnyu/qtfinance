"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize base strategy.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config or {}
        self.positions = {}
        self.signals = pd.DataFrame()
        self.performance = {}
        
    @abstractmethod
    def generate_signals(
        self,
        data: DataFrame,
        regime: Optional[Series] = None
    ) -> DataFrame:
        """Generate trading signals.
        
        Args:
            data: Market data (OHLCV)
            regime: Optional regime labels
            
        Returns:
            DataFrame with signals
        """
        pass
        
    @abstractmethod
    def calculate_positions(
        self,
        signals: DataFrame,
        capital: float,
        risk_params: Optional[Dict] = None
    ) -> DataFrame:
        """Calculate position sizes based on signals.
        
        Args:
            signals: Trading signals
            capital: Available capital
            risk_params: Risk management parameters
            
        Returns:
            DataFrame with position sizes
        """
        pass
        
    def apply_regime_filter(
        self,
        signals: DataFrame,
        regime: Series,
        allowed_regimes: List[int]
    ) -> DataFrame:
        """Filter signals based on market regime.
        
        Args:
            signals: Original signals
            regime: Market regime labels
            allowed_regimes: List of regimes where strategy is active
            
        Returns:
            Filtered signals
        """
        filtered = signals.copy()
        
        # Align indices
        common_index = signals.index.intersection(regime.index)
        
        # Set signals to 0 when not in allowed regimes
        mask = ~regime.loc[common_index].isin(allowed_regimes)
        
        for col in filtered.columns:
            if col.endswith('_signal') or col.endswith('_position'):
                filtered.loc[common_index[mask], col] = 0
                
        return filtered
        
    def calculate_performance_metrics(
        self,
        returns: Series,
        positions: Series,
        benchmark: Optional[Series] = None
    ) -> Dict:
        """Calculate strategy performance metrics.
        
        Args:
            returns: Asset returns
            positions: Position sizes
            benchmark: Optional benchmark returns
            
        Returns:
            Dictionary with performance metrics
        """
        # Calculate strategy returns
        strategy_returns = returns * positions.shift(1)
        
        # Basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
        sortino_ratio = self._calculate_sortino_ratio(strategy_returns)
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        
        # Trade statistics
        trades = self._identify_trades(positions)
        win_rate = self._calculate_win_rate(trades, returns)
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len(trades),
            'avg_trade_return': np.mean([t['return'] for t in trades]) if trades else 0
        }
        
        # Benchmark comparison if provided
        if benchmark is not None:
            benchmark_return = (1 + benchmark).prod() - 1
            metrics['excess_return'] = total_return - benchmark_return
            metrics['information_ratio'] = self._calculate_information_ratio(
                strategy_returns, benchmark
            )
            
        return metrics
        
    def _calculate_sharpe_ratio(
        self,
        returns: Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        
        if returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_sortino_ratio(
        self,
        returns: Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
    def _calculate_max_drawdown(self, returns: Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
        
    def _calculate_information_ratio(
        self,
        returns: Series,
        benchmark: Series
    ) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark
        
        if active_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * active_returns.mean() / active_returns.std()
        
    def _identify_trades(self, positions: Series) -> List[Dict]:
        """Identify individual trades from position series."""
        trades = []
        
        # Find position changes
        position_diff = positions.diff()
        
        # Entry points (position increases)
        entries = position_diff[position_diff > 0]
        
        for entry_date, entry_size in entries.items():
            # Find corresponding exit
            future_positions = positions[positions.index > entry_date]
            
            # Look for position decrease
            future_diff = future_positions.diff()
            exits = future_diff[future_diff < 0]
            
            if len(exits) > 0:
                exit_date = exits.index[0]
                exit_size = abs(exits.iloc[0])
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_size': entry_size,
                    'exit_size': exit_size,
                    'duration': (exit_date - entry_date).days
                })
                
        return trades
        
    def _calculate_win_rate(self, trades: List[Dict], returns: Series) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
            
        winning_trades = 0
        
        for trade in trades:
            # Calculate trade return
            trade_returns = returns[trade['entry_date']:trade['exit_date']]
            trade_return = (1 + trade_returns).prod() - 1
            
            trade['return'] = trade_return
            
            if trade_return > 0:
                winning_trades += 1
                
        return winning_trades / len(trades) if trades else 0
        
    def validate_signals(self, signals: DataFrame) -> bool:
        """Validate signal integrity.
        
        Args:
            signals: DataFrame with trading signals
            
        Returns:
            True if signals are valid
        """
        # Check for required columns
        required_cols = ['signal', 'position']
        
        for col in required_cols:
            if not any(c.endswith(f'_{col}') for c in signals.columns):
                logger.error(f"Missing required column ending with '_{col}'")
                return False
                
        # Check for NaN values
        if signals.isnull().any().any():
            logger.warning("Signals contain NaN values")
            
        # Check signal bounds
        signal_cols = [c for c in signals.columns if c.endswith('_signal')]
        for col in signal_cols:
            if (signals[col].abs() > 1).any():
                logger.warning(f"Signal values in {col} exceed [-1, 1] bounds")
                
        return True
        
    def plot_signals(
        self,
        prices: Series,
        signals: DataFrame,
        positions: Optional[Series] = None,
        save_path: Optional[str] = None
    ):
        """Plot strategy signals and positions.
        
        Args:
            prices: Price series
            signals: Signal DataFrame
            positions: Optional position series
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Price plot
        ax1 = axes[0]
        ax1.plot(prices.index, prices.values, 'b-', label='Price')
        
        # Mark buy/sell signals
        signal_col = [c for c in signals.columns if c.endswith('_signal')][0]
        buy_signals = signals[signals[signal_col] > 0]
        sell_signals = signals[signals[signal_col] < 0]
        
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals.index, prices.loc[buy_signals.index],
                       color='green', marker='^', s=100, label='Buy')
                       
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals.index, prices.loc[sell_signals.index],
                       color='red', marker='v', s=100, label='Sell')
                       
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Signal strength plot
        ax2 = axes[1]
        ax2.plot(signals.index, signals[signal_col], 'g-', label='Signal')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Signal Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Position plot
        ax3 = axes[2]
        if positions is not None:
            ax3.plot(positions.index, positions.values, 'b-', label='Position')
        else:
            position_col = [c for c in signals.columns if c.endswith('_position')][0]
            ax3.plot(signals.index, signals[position_col], 'b-', label='Position')
            
        ax3.set_ylabel('Position Size')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Signal plot saved to {save_path}")
            
        return fig