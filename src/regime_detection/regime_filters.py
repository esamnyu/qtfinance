"""Regime filtering and smoothing utilities."""

from typing import Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


class RegimeFilter:
    """Filter and smooth regime predictions to reduce whipsaws."""
    
    def __init__(
        self,
        min_regime_duration: int = 5,
        probability_threshold: float = 0.7,
        smoothing_window: int = 3
    ):
        """Initialize regime filter.
        
        Args:
            min_regime_duration: Minimum days a regime must persist
            probability_threshold: Minimum probability to confirm regime change
            smoothing_window: Window for probability smoothing
        """
        self.min_regime_duration = min_regime_duration
        self.probability_threshold = probability_threshold
        self.smoothing_window = smoothing_window
        
    def filter_regimes(
        self,
        regimes: Series,
        probabilities: Optional[DataFrame] = None
    ) -> Series:
        """Filter regime predictions to reduce noise.
        
        Args:
            regimes: Raw regime predictions
            probabilities: Optional regime probabilities
            
        Returns:
            Filtered regime predictions
        """
        filtered = regimes.copy()
        
        # Apply minimum duration filter
        filtered = self._apply_duration_filter(filtered)
        
        # Apply probability threshold if available
        if probabilities is not None:
            filtered = self._apply_probability_filter(filtered, probabilities)
            
        return filtered
        
    def _apply_duration_filter(self, regimes: Series) -> Series:
        """Remove regime changes that don't meet minimum duration."""
        filtered = regimes.copy()
        
        # Find regime change points
        changes = regimes != regimes.shift(1)
        change_indices = regimes.index[changes].tolist()
        
        if len(change_indices) <= 1:
            return filtered
            
        # Check duration of each regime
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            
            # Calculate duration in trading days
            duration = len(regimes.loc[start_idx:end_idx]) - 1
            
            if duration < self.min_regime_duration:
                # Revert to previous regime
                if i > 0:
                    prev_regime = regimes.iloc[regimes.index.get_loc(change_indices[i-1])]
                    filtered.loc[start_idx:end_idx] = prev_regime
                    
        return filtered
        
    def _apply_probability_filter(
        self,
        regimes: Series,
        probabilities: DataFrame
    ) -> Series:
        """Filter regimes based on probability threshold."""
        filtered = regimes.copy()
        
        for i in range(len(regimes)):
            current_regime = regimes.iloc[i]
            
            # Get probability of current regime
            prob_col = f'prob_{current_regime}' if isinstance(current_regime, str) else None
            if prob_col and prob_col in probabilities.columns:
                prob = probabilities.iloc[i][prob_col]
            else:
                # Try numeric column access
                max_prob = probabilities.iloc[i].max()
                prob = max_prob
                
            # If probability below threshold, revert to previous regime
            if prob < self.probability_threshold and i > 0:
                filtered.iloc[i] = filtered.iloc[i-1]
                
        return filtered
        
    def smooth_probabilities(
        self,
        probabilities: DataFrame,
        method: str = 'ewm'
    ) -> DataFrame:
        """Smooth regime probabilities.
        
        Args:
            probabilities: Raw regime probabilities
            method: Smoothing method ('ewm', 'rolling', 'uniform')
            
        Returns:
            Smoothed probabilities
        """
        if method == 'ewm':
            # Exponential weighted moving average
            return probabilities.ewm(span=self.smoothing_window, adjust=False).mean()
            
        elif method == 'rolling':
            # Simple rolling average
            return probabilities.rolling(window=self.smoothing_window, center=True).mean().fillna(method='bfill')
            
        elif method == 'uniform':
            # Uniform filter (scipy)
            smoothed = pd.DataFrame(index=probabilities.index)
            for col in probabilities.columns:
                smoothed[col] = uniform_filter1d(
                    probabilities[col].values,
                    size=self.smoothing_window,
                    mode='nearest'
                )
            return smoothed
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


class RegimePersistence:
    """Analyze and enhance regime persistence."""
    
    def __init__(self):
        """Initialize regime persistence analyzer."""
        self.regime_stats = {}
        
    def calculate_persistence_stats(self, regimes: Series) -> Dict:
        """Calculate regime persistence statistics.
        
        Args:
            regimes: Series of regime labels
            
        Returns:
            Dictionary with persistence statistics
        """
        stats = {}
        
        # Calculate regime durations
        regime_changes = regimes != regimes.shift(1)
        regime_segments = []
        
        current_regime = None
        current_start = None
        
        for idx, (timestamp, regime) in enumerate(regimes.items()):
            if regime != current_regime:
                if current_regime is not None:
                    duration = idx - current_start
                    regime_segments.append({
                        'regime': current_regime,
                        'start': regimes.index[current_start],
                        'end': regimes.index[idx-1],
                        'duration': duration
                    })
                current_regime = regime
                current_start = idx
                
        # Add final segment
        if current_regime is not None:
            duration = len(regimes) - current_start
            regime_segments.append({
                'regime': current_regime,
                'start': regimes.index[current_start],
                'end': regimes.index[-1],
                'duration': duration
            })
            
        # Calculate statistics by regime
        regime_durations = {}
        for segment in regime_segments:
            regime = segment['regime']
            if regime not in regime_durations:
                regime_durations[regime] = []
            regime_durations[regime].append(segment['duration'])
            
        # Compute stats
        for regime, durations in regime_durations.items():
            stats[regime] = {
                'mean_duration': np.mean(durations),
                'median_duration': np.median(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'count': len(durations),
                'total_days': sum(durations),
                'percentage': sum(durations) / len(regimes) * 100
            }
            
        # Overall statistics
        all_durations = [seg['duration'] for seg in regime_segments]
        stats['overall'] = {
            'total_changes': len(regime_segments) - 1,
            'avg_duration': np.mean(all_durations),
            'median_duration': np.median(all_durations),
            'min_duration': np.min(all_durations),
            'max_duration': np.max(all_durations)
        }
        
        self.regime_stats = stats
        return stats
        
    def enhance_persistence(
        self,
        regimes: Series,
        probabilities: DataFrame,
        method: str = 'threshold_decay'
    ) -> Tuple[Series, DataFrame]:
        """Enhance regime persistence using various methods.
        
        Args:
            regimes: Original regime predictions
            probabilities: Regime probabilities
            method: Enhancement method
            
        Returns:
            Tuple of (enhanced regimes, adjusted probabilities)
        """
        if method == 'threshold_decay':
            return self._threshold_decay_method(regimes, probabilities)
        elif method == 'markov_smooth':
            return self._markov_smoothing(regimes, probabilities)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
            
    def _threshold_decay_method(
        self,
        regimes: Series,
        probabilities: DataFrame
    ) -> Tuple[Series, DataFrame]:
        """Use decaying threshold to enhance persistence."""
        enhanced_regimes = regimes.copy()
        adjusted_probs = probabilities.copy()
        
        # Initial threshold
        base_threshold = 0.7
        decay_rate = 0.95
        
        current_regime = regimes.iloc[0]
        regime_duration = 0
        
        for i in range(1, len(regimes)):
            proposed_regime = regimes.iloc[i]
            
            if proposed_regime == current_regime:
                regime_duration += 1
            else:
                # Calculate dynamic threshold
                dynamic_threshold = base_threshold * (decay_rate ** regime_duration)
                
                # Get probability of proposed regime
                prob_cols = [col for col in probabilities.columns if 'prob_' in col]
                regime_probs = probabilities.iloc[i][prob_cols].values
                max_prob_idx = np.argmax(regime_probs)
                max_prob = regime_probs[max_prob_idx]
                
                # Only change regime if probability exceeds dynamic threshold
                if max_prob > dynamic_threshold:
                    current_regime = proposed_regime
                    regime_duration = 0
                else:
                    enhanced_regimes.iloc[i] = current_regime
                    regime_duration += 1
                    
        return enhanced_regimes, adjusted_probs
        
    def _markov_smoothing(
        self,
        regimes: Series,
        probabilities: DataFrame
    ) -> Tuple[Series, DataFrame]:
        """Apply Markov chain smoothing to enhance persistence."""
        # Estimate transition matrix from historical data
        transition_matrix = self._estimate_transition_matrix(regimes)
        
        enhanced_regimes = regimes.copy()
        adjusted_probs = probabilities.copy()
        
        # Apply Viterbi-like algorithm for smoothing
        n_regimes = len(regimes.unique())
        n_obs = len(regimes)
        
        # Forward pass
        forward_probs = np.zeros((n_obs, n_regimes))
        
        # Initialize
        prob_cols = [col for col in probabilities.columns if 'prob_' in col]
        forward_probs[0] = probabilities.iloc[0][prob_cols].values
        
        for t in range(1, n_obs):
            for j in range(n_regimes):
                # Emission probability
                emission = probabilities.iloc[t][prob_cols[j]]
                
                # Transition probabilities
                trans_probs = [forward_probs[t-1, i] * transition_matrix[i, j] 
                             for i in range(n_regimes)]
                
                forward_probs[t, j] = emission * np.sum(trans_probs)
                
            # Normalize
            forward_probs[t] = forward_probs[t] / np.sum(forward_probs[t])
            
        # Update regimes based on smoothed probabilities
        for t in range(n_obs):
            enhanced_regimes.iloc[t] = np.argmax(forward_probs[t])
            for j, col in enumerate(prob_cols):
                adjusted_probs.iloc[t, adjusted_probs.columns.get_loc(col)] = forward_probs[t, j]
                
        return enhanced_regimes, adjusted_probs
        
    def _estimate_transition_matrix(self, regimes: Series) -> np.ndarray:
        """Estimate regime transition matrix from historical data."""
        unique_regimes = sorted(regimes.unique())
        n_regimes = len(unique_regimes)
        
        # Create mapping
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        
        # Count transitions
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            from_idx = regime_to_idx[regimes.iloc[i]]
            to_idx = regime_to_idx[regimes.iloc[i + 1]]
            transitions[from_idx, to_idx] += 1
            
        # Convert to probabilities
        for i in range(n_regimes):
            row_sum = transitions[i].sum()
            if row_sum > 0:
                transitions[i] = transitions[i] / row_sum
            else:
                # If no transitions from this state, assume uniform
                transitions[i] = 1.0 / n_regimes
                
        return transitions
        
    def plot_regime_durations(self, regimes: Series, save_path: Optional[str] = None):
        """Plot regime duration distributions.
        
        Args:
            regimes: Series of regime labels
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        if not self.regime_stats:
            self.calculate_persistence_stats(regimes)
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Plot duration histogram for each regime
        regime_changes = regimes != regimes.shift(1)
        
        for idx, (regime, stats) in enumerate(self.regime_stats.items()):
            if regime == 'overall':
                continue
                
            ax = axes[idx]
            
            # Get durations for this regime
            durations = []
            current_regime = None
            start_idx = None
            
            for i, (_, r) in enumerate(regimes.items()):
                if r != current_regime:
                    if current_regime == regime and start_idx is not None:
                        durations.append(i - start_idx)
                    if r == regime:
                        start_idx = i
                    current_regime = r
                    
            # Add final duration if needed
            if current_regime == regime and start_idx is not None:
                durations.append(len(regimes) - start_idx)
                
            # Plot histogram
            ax.hist(durations, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(stats['mean_duration'], color='red', linestyle='--', 
                      label=f"Mean: {stats['mean_duration']:.1f}")
            ax.axvline(stats['median_duration'], color='green', linestyle='--',
                      label=f"Median: {stats['median_duration']:.1f}")
            
            ax.set_xlabel('Duration (days)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{regime} Regime Durations')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Overall statistics in the last subplot
        ax = axes[-1]
        stats_text = "Overall Regime Statistics\n\n"
        stats_text += f"Total regime changes: {self.regime_stats['overall']['total_changes']}\n"
        stats_text += f"Average duration: {self.regime_stats['overall']['avg_duration']:.1f} days\n"
        stats_text += f"Median duration: {self.regime_stats['overall']['median_duration']:.1f} days\n"
        stats_text += f"Min duration: {self.regime_stats['overall']['min_duration']} days\n"
        stats_text += f"Max duration: {self.regime_stats['overall']['max_duration']} days\n\n"
        
        for regime, stats in self.regime_stats.items():
            if regime != 'overall':
                stats_text += f"\n{regime} Regime:\n"
                stats_text += f"  Percentage of time: {stats['percentage']:.1f}%\n"
                stats_text += f"  Number of occurrences: {stats['count']}\n"
                
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='center')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime duration plot saved to {save_path}")
            
        return fig