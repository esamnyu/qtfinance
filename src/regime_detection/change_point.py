"""Change point detection for regime identification."""

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import ruptures as rpt

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detect change points in time series for regime identification."""
    
    def __init__(
        self,
        model: str = 'rbf',
        min_segment_length: int = 20,
        penalty: Optional[float] = None,
        n_breakpoints: Optional[int] = None
    ):
        """Initialize change point detector.
        
        Args:
            model: Detection model ('rbf', 'l1', 'l2', 'normal', 'ar')
            min_segment_length: Minimum length between change points
            penalty: Penalty value for number of breakpoints (if None, use n_breakpoints)
            n_breakpoints: Fixed number of breakpoints to detect
        """
        self.model = model
        self.min_segment_length = min_segment_length
        self.penalty = penalty
        self.n_breakpoints = n_breakpoints
        
    def detect_changepoints(
        self,
        series: Series,
        method: str = 'pelt'
    ) -> List[int]:
        """Detect change points in a time series.
        
        Args:
            series: Time series data
            method: Detection method ('pelt', 'binseg', 'window', 'dynp')
            
        Returns:
            List of change point indices
        """
        # Convert to numpy array
        signal = series.values.reshape(-1, 1)
        
        # Select algorithm
        if method == 'pelt':
            if self.penalty is None:
                raise ValueError("PELT method requires penalty parameter")
            algo = rpt.Pelt(model=self.model, min_size=self.min_segment_length)
            changepoints = algo.fit_predict(signal, pen=self.penalty)
            
        elif method == 'binseg':
            if self.n_breakpoints is None:
                raise ValueError("Binary segmentation requires n_breakpoints parameter")
            algo = rpt.Binseg(model=self.model, min_size=self.min_segment_length)
            changepoints = algo.fit_predict(signal, n_bkps=self.n_breakpoints)
            
        elif method == 'window':
            if self.n_breakpoints is None:
                raise ValueError("Window method requires n_breakpoints parameter")
            width = max(self.min_segment_length * 2, len(signal) // 10)
            algo = rpt.Window(width=width, model=self.model)
            changepoints = algo.fit_predict(signal, n_bkps=self.n_breakpoints)
            
        elif method == 'dynp':
            if self.n_breakpoints is None:
                raise ValueError("Dynamic programming requires n_breakpoints parameter")
            algo = rpt.Dynp(model=self.model, min_size=self.min_segment_length)
            changepoints = algo.fit_predict(signal, n_bkps=self.n_breakpoints)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Remove the last element (which is always the length of the signal)
        if changepoints and changepoints[-1] == len(signal):
            changepoints = changepoints[:-1]
            
        logger.info(f"Detected {len(changepoints)} change points using {method} method")
        
        return changepoints
        
    def detect_multivariate_changepoints(
        self,
        features: DataFrame,
        method: str = 'kernel'
    ) -> List[int]:
        """Detect change points in multivariate time series.
        
        Args:
            features: DataFrame with multiple features
            method: Detection method for multivariate data
            
        Returns:
            List of change point indices
        """
        # Normalize features
        normalized = (features - features.mean()) / features.std()
        signal = normalized.values
        
        if method == 'kernel':
            # Use kernel-based detection
            algo = rpt.KernelCPD(kernel='rbf', min_size=self.min_segment_length)
            
            if self.n_breakpoints is not None:
                changepoints = algo.fit_predict(signal, n_bkps=self.n_breakpoints)
            elif self.penalty is not None:
                # For kernel methods, we need to use Pelt
                algo = rpt.Pelt(model='rbf', min_size=self.min_segment_length)
                changepoints = algo.fit_predict(signal, pen=self.penalty)
            else:
                raise ValueError("Either n_breakpoints or penalty must be specified")
                
        else:
            # Fall back to univariate detection on first principal component
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(signal).flatten()
            
            return self.detect_changepoints(pd.Series(pc1), method='binseg')
            
        # Remove the last element
        if changepoints and changepoints[-1] == len(signal):
            changepoints = changepoints[:-1]
            
        return changepoints
        
    def segment_by_changepoints(
        self,
        series: Series,
        changepoints: List[int]
    ) -> List[Dict]:
        """Segment time series by change points.
        
        Args:
            series: Original time series
            changepoints: List of change point indices
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Add start and end points
        boundaries = [0] + changepoints + [len(series)]
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            segment_data = series.iloc[start_idx:end_idx]
            
            segments.append({
                'start': series.index[start_idx],
                'end': series.index[end_idx - 1] if end_idx < len(series) else series.index[-1],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'length': end_idx - start_idx,
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'min': segment_data.min(),
                'max': segment_data.max()
            })
            
        return segments
        
    def assign_regimes_to_segments(
        self,
        segments: List[Dict],
        n_regimes: int = 3
    ) -> Series:
        """Assign regime labels to segments based on characteristics.
        
        Args:
            segments: List of segment dictionaries
            n_regimes: Number of regimes to assign
            
        Returns:
            Series with regime assignments
        """
        # Extract segment means for clustering
        segment_means = [seg['mean'] for seg in segments]
        
        # Simple quantile-based assignment
        if n_regimes == 2:
            threshold = np.median(segment_means)
            regime_labels = [0 if mean < threshold else 1 for mean in segment_means]
            
        elif n_regimes == 3:
            q33 = np.percentile(segment_means, 33)
            q67 = np.percentile(segment_means, 67)
            
            regime_labels = []
            for mean in segment_means:
                if mean < q33:
                    regime_labels.append(0)  # Bear
                elif mean < q67:
                    regime_labels.append(1)  # Neutral
                else:
                    regime_labels.append(2)  # Bull
                    
        else:
            # Use k-means for more regimes
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            features = np.array([[seg['mean'], seg['std']] for seg in segments])
            regime_labels = kmeans.fit_predict(features)
            
            # Sort labels by mean return
            label_means = {}
            for label, seg in zip(regime_labels, segments):
                if label not in label_means:
                    label_means[label] = []
                label_means[label].append(seg['mean'])
                
            mean_by_label = {label: np.mean(means) for label, means in label_means.items()}
            sorted_labels = sorted(mean_by_label.keys(), key=lambda x: mean_by_label[x])
            
            # Remap labels
            label_map = {old: new for new, old in enumerate(sorted_labels)}
            regime_labels = [label_map[label] for label in regime_labels]
            
        return regime_labels
        
    def create_regime_series(
        self,
        original_series: Series,
        segments: List[Dict],
        regime_labels: List[int]
    ) -> Series:
        """Create full regime series from segments and labels.
        
        Args:
            original_series: Original time series (for index)
            segments: List of segments
            regime_labels: Regime label for each segment
            
        Returns:
            Series with regime labels for each time point
        """
        regimes = pd.Series(index=original_series.index, dtype=int)
        
        for segment, label in zip(segments, regime_labels):
            start_idx = segment['start_idx']
            end_idx = segment['end_idx']
            regimes.iloc[start_idx:end_idx] = label
            
        return regimes
        
    def detect_online_changepoint(
        self,
        history: Series,
        new_data: Series,
        threshold: float = 0.95
    ) -> bool:
        """Detect if new data represents a change point (online detection).
        
        Args:
            history: Historical data
            new_data: New data points
            threshold: Detection threshold
            
        Returns:
            True if change point detected
        """
        # Combine history and new data
        combined = pd.concat([history, new_data])
        
        # Calculate statistics for detection
        history_mean = history.mean()
        history_std = history.std()
        
        # Check if new data is significantly different
        new_mean = new_data.mean()
        
        # Z-score test
        if history_std > 0:
            z_score = abs(new_mean - history_mean) / history_std
            
            # Convert z-score to probability
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            
            return (1 - p_value) > threshold
            
        return False
        
    def plot_changepoints(
        self,
        series: Series,
        changepoints: List[int],
        segments: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ):
        """Plot time series with detected change points.
        
        Args:
            series: Time series data
            changepoints: List of change point indices
            segments: Optional segment information
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot time series
        ax.plot(series.index, series.values, 'b-', alpha=0.7, linewidth=1)
        
        # Plot change points
        for cp in changepoints:
            if cp < len(series):
                ax.axvline(x=series.index[cp], color='red', linestyle='--', 
                         alpha=0.8, label='Change point' if cp == changepoints[0] else '')
                
        # Plot segment means if available
        if segments:
            for seg in segments:
                start_idx = seg['start_idx']
                end_idx = seg['end_idx']
                mean_val = seg['mean']
                
                segment_indices = series.index[start_idx:end_idx]
                ax.plot(segment_indices, [mean_val] * len(segment_indices),
                       'g-', linewidth=2, alpha=0.8)
                
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Change Point Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Change point plot saved to {save_path}")
            
        return fig
        
    def compare_with_hmm_regimes(
        self,
        changepoint_regimes: Series,
        hmm_regimes: Series
    ) -> Dict:
        """Compare change point detection with HMM regime detection.
        
        Args:
            changepoint_regimes: Regimes from change point detection
            hmm_regimes: Regimes from HMM
            
        Returns:
            Dictionary with comparison metrics
        """
        # Ensure same index
        common_index = changepoint_regimes.index.intersection(hmm_regimes.index)
        cp_regimes = changepoint_regimes.loc[common_index]
        hmm_reg = hmm_regimes.loc[common_index]
        
        # Calculate agreement
        agreement = (cp_regimes == hmm_reg).sum() / len(common_index) * 100
        
        # Calculate transition agreement
        cp_transitions = cp_regimes != cp_regimes.shift(1)
        hmm_transitions = hmm_reg != hmm_reg.shift(1)
        
        transition_agreement = (cp_transitions == hmm_transitions).sum() / len(common_index) * 100
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        
        # Align regime labels if needed
        unique_cp = sorted(cp_regimes.unique())
        unique_hmm = sorted(hmm_reg.unique())
        
        if len(unique_cp) == len(unique_hmm):
            cm = confusion_matrix(hmm_reg, cp_regimes)
        else:
            cm = None
            
        metrics = {
            'overall_agreement': agreement,
            'transition_agreement': transition_agreement,
            'n_changepoint_transitions': cp_transitions.sum(),
            'n_hmm_transitions': hmm_transitions.sum(),
            'confusion_matrix': cm
        }
        
        logger.info(f"Change point vs HMM agreement: {agreement:.1f}%")
        logger.info(f"Transition agreement: {transition_agreement:.1f}%")
        
        return metrics