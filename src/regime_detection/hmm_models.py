"""Hidden Markov Model implementations for regime detection."""

from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn import hmm
import joblib

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class HMMRegimeModel:
    """Base Hidden Markov Model for regime detection."""
    
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        """Initialize HMM model.
        
        Args:
            n_regimes: Number of hidden states/regimes
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum number of iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.regime_names = {
            0: "Bear",
            1: "Neutral", 
            2: "Bull"
        }
        
    def fit(self, features: DataFrame) -> 'HMMRegimeModel':
        """Fit HMM model to features.
        
        Args:
            features: DataFrame with features for regime detection
            
        Returns:
            Fitted model instance
        """
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Initialize and fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        logger.info(f"Fitting HMM with {self.n_regimes} regimes on {len(features)} samples")
        self.model.fit(scaled_features)
        
        # Sort states by mean returns (assuming returns is first feature)
        if 'returns' in self.feature_names or self.feature_names[0].startswith('returns'):
            self._sort_states_by_returns()
            
        logger.info("HMM model fitted successfully")
        return self
        
    def _sort_states_by_returns(self):
        """Sort states by mean returns to ensure consistent labeling."""
        returns_idx = 0  # Assume returns is first feature
        mean_returns = self.model.means_[:, returns_idx]
        
        # Get sorting indices (ascending order: bear -> neutral -> bull)
        sort_idx = np.argsort(mean_returns)
        
        # Reorder model parameters
        self.model.means_ = self.model.means_[sort_idx]
        self.model.covars_ = self.model.covars_[sort_idx]
        self.model.startprob_ = self.model.startprob_[sort_idx]
        self.model.transmat_ = self.model.transmat_[sort_idx][:, sort_idx]
        
    def predict(self, features: DataFrame) -> Series:
        """Predict regime labels.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Series with predicted regime labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        scaled_features = self.scaler.transform(features)
        states = self.model.predict(scaled_features)
        
        return pd.Series(states, index=features.index, name='regime')
        
    def predict_proba(self, features: DataFrame) -> DataFrame:
        """Predict regime probabilities.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with probability for each regime
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        scaled_features = self.scaler.transform(features)
        log_prob, posteriors = self.model.score_samples(scaled_features)
        
        # Create DataFrame with regime probabilities
        prob_df = pd.DataFrame(
            posteriors,
            index=features.index,
            columns=[f'prob_{self.regime_names[i]}' for i in range(self.n_regimes)]
        )
        
        return prob_df
        
    def get_regime_statistics(self) -> Dict:
        """Get statistics for each regime.
        
        Returns:
            Dictionary with regime statistics
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
            
        stats = {}
        
        for i in range(self.n_regimes):
            regime_name = self.regime_names[i]
            stats[regime_name] = {
                'mean': dict(zip(self.feature_names, self.model.means_[i])),
                'std': dict(zip(self.feature_names, np.sqrt(np.diag(self.model.covars_[i])))),
                'start_prob': self.model.startprob_[i],
                'persistence': self.model.transmat_[i, i]
            }
            
        return stats
        
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'n_regimes': self.n_regimes,
            'regime_names': self.regime_names
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.n_regimes = model_data['n_regimes']
        self.regime_names = model_data['regime_names']
        logger.info(f"Model loaded from {path}")


class MarkovRegimeModel:
    """Markov Regime Switching model using statsmodels."""
    
    def __init__(
        self,
        n_regimes: int = 3,
        order: int = 0,
        switching_variance: bool = True,
        switching_mean: bool = True
    ):
        """Initialize Markov Regime model.
        
        Args:
            n_regimes: Number of regimes
            order: Order of autoregression
            switching_variance: Whether variance switches between regimes
            switching_mean: Whether mean switches between regimes
        """
        self.n_regimes = n_regimes
        self.order = order
        self.switching_variance = switching_variance
        self.switching_mean = switching_mean
        self.model = None
        self.results = None
        self.regime_names = {
            0: "Bear",
            1: "Neutral",
            2: "Bull"
        }
        
    def fit(self, returns: Series, exog: Optional[DataFrame] = None) -> 'MarkovRegimeModel':
        """Fit Markov Regime Switching model.
        
        Args:
            returns: Series of returns
            exog: Optional exogenous variables
            
        Returns:
            Fitted model instance
        """
        # Prepare data
        endog = returns.values
        
        # Create model
        if self.switching_mean and not self.switching_variance:
            # Switching mean only
            self.model = MarkovRegression(
                endog=endog,
                k_regimes=self.n_regimes,
                order=self.order,
                switching_variance=False
            )
        elif self.switching_variance and not self.switching_mean:
            # Switching variance only
            self.model = MarkovRegression(
                endog=endog,
                k_regimes=self.n_regimes,
                order=self.order,
                switching_variance=True
            )
        else:
            # Both switching (default)
            self.model = MarkovRegression(
                endog=endog,
                k_regimes=self.n_regimes,
                order=self.order,
                switching_variance=True
            )
            
        logger.info(f"Fitting Markov Regime model with {self.n_regimes} regimes")
        
        # Fit model
        self.results = self.model.fit(disp=False)
        
        # Sort regimes by mean returns
        self._sort_regimes_by_mean()
        
        logger.info("Markov Regime model fitted successfully")
        return self
        
    def _sort_regimes_by_mean(self):
        """Sort regimes by mean to ensure consistent labeling."""
        # Get regime means
        params = self.results.params
        regime_means = []
        
        for i in range(self.n_regimes):
            # Extract intercept for each regime
            param_name = f'regime{i}.const'
            if param_name in params:
                regime_means.append(params[param_name])
            else:
                regime_means.append(0)
                
        # Create mapping from old to new indices
        sort_idx = np.argsort(regime_means)
        self.regime_mapping = {old: new for new, old in enumerate(sort_idx)}
        
    def predict(self, returns: Series) -> Series:
        """Predict regime labels.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series with predicted regime labels
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Get smoothed probabilities
        smoothed_probs = self.results.smoothed_marginal_probabilities
        
        # Get most likely regime
        regimes = smoothed_probs.idxmax(axis=1)
        
        # Map to sorted regime indices
        if hasattr(self, 'regime_mapping'):
            regimes = regimes.map(self.regime_mapping)
            
        return pd.Series(regimes.values, index=returns.index, name='regime')
        
    def predict_proba(self, returns: Series) -> DataFrame:
        """Predict regime probabilities.
        
        Args:
            returns: Series of returns
            
        Returns:
            DataFrame with probability for each regime
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
            
        smoothed_probs = self.results.smoothed_marginal_probabilities
        
        # Reorder columns if regimes were sorted
        if hasattr(self, 'regime_mapping'):
            new_columns = [None] * self.n_regimes
            for old, new in self.regime_mapping.items():
                new_columns[new] = smoothed_probs.columns[old]
            smoothed_probs = smoothed_probs[new_columns]
            
        # Rename columns
        smoothed_probs.columns = [f'prob_{self.regime_names[i]}' for i in range(self.n_regimes)]
        
        return smoothed_probs
        
    def get_regime_statistics(self) -> Dict:
        """Get statistics for each regime.
        
        Returns:
            Dictionary with regime statistics
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")
            
        stats = {}
        params = self.results.params
        
        for i in range(self.n_regimes):
            regime_name = self.regime_names[i]
            
            # Get parameters for this regime
            regime_params = {}
            for param_name, value in params.items():
                if f'regime{i}' in param_name:
                    regime_params[param_name] = value
                    
            # Get transition probabilities
            trans_probs = self.results.regime_transition[i]
            
            stats[regime_name] = {
                'parameters': regime_params,
                'persistence': trans_probs[i],  # Probability of staying in same regime
                'transition_probs': dict(enumerate(trans_probs))
            }
            
        return stats


class RegimeDetector:
    """High-level regime detection interface combining multiple models."""
    
    def __init__(
        self,
        primary_model: str = 'hmm',
        n_regimes: int = 3,
        validation_model: Optional[str] = 'markov',
        config: Optional[Dict] = None
    ):
        """Initialize regime detector.
        
        Args:
            primary_model: Primary model type ('hmm' or 'markov')
            n_regimes: Number of regimes to detect
            validation_model: Optional validation model
            config: Additional configuration
        """
        self.primary_model_type = primary_model
        self.n_regimes = n_regimes
        self.validation_model_type = validation_model
        self.config = config or {}
        
        # Initialize models
        if primary_model == 'hmm':
            self.primary_model = HMMRegimeModel(n_regimes=n_regimes)
        elif primary_model == 'markov':
            self.primary_model = MarkovRegimeModel(n_regimes=n_regimes)
        else:
            raise ValueError(f"Unknown model type: {primary_model}")
            
        # Initialize validation model if specified
        self.validation_model = None
        if validation_model:
            if validation_model == 'hmm':
                self.validation_model = HMMRegimeModel(n_regimes=n_regimes)
            elif validation_model == 'markov':
                self.validation_model = MarkovRegimeModel(n_regimes=n_regimes)
                
    def fit(self, data: Union[DataFrame, Series], features: Optional[DataFrame] = None):
        """Fit regime detection models.
        
        Args:
            data: Returns data or OHLCV DataFrame
            features: Optional pre-computed features
        """
        # Prepare features if not provided
        if features is None:
            if isinstance(data, Series):
                features = pd.DataFrame({'returns': data})
            else:
                # Assume OHLCV data, extract returns
                features = pd.DataFrame({'returns': np.log(data['Close'] / data['Close'].shift(1))})
                
        # Fit primary model
        if isinstance(self.primary_model, HMMRegimeModel):
            self.primary_model.fit(features)
        else:
            # MarkovRegimeModel needs returns series
            self.primary_model.fit(features['returns'])
            
        # Fit validation model if specified
        if self.validation_model:
            if isinstance(self.validation_model, HMMRegimeModel):
                self.validation_model.fit(features)
            else:
                self.validation_model.fit(features['returns'])
                
        return self
        
    def predict(self, data: Union[DataFrame, Series], features: Optional[DataFrame] = None) -> Series:
        """Predict regimes.
        
        Args:
            data: Returns data or OHLCV DataFrame
            features: Optional pre-computed features
            
        Returns:
            Series with regime predictions
        """
        # Prepare features if not provided
        if features is None:
            if isinstance(data, Series):
                features = pd.DataFrame({'returns': data})
            else:
                features = pd.DataFrame({'returns': np.log(data['Close'] / data['Close'].shift(1))})
                
        # Get primary predictions
        if isinstance(self.primary_model, HMMRegimeModel):
            predictions = self.primary_model.predict(features)
        else:
            predictions = self.primary_model.predict(features['returns'])
            
        return predictions
        
    def predict_proba(self, data: Union[DataFrame, Series], features: Optional[DataFrame] = None) -> DataFrame:
        """Predict regime probabilities.
        
        Args:
            data: Returns data or OHLCV DataFrame
            features: Optional pre-computed features
            
        Returns:
            DataFrame with regime probabilities
        """
        # Prepare features if not provided
        if features is None:
            if isinstance(data, Series):
                features = pd.DataFrame({'returns': data})
            else:
                features = pd.DataFrame({'returns': np.log(data['Close'] / data['Close'].shift(1))})
                
        # Get probability predictions
        if isinstance(self.primary_model, HMMRegimeModel):
            probs = self.primary_model.predict_proba(features)
        else:
            probs = self.primary_model.predict_proba(features['returns'])
            
        return probs
        
    def get_combined_signals(
        self,
        data: Union[DataFrame, Series],
        features: Optional[DataFrame] = None
    ) -> DataFrame:
        """Get combined signals from multiple models.
        
        Args:
            data: Market data
            features: Optional pre-computed features
            
        Returns:
            DataFrame with combined regime signals
        """
        results = pd.DataFrame(index=data.index)
        
        # Primary model predictions
        results['primary_regime'] = self.predict(data, features)
        probs = self.predict_proba(data, features)
        results = pd.concat([results, probs], axis=1)
        
        # Validation model predictions if available
        if self.validation_model:
            if features is None:
                if isinstance(data, Series):
                    features = pd.DataFrame({'returns': data})
                else:
                    features = pd.DataFrame({'returns': np.log(data['Close'] / data['Close'].shift(1))})
                    
            if isinstance(self.validation_model, HMMRegimeModel):
                val_regime = self.validation_model.predict(features)
                val_probs = self.validation_model.predict_proba(features)
            else:
                val_regime = self.validation_model.predict(features['returns'])
                val_probs = self.validation_model.predict_proba(features['returns'])
                
            results['validation_regime'] = val_regime
            val_probs.columns = [col.replace('prob_', 'val_prob_') for col in val_probs.columns]
            results = pd.concat([results, val_probs], axis=1)
            
            # Consensus regime (when both models agree)
            results['consensus_regime'] = results['primary_regime']
            disagreement = results['primary_regime'] != results['validation_regime']
            results.loc[disagreement, 'consensus_regime'] = 1  # Default to neutral when disagreement
            
        return results