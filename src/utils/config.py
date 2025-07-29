"""Configuration management utilities."""

import os
from typing import Any, Dict, Optional
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files and environment variables."""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config = {}
        
        # Load environment variables
        load_dotenv()
        
    def load_config(self, config_name: str = "base_config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_name: Name of configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Override with environment variables
            config = self._override_with_env(config)
            
            self.config = config
            logger.info(f"Loaded configuration from {config_path}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration values with environment variables.
        
        Environment variables should be prefixed with 'QTF_' and use
        double underscores for nested values.
        
        Example:
            QTF_DATA__YAHOO_API_KEY -> config['data']['yahoo_api_key']
        """
        def update_nested_dict(d: dict, keys: list, value: Any):
            """Update nested dictionary with list of keys."""
            for key in keys[:-1]:
                d = d.setdefault(key.lower(), {})
            d[keys[-1].lower()] = value
            
        for env_key, env_value in os.environ.items():
            if env_key.startswith('QTF_'):
                # Remove prefix and split by double underscore
                config_keys = env_key[4:].split('__')
                
                # Convert value to appropriate type
                if env_value.lower() in ('true', 'false'):
                    value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    value = int(env_value)
                elif '.' in env_value and env_value.replace('.', '').isdigit():
                    value = float(env_value)
                else:
                    value = env_value
                    
                update_nested_dict(config, config_keys, value)
                logger.debug(f"Override from env: {env_key} = {value}")
                
        return config
        
    def save_config(self, config: Dict[str, Any], config_name: str):
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of configuration file
        """
        config_path = self.config_dir / config_name
        
        # Create directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"Saved configuration to {config_path}")
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'regime_detection.n_regimes')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def update(self, key: str, value: Any):
        """Update configuration value.
        
        Args:
            key: Configuration key (dot-notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
            
        config[keys[-1]] = value
        
    def merge_configs(self, *config_files: str) -> Dict[str, Any]:
        """Merge multiple configuration files.
        
        Later files override earlier ones.
        
        Args:
            *config_files: Configuration file names
            
        Returns:
            Merged configuration
        """
        merged = {}
        
        for config_file in config_files:
            config = self.load_config(config_file)
            merged = self._deep_merge(merged, config)
            
        self.config = merged
        return merged
        
    def _deep_merge(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration against schema.
        
        Args:
            schema: Optional schema dictionary
            
        Returns:
            True if valid
        """
        # Basic validation rules
        required_keys = [
            'regime_detection.n_regimes',
            'data.start_date',
            'strategies.momentum.lookback',
            'risk.max_drawdown',
            'risk.position_size'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
                
        # Validate value ranges
        validations = [
            ('regime_detection.n_regimes', lambda x: 2 <= x <= 5),
            ('risk.max_drawdown', lambda x: 0 < x <= 1),
            ('risk.position_size', lambda x: 0 < x <= 1),
            ('regime_detection.min_regime_duration', lambda x: x >= 1)
        ]
        
        for key, validator in validations:
            value = self.get(key)
            if value is not None and not validator(value):
                logger.error(f"Invalid configuration value for {key}: {value}")
                return False
                
        return True


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        'regime_detection': {
            'n_regimes': 3,
            'model_type': 'hmm',
            'covariance_type': 'full',
            'n_iter': 100,
            'min_regime_duration': 5,
            'probability_threshold': 0.7,
            'smoothing_window': 3,
            'features': {
                'groups': ['price', 'statistical', 'regime', 'technical'],
                'lookback_periods': {
                    'returns': 1,
                    'volatility_short': 20,
                    'volatility_long': 60,
                    'volume': 20,
                    'correlation': 60
                }
            }
        },
        'data': {
            'provider': 'yfinance',
            'start_date': '2010-01-01',
            'end_date': None,  # None means today
            'symbols': ['SPY'],  # Can be extended to S&P 500 constituents
            'interval': '1d',
            'cache_dir': 'data/cache'
        },
        'strategies': {
            'momentum': {
                'lookback': 20,
                'holding_period': 5,
                'n_stocks': 10,
                'rebalance_frequency': 'weekly'
            },
            'mean_reversion': {
                'lookback': 20,
                'entry_threshold': -2.0,  # Z-score
                'exit_threshold': 0.0,
                'stop_loss': -3.0
            },
            'pairs_trading': {
                'lookback': 60,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'max_half_life': 20
            }
        },
        'risk': {
            'max_drawdown': 0.20,
            'position_size': 0.02,  # 2% per position
            'max_positions': 10,
            'portfolio_heat': 0.06,  # 6% portfolio VaR
            'use_stops': True,
            'stop_loss': 0.02,  # 2% stop loss
            'regime_scaling': {
                'bull': 1.0,
                'neutral': 0.6,
                'bear': 0.3
            }
        },
        'backtest': {
            'initial_capital': 100000,
            'commission': 0.001,  # 10 bps
            'slippage': 0.0005,  # 5 bps
            'annual_risk_free_rate': 0.02
        },
        'reporting': {
            'output_dir': 'reports',
            'save_plots': True,
            'plot_dpi': 300,
            'generate_pdf': True,
            'generate_excel': True
        },
        'live_trading': {
            'enabled': False,
            'paper_trading': True,
            'broker': 'alpaca',
            'api_key': None,
            'api_secret': None,
            'base_url': 'https://paper-api.alpaca.markets'
        },
        'monitoring': {
            'metrics_port': 8000,
            'log_level': 'INFO',
            'alert_email': None,
            'alert_thresholds': {
                'drawdown': 0.15,
                'daily_loss': 0.05,
                'regime_changes': 3  # Alert if more than 3 changes per day
            }
        }
    }