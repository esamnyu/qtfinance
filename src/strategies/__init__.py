"""Trading strategies for different market regimes."""

from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .pairs_trading import PairsTradingStrategy
from .vol_arbitrage import VolatilityArbitrageStrategy
from .base import BaseStrategy

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "VolatilityArbitrageStrategy",
]