"""QTFinance: Markov-Switching Regime Trading System

A production-ready quantitative trading system that combines Hidden Markov Model
regime detection with dynamic strategy selection for robust performance across
market conditions.
"""

__version__ = "0.1.0"
__author__ = "QTFinance Team"

from .regime_detection import RegimeDetector
from .portfolio import Portfolio, RiskManager
from .backtest import BacktestEngine

__all__ = [
    "RegimeDetector",
    "Portfolio",
    "RiskManager",
    "BacktestEngine",
]