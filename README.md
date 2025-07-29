# QTFinance: Markov-Switching Regime Trading System

A production-ready quantitative trading system that combines Hidden Markov Model (HMM) regime detection with dynamic strategy selection for robust performance across different market conditions.

## 🎯 Overview

QTFinance is an institutional-grade trading system designed for retail traders who want hedge fund-level sophistication without the complexity. It automatically detects market regimes (bull, bear, neutral) and adapts trading strategies accordingly.

### Key Features

- **3-State Regime Detection**: Uses Hidden Markov Models to classify market conditions
- **Dynamic Strategy Selection**: Automatically switches between momentum, mean-reversion, and defensive strategies
- **Risk Management**: Hard limits on drawdowns (20%) and position sizing (2% per trade)
- **Production Ready**: Event-driven architecture, comprehensive logging, and monitoring
- **Professional Reporting**: Generates PDF tear sheets and Excel reports

## 📊 Performance Targets

- **CAGR**: 12-15% (realistic for regime-switching strategies)
- **Sharpe Ratio**: 1.2-1.5 (with proper risk management)
- **Max Drawdown**: ≤ 20% (hard limit with circuit breakers)
- **Win Rate**: 55-60% (regime-filtered trades)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qtfinance.git
cd qtfinance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from qtfinance import RegimeDetector, BacktestEngine
from qtfinance.data import YFinanceLoader
from qtfinance.utils import ConfigLoader

# Load configuration
config = ConfigLoader().load_config('configs/base_config.yaml')

# Fetch data
loader = YFinanceLoader()
data = loader.fetch_data(
    symbols=['SPY', 'QQQ', 'IWM'],
    start_date='2015-01-01',
    end_date='2023-12-31'
)

# Detect regimes
detector = RegimeDetector(n_regimes=3)
detector.fit(data)
regimes = detector.predict(data)

# Run backtest
engine = BacktestEngine(config)
results = engine.run(data, regimes)

# Generate report
results.generate_tearsheet('reports/backtest_results.pdf')
```

## 📁 Project Structure

```
qtfinance/
├── src/
│   ├── regime_detection/     # HMM and regime filtering
│   ├── strategies/           # Trading strategies
│   ├── portfolio/            # Position sizing and risk management
│   ├── data/                 # Data loading and feature engineering
│   ├── backtest/             # Backtesting engine
│   └── utils/                # Configuration and helpers
├── configs/                  # YAML configuration files
├── notebooks/                # Jupyter notebooks for analysis
├── tests/                    # Unit and integration tests
├── scripts/                  # Automation scripts
└── docs/                     # Documentation
```

## 🔧 Configuration

The system is configured via YAML files in the `configs/` directory:

```yaml
regime_detection:
  n_regimes: 3              # Bull, Bear, Neutral
  min_regime_duration: 5    # Minimum days to confirm regime
  probability_threshold: 0.7 # Confidence required for regime change

strategies:
  momentum:
    lookback: 20           # Days for momentum calculation
    n_stocks: 10           # Portfolio size
    rebalance_frequency: weekly

risk:
  max_drawdown: 0.20       # 20% circuit breaker
  position_size: 0.02      # 2% per position
  regime_scaling:
    bull: 1.0              # Full size in bull markets
    neutral: 0.6           # Reduced in neutral
    bear: 0.3              # Minimal in bear markets
```

## 📈 Strategies

### 1. Momentum (Bull Regime)
- Top decile momentum stocks
- Weekly rebalancing
- Trend and volume filters

### 2. Mean Reversion (Neutral Regime)
- Z-score based entry/exit
- Pairs trading with cointegration
- Strict stop losses

### 3. Defensive (Bear Regime)
- Quality factors
- Low volatility stocks
- Optional long volatility hedge

## 🛡️ Risk Management

- **Position Limits**: Max 2% per position, 10 positions total
- **Portfolio VaR**: Limited to 6% of NAV
- **Drawdown Control**: Pause at 15%, full stop at 20%
- **Regime Scaling**: Automatic position reduction in uncertain markets
- **Transaction Costs**: Realistic modeling (10bps commission, 5bps slippage)

## 📊 Backtesting

The system includes a sophisticated backtesting engine with:

- **Walk-Forward Analysis**: Rolling train/test windows
- **Purged Cross-Validation**: Prevents data leakage
- **Bootstrap Confidence Intervals**: Statistical significance testing
- **Multiple Scenarios**: Tests across different market conditions

```python
# Run walk-forward backtest
from qtfinance.backtest import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    train_period=252,  # 1 year
    test_period=63,    # 3 months
    step_size=21       # 1 month
)

results = wfa.run(data, strategy)
print(results.summary())
```

## 📝 Reporting

Generate professional reports with:

```python
from qtfinance.reports import TearsheetGenerator

generator = TearsheetGenerator()
generator.create_report(
    results,
    output_format='pdf',  # or 'excel'
    save_path='reports/strategy_tearsheet.pdf'
)
```

Reports include:
- Performance metrics (CAGR, Sharpe, Sortino, Calmar)
- Regime analysis and transitions
- Drawdown periods
- Monthly/yearly returns heatmap
- Rolling performance windows

## 🔄 Live Trading

### Paper Trading (Recommended Start)

```python
from qtfinance.live import PaperTradingEngine

engine = PaperTradingEngine(config)
engine.start()  # Runs in paper mode with simulated execution
```

### Live Trading (Alpaca Integration)

```python
# Set environment variables
export QTF_LIVE_TRADING__API_KEY=your_api_key
export QTF_LIVE_TRADING__API_SECRET=your_secret

# Run live trading
from qtfinance.live import LiveTradingEngine

engine = LiveTradingEngine(config)
engine.start()  # Real money trading - use with caution!
```

## 📊 Monitoring

The system includes Prometheus metrics for monitoring:

- Regime transitions
- Strategy performance
- Risk metrics
- System health

Access metrics at `http://localhost:8000/metrics`

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# All tests with coverage
pytest --cov=qtfinance tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by techniques from AQR, Two Sigma, and Renaissance Technologies
- Built on the shoulders of giants: scikit-learn, statsmodels, pandas
- Special thanks to the quantitative finance community

## 📚 Further Reading

- [Hidden Markov Models in Finance](https://www.amazon.com/Hidden-Markov-Models-Finance-International/dp/0387710817)
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Quantitative Portfolio Management](https://www.amazon.com/Quantitative-Portfolio-Optimisation-Management-Backtesting/dp/1119821320)

---

**Ready to start?** Check out the [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb) for a hands-on tutorial!