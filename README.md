# Trading Bot: LSTM-Powered Multi-Asset Trading

A professional machine learning-based trading bot that downloads market data from multiple assets with different trading schedules, synchronizes them, engineers features, trains an LSTM model, and backtests a trading strategy.

## Features

### 🔄 Multi-Asset Data Synchronization
- **Handles Different Trading Schedules**: Synchronizes assets with varying trading hours
  - Crypto (BTC, ETH): 24/7 trading
  - Forex (EUR/USD, GBP/USD): Limited hours (closed weekends)
  - Commodities (Gold): Market-specific hours
- **Forward-Fill Strategy**: Uses last known price when markets are closed
- **Unified Timestamps**: All assets aligned to common datetime index for coherent feature engineering

### 🎯 Feature Engineering Pipeline
Automatically computes from OHLCV data:
- **Technical Indicators**: SMA (20, 50), EMA (12, 26), RSI, MACD, Bollinger Bands
- **Lagged Features**: Historical returns and prices (1-5 periods)
- **Volatility Measures**: Rolling volatility and squared returns

### 🤖 LSTM Model
- **Architecture**: Deep learning time series model via Darts
- **Input**: 10-timestep sequences of engineered features
- **Output**: Binary prediction (up/down next period)
- **Training**: Early stopping with validation set
- **Advantage over traditional ML**: Captures temporal dependencies and market momentum

### 📊 Backtesting Engine
- Simulates trades based on model predictions
- Calculates key metrics:
  - **Total Return %**: Profit relative to initial capital
  - **Sharpe Ratio**: Risk-adjusted returns
  - **Max Drawdown %**: Worst peak-to-trough decline
  - **Trade Count**: Number of executed trades
  - **Final Portfolio Value**: Ending wealth

## Project Structure

```
trading_bot/
├── data/
│   ├── raw/                          # Downloaded hourly OHLCV data
│   │   ├── BTC_USD.csv
│   │   ├── ETH_USD.csv
│   │   ├── EURUSD.csv
│   │   ├── GBPUSD.csv
│   │   └── GC_F.csv (Gold)
│   └── processed/                    # Feature-engineered synchronized data
│       ├── synchronized_data.csv     # All assets with aligned timestamps
│       ├── BTC_USD.csv               # Individual asset files (for model training)
│       ├── ETH_USD.csv
│       ├── EURUSD.csv
│       ├── GBPUSD.csv
│       └── GC_F.csv
├── src/
│   ├── data_downloader.py            # Phase 1a: Market data download
│   ├── preprocess.py                 # Phase 1b: Synchronization & feature engineering
│   ├── model.py                      # Phase 2: LSTM model (Darts-based)
│   ├── environment.py                # Trading environment simulation
│   ├── backtest.py                   # Phase 3: Backtesting engine
│   ├── performance.py                # Metrics calculation and visualization
│   ├── generate_test_data.py         # Generate synthetic test data for testing
│   ├── test_suite.py                 # Comprehensive test suite for all modules
│   └── main.py                       # Orchestration (runs all phases)
├── models/
│   └── model                         # Trained LSTM weights
├── results/
│   └── backtest_results.png          # Equity curve visualization
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd trading_bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `torch`: Deep learning framework (PyTorch)
   - `darts`: Time series forecasting library with LSTM support
   - `pandas`, `numpy`: Data manipulation
   - `scikit-learn`: Feature scaling and train/test splitting
   - `yfinance`: Market data download
   - `matplotlib`, `quantstats`: Visualization and analytics

## Quick Start

### Run the Full Pipeline

```bash
cd src
python main.py
```

This will:
1. **Phase 1**: Download 2 years of hourly data for 5 assets
2. **Phase 1**: Synchronize assets and engineer 20+ features
3. **Phase 2**: Train LSTM model (1-2 minutes)
4. **Phase 3**: Run backtest on full historical data
5. **Output**: Trading metrics and equity curve visualization

### Testing

Run the test suite to validate all modules:

```bash
cd src
python test_suite.py
```

Generate synthetic test data for quick validation:

```bash
python generate_test_data.py
```

This creates sample OHLCV data useful for testing without downloading real market data.

### Expected Output

```
============================================================
TRADING BOT - FULL PIPELINE
============================================================

[Phase 1] Downloading market data...
[OK] Data for BTC-USD saved...
[OK] Data for ETH-USD saved...
[OK] Data for EURUSD=X saved...
[OK] Data for GBPUSD=X saved...
[OK] Data for GC=F saved...

[Phase 1] Preprocessing data and engineering features...
Synchronizing assets...
Processing BTC_USD...
Processing ETH_USD...
Processing EURUSD...
Processing GBPUSD...
Processing GC_F...
Saved synchronized data...
Saved BTC_USD to...

[Phase 2] Training ML model...
Training LSTM model...
Train Accuracy: 0.52
Test Accuracy: 0.49

[Phase 3] Running backtest...
Total Return (%): 5.23%
Sharpe Ratio: 0.85
Max Drawdown (%): 12.4%
Final Value: $10523.00
Total Trades: 127.00

Generating performance visualization...
Visualization saved to trading_bot/results/backtest_results.png

============================================================
PIPELINE COMPLETE!
============================================================
```

## How It Works

### 1. Data Download & Synchronization

The pipeline downloads hourly OHLCV (Open, High, Low, Close, Volume) data:

```
BTC-USD   (24/7)              ETH-USD   (24/7)
├─────────────────────────    ├─────────────────────────
│         │                   │         │
├─────────────────────────    ├─────────────────────────
│                             │
├─────────────────────────────┤  ← Synchronized to common timestamps
│                             │
GBP/USD (Closed Weekends)    Gold (Limited Hours)
├────────┤                   ├────────┤
    │────────┤────────┤           │────────┤
```

**Synchronization Process**:
1. Load all 5 asset CSVs
2. Create outer join on timestamps (keeps all timestamps from all assets)
3. Forward-fill missing values (use last price when market closed)
4. Result: Single aligned dataset with no gaps

### 2. Feature Engineering

For each asset at each timestamp, compute:

| Category | Features |
|----------|----------|
| **Trend** | SMA(20), SMA(50), EMA(12), EMA(26) |
| **Momentum** | RSI(14), MACD, MACD Signal, MACD Difference |
| **Volatility** | Bollinger Bands (upper, middle, lower), Rolling Volatility |
| **History** | Returns (1-5 lags), Close Price (1-5 lags), Squared Returns |

Total: **20+ features** per timestamp

### 3. LSTM Training

The LSTM model:
- **Accepts**: 10-timestep windows of engineered features
- **Processes**: Temporal patterns and dependencies
- **Outputs**: Probability that next price will be up (0-1 scale)
- **Training**: 80% of data, 20% validation/test split
- **Optimization**: Adam optimizer, early stopping on validation loss

**Why LSTM?**
- Captures market momentum over time
- Better than Random Forest for sequential data
- Learns long-term price dependencies
- Handles multiple assets with shared patterns

### 4. Backtesting

The backtest simulates realistic trading:

```python
for each timestamp t:
    features = last 10 timesteps of engineered features
    prediction = model(features)  # Returns 0-1 probability
    
    if prediction > 0.55:      # Strong buy signal
        buy 1 share
    elif prediction < 0.45:    # Strong sell signal
        close position
    
    update portfolio value
```

**Key Assumptions**:
- $10,000 initial capital
- 0.1% transaction costs
- Instant execution at market price
- No slippage (simplified)
- No position sizing (fixed 1 share)

## Configuration

### Modify Assets

Edit `src/data_downloader.py`:
```python
TICKERS = {
    'BTC-USD': 'BTC_USD',
    'ETH-USD': 'ETH_USD',
    'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD',
    'GC=F': 'GC_F'
}
```

### Tune LSTM Hyperparameters

Edit `src/model.py`:
```python
self.model = NLinearModel(
    input_chunk_length=10,        # Sequence length (increase for longer context)
    output_chunk_length=1,         # Predict 1 step ahead
    n_epochs=100,                  # Training epochs
    batch_size=32,                 # Samples per update
)
```

### Adjust Trading Threshold

Edit `src/main.py`:
```python
backtester = Backtester(model, environment, threshold=0.55)
# Lower threshold = more trades (higher risk)
# Higher threshold = fewer trades (conservative)
```

## Performance Expectations

**Based on 2 years of hourly data (BTC_USD):**
- **Sharpe Ratio**: 0.5 - 1.5 (decent risk-adjusted returns)
- **Max Drawdown**: 10-25% (normal for trading strategies)
- **Win Rate**: 45-55% (directional predictions are hard)
- **Total Return**: Highly variable (model is exploratory)

**Note**: Past performance ≠ future results. This is for educational purposes.

## Customization

### Add a New Asset

1. Add ticker to `TICKERS` dict in `download_data.py`
2. Run `main.py` - will download and include in pipeline
3. New asset automatically synchronizes with existing ones

### Experiment with Different Models

Replace the LSTM with Darts alternatives:
```python
# In model.py
from darts.models import TCNModel, NBeatsModel, TransformerModel

self.model = TCNModel(...)    # Temporal Convolutional Network
self.model = NBeatsModel(...) # Neural Basis Expansion
self.model = TransformerModel(...) # Transformer-based
```

### Change Loss Function

For regression (predict price, not direction):
```python
# In model.py
self.model.fit(..., loss_fn=MSELoss())  # Minimize squared error
# Target: raw returns instead of binary
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'darts'"

**Solution**: Reinstall dependencies:
```bash
pip install --upgrade darts torch
```

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size or sequence length in `model.py`:
```python
batch_size=16,  # Reduce from 32
input_chunk_length=5,  # Reduce from 10
```

### Issue: Poor Backtest Performance (Negative Returns)

**Solution**:
1. Check feature engineering (run phase 1 individually)
2. Lower trading threshold (more trades)
3. Try different assets
4. Note: High-frequency trading is inherently difficult

## Files Overview

| File | Purpose |
|------|---------|
| `data_downloader.py` | Downloads hourly OHLCV data from yfinance |
| `preprocess.py` | Synchronizes assets, engineers features |
| `model.py` | Darts LSTM model wrapper with train/predict API |
| `environment.py` | Simulates trading account and position tracking |
| `backtest.py` | Historical backtesting engine |
| `performance.py` | Calculates metrics and visualizes results |
| `generate_test_data.py` | Creates synthetic OHLCV data for testing and validation |
| `test_suite.py` | Unit tests and integration tests for all modules |
| `main.py` | Orchestrates all phases end-to-end |

## Dependencies

- **torch** (2.0+): Deep learning framework
- **darts** (0.25+): Time series forecasting with LSTM
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Feature scaling
- **yfinance**: Market data download
- **matplotlib**: Plotting
- **quantstats**: Performance analytics

## Future Enhancements

- [ ] Multi-step ahead predictions (t+1, t+2, etc.)
- [ ] Position sizing based on prediction confidence
- [ ] Transaction cost optimization
- [ ] Walk-forward validation
- [ ] Ensemble methods (LSTM + traditional ML)
- [ ] Real-time trading execution
- [ ] Risk management (stop-loss, take-profit)
- [ ] Portfolio optimization across multiple assets

## License

MIT License - See LICENSE file for details

## References

- Darts Documentation: https://unit8co.github.io/darts/
- PyTorch LSTM: https://pytorch.org/docs/stable/nn.html#lstm
- Technical Analysis: https://en.wikipedia.org/wiki/Technical_analysis
- Backtesting Best Practices: https://en.wikipedia.org/wiki/Backtesting

---

**Disclaimer**: This is a learning project. Do not use for live trading without extensive validation. Past performance is not indicative of future results.
