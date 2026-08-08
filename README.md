# Trading Bot: HistData FX + Temporal Fusion Transformer

A professional FX trading bot powered by **HistData.com 1-minute data**, **TemporalFusionTransformer** (LSTM + Attention), and **Triple Barrier labeling** with ATR-based risk management.

## Key Features

### 📊 Advanced Data Pipeline
- **HistData.com Integration**: Downloads raw 1-minute FX data from HistData.com
- **Configurable Resampling**: Dynamically resample 1-minute candles to any timeframe (default: 1-hour)
- **Multi-Asset Pipeline**: Main asset (EURUSD) gets full feature engineering, exogenous assets (GBPUSD, USDJPY) get essential features
- **Date Range Control**: Download any period from 2008 to present via .env configuration

### 🎯 Intelligent Labeling
- **Triple Barrier Method**: Uses dynamic ATR (Average True Range) to set profit targets & stop losses
  - TP multiplier: 1.5× ATR → enforces 2:1 risk/reward ratio
  - SL multiplier: 0.75× ATR → balanced risk
  - Max hold period: 6 hours before label expires
- **3-Class Output**: Flat (0), Up (1), Down (2) for robust market regime detection

### 🤖 Neural Architecture (Dual-Brain TFT + PPO)
- **Deep Reinforcement Learning (PPO)**: The core trading agent uses Proximal Policy Optimization (PPO) from Stable-Baselines3 to make discrete actions (Flat, Long, Short).
- **Dual-Brain TemporalFusionTransformer**: Two independently trained TFTs (`tft_long.pth` and `tft_short.pth`) act as feature extractors.
  - The models process 30-timestep sequences of engineered features.
  - Their outputs are concatenated into a **256-dimensional state embedding** that provides the PPO agent with a unified view of bullish and bearish momentum.
- **Self-Supervised MTF & PyTorch ENN**:
  - Uses Multi-Timeframe (MTF) Supervised Contrastive Learning.
  - Custom **PyTorch ENN** (Edited Nearest Neighbors) to aggressively clean contradictory labels and eliminate noise from the dataset.
  - Implements **Macro F1 Early Stopping** to maintain balance and prevent overfitting.

### 📈 Risk Management & Reward Function
- **SOTA Differential Sharpe Ratio (DSR)**: The agent is trained using a continuous, dense reward function based on the Differential Sharpe Ratio. This mathematically penalizes high-variance equity curves and drawdowns step-by-step.
- **Asymmetrical R:R Trading Rules**: 
  - Long trades strictly enforce a `1:2.5` Risk-Reward ratio.
  - Short trades strictly enforce a `1:4` Risk-Reward ratio.
  - Trades must hit either the full Take Profit or full Stop Loss (no specification loopholes or trailing stops).
- **Timeout Momentum Penalty**: Trades are given a 96-hour window to resolve. If they fail to hit targets, they are closed with a fixed negative reward to enforce momentum trading.

## Project Structure

```
trading_bot/
├── data/
│   ├── raw/                          # 1-minute HistData CSV files
│   │   ├── EURUSD.csv
│   │   ├── GBPUSD.csv
│   │   └── USDJPY.csv
│   └── processed/                    # Feature-engineered data
│       ├── EURUSD_master.csv         # Main asset with triple barrier labels + all features
│       ├── train_split.csv           # 80% of data for training
│       └── test_split.csv            # 20% for out-of-sample backtest
├── src/
│   ├── data_downloader.py            # Download 1-minute FX data from HistData.com
│   ├── preprocess.py                 # Resample, engineer features, apply triple barrier labels
│   ├── model.py                      # TemporalFusionTransformer architecture & PricePredictor
│   ├── environment.py                # Simulates trading account, positions, PnL
│   ├── backtest.py                   # Out-of-sample backtesting engine with dynamic risk management
│   ├── performance.py                # Calculates metrics and generates visualizations
│   ├── generate_test_data.py         # Generates synthetic OHLCV data for quick testing
│   ├── test_suite.py                 # Unit tests for all modules
│   └── main.py                       # Orchestrates full pipeline
├── models/
│   ├── model.pkl                     # Trained model weights
│   └── model.scaler.pkl              # Feature StandardScaler for inference
├── results/
│   ├── backtest_results.png          # Equity curve + trading signals
│   ├── training_history.png          # Train/val loss curves
│   └── confusion_matrix.png          # 3-class prediction confusion matrix
├── .env                              # Configuration (tickers, dates, timeframe)
├── .env.example                      # Template for .env setup
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
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

2. **Create and configure .env file:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env`:
   ```env
   # FX pairs to download
   HISTDATA_TICKERS=EURUSD,GBPUSD,USDJPY
   
   # Date range
   HISTDATA_START_YEAR=2020
   HISTDATA_END_YEAR=2026
   
   # Preprocessing
   TARGET_TIMEFRAME=1h        # Resample to hourly (options: 1min, 5min, 15min, 1h, 4h, 1d)
   MAIN_ASSET=EURUSD         # Primary asset for full feature engineering
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key packages:
   - `histdata` — Download 1-minute FX data
   - `torch` — Deep learning framework
   - `pandas`, `numpy` — Data manipulation
   - `scikit-learn` — Feature scaling
   - `matplotlib` — Visualization
   - `python-dotenv` — Environment variables

## Quick Start

### Full Pipeline Execution

```bash
cd src
python main.py
```

This will:
1. **Load preprocessed data** from `data/processed/{MAIN_ASSET}_master.csv`
   - (Run `python data_downloader.py` then `python preprocess.py` on first setup)
2. **Train TemporalFusionTransformer** on 80% of last 10k rows
3. **Generate training visualizations** (learning curves, confusion matrix)
4. **Run out-of-sample backtest** on remaining 20% with dynamic ATR risk management
5. **Output performance report** with metrics and equity curve plot

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

[Phase 1] Preprocessing data and engineering features...
(Optional - data already processed)

[Phase 2] Splitting Data & Training ML model...
Training Multi-Class TFT (Features: 38, Target Classes: 3)...
  Epoch 001/150 | Train Loss: 1.0823 | Val Loss: 1.0547 | Val Acc: 0.3847 | LR: 0.0010
  Epoch 006/150 | Train Loss: 0.9234 | Val Loss: 0.9012 | Val Acc: 0.4521 | LR: 0.0010
  ...
Early stopping triggered at epoch 87. Restoring best weights.
Model trained on 80% split of EURUSD_master.csv
Internal Validation Score: 0.4823

Confusion Matrix saved to trading_bot/results/confusion_matrix.png

[Phase 3] Running out-of-sample backtest...
Starting Multi-Class Dynamic ATR Backtest on trading_bot/data/processed/test_split.csv...

==================================================
BACKTEST PERFORMANCE REPORT
==================================================
Total Return (%): 3.47%
Sharpe Ratio: 0.92
Max Drawdown (%): -8.23%
Win Rate (%): 52.34%
Final Value: 10347.00
Total Trades: 47.00
==================================================

Generating performance visualization...
Visualization saved to trading_bot/results/backtest_results.png

============================================================
PIPELINE COMPLETE!
============================================================
```

### Testing

Run the test suite:
```bash
python test_suite.py
```

Generate synthetic test data for validation:
```bash
python generate_test_data.py
```

## How It Works

### Phase 1: Data Download & Preprocessing

#### 1a. Download Raw 1-Minute Data
`data_downloader.py` downloads EURUSD, GBPUSD, USDJPY (1-minute candles) from HistData.com:

```python
from histdata import download_hist_data
# Downloads full years with month-by-month fallback for current year
# Outputs: trading_bot/data/raw/{TICKER}.csv
```

Format: `YYYYMMDD HHMMSS;Open;High;Low;Close;Volume`

#### 1b. Resample, Engineer Features, and Label
`preprocess.py` processes raw 1-minute data:

**Step 1: Resample**
- Aggregate 1-minute candles to target timeframe (default 1-hour):
  - Open: first of minute
  - High: max of minutes
  - Low: min of minutes
  - Close: last of minutes

**Step 2: Feature Engineering (MAIN_ASSET only)**

For EURUSD (main asset), compute full suite:

| Category | Features |
|----------|----------|
| **Trend** | SMA(20), SMA(50), EMA(12), EMA(26) |
| **Momentum** | RSI(14), MACD, MACD Signal, MACD Difference |
| **Volatility** | Bollinger Bands (upper, middle, lower), Rolling Volatility(20) |
| **Returns** | 1-5 lag returns, 1-5 lag closes, squared returns |
| **Total** | 20+ features |

For GBPUSD, USDJPY (exogenous assets), compute essential only:
- Close, returns, SMA(20), EMA(12), EMA(26), MACD

**Step 3: Triple Barrier Labels (MAIN_ASSET only)**

Dynamic ATR-based labeling assigns one of 3 classes per timestamp:

```
Entry Price = Close[t]
Upper Barrier = Entry + (1.5 × ATR[t])    → Label = 1 (Up)
Lower Barrier = Entry - (0.75 × ATR[t])   → Label = 2 (Down)
Max Hold = 6 hours

For each candle t to t+5:
  If High >= Upper:  label[t] = 1 (target hit)
  Else If Low <= Lower:  label[t] = 2 (stop hit)
  Else If t+6 expired:  label[t] = 0 (flat/timeout)
```

Enforces **2:1 risk/reward** (1.5 TP vs 0.75 SL).

**Step 4: Synchronization & Output**

```
EURUSD_master.csv columns:
  Open, High, Low, Close, Volume,      # OHLCV
  ATR, SMA_20, SMA_50, EMA_12, EMA_26, # Trend
  RSI, MACD, MACD_signal, MACD_diff,   # Momentum
  BB_upper, BB_lower, BB_middle,       # Volatility
  returns, returns_lag_1..5,           # Lagged returns
  close_lag_1..5, volatility,          # Lagged prices
  returns_squared,                     # Squared returns
  GBPUSD_Close, GBPUSD_returns, ...,   # Exogenous features
  USDJPY_Close, USDJPY_returns, ...,
  hour_of_day, day_of_week, month,     # Time features
  target                               # 0/1/2 labels
```

### Phase 2: Training TemporalFusionTransformer

The model architecture:

```python
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        # LSTM: 30-step input → 64-dim hidden state
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        
        # Attention: Focus on key timesteps across feature interactions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=dropout)
        
        # Classifier: Map to 3 classes (Flat, Up, Down)
        self.fc_out = Sequential(Linear(64, 64), ReLU(), Linear(64, 3))
```

**Training Procedure:**
- Input: 30-timestep windows of 38 features
- Target: 3-class labels {0, 1, 2}
- Loss: CrossEntropyLoss
- Optimizer: Adam(lr=0.001)
- Learning rate scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: patience=15 epochs on validation loss
- Train/Val split: 80/20 of last 10,000 rows (strictly split *before* scaling to prevent data leakage)
- Hardware: GPU (CUDA) is strictly enforced for training

Output: Probability distribution [P_Flat, P_Up, P_Down]

### Phase 3: Out-of-Sample Backtesting

`backtest.py` simulates trading on unseen test data:

**Trading Logic:**
```python
# At each hourly candle:
1. Extract last 30 hours of features
2. Predict probabilities: [P_flat, P_up, P_down]
3. Check dynamic risk management:
   - If position hit take-profit (unrealized_return >= tp_pct): CLOSE
   - If position hit stop-loss (unrealized_return <= sl_pct): CLOSE
4. Place new orders based on conviction:
   - If P_up > threshold(0.35) AND P_up > P_down: BUY (+1 share)
   - If P_down > threshold(0.35) AND P_down > P_up: SELL (-1 share)
   - If P_flat is highest: CLOSE ALL
5. Track position, entry price, fees, PnL
6. Update portfolio value at close
```

**Risk Management:**
- Dynamic stops/targets based on hourly ATR
- Break-even plus: Move stop to +50% of TP once at 70% unrealized gain
- Forex fee rate: 0.01% (spreads for EURUSD)
- Position size: Dynamic sizing based on ATR and Risk Percentage (e.g. risk 2% of portfolio per trade)
- Margin & Leverage: Enforces standard 50:1 leverage checks before executing trades to prevent over-margining

### Phase 4: Performance Metrics

Calculates out-of-sample metrics:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Total Return** | (Final Value - Initial) / Initial × 100 | Profit percentage |
| **Sharpe Ratio** | sqrt(252×24) × mean(returns) / std(returns) | Risk-adjusted returns (hourly annualization) |
| **Max Drawdown** | min(Portfolio / Running Max) × 100 | Worst peak-to-trough loss |
| **Win Rate** | # Winning Trades / # Total Trades × 100 | Percentage of profitable round-trips |

## Configuration

### Environment Variables (.env)

```env
# Data Download
HISTDATA_TICKERS=EURUSD,GBPUSD,USDJPY
HISTDATA_START_YEAR=2020
HISTDATA_END_YEAR=2026

# Preprocessing
TARGET_TIMEFRAME=1h              # Resample frequency
MAIN_ASSET=EURUSD                # Asset with full features + labels

# Model (via src/model.py)
# - input_chunk_length=30  (sequence length)
# - hidden_size=64
# - num_layers=2
# - epochs=150

# Backtesting (via src/backtest.py)
# - threshold=0.35         (probability threshold for conviction)
# - risk_percentage=0.2    (for position sizing if needed)

### Customize Model Architecture

Edit `src/model.py` > `PricePredictor.__init__()`:

```python
# Sequence length (longer = more context, slower training)
self.input_chunk_length = 30

# Hidden state dimension (larger = more capacity)
self.hidden_size = 64

# LSTM depth (2 = good balance)
self.num_layers = 2
```

### Adjust Trading Thresholds

Edit `src/main.py` > backtester initialization:

```python
backtester = Backtester(
    model, 
    environment, 
    threshold=0.35,        # 0.33 is 3-class baseline; lower = more aggressive
    risk_percentage=0.2
)
```

### Change Data Download Range

Edit `.env`:

```env
HISTDATA_START_YEAR=2018      # Start year
HISTDATA_END_YEAR=2024        # End year (≤ current year)
TARGET_TIMEFRAME=4h           # Change to 4-hour or 1-day data
```

## Performance Expectations

Based on EUR/USD 1-hour data (2020-2026):

- **Sharpe Ratio**: 0.5 - 1.5 (depends on market regime)
- **Max Drawdown**: 5-15% (typical for hourly trading)
- **Win Rate**: 45-55% (directional predictions are inherently difficult)
- **Total Return**: Highly variable (0-20% annually on backtests)

**Important**: Past performance ≠ future results. This is educational; live trading requires extensive out-of-sample validation.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'histdata'"

**Solution**: Install the histdata package:
```bash
pip install histdata
```

### Issue: "No data could be processed for EURUSD"

**Solution**: HistData.com may have rate limits. Try:
1. Increase wait time between downloads
2. Download smaller year ranges
3. Verify internet connection

### Issue: Memory Error During Training

**Solution**: Reduce batch size or sequence length in `model.py`:
```python
# In PricePredictor.__init__()
self.input_chunk_length = 15  # Reduce from 30
```

And in `main.py`:
```python
train_score, val_score = model.train(str(train_file), epochs=100, batch_size=32)  # Reduce from 64
```

### Issue: Poor Backtest Performance (Negative Returns)

**Checklist:**
1. Verify training completed (check confusion matrix for 3-class coverage)
2. Lower the probability threshold (more trades):
   ```python
   backtester = Backtester(model, environment, threshold=0.33)
   ```
3. Inspect backtest results CSV for entry/exit signals:
   ```bash
   tail test_split.csv | cut -d',' -f1-5  # Check timestamps & prices
   ```
4. Check for data look-ahead bias (ensure train/test strictly separated)

## Files Overview

| File | Purpose |
|------|---------|
| `data_downloader.py` | Downloads 1-minute OHLCV data from HistData.com |
| `preprocess.py` | Resamples to target timeframe, engineers features, applies Triple Barrier labels |
| `model.py` | TemporalFusionTransformer architecture + PricePredictor training/inference |
| `environment.py` | Simulates trading account, positions, PnL, fees, and VWAP entry prices |
| `backtest.py` | Out-of-sample backtesting with dynamic ATR-based risk management |
| `performance.py` | Calculates Sharpe, Drawdown, Win Rate; generates visualizations |
| `generate_test_data.py` | Creates synthetic OHLCV data for quick validation |
| `test_suite.py` | Unit tests for preprocessing, modeling, and backtesting |
| `main.py` | Orchestrates full pipeline: train → visualize → backtest |

## Dependencies

- **torch** (2.0+) — Deep learning framework
- **histdata** — Download FX data from HistData.com
- **pandas**, **numpy** — Data manipulation
- **scikit-learn** — StandardScaler for feature normalization
- **matplotlib**, **seaborn** — Visualization
- **joblib** — Serialize scaler for inference
- **python-dotenv** — Load .env configuration

## Future Enhancements

- [ ] Portfolio optimization (multiple pairs simultaneously)
- [ ] Real-time trading execution (broker integration)
- [ ] Advanced walk-forward validation (rolling windows)
- [ ] Ensemble methods (multiple models voting)
- [ ] Reinforcement learning for dynamic position sizing
- [ ] Sentiment analysis integration (news/macro)
- [ ] Genetic algorithm for hyperparameter optimization

## License

MIT License — See LICENSE file for details

## References

- HistData.com API: https://histdata.com/
- PyTorch LSTM: https://pytorch.org/docs/stable/nn.html#lstm
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- Triple Barrier Method: López de Prado, M. (Advances in Financial ML)
- Sharpe Ratio: https://en.wikipedia.org/wiki/Sharpe_ratio

---

**Disclaimer**: This is a learning project for educational purposes only. Do not use for live trading without extensive validation, paper trading, and risk management. Past performance is not indicative of future results. FX trading involves substantial risk of loss.
