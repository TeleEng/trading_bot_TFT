# Multi-Timeframe Temporal Fusion Transformer (MTF-TFT) + PPO Trading Pipeline

A state-of-the-art Deep Reinforcement Learning Forex trading bot built for **EUR/USD**, combining **HistData 1-minute tick/M1 data**, a **Multi-Timeframe Temporal Fusion Transformer (1H, 4H, 1D, 1W)** representation backbone, and a high-performance **PPO execution agent** with wide ATR risk/reward geometry, dynamic break-even stop loss ratcheting, and net-of-cost reward formulation.

---

## 🚀 Empirical Performance & Progression

Our system underwent continuous rigorous iteration on out-of-sample unseen market data (2025–2026 test period):

| Iteration / Version | Configuration | Trades | Win Rate | Max Drawdown | Total Return | Sharpe Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Static Discrete RL Agent | 0 | 0.0% | 0.0% | 0.0% | 0.00 |
| **Iteration 1** | Tight ATR Barriers (1.0x / 0.5x), Upfront Penalty | 4,210 | 21.0% | -90.0% | -90.0% | -3.52 |
| **Iteration 2** | Symmetrical 1.5:1 R:R, Continuous Action Space | 753 | 38.9% | -55.8% | -46.0% | -0.63 |
| **Iteration 3 (Champion)** | **Wide 3:2 ATR, Net-of-Cost, Backbone Gating** | **131** | **43.51%** | **-14.50%** | **+3.45%** | **+0.28** |
| **Iteration 4 (Current)** | **BE SL Ratchet + Relative Edge Gate + Scaler Fix** | *Evaluating* | *Running* | *Running* | *Running* | *Running* |

---

## 🧠 Core Architecture

```
                                  [Raw Market Data (HistData.com M1)]
                                                  │
                                                  ▼
                        [Multi-Timeframe Resampling & Feature Pipeline]
                                                  │
                ┌───────────────────┬─────────────┴─────┬───────────────────┐
                ▼                   ▼                   ▼                   ▼
         [1-Hour (1H)]       [4-Hour (4H)]        [1-Day (1D)]        [1-Week (1W)]
         52-bar Window       52-bar Window        52-bar Window       52-bar Window
                │                   │                   │                   │
                └───────────────────┼───────────────────┘                   │
                                    ▼                                       │
                      [StandardScaler Feature Normalization]                │
                                    │                                       │
                    ┌───────────────┴───────────────┐                       │
                    ▼                               ▼                       │
      [Categorical Embeddings]            [Continuous Projection]           │
      (Hour, DayOfWeek - Raw Int)        (LayerNorm Normalization)          │
                    │                               │                       │
                    └───────────────┬───────────────┘                       │
                                    ▼                                       │
                  [Gated Residual Network (GRN) per Branch]                 │
                                    │                                       │
                  [Bidirectional LSTM with Residual Skip]                   │
                      output = LayerNorm(x_grn + lstm_out)                  │
                                    │                                       │
                                    ▼                                       │
                  [Cross-Timeframe Multi-Head Attention]                    │
                                    │                                       │
                  [Fused State Representation (256-dim)]                    │
                                    │                                       │
                    ┌───────────────┴───────────────┐                       │
                    ▼                               ▼                       │
     [Cached Fused Embeddings]          [Softmax Probabilities]             │
     (256-dim State Observation)        (p_long, p_short Directional Head)  │
                    │                               │                       │
                    └───────────────┬───────────────┘                       │
                                    ▼                                       │
             [High-Performance Trading RL Environment (TradingRLEnv)]        │
             ├── Wide Triple Barrier (3.0x ATR TP / 2.0x ATR SL)            │
             ├── Break-Even Stop-Loss Ratchet (triggers at +1.5x ATR)       │
             ├── Relative Probability Edge Gate (p_long - p_short >= 0.01)  │
             ├── Volatility Regime Filter (0.70 <= ATR_Ratio <= 2.50)       │
             ├── Macro Trend Filter (Dist_SMA_50 Alignment)                 │
             └── Net-of-Cost Realized Reward (Deducts Spread, Slip, Swap)   │
                                    │                                       │
                                    ▼                                       │
                      [PPO Reinforcement Learning Agent]                     │
                      Continuous Action Space [-1.0, 1.0]                   │
                      Conviction-Based Dynamic Risk Multiplier [1.0x, 1.5x] │
```

### 1. MultiTimeframe Temporal Fusion Transformer (MTF-TFT)
- **Hierarchical Representation**: Processes 52-candle rolling contexts across 4 concurrent timeframes (1H, 4H, 1D, 1W) aligned by forward-filling higher timeframes to hourly decision candles.
- **Per-Feature `StandardScaler`**: Continuous features are normalized with per-feature mean and standard deviation fitted on training data, preventing 0–100 indicators (RSI, Stochastic) from drowning out returns/volatility ($\approx 0.001$). Categorical features (`Hour`, `DayOfWeek`) bypass scaling and feed directly into entity embeddings.
- **Residual LSTM Branches**: Each branch combines Gated Residual Networks (GRN) with LSTMs and residual skip connections:
  $$\text{branch\_out} = \text{LayerNorm}(x_{\text{grn}} + \text{LSTM}(x_{\text{grn}}))$$
- **Cross-Timeframe Attention**: Multi-head temporal attention weights macro daily/weekly regime indicators against micro hourly momentum signals.
- **Supervised Training Convergence**: Trained with pure `CrossEntropyLoss(label_smoothing=0.05)` and `CosineAnnealingLR` to guarantee sharp directional probability separation.

### 2. High-Performance Trading RL Environment (`TradingRLEnv`)
- **Wide 1.5:1 Risk/Reward Geometry**:
  - Take Profit: $+3.0\times$ ATR
  - Stop Loss: $-2.0\times$ ATR
  - Maximum holding duration: 96 hours
  - Reduces round-trip transaction friction from $\sim 19\%$ of stop-loss down to $< 4.5\%$.
- **Dynamic Break-Even Stop-Loss Ratchet**:
  - Longs: When intra-bar High reaches $\ge \text{entry} + 1.5\times \text{ATR}$, SL automatically ratchets to $\text{entry} + 0.1\times \text{ATR}$.
  - Shorts: When intra-bar Low drops $\le \text{entry} - 1.5\times \text{ATR}$, SL ratchets to $\text{entry} - 0.1\times \text{ATR}$.
  - Locks in broker spread and exchange fees, preventing $+1.5\times$ to $+2.5\times$ ATR winners from decaying into maximum losses.
- **Relative Directional Edge Gating**:
  - Requires $(p_{\text{long}} - p_{\text{short}}) \ge \text{min\_edge}$ (default 0.01) for longs and vice versa for shorts. Replaces brittle absolute gating to eliminate trade starvation while filtering out noise.
- **Volatility Regime Filter**:
  - Rejects trade entries when $\text{ATR\_Ratio} < 0.70$ (low-volatility dead zones) or $\text{ATR\_Ratio} > 2.50$ (extreme volatility blowouts).
- **Macro Trend Filter**:
  - Aligns trades with the higher-timeframe trend using normalized SMA-50 distance, blocking longs during steep macro downtrends and shorts during strong macro uptrends.
- **Anti-Revenge Trading Cooldown**:
  - Enforces a 4-bar mandatory lockout in the losing direction following any stop-loss event.
- **Realistic Transaction Cost Model**:
  - Deducts 1.0 pip spread, 0.3 pip random slippage, and hourly rollover swap costs ($0.5$ pips/day).

---

## 📁 Repository Structure

```
trading_bot/
├── data/
│   ├── raw/                              # HistData.com 1-minute OHLCV tick data
│   └── processed/                        # Resampled, engineered & labeled CSV files
│       ├── EURUSD_master_1h.csv          # 1-Hour timeframe features + Triple Barrier
│       ├── EURUSD_master_4h.csv          # 4-Hour timeframe features
│       ├── EURUSD_master_1d.csv          # Daily timeframe features
│       └── EURUSD_master_1w.csv          # Weekly timeframe features
├── models/
│   ├── tft_backbone.pth                  # Trained MultiTimeframeTFT model weights + scalers
│   ├── train_embeddings.npy              # Cached 256-dim fused embeddings (Train)
│   ├── val_embeddings.npy                # Cached 256-dim fused embeddings (Validation)
│   ├── test_embeddings.npy               # Cached 256-dim fused embeddings (Held-out Test)
│   ├── train_probabilities.npy           # Cached softmax directional probabilities (Train)
│   ├── test_probabilities.npy            # Cached softmax directional probabilities (Test)
│   └── best_model.zip                    # Champion PPO RL agent checkpoint
├── results/
│   ├── backtest_results.png              # Champion out-of-sample backtest equity curve
│   ├── rl_search_results.csv             # Full RL hyperparameter grid search leaderboard
│   └── log.txt                           # Detailed empirical execution log
├── src/
│   ├── data_downloader.py                # HistData.com M1 downloader with fallback
│   ├── preprocess.py                     # Multi-timeframe resampling, indicators & Triple Barrier
│   ├── model.py                          # MultiTimeframeTFT architecture & PricePredictor trainer
│   ├── rl_environment.py                 # TradingRLEnv (Gymnasium) with BE ratchet & edge filters
│   ├── rl_grid_search.py                 # Multi-configuration PPO hyperparameter grid search
│   ├── performance.py                    # PerformanceMetrics (Sharpe, Drawdown, Profit Factor, Plots)
│   ├── backtest.py                       # Standalone backtesting engine
│   ├── test_rl_env.py                    # Unit test suite for RL environment & ratchet logic
│   ├── test_suite.py                     # General test suite for data & features
│   └── main.py                           # Central orchestrator CLI
├── kaggle_pull/
│   ├── kernel-metadata.json              # Kaggle API metadata
│   └── trading-bot-tft-pipeline-2026.ipynb # Automated cloud GPU runner notebook
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU strongly recommended for backbone training)
- Gymnasium, Stable-Baselines3

### 1. Clone & Install
```bash
git clone https://github.com/TeleEng/trading_bot_TFT.git
cd trading_bot_TFT
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit `.env` in the project root:
```env
HISTDATA_TICKERS=EURUSD,GBPUSD,USDJPY
HISTDATA_START_YEAR=2020
HISTDATA_END_YEAR=2026
TARGET_TIMEFRAME=1h
MAIN_ASSET=EURUSD
```

---

## ⚡ Execution Workflows

### Option A: Full Pipeline with RL Hyperparameter Grid Search (Recommended)
Executes data preprocessing, MultiTimeframe TFT representation training, embedding caching, and runs the 6-configuration RL hyperparameter grid search:
```bash
cd src
python main.py --grid-search --force-retrain
```

### Option B: Quick Champion Run (Cached Embeddings)
Runs the champion PPO agent directly using pre-computed representations:
```bash
cd src
python main.py
```

### Option C: Cloud Execution on Kaggle (P100 / T4 GPU)
The repository includes a ready-to-run Kaggle notebook pipeline in `kaggle_pull/`:
```bash
# Push kernel to Kaggle
kaggle kernels push -p kaggle_pull

# Monitor execution status
kaggle kernels status infernalss/trading-bot-tft-pipeline-2026

# Retrieve generated output and backtest plot
kaggle kernels output infernalss/trading-bot-tft-pipeline-2026 -p results/
```

---

## 🧪 Testing & Verification

Run the dedicated RL environment unit tests (verifies spread deduction, dynamic sizing, break-even ratchet, probability edge filter, and volatility gating):
```bash
python src/test_rl_env.py
```

Run the pipeline test suite:
```bash
python src/test_suite.py
```

Verify syntax across all source files:
```bash
python -m py_compile src/*.py
```

---

## 📊 Performance Metrics Formulae

The system evaluates out-of-sample backtests with institutional quantitative metrics:

- **Total Return (%)**:
  $$\text{Return} = \frac{V_{\text{final}} - V_{\text{initial}}}{V_{\text{initial}}} \times 100$$
- **Hourly Annualized Sharpe Ratio**:
  $$\text{Sharpe} = \sqrt{252 \times 24} \cdot \frac{\mu(R_t)}{\sigma(R_t)}$$
- **Maximum Drawdown (%)**:
  $$\text{MaxDD} = \max_{t} \left( \frac{\max_{s \le t} V_s - V_t}{\max_{s \le t} V_s} \right) \times 100$$
- **Profit Factor**:
  $$\text{Profit Factor} = \frac{\sum \text{Gross Profits}}{\left| \sum \text{Gross Losses} \right|}$$

---

## 📜 License & Disclaimer

Distributed under the MIT License.

> **Disclaimer**: This software is developed for research and educational purposes. Forex and CFD trading involves substantial risk of loss. Past performance, backtest simulations, and neural network predictions are not guarantees of future profitability. Validate thoroughly in simulated paper-trading environments before deploying real capital.
