import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
from src.model import PricePredictor
from src.main import OUTPUT_PATH
import numpy as np

# 1. Load small subset of data
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Loading data...")
df_1h = pd.read_csv("data/processed/EURUSD_master_1h.csv", index_col=0, parse_dates=True).iloc[-5000:]
df_4h = pd.read_csv("data/processed/EURUSD_master_4h.csv", index_col=0, parse_dates=True)
df_1d = pd.read_csv("data/processed/EURUSD_master_1d.csv", index_col=0, parse_dates=True)
df_1w = pd.read_csv("data/processed/EURUSD_master_1w.csv", index_col=0, parse_dates=True)

# 2. Split
val_idx = int(len(df_1h) * 0.6)
test_idx = int(len(df_1h) * 0.8)

model = PricePredictor(input_chunk_length=52, hidden_size=128, num_layers=3)
train_1h = df_1h.iloc[:val_idx]
val_1h = df_1h.iloc[val_idx - model.input_chunk_length : test_idx]

print("Training...")
# 3. Train
train_score, val_score = model.train(
    (train_1h, df_4h, df_1d, df_1w),
    (val_1h, df_4h, df_1d, df_1w),
    epochs=15
)

print("Generating CM...")
# 4. CM
val_c_probs = model.predict_batch_classified((val_1h, df_4h, df_1d, df_1w))
if len(val_c_probs) > 0:
    val_preds = val_c_probs.argmax(axis=1)
    _, _, _, _, y_val = model.create_sequences(val_1h, df_4h, df_1d, df_1w, clean_noise=False)
    from sklearn.metrics import confusion_matrix
    print("CM:")
    print(confusion_matrix(y_val, val_preds))
