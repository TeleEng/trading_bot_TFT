import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Append src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from preprocess import add_triple_barrier_labels

def main():
    print("Loading test data...")
    data_dir = Path("d:/Work/trading_bot/data/processed")
    df_1h = pd.read_csv(data_dir / "EURUSD_master_1h.csv", index_col=0, parse_dates=True)
    df_4h = pd.read_csv(data_dir / "EURUSD_master_4h.csv", index_col=0, parse_dates=True)
    df_1d = pd.read_csv(data_dir / "EURUSD_master_1d.csv", index_col=0, parse_dates=True)
    
    print("Adding labels...")
    df_1h = add_triple_barrier_labels(df_1h)
    
    print("Splitting data...")
    test_idx = int(len(df_1h) * 0.8)
    test_1h = df_1h.iloc[test_idx - 30:]
    
    print("Loading model...")
    model = PricePredictor(input_chunk_length=30)
    # Don't strictly need to load weights just to test logic, random weights will produce some signals
    model.model = model.model = __import__('model').MultiTimeframeTFT(
        input_size=test_1h.drop(columns=['target'], errors='ignore').shape[1],
        hidden_size=64,
        num_layers=2
    ).to(model.device)
    
    print("Running backtest...")
    environment = TradingEnvironment(initial_capital=10000)
    backtester = Backtester(model, environment, threshold=0.35, risk_percentage=0.2)
    
    # Enable verbose printing
    print("Patching Backtester to print logic...")
    original_run = backtester.run
    
    results = backtester.run((test_1h, df_4h, df_1d))
    
    print(f"Total rows in results: {len(results)}")
    
    positions = results['position'].values
    print(f"Number of times position != 0: {(positions != 0).sum()}")
    print(f"Unique positions: {np.unique(positions)}")

if __name__ == '__main__':
    main()
