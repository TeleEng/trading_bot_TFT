#!/usr/bin/env python3
"""
Trading Bot - End-to-end pipeline: download data -> preprocess -> train model -> backtest
"""
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from download_data import download_data, TICKERS, INTERVAL, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics

def main():
    print("="*60)
    print("TRADING BOT - FULL PIPELINE")
    print("="*60)

    # Setup directories safely using Pathlib
    base_dir = Path("trading_bot")
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Download Data
    print("\n[Phase 1] Downloading market data...")
    # download_data(TICKERS, INTERVAL, DATA_PATH)

    # Phase 1: Preprocess Data
    print("\n[Phase 1] Preprocessing data and engineering features...")
    # process_all_data()

    # Phase 2: Train Model
    print("\n[Phase 2] Splitting Data & Training ML model...")
    model = PricePredictor()

    # FIX: Explicitly load the correct master dataset
    load_dotenv()
    MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")
    first_ticker_file = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master.csv"
    
    if not first_ticker_file.exists():
        print(f"[ERROR] {first_ticker_file} not found. Did you run preprocess.py?")
        sys.exit(1)
        
    # Strictly separate out-of-sample data
    df = pd.read_csv(first_ticker_file, index_col=0, parse_dates=True).iloc[-10_000:]
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    
    # FIX: Prevent overlap leakage. By backing up exactly `input_chunk_length`, 
    # the backtester's first trade will land exactly on the first unseen row (split_idx).
    test_df = df.iloc[split_idx - model.input_chunk_length:] 

    train_file = Path(OUTPUT_PATH) / "train_split.csv"
    test_file = Path(OUTPUT_PATH) / "test_split.csv"
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    print(train_df.target.value_counts())
    train_score, val_score = model.train(str(train_file))
    print(f"Model trained on 80% split of {first_ticker_file.name}")
    print(f"Internal Validation Score: {val_score:.4f}")

    # Save model safely
    model_path = models_dir / "model.pkl"
    model.save(str(model_path))

    # Plot learning curves safely
    history_path = results_dir / "training_history.png"
    model.plot_training_history(save_path=str(history_path))
    model.plot_confusion_matrix(save_path=str(results_dir / "confusion_matrix.png"))
    # Phase 3: Backtesting
    print("\n[Phase 3] Running out-of-sample backtest...")
    environment = TradingEnvironment(initial_capital=10000)
    backtester = Backtester(model, environment, threshold=0.55, risk_percentage=0.2)

    backtest_results = backtester.run(str(test_file))

    # Calculate metrics
    metrics = PerformanceMetrics.calculate_metrics(
        backtest_results,
        initial_capital=10000
    )
    PerformanceMetrics.print_report(metrics)

    # Plot results safely
    print("Generating performance visualization...")
    viz_path = results_dir / "backtest_results.png"
    PerformanceMetrics.plot_results(backtest_results, save_path=str(viz_path))

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()