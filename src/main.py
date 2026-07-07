#!/usr/bin/env python3
"""
Trading Bot - End-to-end pipeline: download data -> preprocess -> train model -> backtest
"""
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from data_downloader import download_histdata, TICKERS, START_YEAR, END_YEAR, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics
from viz import plot_tsne_and_confusion_matrix

def main():
    print("="*60)
    print("TRADING BOT - FULL PIPELINE (SELF-SUPERVISED)")
    print("="*60)

    # Setup directories safely using Pathlib
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 1] Downloading & Preprocessing Data...")
    # download_histdata(TICKERS, START_YEAR, END_YEAR, DATA_PATH)
    # process_all_data()

    print("\n[Phase 2] Splitting Data (60/20/20)...")
    model = PricePredictor()

    load_dotenv()
    MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")
    first_ticker_file = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master.csv"
    
    if not first_ticker_file.exists():
        print(f"[ERROR] {first_ticker_file} not found. Did you run preprocess.py?")
        sys.exit(1)
        
    df = pd.read_csv(first_ticker_file, index_col=0, parse_dates=True).iloc[-10_000:]
    
    val_idx = int(len(df) * 0.6)
    test_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx - model.input_chunk_length : test_idx]
    test_df = df.iloc[test_idx - model.input_chunk_length :] 

    train_file = Path(OUTPUT_PATH) / "train_split.csv"
    val_file = Path(OUTPUT_PATH) / "val_split.csv"
    test_file = Path(OUTPUT_PATH) / "test_split.csv"
    
    train_df.to_csv(train_file)
    val_df.to_csv(val_file)
    test_df.to_csv(test_file)
    
    print("\n[Phase 3] Self-Supervised Contrastive Training...")
    train_score, val_score = model.train(str(train_file), str(val_file), epochs=10)
    print(f"Model trained on 60% split of {first_ticker_file.name}")
    print(f"Final InfoNCE Loss - Train: {train_score:.4f} | Val: {val_score:.4f}")

    model_path = models_dir / "model.pkl"
    model.save(str(model_path))

    print("\n[Phase 4] Generating t-SNE & Confusion Matrix...")
    plot_tsne_and_confusion_matrix(model, str(test_file), str(results_dir))

    print("\n[Phase 5] Running out-of-sample backtest...")
    environment = TradingEnvironment(initial_capital=10000)
    backtester = Backtester(model, environment, threshold=0.35, risk_percentage=0.2)

    backtest_results = backtester.run(str(test_file))

    metrics = PerformanceMetrics.calculate_metrics(
        backtest_results,
        initial_capital=10000
    )
    PerformanceMetrics.print_report(metrics)

    print("Generating performance visualization...")
    viz_path = results_dir / "backtest_results.png"
    PerformanceMetrics.plot_results(backtest_results, save_path=str(viz_path))

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()