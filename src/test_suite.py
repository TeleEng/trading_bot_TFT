#!/usr/bin/env python3
"""Test script for trading bot components using synthetic data."""
import os
import sys
import pandas as pd
from pathlib import Path

# Generate test data first
print("="*60)
print("TRADING BOT - TEST SUITE")
print("="*60)

print("\n[1/6] Generating synthetic test data...")
from generate_test_data import create_test_data
create_test_data()

# Preprocess test data
print("\n[2/6] Testing preprocessing pipeline...")
from preprocess import process_all_data
try:
    process_all_data()
    print("[OK] Data preprocessing successful")
except Exception as e:
    print(f"[FAIL] Preprocessing failed: {e}")
    sys.exit(1)

# Test model training
print("\n[3/6] Testing model training...")
from model import PricePredictor
try:
    # FIX: Initialize the TFT model properly (removing invalid 'random_forest' param)
    model = PricePredictor(input_chunk_length=10)
    data_files = list(Path("trading_bot/data/processed").glob("*.csv"))
    if not data_files:
        raise FileNotFoundError("No processed data files found")

    # Limit train size for faster testing
    df = pd.read_csv(str(data_files[0]), index_col=0, parse_dates=True)
    train_file = "trading_bot/data/processed/test_split_for_suite.csv"
    df.iloc[:200].to_csv(train_file)

    train_score, test_score = model.train(train_file, epochs=2)
    print(f"[OK] Model training successful (Test Acc: {test_score:.4f})")

    # Save model
    os.makedirs("trading_bot/models", exist_ok=True)
    model.save("trading_bot/models/test_model.pkl")
    print("[OK] Model saved successfully")
except Exception as e:
    print(f"[FAIL] Model training failed: {e}")
    sys.exit(1)

# Test environment
print("\n[4/6] Testing trading environment...")
from environment import TradingEnvironment
try:
    # FIX: Transaction costs are now dynamically fetched inside the environment based on ticker
    env = TradingEnvironment(initial_capital=10000)

    # Test buy
    ts = pd.Timestamp.now()
    env.execute_trade("BTC_USD", 1, 100, ts)
    assert env.positions["BTC_USD"] == 1, "Buy trade failed"
    assert env.capital < 10000, "Capital not deducted"

    # Test sell
    env.execute_trade("BTC_USD", -1, 105, ts)
    assert env.positions["BTC_USD"] == 0, "Sell trade failed"

    # Test portfolio value
    prices = {"BTC_USD": 105}
    value = env.get_portfolio_value(prices)
    assert value > 0, "Portfolio value invalid"

    print("[OK] Trading environment working correctly")
except Exception as e:
    print(f"[FAIL] Environment test failed: {e}")
    sys.exit(1)

# Test backtester
print("\n[5/6] Testing backtester...")
from backtest import Backtester
try:
    env = TradingEnvironment(initial_capital=10000)
    # FIX: Include risk_percentage parameter
    backtester = Backtester(model, env, threshold=0.55, risk_percentage=0.2)

    backtest_results = backtester.run(str(data_files[0]))
    assert len(backtest_results) > 0, "No backtest results"
    assert 'portfolio_value' in backtest_results.columns, "Missing portfolio_value column"

    print(f"[OK] Backtester completed ({len(backtest_results)} timesteps)")
except Exception as e:
    print(f"[FAIL] Backtester failed: {e}")
    sys.exit(1)

# Test performance metrics
print("\n[6/6] Testing performance metrics...")
from performance import PerformanceMetrics
try:
    metrics = PerformanceMetrics.calculate_metrics(backtest_results, initial_capital=10000)

    print("[OK] Performance metrics calculated:")
    for key, value in metrics.items():
        if '%' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.2f}")

    # Generate visualization
    os.makedirs("trading_bot/results", exist_ok=True)
    PerformanceMetrics.plot_results(backtest_results, save_path="trading_bot/results/test_backtest.png")
    print("[OK] Visualization saved to trading_bot/results/test_backtest.png")

except Exception as e:
    print(f"[FAIL] Performance metrics failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)