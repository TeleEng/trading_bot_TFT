import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_synthetic_data(ticker_name, num_periods=500, start_price=100, volatility=0.02):
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_periods, freq='h')

    np.random.seed(42)
    returns = np.random.normal(0, volatility, num_periods)
    prices = start_price * np.exp(np.cumsum(returns))

    data = {
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, num_periods)),
        'High': prices * (1 + np.random.uniform(0, 0.01, num_periods)),
        'Low': prices * (1 - np.random.uniform(0, 0.01, num_periods)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, num_periods),
        'Adj Close': prices,
    }

    df = pd.DataFrame(data, index=dates)
    return df.sort_index()

def create_test_data():
    """Create synthetic data for testing."""
    data_path = "../data/raw"
    os.makedirs(data_path, exist_ok=True)

    tickers = {
        'EURUSD': 1.08,
        'BTC_USD': 45000,
        'ETH_USD': 2500,
        'GC_F': 2000,
        'GBPUSD': 1.27,
    }

    for ticker, start_price in tickers.items():
        df = generate_synthetic_data(ticker, num_periods=500, start_price=start_price)
        filepath = os.path.join(data_path, f'{ticker}.csv')
        df.to_csv(filepath)
        print(f"Generated synthetic data for {ticker}: {filepath}")

if __name__ == "__main__":
    create_test_data()
