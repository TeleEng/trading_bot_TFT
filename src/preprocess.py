import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path("D:/Work/trading_bot/archive/trading_bot_TFT/data/raw")
OUTPUT_PATH = BASE_DIR / "data" / "processed"
TARGET_TIMEFRAME = os.getenv("TARGET_TIMEFRAME", "1h")
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def add_triple_barrier_labels(df, atr_window=14, tp_mult=1.5, sl_mult=0.75, max_wait=6):
    """
    Implements a 6-Hour Triple Barrier Method using dynamic ATR.
    tp_mult=1.5 and sl_mult=0.75 enforces the R/R = 2 rule.
    """
    # 1. Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_window).mean()

    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    atr_arr = df['ATR'].values

    labels = np.zeros(len(df)) # Class 0: Flat

    # 2. Evaluate paths
    for i in range(len(df) - max_wait):
        if np.isnan(atr_arr[i]):
            continue

        current_price = close_arr[i]
        upper_barrier = current_price + (tp_mult * atr_arr[i])
        lower_barrier = current_price - (sl_mult * atr_arr[i])

        label = 0 
        for j in range(i + 1, i + 1 + max_wait):
            hit_upper = high_arr[j] >= upper_barrier
            hit_lower = low_arr[j] <= lower_barrier

            if hit_upper and hit_lower:
                label = 0 
                break
            elif hit_upper:
                label = 1 # Up
                break
            elif hit_lower:
                label = 2 # Down
                break

        labels[i] = label

    df['target'] = labels
    return df

def add_all_features(df):
    """FULL feature engineering applied exclusively to the MAIN asset."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    bb_sma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = bb_sma + (bb_std * 2)
    df['BB_lower'] = bb_sma - (bb_std * 2)
    df['BB_middle'] = bb_sma

    df['returns'] = df['Close'].pct_change()
    for lag in range(1, 6):
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['returns_squared'] = df['returns'] ** 2
    
    # 6-Hour Triple Barrier
    df = add_triple_barrier_labels(df, max_wait=6)
    
    return df

def add_essential_features(df):
    """ESSENTIAL features applied to exogenous non-main assets."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['returns'] = df['Close'].pct_change()
    return df[['Close', 'returns', 'SMA_20', 'EMA_12', 'EMA_26', 'MACD']]

def generate_timeframe_data(timeframe_str, is_base=False):
    print(f"--- Processing {timeframe_str} Timeframe ---")
    raw_files = list(Path(DATA_PATH).glob("*.csv"))
    if not raw_files:
        raise ValueError(f"No CSV files found in {DATA_PATH}")

    resample_freq = timeframe_str.lower().replace('m', 'min')

    main_df = None
    exogenous_dfs = []

    for file in raw_files:
        ticker = file.stem
        df = pd.read_csv(file, index_col=0, parse_dates=True).sort_index()
        df = df.resample(resample_freq).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        if ticker == MAIN_ASSET:
            main_df = df
        else:
            exogenous_dfs.append((ticker, df))

    main_df = add_all_features(main_df)
    
    # Only the base timeframe needs the target labels
    if is_base:
        main_df = add_triple_barrier_labels(main_df, max_wait=6)
    
    processed_exogenous = {}
    for ticker, df in exogenous_dfs:
        df = add_essential_features(df)
        df.columns = [f"{ticker}_{col}" for col in df.columns]
        processed_exogenous[ticker] = df

    if processed_exogenous:
        exo_concat = pd.concat(processed_exogenous.values(), axis=1, join='outer')
        unified = pd.concat([main_df, exo_concat], axis=1, join='outer')
    else:
        unified = main_df

    unified.ffill(limit=12, inplace=True)
    unified.dropna(inplace=True)
    if is_base:
        unified = unified.iloc[:-6] # Drop final rows with unresolved labels

    unified['hour_sin'] = np.sin(2 * np.pi * unified.index.hour / 24)
    unified['hour_cos'] = np.cos(2 * np.pi * unified.index.hour / 24)
    unified['day_sin'] = np.sin(2 * np.pi * unified.index.dayofweek / 7)
    unified['day_cos'] = np.cos(2 * np.pi * unified.index.dayofweek / 7)
    unified['month_sin'] = np.sin(2 * np.pi * unified.index.month / 12)
    unified['month_cos'] = np.cos(2 * np.pi * unified.index.month / 12)

    output_file = OUTPUT_PATH / f"{MAIN_ASSET}_master_{timeframe_str}.csv"
    unified.to_csv(output_file)
    print(f"Saved to {output_file}")

def process_all_data():
    generate_timeframe_data("1h", is_base=True)
    generate_timeframe_data("4h", is_base=False)
    generate_timeframe_data("1d", is_base=False)
    print(f"\n[OK] MTF Pipeline Complete!")

if __name__ == "__main__":
    process_all_data()