import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

local_raw_path = Path("D:/Work/trading_bot/archive/trading_bot_TFT/data/raw")
kaggle_raw_path1 = Path("/kaggle/input/datasets/infernalss/tft-pipeline-dataset/trading_bot_TFT/data/raw")
kaggle_raw_path2 = Path("/kaggle/input/tft-pipeline-dataset/trading_bot_TFT/data/raw")
kaggle_raw_path3 = Path("/kaggle/input/tft-pipeline-dataset/data/raw")

if local_raw_path.exists():
    DATA_PATH = local_raw_path
elif kaggle_raw_path1.exists():
    DATA_PATH = kaggle_raw_path1
elif kaggle_raw_path2.exists():
    DATA_PATH = kaggle_raw_path2
elif kaggle_raw_path3.exists():
    DATA_PATH = kaggle_raw_path3
else:
    DATA_PATH = BASE_DIR / "data" / "raw"

OUTPUT_PATH = BASE_DIR / "data" / "processed"
TARGET_TIMEFRAME = os.getenv("TARGET_TIMEFRAME", "1h")
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def add_triple_barrier_labels(df, atr_window=14, 
                              long_tp_mult=2.5, long_sl_mult=1.0, 
                              short_tp_mult=4.0, short_sl_mult=1.0, 
                              max_wait=48):
    """
    Implements a decoupled Triple Barrier Method using dynamic ATR.
    Longs use 1:2.5 R/R. Shorts use 1:4 R/R.
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

    labels_long = np.zeros(len(df))
    labels_short = np.zeros(len(df))

    # 2. Evaluate paths
    for i in range(len(df) - max_wait):
        if np.isnan(atr_arr[i]):
            continue

        current_price = close_arr[i]
        
        # Long Barriers
        tp_long = current_price + (long_tp_mult * atr_arr[i])
        sl_long = current_price - (long_sl_mult * atr_arr[i])
        
        # Short Barriers
        tp_short = current_price - (short_tp_mult * atr_arr[i])
        sl_short = current_price + (short_sl_mult * atr_arr[i])

        # Evaluate Long Path
        label_l = 0
        for j in range(i + 1, i + 1 + max_wait):
            hit_tp = high_arr[j] >= tp_long
            hit_sl = low_arr[j] <= sl_long
            
            if hit_tp and hit_sl:
                break # Collision
            elif hit_tp:
                label_l = 1
                break
            elif hit_sl:
                break
        labels_long[i] = label_l

        # Evaluate Short Path
        label_s = 0
        for j in range(i + 1, i + 1 + max_wait):
            hit_tp = low_arr[j] <= tp_short
            hit_sl = high_arr[j] >= sl_short
            
            if hit_tp and hit_sl:
                break # Collision
            elif hit_tp:
                label_s = 1
                break
            elif hit_sl:
                break
        labels_short[i] = label_s

    df['target_long'] = labels_long
    df['target_short'] = labels_short
    return df

def add_all_features(df):
    """FULL feature engineering applied exclusively to the MAIN asset."""
    # 1. Calculate ATR for normalization
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean().replace(0, 1e-10) # Avoid division by zero
    df['SMA_ATR_50'] = df['ATR'].rolling(50).mean().replace(0, 1e-10)
    df['ATR_Ratio'] = df['ATR'] / df['SMA_ATR_50']

    # 2. Stationary distances for Moving Averages
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_50 = df['Close'].rolling(window=50).mean()
    df['Dist_SMA_20'] = (df['Close'] - sma_20) / df['ATR']
    df['Dist_SMA_50'] = (df['Close'] - sma_50) / df['ATR']

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['Dist_EMA_12'] = (df['Close'] - ema_12) / df['ATR']
    df['Dist_EMA_26'] = (df['Close'] - ema_26) / df['ATR']

    # RSI (Already stationary 0-100)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Normalize by ATR)
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_norm'] = macd / df['ATR']
    df['MACD_signal_norm'] = macd_signal / df['ATR']
    df['MACD_diff_norm'] = (macd - macd_signal) / df['ATR']

    # Bollinger Bands
    bb_sma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std().replace(0, 1e-10)
    # Feature 1: Position within bands (-1 to 1)
    df['BB_Position'] = (df['Close'] - bb_sma) / (bb_std * 2)
    # Feature 2: Band width normalized by ATR
    df['BB_Width_norm'] = (bb_std * 4) / df['ATR']

    # Returns (Already stationary)
    df['returns'] = df['Close'].pct_change()
    for lag in range(1, 6):
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['returns_squared'] = df['returns'] ** 2
    
    # Stochastic Oscillator (Already stationary 0-100)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14).replace(0, 1e-10))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Momentum (Normalize by ATR)
    df['Momentum_10_norm'] = (df['Close'] - df['Close'].shift(10)) / df['ATR']
    
    # Categorical Time Features for TFT
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    
    # 6-Hour Triple Barrier
    df = add_triple_barrier_labels(df)
    
    return df

def add_essential_features(df):
    """ESSENTIAL features applied to exogenous non-main assets."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().replace(0, 1e-10)

    sma_20 = df['Close'].rolling(window=20).mean()
    df['Dist_SMA_20'] = (df['Close'] - sma_20) / atr
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df['MACD_norm'] = macd / atr
    df['returns'] = df['Close'].pct_change()
    
    # Do NOT return raw non-stationary columns
    return df[['returns', 'Dist_SMA_20', 'MACD_norm']]

def generate_timeframe_data(timeframe_str, is_base=False):
    print(f"--- Processing {timeframe_str} Timeframe ---")
    raw_files = list(Path(DATA_PATH).glob("*.csv"))
    if not raw_files:
        raise ValueError(f"No CSV files found in {DATA_PATH}")

    resample_freq = timeframe_str.lower().replace('m', 'min')
    if resample_freq == '1w':
        resample_freq = '1W'
        
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
        main_df = add_triple_barrier_labels(main_df)
    
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
        unified = unified.iloc[:-48] # Drop final rows with unresolved labels

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
    generate_timeframe_data("1w", is_base=False)
    print(f"\n[OK] MTF Pipeline Complete!")

if __name__ == "__main__":
    process_all_data()