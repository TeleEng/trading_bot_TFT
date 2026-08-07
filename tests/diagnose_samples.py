#!/usr/bin/env python3
"""Diagnostic: trace sample counts through the entire preprocessing pipeline."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "raw"
processed_path = base_dir / "data" / "processed"

MAIN_ASSET = os.environ.get("MAIN_ASSET", "EURUSD")

print("=" * 60)
print("DATA PIPELINE DIAGNOSTIC")
print("=" * 60)

# Step 1: Check raw data
print("\n[1] RAW DATA FILES:")
raw_files = list(data_path.glob("*.csv"))
if not raw_files:
    print(f"  No CSV files found in {data_path}")
    print(f"  Checking processed data instead...")
else:
    for f in raw_files:
        df_raw = pd.read_csv(f, index_col=0, parse_dates=True)
        print(f"  {f.stem}: {len(df_raw)} rows, date range: {df_raw.index.min()} to {df_raw.index.max()}")

# Step 2: Check processed data
print("\n[2] PROCESSED DATA FILES:")
for tf in ['1h', '4h', '1d', '1w']:
    fpath = processed_path / f"{MAIN_ASSET}_master_{tf}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        print(f"  {tf}: {len(df)} rows, {len(df.columns)} cols, date range: {df.index.min()} to {df.index.max()}")
    else:
        print(f"  {tf}: FILE NOT FOUND at {fpath}")

# Step 3: Simulate create_sequences to count valid samples
print("\n[3] SEQUENCE CREATION SIMULATION:")
df_1h_path = processed_path / f"{MAIN_ASSET}_master_1h.csv"
df_4h_path = processed_path / f"{MAIN_ASSET}_master_4h.csv"
df_1d_path = processed_path / f"{MAIN_ASSET}_master_1d.csv"
df_1w_path = processed_path / f"{MAIN_ASSET}_master_1w.csv"

if all(p.exists() for p in [df_1h_path, df_4h_path, df_1d_path, df_1w_path]):
    df_1h = pd.read_csv(df_1h_path, index_col=0, parse_dates=True)
    df_4h = pd.read_csv(df_4h_path, index_col=0, parse_dates=True)
    df_1d = pd.read_csv(df_1d_path, index_col=0, parse_dates=True)
    df_1w = pd.read_csv(df_1w_path, index_col=0, parse_dates=True)
    
    input_chunk_length = 52
    
    idx_4h = df_4h.index.get_indexer(df_1h.index, method='ffill')
    idx_1d = df_1d.index.get_indexer(df_1h.index, method='ffill')
    idx_1w = df_1w.index.get_indexer(df_1h.index, method='ffill')
    
    total_1h = len(df_1h)
    skipped_chunk = 0
    skipped_4h = 0
    skipped_1d = 0
    skipped_1w = 0
    valid = 0
    first_valid_idx = None
    
    for i in range(total_1h):
        if i < input_chunk_length:
            skipped_chunk += 1
            continue
        i_4h = idx_4h[i]
        i_1d = idx_1d[i]
        i_1w = idx_1w[i]
        
        if i_4h < 52:
            skipped_4h += 1
        elif i_1d < 52:
            skipped_1d += 1
        elif i_1w < 52:
            skipped_1w += 1
        else:
            valid += 1
            if first_valid_idx is None:
                first_valid_idx = i
    
    print(f"  Total 1H rows:             {total_1h}")
    print(f"  Skipped (input_chunk < 52): {skipped_chunk}")
    print(f"  Skipped (4H index < 52):    {skipped_4h}")
    print(f"  Skipped (1D index < 52):    {skipped_1d}")
    print(f"  Skipped (1W index < 52):    {skipped_1w}")
    print(f"  ----------------------------------------")
    print(f"  VALID SAMPLES:              {valid}")
    print(f"  First valid at 1H index:    {first_valid_idx}")
    if first_valid_idx:
        print(f"  First valid date:           {df_1h.index[first_valid_idx]}")
        print(f"  Data lost to warmup:        {first_valid_idx} rows = {first_valid_idx/24:.0f} days")
    
    # Step 4: Check split sizes
    print("\n[4] TRAIN/VAL/TEST SPLIT (60/20/20):")
    n = total_1h
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    print(f"  Train: rows 0-{train_end} ({train_end} rows)")
    print(f"  Val:   rows {train_end}-{val_end} ({val_end - train_end} rows)")
    print(f"  Test:  rows {val_end}-{n} ({n - val_end} rows)")
    
    # Count valid per split
    valid_train = sum(1 for i in range(input_chunk_length, train_end) 
                      if idx_4h[i] >= 52 and idx_1d[i] >= 52 and idx_1w[i] >= 52)
    valid_val = sum(1 for i in range(max(input_chunk_length, train_end), val_end) 
                    if idx_4h[i] >= 52 and idx_1d[i] >= 52 and idx_1w[i] >= 52)
    valid_test = sum(1 for i in range(max(input_chunk_length, val_end), n) 
                     if idx_4h[i] >= 52 and idx_1d[i] >= 52 and idx_1w[i] >= 52)
    print(f"  Valid train samples: {valid_train}")
    print(f"  Valid val samples:   {valid_val}")
    print(f"  Valid test samples:  {valid_test}")
    print(f"  Total valid:         {valid_train + valid_val + valid_test}")
    
    # Step 5: Label distribution
    print("\n[5] LABEL DISTRIBUTION (1H base):")
    if 'target_long' in df_1h.columns and 'target_short' in df_1h.columns:
        counts_long = df_1h['target_long'].value_counts().sort_index()
        counts_short = df_1h['target_short'].value_counts().sort_index()
        
        print("  --- Long Target ---")
        for label, count in counts_long.items():
            pct = count / len(df_1h) * 100
            name = {0: 'Flat', 1: 'Up'}.get(int(label), f'Label {label}')
            print(f"  {name} ({int(label)}): {count} ({pct:.1f}%)")
            
        print("  --- Short Target ---")
        for label, count in counts_short.items():
            pct = count / len(df_1h) * 100
            name = {0: 'Flat', 1: 'Down'}.get(int(label), f'Label {label}')
            print(f"  {name} ({int(label)}): {count} ({pct:.1f}%)")
    else:
        print("  No 'target_long' or 'target_short' column found")
else:
    print("  Processed files not found. Run the pipeline first.")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
