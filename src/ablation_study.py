import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

from model import PricePredictor

# --- Setup Paths & Config ---
load_dotenv()
DATA_PATH = Path("../data/processed")
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")

def load_data():
    """Loads and splits the preprocessed data."""
    print("Loading datasets...")
    # Adjust path if script is run from src/
    data_dir = DATA_PATH if DATA_PATH.exists() else Path("data/processed")
    
    df_1h = pd.read_csv(data_dir / f"{MAIN_ASSET}_master_1h.csv", index_col=0, parse_dates=True)
    df_4h = pd.read_csv(data_dir / f"{MAIN_ASSET}_master_4h.csv", index_col=0, parse_dates=True)
    df_1d = pd.read_csv(data_dir / f"{MAIN_ASSET}_master_1d.csv", index_col=0, parse_dates=True)
    df_1w = pd.read_csv(data_dir / f"{MAIN_ASSET}_master_1w.csv", index_col=0, parse_dates=True)
    
    # 60/20/20 split as defined in main.py
    n = len(df_1h)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_1h = df_1h.iloc[:train_end]
    val_1h = df_1h.iloc[train_end:val_end]
    
    return (train_1h, df_4h, df_1d, df_1w), (val_1h, df_4h, df_1d, df_1w)

def run_ablation():
    train_dfs, val_dfs = load_data()
    
    configs = {
        "Baseline (No Augs)": [],
        "Jitter Only": ['jitter'],
        "Block Masking Only": ['block_masking'],
        "Magnitude Warping Only": ['magnitude_warping'],
        "Random Smoothing Only": ['random_smoothing'],
        "Original Augs": ['jitter', 'scaling', 'skew', 'time_step_masking', 'feature_masking'],
        "New Augs Only": ['block_masking', 'magnitude_warping', 'random_smoothing'],
        "All Augmentations": None # None means all available
    }
    
    results = {}
    epochs_per_test = 5 # Short run to test convergence speed and representation quality
    
    print("==================================================")
    print("AUGMENTATION ABLATION STUDY")
    print(f"Testing {len(configs)} configurations for {epochs_per_test} epochs each.")
    print("==================================================")
    
    for name, active_augs in configs.items():
        print(f"\n--- Testing Configuration: {name} ---")
        if active_augs is not None:
            print(f"Active Augs: {active_augs}")
        else:
            print("Active Augs: ALL")
            
        model = PricePredictor(num_classes=2)
        
        # Train Long Brain
        train_loss, val_f1 = model.train(
            train_dfs,
            val_dfs,
            target_col="target_long",
            epochs=epochs_per_test,
            patience=epochs_per_test, # No early stopping
            active_augs=active_augs
        )
        
        results[name] = {
            "Train InfoNCE Loss": train_loss,
            "Peak Val F1": val_f1
        }
        
    print("\n==================================================")
    print("ABLATION STUDY RESULTS")
    print("==================================================")
    print(f"{'Configuration':<35} | {'Train Loss':<12} | {'Val F1 Score':<12}")
    print("-" * 65)
    for name, metrics in results.items():
        loss = metrics["Train InfoNCE Loss"]
        f1 = metrics["Peak Val F1"]
        print(f"{name:<35} | {loss:<12.4f} | {f1:<12.4f}")
    print("==================================================")

if __name__ == "__main__":
    run_ablation()
