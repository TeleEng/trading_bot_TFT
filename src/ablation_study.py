import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

from model import PricePredictor

# --- Setup Paths & Config ---
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed"
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")

def load_data():
    """Loads and splits the preprocessed data."""
    print("Loading datasets...")
    
    df_1h = pd.read_csv(DATA_PATH / f"{MAIN_ASSET}_master_1h.csv", index_col=0, parse_dates=True)
    df_4h = pd.read_csv(DATA_PATH / f"{MAIN_ASSET}_master_4h.csv", index_col=0, parse_dates=True)
    df_1d = pd.read_csv(DATA_PATH / f"{MAIN_ASSET}_master_1d.csv", index_col=0, parse_dates=True)
    df_1w = pd.read_csv(DATA_PATH / f"{MAIN_ASSET}_master_1w.csv", index_col=0, parse_dates=True)
    
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
    
    results_list = []
    epochs_per_test = 20 # Increased epochs for better convergence testing
    
    print("==================================================")
    print("AUGMENTATION ABLATION STUDY")
    print(f"Testing {len(configs)} configurations for {epochs_per_test} epochs each.")
    print("Testing both LONG and SHORT brains.")
    print("==================================================")
    
    for name, active_augs in configs.items():
        print(f"\n--- Testing Configuration: {name} ---")
        if active_augs is not None:
            print(f"Active Augs: {active_augs}")
        else:
            print("Active Augs: ALL")
            
        total_loss = 0
        total_f1 = 0
        
        for target in ["target_long", "target_short"]:
            print(f"  -> Training {target.upper()} Brain...")
            model = PricePredictor(num_classes=2)
            
            # Train Brain
            train_loss, val_f1 = model.train(
                train_dfs,
                val_dfs,
                target_col=target,
                epochs=epochs_per_test,
                patience=epochs_per_test, # No early stopping
                active_augs=active_augs
            )
            
            total_loss += train_loss
            total_f1 += val_f1
            
        # Average the metrics
        avg_loss = total_loss / 2.0
        avg_f1 = total_f1 / 2.0
        
        results_list.append({
            "Configuration": name,
            "Avg Train InfoNCE Loss": avg_loss,
            "Avg Peak Val F1": avg_f1
        })
        
    print("\n==================================================")
    print("ABLATION STUDY RESULTS (Averaged Long/Short)")
    print("==================================================")
    print(f"{'Configuration':<35} | {'Avg Train Loss':<14} | {'Avg Val F1':<12}")
    print("-" * 65)
    
    # Save to CSV
    results_df = pd.DataFrame(results_list)
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    for row in results_list:
        print(f"{row['Configuration']:<35} | {row['Avg Train InfoNCE Loss']:<14.4f} | {row['Avg Peak Val F1']:<12.4f}")
    
    print("==================================================")
    print(f"Results successfully saved to {csv_path}")

if __name__ == "__main__":
    run_ablation()
