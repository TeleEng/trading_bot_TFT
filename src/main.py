#!/usr/bin/env python3
"""
Trading Bot - End-to-end pipeline: download data -> preprocess -> train model -> backtest
"""
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path

from data_downloader import download_histdata, TICKERS, START_YEAR, END_YEAR, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics
from viz import plot_tsne_and_confusion_matrix

from sklearn.cluster import AgglomerativeClustering

from rl_environment import TradingRLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.metrics import confusion_matrix, classification_report

def main():
    print("="*60)
    print("TRADING BOT - FULL PIPELINE (SELF-SUPERVISED MTF)")
    print("="*60)

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 1] Downloading & Preprocessing Data...")
    # download_histdata(TICKERS, START_YEAR, END_YEAR, DATA_PATH)
    process_all_data()

    print("\n[Phase 2] Loading MTF Data & Splitting (60/20/20)...")
    model = PricePredictor()

    load_dotenv()
    MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")
    
    try:
        file_1h = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1h.csv"
        file_4h = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_4h.csv"
        file_1d = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1d.csv"
        file_1w = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1w.csv"
        
        df_1h = pd.read_csv(file_1h, index_col=0, parse_dates=True)
        df_4h = pd.read_csv(file_4h, index_col=0, parse_dates=True)
        df_1d = pd.read_csv(file_1d, index_col=0, parse_dates=True)
        df_1w = pd.read_csv(file_1w, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("[INFO] Local MTF files not found. Attempting to load from Kaggle dataset...")
        kaggle_base = Path("/kaggle/input/datasets/infernalss/tft-pipeline-dataset/trading_bot_TFT/data/processed")
        if not kaggle_base.exists():
            kaggle_base = Path("/kaggle/input/tft-pipeline-dataset/trading_bot_TFT/data/processed")
            
        file_1h = kaggle_base / f"{MAIN_ASSET}_master_1h.csv"
        file_4h = kaggle_base / f"{MAIN_ASSET}_master_4h.csv"
        file_1d = kaggle_base / f"{MAIN_ASSET}_master_1d.csv"
        file_1w = kaggle_base / f"{MAIN_ASSET}_master_1w.csv"
        
        df_1h = pd.read_csv(file_1h, index_col=0, parse_dates=True)
        df_4h = pd.read_csv(file_4h, index_col=0, parse_dates=True)
        df_1d = pd.read_csv(file_1d, index_col=0, parse_dates=True)
        df_1w = pd.read_csv(file_1w, index_col=0, parse_dates=True)
    
    # Temporarily limit data for local run
    df_1h = df_1h.iloc[-int(len(df_1h)*0.25):]
    df_4h = df_4h[df_4h.index >= df_1h.index[0]]
    df_1d = df_1d[df_1d.index >= df_1h.index[0]]
    df_1w = df_1w[df_1w.index >= df_1h.index[0]]

    val_idx = int(len(df_1h) * 0.6)
    test_idx = int(len(df_1h) * 0.8)
    
    train_1h = df_1h.iloc[:val_idx]
    val_1h = df_1h.iloc[val_idx - model.input_chunk_length : test_idx]
    test_1h = df_1h.iloc[test_idx - model.input_chunk_length :] 

    print("\n[Phase 3] Supervised Contrastive Training (MTF)...")
    train_score, val_score = model.train(
        (train_1h, df_4h, df_1d, df_1w),
        (val_1h, df_4h, df_1d, df_1w),
        epochs=50,
        patience=20
    )
    print(f"Final InfoNCE Loss - Train: {train_score:.4f} | Val: {val_score:.4f}")

    print("\n[Phase 3.5] Building Cluster Voting Map via Agglomerative Clustering...")
    W = model.get_over_cluster_weights() # (6, hidden_size)
    agg = AgglomerativeClustering(n_clusters=3)
    macro_labels = agg.fit_predict(W) # (6,)
    
    # Get raw 6-dim predictions AND aligned labels from create_sequences
    train_result = model.create_sequences(train_1h, df_4h, df_1d, df_1w, clean_noise=True)
    X1_t, X4_t, X1d_t, X1w_t, y_train = train_result[0], train_result[1], train_result[2], train_result[3], train_result[4]
    
    # Run batch prediction on training data (pass full DataFrames, including target)
    train_c_probs = model.predict_batch((train_1h, df_4h, df_1d, df_1w)) # (N, 6)
    train_micro_preds = train_c_probs.argmax(axis=1) # (N,)
    train_macro_preds = np.array([macro_labels[m] for m in train_micro_preds]) # (N,)
    
    # y_train is already aligned from create_sequences
    print(f"  train_c_probs shape: {train_c_probs.shape}, y_train shape: {y_train.shape}")
    print(f"  Label distribution in y_train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Calculate baseline distributions
    total_samples = len(y_train)
    baseline_probs = {c: (y_train == c).sum() / total_samples for c in [0, 1, 2]}
    
    macro_to_target = {}
    for macro_idx in range(3):
        idx = (train_macro_preds == macro_idx)
        if idx.sum() > 0:
            cluster_labels = y_train[idx]
            cluster_size = len(cluster_labels)
            
            best_class = 0
            best_relative_density = -1
            
            for c in [0, 1, 2]:
                class_count = (cluster_labels == c).sum()
                if class_count == 0:
                    continue
                cluster_prob = class_count / cluster_size
                relative_density = cluster_prob / baseline_probs[c]
                
                if relative_density > best_relative_density:
                    best_relative_density = relative_density
                    best_class = c
                    
            macro_to_target[macro_idx] = int(best_class)
        else:
            macro_to_target[macro_idx] = 0
            
    print(f"Macro-Cluster to Triple Barrier Label Mapping: {macro_to_target}")
    
    voting_map = {}
    for micro_idx in range(6):
        voting_map[micro_idx] = macro_to_target[macro_labels[micro_idx]]
        
    model.set_voting_map(voting_map)

    model_path = models_dir / "model.pkl"
    model.save(str(model_path))
    
    # Save the voting map for future inference!
    import json
    with open(models_dir / "voting_map.json", "w") as f:
        json.dump(voting_map, f)

    print("\n[Phase 4] Training RL Agent via PPO...")
    train_embeddings = model.predict_batch_embeddings((train_1h, df_4h, df_1d, df_1w))
    val_embeddings = model.predict_batch_embeddings((val_1h, df_4h, df_1d, df_1w))
    
    train_env = TradingRLEnv(train_embeddings, model.get_aligned_df(train_1h, df_4h, df_1d, df_1w))
    val_env = TradingRLEnv(val_embeddings, model.get_aligned_df(val_1h, df_4h, df_1d, df_1w))
    
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path=str(models_dir),
        log_path=str(models_dir), 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    ppo_model = PPO("MlpPolicy", train_env, verbose=0, device="cpu")
    ppo_model.learn(total_timesteps=100000, callback=eval_callback)
    
    try:
        ppo_model = PPO.load(str(models_dir / "best_model.zip"))
        print("Loaded best RL model from validation.")
    except Exception as e:
        print("Failed to load best model, using current model.")

    print("\n[Phase 5] Running out-of-sample backtest with RL Agent...")
    
    test_embeddings = model.predict_batch_embeddings((test_1h, df_4h, df_1d, df_1w))
    test_env = TradingRLEnv(test_embeddings, model.get_aligned_df(test_1h, df_4h, df_1d, df_1w))
    
    obs, _ = test_env.reset()
    done = False
    
    results = []
    
    while not done:
        idx = test_env.current_step
        current_timestamp = test_env.df.index[idx]
        current_price = test_env.df.iloc[idx]['Close']
        
        action, _states = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        
        results.append({
            'timestamp': current_timestamp,
            'price': current_price,
            'position': test_env.position,
            'portfolio_value': test_env.portfolio_value
        })
        
    backtest_results = pd.DataFrame(results)
    
    # Calculate metrics
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