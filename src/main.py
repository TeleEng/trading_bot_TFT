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
    
    import datetime, os
    last_edited = datetime.datetime.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Codebase Last Edited: {last_edited}")
    print("="*60)

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 1] Preprocessing Data...")
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
    

    val_idx = int(len(df_1h) * 0.6)
    test_idx = int(len(df_1h) * 0.8)
    
    train_1h = df_1h.iloc[:val_idx]
    val_1h = df_1h.iloc[val_idx - model.input_chunk_length : test_idx]
    test_1h = df_1h.iloc[test_idx - model.input_chunk_length :] 

    print("\n[Phase 3] Supervised Contrastive Training (Long Brain)...")
    model_long = PricePredictor(num_classes=2)
    train_score_l, val_score_l = model_long.train(
        (train_1h, df_4h, df_1d, df_1w),
        (val_1h, df_4h, df_1d, df_1w),
        target_col="target_long",
        epochs=150,
        patience=20
    )
    print(f"Final InfoNCE Loss (Long) - Train: {train_score_l:.4f} | Val F1: {val_score_l:.4f}")

    print("\n[Phase 3.5] Supervised Contrastive Training (Short Brain)...")
    model_short = PricePredictor(num_classes=2)
    train_score_s, val_score_s = model_short.train(
        (train_1h, df_4h, df_1d, df_1w),
        (val_1h, df_4h, df_1d, df_1w),
        target_col="target_short",
        epochs=150,
        patience=20
    )
    print(f"Final InfoNCE Loss (Short) - Train: {train_score_s:.4f} | Val F1: {val_score_s:.4f}")

    model_long.save(str(models_dir / "tft_long.pth"))
    model_short.save(str(models_dir / "tft_short.pth"))

    print("\n[Phase 4] Training RL Agent via PPO...")
    
    # Concatenate embeddings from both brains for the RL Agent
    train_emb_long = model_long.predict_batch_embeddings((train_1h, df_4h, df_1d, df_1w))
    train_emb_short = model_short.predict_batch_embeddings((train_1h, df_4h, df_1d, df_1w))
    train_embeddings = np.concatenate([train_emb_long, train_emb_short], axis=1)
    
    val_emb_long = model_long.predict_batch_embeddings((val_1h, df_4h, df_1d, df_1w))
    val_emb_short = model_short.predict_batch_embeddings((val_1h, df_4h, df_1d, df_1w))
    val_embeddings = np.concatenate([val_emb_long, val_emb_short], axis=1)
    
    # The models must use the same aligned index logic, so we can use either one's get_aligned_df
    train_env = TradingRLEnv(train_embeddings, model_long.get_aligned_df(train_1h, df_4h, df_1d, df_1w))
    val_env = TradingRLEnv(val_embeddings, model_long.get_aligned_df(val_1h, df_4h, df_1d, df_1w))
    
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
    ppo_model.learn(total_timesteps=2000000, callback=eval_callback)
    
    try:
        ppo_model = PPO.load(str(models_dir / "best_model.zip"))
        print("Loaded best RL model from validation.")
    except Exception as e:
        print("Failed to load best model, using current model.")

    print("\n[Phase 5] Running out-of-sample backtest with RL Agent...")
    
    test_emb_long = model_long.predict_batch_embeddings((test_1h, df_4h, df_1d, df_1w))
    test_emb_short = model_short.predict_batch_embeddings((test_1h, df_4h, df_1d, df_1w))
    test_embeddings = np.concatenate([test_emb_long, test_emb_short], axis=1)
    
    test_env = TradingRLEnv(test_embeddings, model_long.get_aligned_df(test_1h, df_4h, df_1d, df_1w))
    
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