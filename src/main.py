#!/usr/bin/env python3
"""
Trading Bot - End-to-end pipeline:
Feature Engineering -> Supervised Contrastive Multi-Timeframe Embedding Caching -> RL Policy Optimization (PPO) -> Out-Of-Sample Backtesting
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add src to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "src"))

from data_downloader import download_histdata, TICKERS, START_YEAR, END_YEAR, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics
from rl_environment import TradingRLEnv
from rl_grid_search import run_grid_search

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser(description="Trading Bot TFT + RL Pipeline")
    parser.add_argument("--grid-search", action="store_true", help="Run comprehensive RL hyperparameter grid search")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of TFT backbones even if weights/embeddings exist")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total timesteps for RL training")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("TRADING BOT - DUAL-BRAIN TFT + PPO PIPELINE")
    
    import datetime
    last_edited = datetime.datetime.fromtimestamp(os.path.getmtime(__file__)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Codebase Last Edited: {last_edited}")
    print(f"Mode: {'RL Grid Search' if args.grid_search else 'Standard Training & Backtest'}")
    print("="*60)

    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")
    
    # Path to preprocessed data
    try:
        file_1h = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1h.csv"
        file_4h = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_4h.csv"
        file_1d = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1d.csv"
        file_1w = Path(OUTPUT_PATH) / f"{MAIN_ASSET}_master_1w.csv"
        
        if not file_1h.exists():
            print("\n[Phase 1] Preprocessed data not found. Running Preprocessing...")
            process_all_data()
            
        df_1h = pd.read_csv(file_1h, index_col=0, parse_dates=True)
        df_4h = pd.read_csv(file_4h, index_col=0, parse_dates=True)
        df_1d = pd.read_csv(file_1d, index_col=0, parse_dates=True)
        df_1w = pd.read_csv(file_1w, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[INFO] Local MTF files not found ({e}). Attempting to load from Kaggle dataset...")
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

    print("\n[Phase 2] Loading MTF Data & Splitting (Train / 1-Year Val / 1-Year Test)...")
    model_template = PricePredictor()
    
    one_year_bars = 252 * 24
    test_idx = len(df_1h) - one_year_bars
    val_idx = test_idx - one_year_bars
    
    train_1h = df_1h.iloc[:val_idx]
    val_1h = df_1h.iloc[val_idx - model_template.input_chunk_length : test_idx]
    test_1h = df_1h.iloc[test_idx - model_template.input_chunk_length :]

    train_dfs = (train_1h, df_4h, df_1d, df_1w)
    val_dfs = (val_1h, df_4h, df_1d, df_1w)
    test_dfs = (test_1h, df_4h, df_1d, df_1w)
    
    train_df_aligned = model_template.get_aligned_df(train_1h, df_4h, df_1d, df_1w)
    val_df_aligned = model_template.get_aligned_df(val_1h, df_4h, df_1d, df_1w)
    test_df_aligned = model_template.get_aligned_df(test_1h, df_4h, df_1d, df_1w)

    # Embedding Cache Paths
    train_emb_path = models_dir / "train_embeddings.npy"
    val_emb_path = models_dir / "val_embeddings.npy"
    test_emb_path = models_dir / "test_embeddings.npy"
    
    embeddings_cached = (
        train_emb_path.exists() and val_emb_path.exists() and test_emb_path.exists()
    )
    
    if embeddings_cached and not args.force_retrain:
        print("\n[INFO] Found cached embeddings on disk! Loading precomputed embeddings...")
        train_embeddings = np.load(train_emb_path)
        val_embeddings = np.load(val_emb_path)
        test_embeddings = np.load(test_emb_path)
        print(f"  Loaded Train Embeddings: {train_embeddings.shape}")
        print(f"  Loaded Val Embeddings:   {val_embeddings.shape}")
        print(f"  Loaded Test Embeddings:  {test_embeddings.shape}")
    else:
        long_model_path = models_dir / "tft_long.pth"
        short_model_path = models_dir / "tft_short.pth"
        
        model_long = PricePredictor(num_classes=2)
        model_short = PricePredictor(num_classes=2)
        
        # Check if saved model checkpoints exist
        if long_model_path.exists() and short_model_path.exists() and not args.force_retrain:
            print("\n[INFO] Found saved TFT model checkpoints! Loading weights...")
            drop_cols = ['target_long', 'target_short', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']
            input_size = train_1h.drop(columns=drop_cols, errors='ignore').shape[1]
            model_long.load(str(long_model_path), input_size=input_size)
            model_short.load(str(short_model_path), input_size=input_size)
        else:
            print("\n[Phase 3] Supervised Contrastive Training (Long Brain)...")
            train_score_l, val_score_l = model_long.train(
                train_dfs,
                val_dfs,
                target_col="target_long",
                epochs=150,
                patience=20
            )
            print(f"Final InfoNCE Loss (Long) - Train: {train_score_l:.4f} | Val F1: {val_score_l:.4f}")
            model_long.save(str(long_model_path))

            print("\n[Phase 3.5] Supervised Contrastive Training (Short Brain)...")
            train_score_s, val_score_s = model_short.train(
                train_dfs,
                val_dfs,
                target_col="target_short",
                epochs=150,
                patience=20
            )
            print(f"Final InfoNCE Loss (Short) - Train: {train_score_s:.4f} | Val F1: {val_score_s:.4f}")
            model_short.save(str(short_model_path))

        print("\n[Phase 3.8] Generating & Caching 256-d Multi-Timeframe Embeddings...")
        train_emb_l = model_long.predict_batch_embeddings(train_dfs)
        train_emb_s = model_short.predict_batch_embeddings(train_dfs)
        train_embeddings = np.concatenate([train_emb_l, train_emb_s], axis=1)

        val_emb_l = model_long.predict_batch_embeddings(val_dfs)
        val_emb_s = model_short.predict_batch_embeddings(val_dfs)
        val_embeddings = np.concatenate([val_emb_l, val_emb_s], axis=1)

        test_emb_l = model_long.predict_batch_embeddings(test_dfs)
        test_emb_s = model_short.predict_batch_embeddings(test_dfs)
        test_embeddings = np.concatenate([test_emb_l, test_emb_s], axis=1)

        np.save(train_emb_path, train_embeddings)
        np.save(val_emb_path, val_embeddings)
        np.save(test_emb_path, test_embeddings)
        print("Embeddings successfully cached to disk!")

    # Check if user requested Grid Search
    if args.grid_search:
        print("\n[Phase 4] Executing Comprehensive RL Grid Search...")
        run_grid_search(
            train_embeddings, val_embeddings, test_embeddings,
            train_df_aligned, val_df_aligned, test_df_aligned,
            timesteps_per_config=args.timesteps
        )
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETE! Results saved to results/rl_search_results.csv")
        print("="*60)
        return

    # Standard Execution: Train Champion PPO Configuration
    print("\n[Phase 4] Training Champion RL Agent (PPO with Active Exploration & Decoupled R:R)...")
    
    train_env = TradingRLEnv(
        train_embeddings, train_df_aligned,
        deadzone=0.10,
        reward_mode="asymmetric_rr",
        flat_penalty=0.005,
        max_risk_multiplier=5.0,
        is_eval=False
    )
    val_env = TradingRLEnv(
        val_embeddings, val_df_aligned,
        deadzone=0.10,
        reward_mode="asymmetric_rr",
        flat_penalty=0.005,
        max_risk_multiplier=5.0,
        is_eval=True
    )
    
    train_env = Monitor(train_env)
    val_env = Monitor(val_env)
    
    eval_callback = EvalCallback(
        val_env, 
        best_model_save_path=str(models_dir),
        log_path=str(models_dir), 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Configure PPO with non-zero entropy coefficient to guarantee exploration
    ppo_model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        clip_range=0.2,
        verbose=0,
        device="cpu"
    )
    
    print(f"Training PPO for {args.timesteps:,} steps...")
    ppo_model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    
    try:
        best_zip = models_dir / "best_model.zip"
        if best_zip.exists():
            ppo_model = PPO.load(str(best_zip))
            print("Successfully loaded best RL model from validation checkpoint.")
    except Exception as e:
        print(f"Using current model ({e}).")

    print("\n[Phase 5] Running Out-Of-Sample Backtest with RL Agent...")
    test_env = TradingRLEnv(
        test_embeddings, test_df_aligned,
        deadzone=0.10,
        reward_mode="asymmetric_rr",
        flat_penalty=0.005,
        max_risk_multiplier=5.0,
        is_eval=True
    )
    
    obs, _ = test_env.reset()
    done = False
    results = []
    
    while not done:
        idx = test_env.current_step
        current_timestamp = test_env.df.index[idx]
        current_price = test_env.close_prices[idx]
        
        action, _states = ppo_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        results.append({
            'timestamp': current_timestamp,
            'price': current_price,
            'position': test_env.position,
            'portfolio_value': test_env.portfolio_value
        })
        
    backtest_results = pd.DataFrame(results)
    
    metrics = PerformanceMetrics.calculate_metrics(
        backtest_results,
        initial_capital=10000
    )
    PerformanceMetrics.print_report(metrics)
    
    print(f"Total Out-Of-Sample Trades Taken: {test_env.total_trades}")
    print(f"Winning Trades: {test_env.winning_trades} | Losing Trades: {test_env.losing_trades} | Timeouts: {test_env.timeout_trades}")

    print("\nGenerating performance visualization...")
    viz_path = results_dir / "backtest_results.png"
    PerformanceMetrics.plot_results(backtest_results, save_path=str(viz_path))
    print(f"Visualization saved to {viz_path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
