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

import random
import torch
from data_downloader import download_histdata, TICKERS, START_YEAR, END_YEAR, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor, DROP_COLS
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics
from rl_environment import TradingRLEnv
from rl_grid_search import run_grid_search
from baselines import run_deterministic_baseline, run_random_baseline

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


def seed_everything(seed=42):
    """Sets global seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_probability_calibration(df, probabilities, split_name="Test Split"):
    """Evaluates probability distributions and empirical edge lift against ground truth."""
    if probabilities is None or len(probabilities) == 0:
        return
    p_long = probabilities[:, 0]
    p_short = probabilities[:, 1]
    edge = p_long - p_short
    
    print("\n" + "="*70)
    print(f"BACKBONE PROBABILITY & CALIBRATION REPORT [{split_name.upper()}]")
    print("="*70)
    print(f"  p_long  distribution -> Mean: {np.mean(p_long):.4f} | Std: {np.std(p_long):.4f} | Quantiles (10/50/90%): {np.percentile(p_long, 10):.4f} / {np.percentile(p_long, 50):.4f} / {np.percentile(p_long, 90):.4f}")
    print(f"  p_short distribution -> Mean: {np.mean(p_short):.4f} | Std: {np.std(p_short):.4f} | Quantiles (10/50/90%): {np.percentile(p_short, 10):.4f} / {np.percentile(p_short, 50):.4f} / {np.percentile(p_short, 90):.4f}")
    print(f"  edge (p_l - p_s)     -> Mean: {np.mean(edge):+.4f} | Std: {np.std(edge):.4f} | Range: [{np.min(edge):+.4f}, {np.max(edge):+.4f}]")
    
    target_long = df['target_long'].values if 'target_long' in df.columns else None
    target_short = df['target_short'].values if 'target_short' in df.columns else None
    
    if target_long is not None and len(target_long) == len(probabilities):
        base_long_rate = (target_long == 1).mean() * 100.0
        print(f"\n  [LONG EDGE LIFT vs Base Rate {base_long_rate:.1f}%]")
        for thresh in [0.00, 0.01, 0.02, 0.04]:
            mask = (edge >= thresh)
            count = int(np.sum(mask))
            pct_bars = (count / len(edge)) * 100.0
            if count > 0:
                hit_rate = (target_long[mask] == 1).mean() * 100.0
                lift = hit_rate - base_long_rate
                print(f"    Edge >= {thresh:+.2f} | Bars: {count:,} ({pct_bars:4.1f}%) | Actual Hit Rate: {hit_rate:4.1f}% (Lift: {lift:+4.1f}%)")
            else:
                print(f"    Edge >= {thresh:+.2f} | Bars: 0 (0.0%) | Actual Hit Rate: N/A")
                
    if target_short is not None and len(target_short) == len(probabilities):
        base_short_rate = (target_short == 1).mean() * 100.0
        print(f"\n  [SHORT EDGE LIFT vs Base Rate {base_short_rate:.1f}%]")
        for thresh in [0.00, 0.01, 0.02, 0.04]:
            mask = (-edge >= thresh)
            count = int(np.sum(mask))
            pct_bars = (count / len(edge)) * 100.0
            if count > 0:
                hit_rate = (target_short[mask] == 1).mean() * 100.0
                lift = hit_rate - base_short_rate
                print(f"    Edge >= {thresh:+.2f} | Bars: {count:,} ({pct_bars:4.1f}%) | Actual Hit Rate: {hit_rate:4.1f}% (Lift: {lift:+4.1f}%)")
            else:
                print(f"    Edge >= {thresh:+.2f} | Bars: 0 (0.0%) | Actual Hit Rate: N/A")
    print("="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Trading Bot TFT + RL Pipeline")
    parser.add_argument("--grid-search", action="store_true", help="Run comprehensive RL hyperparameter grid search")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining of TFT backbones even if weights/embeddings exist")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total timesteps for RL training")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(seed=42)
    
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
        
        if not file_1h.exists() or args.force_retrain:
            print("\n[Phase 1] Processing MTF data with 3.0x/2.0x ATR Triple Barrier labels (96-bar horizon)...")
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
    max_wait = 96
    test_idx = len(df_1h) - one_year_bars
    val_idx = test_idx - one_year_bars
    
    # Strictly purged split boundaries (purging 96 bars = max_wait label lookahead horizon)
    train_1h = df_1h.iloc[: val_idx - max_wait]
    val_1h = df_1h.iloc[val_idx - model_template.input_chunk_length : test_idx - max_wait]
    test_1h = df_1h.iloc[test_idx - model_template.input_chunk_length :]

    train_end = train_1h.index[-1]
    val_end = val_1h.index[-1]

    # Strictly causal higher-timeframe slicing to prevent scaler leakage
    train_dfs = (train_1h, df_4h.loc[:train_end], df_1d.loc[:train_end], df_1w.loc[:train_end])
    val_dfs = (val_1h, df_4h.loc[:val_end], df_1d.loc[:val_end], df_1w.loc[:val_end])
    test_dfs = (test_1h, df_4h, df_1d, df_1w)
    
    train_df_aligned = model_template.get_aligned_df(train_1h, df_4h.loc[:train_end], df_1d.loc[:train_end], df_1w.loc[:train_end])
    val_df_aligned = model_template.get_aligned_df(val_1h, df_4h.loc[:val_end], df_1d.loc[:val_end], df_1w.loc[:val_end])
    test_df_aligned = model_template.get_aligned_df(test_1h, df_4h, df_1d, df_1w)

    # Embedding & Probability Cache Paths
    train_emb_path = models_dir / "train_embeddings.npy"
    val_emb_path = models_dir / "val_embeddings.npy"
    test_emb_path = models_dir / "test_embeddings.npy"
    train_prob_path = models_dir / "train_probabilities.npy"
    val_prob_path = models_dir / "val_probabilities.npy"
    test_prob_path = models_dir / "test_probabilities.npy"
    train_ev_prob_path = models_dir / "train_ev_probabilities.npy"
    val_ev_prob_path = models_dir / "val_ev_probabilities.npy"
    test_ev_prob_path = models_dir / "test_ev_probabilities.npy"
    
    embeddings_cached = all(p.exists() for p in [
        train_emb_path, val_emb_path, test_emb_path,
        train_prob_path, val_prob_path, test_prob_path,
        train_ev_prob_path, val_ev_prob_path, test_ev_prob_path,
    ])
    
    if embeddings_cached and not args.force_retrain:
        print("\n[INFO] Found cached embeddings and probabilities on disk! Loading...")
        train_embeddings = np.load(train_emb_path)
        val_embeddings = np.load(val_emb_path)
        test_embeddings = np.load(test_emb_path)
        train_probabilities = np.load(train_prob_path)
        val_probabilities = np.load(val_prob_path)
        test_probabilities = np.load(test_prob_path)
        train_ev_probabilities = np.load(train_ev_prob_path)
        val_ev_probabilities = np.load(val_ev_prob_path)
        test_ev_probabilities = np.load(test_ev_prob_path)
        print(f"  Loaded Train Embeddings: {train_embeddings.shape}, Probs: {train_probabilities.shape}")
        print(f"  Loaded Val Embeddings:   {val_embeddings.shape}, Probs: {val_probabilities.shape}")
        print(f"  Loaded Test Embeddings:  {test_embeddings.shape}, Probs: {test_probabilities.shape}")
    else:
        long_model_path = models_dir / "tft_long.pth"
        short_model_path = models_dir / "tft_short.pth"
        
        model_long = PricePredictor(num_classes=2)
        model_short = PricePredictor(num_classes=2)
        
        # Check if saved model checkpoints exist
        if long_model_path.exists() and short_model_path.exists() and not args.force_retrain:
            print("\n[INFO] Found saved TFT model checkpoints! Loading weights...")
            input_size = train_1h.drop(columns=DROP_COLS, errors='ignore').shape[1]
            model_long.load(str(long_model_path), input_size=input_size)
            model_short.load(str(short_model_path), input_size=input_size)
        else:
            print("\n[Phase 3] Supervised MTF-TFT Training (Long Brain)...")
            train_score_l, val_score_l = model_long.train(
                train_dfs,
                val_dfs,
                target_col="target_long",
                epochs=50,
                patience=10
            )
            print(f"Long Brain Trained - Train Loss: {train_score_l:.4f} | Best Val AUC: {val_score_l:.4f}")

            print("\n[Phase 3.5] Supervised MTF-TFT Training (Short Brain)...")
            train_score_s, val_score_s = model_short.train(
                train_dfs,
                val_dfs,
                target_col="target_short",
                epochs=50,
                patience=10
            )
            print(f"Short Brain Trained - Train Loss: {train_score_s:.4f} | Best Val AUC: {val_score_s:.4f}")

            print("\n[Phase 3.6] Fitting Probability Calibrators on Validation Split...")
            model_long.fit_calibrator(val_dfs, target_col="target_long")
            model_short.fit_calibrator(val_dfs, target_col="target_short")
            model_long.save(str(long_model_path))
            model_short.save(str(short_model_path))

        print("\n[Phase 3.8] Generating & Caching 128-d Multi-Timeframe Embeddings & Calibrated Probabilities...")
        train_emb_l = model_long.predict_batch_embeddings(train_dfs)
        train_emb_s = model_short.predict_batch_embeddings(train_dfs)
        train_embeddings = np.concatenate([train_emb_l, train_emb_s], axis=1)

        val_emb_l = model_long.predict_batch_embeddings(val_dfs)
        val_emb_s = model_short.predict_batch_embeddings(val_dfs)
        val_embeddings = np.concatenate([val_emb_l, val_emb_s], axis=1)

        test_emb_l = model_long.predict_batch_embeddings(test_dfs)
        test_emb_s = model_short.predict_batch_embeddings(test_dfs)
        test_embeddings = np.concatenate([test_emb_l, test_emb_s], axis=1)

        # Balanced (prior-corrected, 0.50 == validation base rate) -> PPO observations + confidence/edge gates
        # True calibrated P(TP hit)                                 -> EV-in-R gate only
        def _calibrated_pair(dfs, balanced):
            p_l = model_long.predict_batch_calibrated(dfs, balanced=balanced)[:, 1]
            p_s = model_short.predict_batch_calibrated(dfs, balanced=balanced)[:, 1]
            return np.column_stack([p_l, p_s])

        train_probabilities = _calibrated_pair(train_dfs, balanced=True)
        val_probabilities = _calibrated_pair(val_dfs, balanced=True)
        test_probabilities = _calibrated_pair(test_dfs, balanced=True)
        train_ev_probabilities = _calibrated_pair(train_dfs, balanced=False)
        val_ev_probabilities = _calibrated_pair(val_dfs, balanced=False)
        test_ev_probabilities = _calibrated_pair(test_dfs, balanced=False)

        np.save(train_emb_path, train_embeddings)
        np.save(val_emb_path, val_embeddings)
        np.save(test_emb_path, test_embeddings)
        np.save(train_prob_path, train_probabilities)
        np.save(val_prob_path, val_probabilities)
        np.save(test_prob_path, test_probabilities)
        np.save(train_ev_prob_path, train_ev_probabilities)
        np.save(val_ev_prob_path, val_ev_probabilities)
        np.save(test_ev_prob_path, test_ev_probabilities)
        print("Embeddings & probabilities successfully cached to disk!")

    # Evaluate Backbone Probability Distribution & Empirical Edge Calibration
    if train_embeddings.shape[1] > 16:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=16, random_state=42)
        train_embeddings = pca.fit_transform(train_embeddings).astype(np.float32)
        val_embeddings = pca.transform(val_embeddings).astype(np.float32)
        test_embeddings = pca.transform(test_embeddings).astype(np.float32)
        print(f"  [PCA COMPRESSION] Compressed backbone embeddings to 16-d (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    evaluate_probability_calibration(val_df_aligned, val_probabilities, split_name="Validation Split (Balanced Probs)"); evaluate_probability_calibration(test_df_aligned, test_probabilities, split_name="Held-Out Test Split (Balanced Probs)")
    print(f"  [TRUE CALIBRATED P(TP)] Test mean p_long: {np.mean(test_ev_probabilities[:, 0]):.4f} | "
          f"mean p_short: {np.mean(test_ev_probabilities[:, 1]):.4f} | "
          f"bars with p > 0.40 (3:2 EV breakeven) -> long: {(test_ev_probabilities[:, 0] > 0.40).mean()*100:.1f}% | "
          f"short: {(test_ev_probabilities[:, 1] > 0.40).mean()*100:.1f}%")

    # Check if user requested Grid Search
    if args.grid_search:
        print("\n[Phase 4] Executing Comprehensive RL Grid Search...")
        run_grid_search(
            train_embeddings, val_embeddings, test_embeddings,
            train_df_aligned, val_df_aligned, test_df_aligned,
            train_probs=train_probabilities,
            val_probs=val_probabilities,
            test_probs=test_probabilities,
            timesteps_per_config=args.timesteps,
            train_ev_probs=train_ev_probabilities,
            val_ev_probs=val_ev_probabilities,
            test_ev_probs=test_ev_probabilities
        )
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETE! Results saved to results/rl_val_search_results.csv & results/rl_test_champion.csv")
        print("="*60)
        return

    # Standard Execution: Train Champion PPO Configuration
    print("\n[Phase 4] Training Champion RL Agent (PPO with 3.0x/2.0x ATR R:R, Backbone Gating & Net-of-Cost Rewards)...")
    
    train_env = TradingRLEnv(
        train_embeddings, train_df_aligned,
        probabilities=train_probabilities,
        ev_probabilities=train_ev_probabilities,
        long_tp_mult=3.0, long_sl_mult=2.0,
        short_tp_mult=3.0, short_sl_mult=2.0,
        deadzone=0.25,
        min_confidence=0.50,
        min_edge=0.01,
        use_ev_gate=False,  # EV gate uses TRUE calibrated probs; enable once P(TP) > 0.40 coverage is validated
        min_ev_r=0.0,
        breakeven_sl=False,
        be_trigger_mult=1.2,
        volatility_filter=True,
        reward_mode="net_rr",
        flat_penalty=0.0,
        max_risk_multiplier=1.5,
        cooldown_bars=4,
        trend_filter=True,
        max_wait_bars=96,
        is_eval=False
    )
    val_env = TradingRLEnv(
        val_embeddings, val_df_aligned,
        probabilities=val_probabilities,
        ev_probabilities=val_ev_probabilities,
        long_tp_mult=3.0, long_sl_mult=2.0,
        short_tp_mult=3.0, short_sl_mult=2.0,
        deadzone=0.25,
        min_confidence=0.50,
        min_edge=0.01,
        use_ev_gate=False,  # EV gate uses TRUE calibrated probs; enable once P(TP) > 0.40 coverage is validated
        min_ev_r=0.0,
        breakeven_sl=False,
        be_trigger_mult=1.2,
        volatility_filter=True,
        reward_mode="net_rr",
        flat_penalty=0.0,
        max_risk_multiplier=1.5,
        cooldown_bars=4,
        trend_filter=True,
        max_wait_bars=96,
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
    
    rl_device = "cpu"
    print(f"PPO Policy training device: {rl_device}")

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
        device=rl_device
    )
    
    print(f"Training PPO for {args.timesteps:,} steps...")
    ppo_model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    
    try:
        best_zip = models_dir / "best_model.zip"
        if best_zip.exists():
            ppo_model = PPO.load(str(best_zip), device="cpu")
            print("Successfully loaded best RL model from validation checkpoint.")
    except Exception as e:
        print(f"Using current model ({e}).")

    print("\n[Phase 5] Running Out-Of-Sample Backtest with RL Agent...")
    test_env = TradingRLEnv(
        test_embeddings, test_df_aligned,
        probabilities=test_probabilities,
        ev_probabilities=test_ev_probabilities,
        long_tp_mult=3.0, long_sl_mult=2.0,
        short_tp_mult=3.0, short_sl_mult=2.0,
        deadzone=0.25,
        min_confidence=0.50,
        min_edge=0.01,
        use_ev_gate=False,  # EV gate uses TRUE calibrated probs; enable once P(TP) > 0.40 coverage is validated
        min_ev_r=0.0,
        breakeven_sl=False,
        be_trigger_mult=1.2,
        volatility_filter=True,
        reward_mode="net_rr",
        flat_penalty=0.0,
        max_risk_multiplier=1.5,
        cooldown_bars=4,
        trend_filter=True,
        max_wait_bars=96,
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
    print("\n" + "="*50)
    print("CHAMPION PPO AGENT PERFORMANCE (HELD-OUT TEST)")
    print("="*50)
    PerformanceMetrics.print_report(metrics)
    test_env.print_funnel_report()
    test_env.print_exit_and_cost_report()

    print("\nGenerating performance visualization...")
    viz_path = results_dir / "backtest_results.png"
    PerformanceMetrics.plot_results(backtest_results, save_path=str(viz_path))
    print(f"Visualization saved to {viz_path}")

    # Phase 6: Benchmark against Deterministic Rule Baseline
    print("\n[Phase 6] Running Deterministic Rule-Based Baseline Benchmark...")
    run_deterministic_baseline(
        test_embeddings, test_df_aligned,
        probabilities=test_probabilities,
        ev_probabilities=test_ev_probabilities,
        be_trigger_mult=1.2,
        split_name="Held-Out Test Split",
        use_ev=False
    )
    run_deterministic_baseline(
        test_embeddings, test_df_aligned,
        probabilities=test_probabilities,
        ev_probabilities=test_ev_probabilities,
        be_trigger_mult=1.2,
        split_name="Held-Out Test Split",
        use_ev=True,
        min_ev_r=0.0
    )

    print("\n" + "="*60)
    run_random_baseline(test_embeddings, test_df_aligned, breakeven_sl=False, be_trigger_mult=1.2, split_name="Held-Out Test Split")
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
