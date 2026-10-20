#!/usr/bin/env python3
"""
RL Hyperparameter Grid Search & Benchmark Runner
Tests multiple reward formulations, action space deadzones, and exploration settings.
Optimized to execute within the GPU/CPU training budget using precomputed embeddings.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add current directory to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "src"))

from rl_environment import TradingRLEnv
from performance import PerformanceMetrics
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def run_grid_search(train_embeddings, val_embeddings, test_embeddings,
                     train_df, val_df, test_df,
                     timesteps_per_config=250000):
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid Search Configurations
    configs = [
        {
            "name": "AsymRR_dz0.10_ent0.01",
            "reward_mode": "asymmetric_rr",
            "deadzone": 0.10,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        },
        {
            "name": "AsymRR_dz0.15_ent0.02",
            "reward_mode": "asymmetric_rr",
            "deadzone": 0.15,
            "ent_coef": 0.02,
            "lr": 3e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        },
        {
            "name": "AsymRR_dz0.08_ent0.01_lr1e4",
            "reward_mode": "asymmetric_rr",
            "deadzone": 0.08,
            "ent_coef": 0.01,
            "lr": 1e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        },
        {
            "name": "DSR_dz0.10_ent0.01",
            "reward_mode": "dsr",
            "deadzone": 0.10,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        },
        {
            "name": "DSR_dz0.15_ent0.02",
            "reward_mode": "dsr",
            "deadzone": 0.15,
            "ent_coef": 0.02,
            "lr": 3e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        },
        {
            "name": "DensePnL_dz0.10_ent0.01",
            "reward_mode": "pnl_dense",
            "deadzone": 0.10,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.005,
            "max_risk_multiplier": 5.0
        }
    ]
    
    print("\n" + "="*70)
    print(f"STARTING RL HYPERPARAMETER GRID SEARCH ({len(configs)} CONFIGURATIONS)")
    print(f"Training Budget: {timesteps_per_config:,} timesteps per configuration")
    print("="*70)
    
    leaderboard = []
    best_sharpe = -float('inf')
    champion_config = None
    champion_results_df = None
    
    for idx, cfg in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Evaluating Config: {cfg['name']}")
        print(f"  -> Reward Mode: {cfg['reward_mode']} | Deadzone: {cfg['deadzone']} | Ent Coef: {cfg['ent_coef']} | LR: {cfg['lr']}")
        
        train_env = TradingRLEnv(
            train_embeddings, train_df,
            deadzone=cfg['deadzone'],
            reward_mode=cfg['reward_mode'],
            flat_penalty=cfg['flat_penalty'],
            max_risk_multiplier=cfg['max_risk_multiplier'],
            is_eval=False
        )
        train_env = Monitor(train_env)
        
        # PPO Agent Initialization
        agent = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg['lr'],
            ent_coef=cfg['ent_coef'],
            gamma=0.99,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            clip_range=0.2,
            verbose=0,
            device="cpu"
        )
        
        # Train PPO Agent
        agent.learn(total_timesteps=timesteps_per_config)
        
        # Save config checkpoint
        model_save_path = models_dir / f"ppo_{cfg['name']}.zip"
        agent.save(str(model_save_path))
        
        # Run Out-Of-Sample Backtest on Held-Out Test Set
        test_env = TradingRLEnv(
            test_embeddings, test_df,
            deadzone=cfg['deadzone'],
            reward_mode=cfg['reward_mode'],
            flat_penalty=cfg['flat_penalty'],
            max_risk_multiplier=cfg['max_risk_multiplier'],
            is_eval=True
        )
        
        obs, _ = test_env.reset()
        done = False
        test_records = []
        
        while not done:
            step_idx = test_env.current_step
            timestamp = test_env.df.index[step_idx]
            close_price = test_env.close_prices[step_idx]
            
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            test_records.append({
                'timestamp': timestamp,
                'price': close_price,
                'position': test_env.position,
                'portfolio_value': test_env.portfolio_value
            })
            
        test_results = pd.DataFrame(test_records)
        metrics = PerformanceMetrics.calculate_metrics(test_results, initial_capital=10000)
        
        total_trades = test_env.total_trades
        win_rate = metrics.get('Win Rate (%)', 0.0)
        total_return = metrics.get('Total Return (%)', 0.0)
        sharpe = metrics.get('Sharpe Ratio', 0.0)
        max_dd = metrics.get('Max Drawdown (%)', 0.0)
        profit_factor = metrics.get('Profit Factor', 0.0)
        
        print(f"  [RESULT] Trades: {total_trades} | Return: {total_return:.2f}% | Sharpe: {sharpe:.2f} | WinRate: {win_rate:.1f}% | MaxDD: {max_dd:.2f}%")
        
        leaderboard.append({
            "Configuration": cfg['name'],
            "Reward Mode": cfg['reward_mode'],
            "Deadzone": cfg['deadzone'],
            "Ent Coef": cfg['ent_coef'],
            "Total Trades": total_trades,
            "Total Return (%)": total_return,
            "Sharpe Ratio": sharpe,
            "Win Rate (%)": win_rate,
            "Max Drawdown (%)": max_dd,
            "Profit Factor": profit_factor
        })
        
        # Select champion model (favoring positive trades and superior Sharpe)
        if total_trades > 0 and sharpe > best_sharpe:
            best_sharpe = sharpe
            champion_config = cfg
            champion_results_df = test_results
            # Save as official best model
            agent.save(str(models_dir / "best_model.zip"))
            print(f"  *** NEW CHAMPION MODEL: {cfg['name']} (Sharpe: {sharpe:.2f}) ***")
            
    # Compile Leaderboard
    df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="Sharpe Ratio", ascending=False)
    csv_path = results_dir / "rl_search_results.csv"
    df_leaderboard.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("RL HYPERPARAMETER SEARCH LEADERBOARD")
    print("="*80)
    print(df_leaderboard.to_string(index=False))
    print("="*80)
    print(f"Leaderboard saved to {csv_path}")
    
    if champion_config is not None and champion_results_df is not None:
        print(f"\nFinalizing Champion Backtest: {champion_config['name']}")
        viz_path = results_dir / "backtest_results.png"
        PerformanceMetrics.plot_results(champion_results_df, save_path=str(viz_path))
        print(f"Backtest visualization saved to {viz_path}")
    
    return df_leaderboard


if __name__ == "__main__":
    print("Direct execution requires embeddings. Run via main.py --grid-search.")
