#!/usr/bin/env python3
"""
RL Hyperparameter Grid Search & Benchmark Runner
Tests balanced 1.5:1 Risk/Reward, zero flat penalty (patience), trend filtering, and cooldown rules.
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
from stable_baselines3.common.callbacks import BaseCallback


class DiagnosticsCallback(BaseCallback):
    def __init__(self, check_freq=50000, total_timesteps=300000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.last_reported_step = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.last_reported_step == 0 or (self.num_timesteps - self.last_reported_step) >= self.check_freq:
            self.last_reported_step = self.num_timesteps
            try:
                rollout = getattr(self.model, "rollout_buffer", None)
                if rollout is not None:
                    # rollout.actions has shape (buffer_size, n_envs, action_dim)
                    actions = rollout.actions.flatten()
                    act_mean = float(np.mean(actions))
                    act_std = float(np.std(actions))
                    
                    returns = rollout.returns.flatten()
                    values = rollout.values.flatten()
                    var_y = np.var(returns)
                    expl_var = float(1.0 - np.var(returns - values) / var_y) if var_y > 1e-8 else 0.0
                    print(f"    [PPO Step {self.num_timesteps:,}/{self.total_timesteps:,}] Action Mean: {act_mean:+.3f} (Std: {act_std:.3f}) | ExplVar: {expl_var:+.2f}")
            except Exception:
                pass


def run_grid_search(train_embeddings, val_embeddings, test_embeddings,
                     train_df, val_df, test_df,
                     train_probs=None, val_probs=None, test_probs=None,
                     timesteps_per_config=300000):
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid Search Configurations centered on the Profitable Champion Architecture
    # Features: Wide 3:2 ATR R:R, Backbone Prob Edge, Break-Even Ratchet (1.2x vs 1.5x), Volatility & Trend Filters
    configs = [
        {
            "name": "NetRR_dz0.25_edge0.01_BE1.2",
            "reward_mode": "net_rr",
            "deadzone": 0.25,
            "min_confidence": 0.50,
            "min_edge": 0.01,
            "breakeven_sl": True,
            "be_trigger_mult": 1.2,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        },
        {
            "name": "NetRR_dz0.25_edge0.02_BE1.2",
            "reward_mode": "net_rr",
            "deadzone": 0.25,
            "min_confidence": 0.50,
            "min_edge": 0.02,
            "breakeven_sl": True,
            "be_trigger_mult": 1.2,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        },
        {
            "name": "NetRR_dz0.20_edge0.01_BE1.2",
            "reward_mode": "net_rr",
            "deadzone": 0.20,
            "min_confidence": 0.50,
            "min_edge": 0.01,
            "breakeven_sl": True,
            "be_trigger_mult": 1.2,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        },
        {
            "name": "NetRR_dz0.20_edge0.02_BE1.2",
            "reward_mode": "net_rr",
            "deadzone": 0.20,
            "min_confidence": 0.50,
            "min_edge": 0.02,
            "breakeven_sl": True,
            "be_trigger_mult": 1.2,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        },
        {
            "name": "NetRR_dz0.25_edge0.01_BE1.5",
            "reward_mode": "net_rr",
            "deadzone": 0.25,
            "min_confidence": 0.50,
            "min_edge": 0.01,
            "breakeven_sl": True,
            "be_trigger_mult": 1.5,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        },
        {
            "name": "DSR_dz0.25_edge0.01_BE1.2",
            "reward_mode": "dsr",
            "deadzone": 0.25,
            "min_confidence": 0.50,
            "min_edge": 0.01,
            "breakeven_sl": True,
            "be_trigger_mult": 1.2,
            "volatility_filter": True,
            "trend_filter": True,
            "cooldown_bars": 4,
            "max_risk_multiplier": 1.5,
            "ent_coef": 0.01,
            "lr": 3e-4,
            "flat_penalty": 0.0,
        }
    ]
    
    print("\n" + "="*70)
    print(f"STARTING RL HYPERPARAMETER GRID SEARCH ({len(configs)} CONFIGURATIONS)")
    print(f"Training Budget: {timesteps_per_config:,} timesteps per configuration")
    print(f"Model Selection Protocol: Best Config Selected Strictly on VALIDATION Split")
    print(f"Held-Out Test Split: Evaluated Once on Champion Model (Zero Leakage)")
    print("="*70)
    
    val_leaderboard = []
    best_val_sharpe = -float('inf')
    champion_config = None
    
    for idx, cfg in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Evaluating Config: {cfg['name']}")
        print(f"  -> Reward: {cfg['reward_mode']} | Deadzone: {cfg['deadzone']} | Edge: {cfg.get('min_edge', 0.01)} | BE: {cfg.get('breakeven_sl', True)} ({cfg.get('be_trigger_mult', 1.2):.1f}x) | Filter: {cfg['trend_filter']} | Cooldown: {cfg['cooldown_bars']}")
        
        train_env = TradingRLEnv(
            train_embeddings, train_df,
            probabilities=train_probs,
            long_tp_mult=3.0, long_sl_mult=2.0,
            short_tp_mult=3.0, short_sl_mult=2.0,
            deadzone=cfg['deadzone'],
            min_confidence=cfg.get('min_confidence', 0.50),
            min_edge=cfg.get('min_edge', 0.01),
            reward_mode=cfg['reward_mode'],
            flat_penalty=cfg['flat_penalty'],
            max_risk_multiplier=cfg['max_risk_multiplier'],
            cooldown_bars=cfg['cooldown_bars'],
            trend_filter=cfg['trend_filter'],
            volatility_filter=cfg.get('volatility_filter', True),
            breakeven_sl=cfg.get('breakeven_sl', True),
            be_trigger_mult=cfg.get('be_trigger_mult', 1.2),
            use_ev_gate=cfg.get('use_ev_gate', False),
            min_ev_r=cfg.get('min_ev_r', 0.0),
            max_wait_bars=96,
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
        
        # Train PPO Agent with Periodic Diagnostic Telemetry
        diag_cb = DiagnosticsCallback(check_freq=50000, total_timesteps=timesteps_per_config)
        agent.learn(total_timesteps=timesteps_per_config, callback=diag_cb)
        
        # Save config checkpoint
        model_save_path = models_dir / f"ppo_{cfg['name']}.zip"
        agent.save(str(model_save_path))
        
        # Evaluate Out-Of-Sample Policy on Validation Split (True Model Selection)
        val_env = TradingRLEnv(
            val_embeddings, val_df,
            probabilities=val_probs,
            long_tp_mult=3.0, long_sl_mult=2.0,
            short_tp_mult=3.0, short_sl_mult=2.0,
            deadzone=cfg['deadzone'],
            min_confidence=cfg.get('min_confidence', 0.50),
            min_edge=cfg.get('min_edge', 0.01),
            reward_mode=cfg['reward_mode'],
            flat_penalty=cfg['flat_penalty'],
            max_risk_multiplier=cfg['max_risk_multiplier'],
            cooldown_bars=cfg['cooldown_bars'],
            trend_filter=cfg['trend_filter'],
            volatility_filter=cfg.get('volatility_filter', True),
            breakeven_sl=cfg.get('breakeven_sl', True),
            be_trigger_mult=cfg.get('be_trigger_mult', 1.2),
            use_ev_gate=cfg.get('use_ev_gate', False),
            min_ev_r=cfg.get('min_ev_r', 0.0),
            max_wait_bars=96,
            is_eval=True
        )
        
        obs, _ = val_env.reset()
        done = False
        val_records = []
        
        while not done:
            step_idx = val_env.current_step
            timestamp = val_env.df.index[step_idx]
            close_price = val_env.close_prices[step_idx]
            
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            
            val_records.append({
                'timestamp': timestamp,
                'price': close_price,
                'position': val_env.position,
                'portfolio_value': val_env.portfolio_value
            })
            
        val_results = pd.DataFrame(val_records)
        metrics = PerformanceMetrics.calculate_metrics(val_results, initial_capital=10000)
        
        total_trades = val_env.total_trades
        win_rate = metrics.get('Win Rate (%)', 0.0)
        total_return = metrics.get('Total Return (%)', 0.0)
        sharpe = metrics.get('Sharpe Ratio', 0.0)
        max_dd = metrics.get('Max Drawdown (%)', 0.0)
        profit_factor = metrics.get('Profit Factor', 0.0)
        
        print(f"  [VAL RESULT] Trades: {total_trades} | Return: {total_return:.2f}% | Sharpe: {sharpe:.2f} | WinRate: {win_rate:.1f}% | MaxDD: {max_dd:.2f}% | PF: {profit_factor:.2f}")
        val_env.print_funnel_report()
        val_env.print_exit_and_cost_report()
        
        val_leaderboard.append({
            "Configuration": cfg['name'],
            "Reward Mode": cfg['reward_mode'],
            "Deadzone": cfg['deadzone'],
            "Min Edge": cfg.get('min_edge', 0.01),
            "BE SL": cfg.get('breakeven_sl', True),
            "BE Mult": cfg.get('be_trigger_mult', 1.2),
            "Filter": cfg['trend_filter'],
            "Val Trades": total_trades,
            "Val Return (%)": total_return,
            "Val Sharpe Ratio": sharpe,
            "Val Win Rate (%)": win_rate,
            "Val Max Drawdown (%)": max_dd,
            "Val Profit Factor": profit_factor
        })
        
        # Select champion model strictly based on Validation Sharpe
        if total_trades > 0 and sharpe > best_val_sharpe:
            best_val_sharpe = sharpe
            champion_config = cfg
            agent.save(str(models_dir / "best_model.zip"))
            print(f"  *** NEW CHAMPION MODEL (Validation): {cfg['name']} (Val Sharpe: {sharpe:.2f}) ***")
            
    # Compile and Save Validation Leaderboard
    df_val_leaderboard = pd.DataFrame(val_leaderboard).sort_values(by="Val Sharpe Ratio", ascending=False)
    csv_path = results_dir / "rl_val_search_results.csv"
    df_val_leaderboard.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("RL VALIDATION SEARCH LEADERBOARD (SELECTION SPLIT)")
    print("="*80)
    print(df_val_leaderboard.to_string(index=False))
    print("="*80)
    print(f"Validation Leaderboard saved to {csv_path}")
    
    # Run Final Single-Shot Evaluation on Untouched Held-Out Test Set
    if champion_config is not None:
        print("\n" + "="*80)
        print(f"SINGLE-SHOT FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SPLIT: {champion_config['name']}")
        print("="*80)
        
        champion_model = PPO.load(str(models_dir / "best_model.zip"))
        
        test_env = TradingRLEnv(
            test_embeddings, test_df,
            probabilities=test_probs,
            long_tp_mult=3.0, long_sl_mult=2.0,
            short_tp_mult=3.0, short_sl_mult=2.0,
            deadzone=champion_config['deadzone'],
            min_confidence=champion_config.get('min_confidence', 0.50),
            min_edge=champion_config.get('min_edge', 0.01),
            reward_mode=champion_config['reward_mode'],
            flat_penalty=champion_config['flat_penalty'],
            max_risk_multiplier=champion_config['max_risk_multiplier'],
            cooldown_bars=champion_config['cooldown_bars'],
            trend_filter=champion_config['trend_filter'],
            volatility_filter=champion_config.get('volatility_filter', True),
            breakeven_sl=champion_config.get('breakeven_sl', True),
            be_trigger_mult=champion_config.get('be_trigger_mult', 1.2),
            use_ev_gate=champion_config.get('use_ev_gate', False),
            min_ev_r=champion_config.get('min_ev_r', 0.0),
            max_wait_bars=96,
            is_eval=True
        )
        
        obs, _ = test_env.reset()
        done = False
        test_records = []
        
        while not done:
            step_idx = test_env.current_step
            timestamp = test_env.df.index[step_idx]
            close_price = test_env.close_prices[step_idx]
            
            action, _ = champion_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            test_records.append({
                'timestamp': timestamp,
                'price': close_price,
                'position': test_env.position,
                'portfolio_value': test_env.portfolio_value
            })
            
        test_results = pd.DataFrame(test_records)
        test_metrics = PerformanceMetrics.calculate_metrics(test_results, initial_capital=10000)
        
        print(f"\n[CHAMPION TEST RESULTS] Config: {champion_config['name']}")
        print(f"  Trades: {test_env.total_trades} | Return: {test_metrics.get('Total Return (%)', 0.0):.2f}% | Sharpe: {test_metrics.get('Sharpe Ratio', 0.0):.2f} | WinRate: {test_metrics.get('Win Rate (%)', 0.0):.1f}% | MaxDD: {test_metrics.get('Max Drawdown (%)', 0.0):.2f}% | PF: {test_metrics.get('Profit Factor', 0.0):.2f}")
        test_env.print_funnel_report()
        test_env.print_exit_and_cost_report()
        
        # Save champion test metrics
        test_csv_path = results_dir / "rl_test_champion.csv"
        pd.DataFrame([test_metrics]).to_csv(test_csv_path, index=False)
        print(f"Champion Test Results saved to {test_csv_path}")
        
        viz_path = results_dir / "backtest_results.png"
        PerformanceMetrics.plot_results(test_results, save_path=str(viz_path))
        print(f"Champion Test Backtest visualization saved to {viz_path}")
        
    # Benchmark against Deterministic Rule Agent on the same test split
    try:
        from baselines import run_deterministic_baseline
        run_deterministic_baseline(test_embeddings, test_df, probabilities=test_probs, be_trigger_mult=1.2, split_name="Held-Out Test")
    except Exception as e:
        print(f"Deterministic baseline check: {e}")
        
    return df_val_leaderboard
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
