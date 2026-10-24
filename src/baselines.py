#!/usr/bin/env python3
"""
Baselines Benchmark Module
1. Deterministic Rule-Based Baseline:
   Executes trades directly using calibrated probabilities/EV and macro/volatility/cooldown filters without RL.
2. Direct comparison helper to benchmark PPO against the deterministic rule.
"""
import numpy as np
import pandas as pd
from rl_environment import TradingRLEnv
from performance import PerformanceMetrics


class DeterministicRuleAgent:
    """
    A deterministic trading policy that acts without RL neural networks:
    - Uses calibrated probabilities to compute Expected Value in R:
        EV_long  = p_long * 3.0 - (1 - p_long) * 2.0 - cost_in_r
        EV_short = p_short * 3.0 - (1 - p_short) * 2.0 - cost_in_r
    - If EV_long > min_ev_r: action = +1.0 (long)
    - If EV_short > min_ev_r: action = -1.0 (short)
    - Otherwise: action = 0.0 (flat)
    Environment filters (Trend, Volatility, Cooldown) handle regime confirmation.
    """
    def __init__(self, min_edge=0.01, min_confidence=0.50, use_ev=True, min_ev_r=0.0):
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.use_ev = use_ev
        self.min_ev_r = min_ev_r

    def predict(self, env):
        step = env.current_step
        if env.prob_long is None or env.prob_short is None:
            return 0.0
            
        p_l = float(env.prob_long[step])
        p_s = float(env.prob_short[step])
        atr = float(env.atr_values[step])
        cost_in_r = (env.spread + env.slippage) / max(1e-5, atr)
        
        ev_l = p_l * env.long_tp_mult - (1.0 - p_l) * env.long_sl_mult - cost_in_r
        ev_s = p_s * env.short_tp_mult - (1.0 - p_s) * env.short_sl_mult - cost_in_r
        
        if self.use_ev:
            if ev_l > self.min_ev_r and ev_l > ev_s:
                return 1.0
            elif ev_s > self.min_ev_r and ev_s > ev_l:
                return -1.0
            return 0.0
        else:
            if p_l >= self.min_confidence and (p_l - p_s) >= self.min_edge:
                return 1.0
            elif p_s >= self.min_confidence and (p_s - p_l) >= self.min_edge:
                return -1.0
            return 0.0


def run_deterministic_baseline(embeddings, df, probabilities=None, be_trigger_mult=1.2, split_name="Held-Out Test"):
    """Runs the deterministic rule agent on the specified split."""
    env = TradingRLEnv(
        embeddings, df,
        probabilities=probabilities,
        long_tp_mult=3.0, long_sl_mult=2.0,
        short_tp_mult=3.0, short_sl_mult=2.0,
        deadzone=0.20,
        breakeven_sl=True,
        be_trigger_mult=be_trigger_mult,
        volatility_filter=True,
        trend_filter=True,
        cooldown_bars=4,
        max_wait_bars=96,
        is_eval=True
    )
    
    agent = DeterministicRuleAgent(min_edge=0.01, min_confidence=0.50, use_ev=True, min_ev_r=0.0)
    obs, _ = env.reset()
    done = False
    records = []
    
    while not done:
        step_idx = env.current_step
        timestamp = env.df.index[step_idx]
        close_price = env.close_prices[step_idx]
        
        action = agent.predict(env)
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        done = terminated or truncated
        
        records.append({
            'timestamp': timestamp,
            'price': close_price,
            'position': env.position,
            'portfolio_value': env.portfolio_value
        })
        
    results_df = pd.DataFrame(records)
    metrics = PerformanceMetrics.calculate_metrics(results_df, initial_capital=10000)
    
    print("\n" + "="*70)
    print(f"BENCHMARK BASELINE: DETERMINISTIC RULE AGENT (NO RL) [{split_name.upper()}]")
    print("="*70)
    print(f"  Trades: {env.total_trades} | Return: {metrics.get('Total Return (%)', 0.0):.2f}% | Sharpe: {metrics.get('Sharpe Ratio', 0.0):.2f} | WinRate: {metrics.get('Win Rate (%)', 0.0):.1f}% | PF: {metrics.get('Profit Factor', 0.0):.2f}")
    env.print_funnel_report()
    env.print_exit_and_cost_report()
    return metrics, results_df
