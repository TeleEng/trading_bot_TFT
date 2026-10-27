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
            
        # Balanced (prior-corrected) probabilities for the edge rule; TRUE calibrated ones for EV.
        p_l = float(env.prob_long[step])
        p_s = float(env.prob_short[step])
        ev_p_l = float(env.ev_prob_long[step]) if getattr(env, 'ev_prob_long', None) is not None else p_l
        ev_p_s = float(env.ev_prob_short[step]) if getattr(env, 'ev_prob_short', None) is not None else p_s
        atr = float(env.atr_values[step])
        cost_in_r = (env.spread + env.slippage) / max(1e-5, atr)
        
        ev_l = ev_p_l * env.long_tp_mult - (1.0 - ev_p_l) * env.long_sl_mult - cost_in_r
        ev_s = ev_p_s * env.short_tp_mult - (1.0 - ev_p_s) * env.short_sl_mult - cost_in_r
        
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


def run_deterministic_baseline(embeddings, df, probabilities=None, breakeven_sl=False, be_trigger_mult=1.2, split_name="Held-Out Test",
                               ev_probabilities=None, use_ev=False, min_ev_r=0.0):
    """Runs the deterministic rule agent on the specified split.

    use_ev=False -> balanced-probability confidence/edge rule (same gate semantics as the PPO configs).
    use_ev=True  -> EV-in-R rule on TRUE calibrated probabilities (ev_probabilities).
    """
    env = TradingRLEnv(
        embeddings, df,
        probabilities=probabilities,
        ev_probabilities=ev_probabilities,
        long_tp_mult=3.0, long_sl_mult=2.0,
        short_tp_mult=3.0, short_sl_mult=2.0,
        deadzone=0.20,
        breakeven_sl=breakeven_sl,
        be_trigger_mult=be_trigger_mult,
        volatility_filter=True,
        trend_filter=True,
        cooldown_bars=4,
        use_ev_gate=use_ev,
        min_ev_r=min_ev_r,
        max_wait_bars=96,
        is_eval=True
    )
    
    agent = DeterministicRuleAgent(min_edge=0.01, min_confidence=0.50, use_ev=use_ev, min_ev_r=min_ev_r)
    be_tag = f"BE={be_trigger_mult:.1f}x" if breakeven_sl else "NoBE"
    rule_name = f"EV-in-R > {min_ev_r:+.2f} ({be_tag})" if use_ev else f"Balanced Prob Edge >= 0.01 ({be_tag})"
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
    print(f"BENCHMARK BASELINE: DETERMINISTIC RULE AGENT (NO RL) [{split_name.upper()}] | Rule: {rule_name}")
    print("="*70)
    print(f"  Trades: {env.total_trades} | Return: {metrics.get('Total Return (%)', 0.0):.2f}% | Sharpe: {metrics.get('Sharpe Ratio', 0.0):.2f} | WinRate: {metrics.get('Win Rate (%)', 0.0):.1f}% | PF: {metrics.get('Profit Factor', 0.0):.2f}")
    env.print_funnel_report()
    env.print_exit_and_cost_report()
    return metrics, results_df


def run_random_baseline(embeddings, df, breakeven_sl=False, be_trigger_mult=1.2,
                        trade_prob=0.15, n_seeds=5, split_name="Held-Out Test"):
    """Runs a zero-skill Random Entry benchmark across multiple seeds under identical filters, barriers, and costs."""
    seed_metrics = []
    be_tag = f"BE={be_trigger_mult:.1f}x" if breakeven_sl else "NoBE"

    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        # Pass probabilities=None so only the structural regime filters (Trend, Volatility, Cooldown) and barriers apply
        env = TradingRLEnv(
            embeddings, df,
            probabilities=None,
            long_tp_mult=3.0, long_sl_mult=2.0,
            short_tp_mult=3.0, short_sl_mult=2.0,
            deadzone=0.20,
            breakeven_sl=breakeven_sl,
            be_trigger_mult=be_trigger_mult,
            volatility_filter=True,
            trend_filter=True,
            cooldown_bars=4,
            use_ev_gate=False,
            max_wait_bars=96,
            is_eval=True
        )
        obs, _ = env.reset()
        done = False
        records = []
        while not done:
            step_idx = env.current_step
            timestamp = env.df.index[step_idx]
            close_price = env.close_prices[step_idx]
            if env.position == 0 and rng.random() < trade_prob:
                action = 1.0 if rng.random() < 0.5 else -1.0
            else:
                action = 0.0
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
            records.append({
                'timestamp': timestamp,
                'price': close_price,
                'position': env.position,
                'portfolio_value': env.portfolio_value
            })
        res_df = pd.DataFrame(records)
        m = PerformanceMetrics.calculate_metrics(res_df, initial_capital=10000)
        exp_r = float(np.mean(env.trade_r_multiples)) if len(env.trade_r_multiples) > 0 else 0.0
        seed_metrics.append({
            'trades': env.total_trades,
            'return': m.get('Total Return (%)', 0.0),
            'sharpe': m.get('Sharpe Ratio', 0.0),
            'win_rate': m.get('Win Rate (%)', 0.0),
            'pf': m.get('Profit Factor', 0.0),
            'exp_r': exp_r,
            'costs': env.total_Friction_costs_usd
        })

    df_m = pd.DataFrame(seed_metrics)
    print("\n" + "="*70)
    print(f"ZERO-SKILL BENCHMARK: RANDOM ENTRY AGENT ({n_seeds} SEEDS MEAN ± STD) [{split_name.upper()}] | {be_tag}")
    print("="*70)
    print(f"  Trades:   {df_m['trades'].mean():.1f} ± {df_m['trades'].std():.1f}")
    print(f"  Return:   {df_m['return'].mean():+.2f}% ± {df_m['return'].std():.2f}% (Range: [{df_m['return'].min():+.2f}%, {df_m['return'].max():+.2f}%])")
    print(f"  Sharpe:   {df_m['sharpe'].mean():+.2f} ± {df_m['sharpe'].std():.2f}")
    print(f"  WinRate:  {df_m['win_rate'].mean():.1f}% ± {df_m['win_rate'].std():.1f}% | PF: {df_m['pf'].mean():.2f} | Expectancy: {df_m['exp_r'].mean():+.3f}R | Mean Costs: ${df_m['costs'].mean():,.2f}")
    return df_m
