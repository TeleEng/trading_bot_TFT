import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingRLEnv(gym.Env):
    """
    High-performance Trading RL Environment with:
    - Balanced 1.5:1 Risk/Reward geometry (3.0x ATR TP / 2.0x ATR SL)
    - Net-of-cost reward formulation deducting spread, slippage, and overnight swap
    - Backbone probability confirmation gate (eliminating noise trades)
    - Zero inactivity penalty (rewards patience, eliminates overtrading)
    - Conservative dynamic risk sizing (capped at 1.5% account risk)
    - Post-StopLoss cooldown to prevent revenge-trading whipsaws
    - Higher-timeframe macro trend filter (SMA-50 alignment)
    """
    def __init__(self, embeddings, df, 
                 probabilities=None,
                 ev_probabilities=None,
                 long_tp_mult=3.0, long_sl_mult=2.0, 
                 short_tp_mult=3.0, short_sl_mult=2.0, 
                 initial_capital=10000,
                 spread_pips=1.0, slippage_pips=0.3, swap_pips_per_day=0.5, 
                 is_eval=False,
                 deadzone=0.20,
                 min_confidence=0.50,
                 min_edge=0.01,
                 reward_mode='net_rr',
                 flat_penalty=0.0,
                 max_risk_multiplier=1.5,
                 cooldown_bars=4,
                 trend_filter=True,
                 volatility_filter=True,
                 min_atr_ratio=0.70,
                 max_atr_ratio=2.50,
                 breakeven_sl=False,
                 be_trigger_mult=1.2,
                 use_ev_gate=False,
                 min_ev_r=0.0,
                 max_wait_bars=96):
        super(TradingRLEnv, self).__init__()
        self.is_eval = is_eval
        self.embeddings = embeddings
        self.df = df
        self.probabilities = probabilities
        
        # Hyperparameters
        self.deadzone = deadzone
        self.min_confidence = min_confidence
        self.min_edge = min_edge
        self.use_ev_gate = use_ev_gate
        self.min_ev_r = min_ev_r
        self.reward_mode = reward_mode
        self.flat_penalty = flat_penalty
        self.max_risk_multiplier = max_risk_multiplier
        self.cooldown_bars = cooldown_bars
        self.trend_filter = trend_filter
        self.volatility_filter = volatility_filter
        self.min_atr_ratio = min_atr_ratio
        self.max_atr_ratio = max_atr_ratio
        self.breakeven_sl = breakeven_sl
        self.be_trigger_mult = be_trigger_mult
        self.max_wait_bars = max_wait_bars
        
        # Extract numpy arrays for O(1) lookups during RL steps
        self.close_prices = self.df['Close'].values
        self.open_prices = self.df['Open'].values if 'Open' in self.df.columns else self.close_prices
        self.high_prices = self.df['High'].values if 'High' in self.df.columns else self.close_prices
        self.low_prices = self.df['Low'].values if 'Low' in self.df.columns else self.close_prices
        self.atr_values = self.df['ATR'].values if 'ATR' in self.df.columns else self.close_prices * 0.005
        self.dist_sma_50 = self.df['Dist_SMA_50'].values if 'Dist_SMA_50' in self.df.columns else None
        self.dist_sma_50_4h = self.df['Dist_SMA_50_4H'].values if 'Dist_SMA_50_4H' in self.df.columns else None
        self.dist_sma_50_1d = self.df['Dist_SMA_50_1D'].values if 'Dist_SMA_50_1D' in self.df.columns else None
        self.atr_ratio = self.df['ATR_Ratio'].values if 'ATR_Ratio' in self.df.columns else None
        self.rsi = self.df['RSI'].values if 'RSI' in self.df.columns else None
        
        # Extract probabilities if available
        # `probabilities`    -> prior-corrected (0.50 == base rate) values for obs + confidence/edge gating.
        # `ev_probabilities` -> prior-neutralized (0.40 == base rate) P(TP hit) values for EV-in-R gating.
        if probabilities is not None and len(probabilities) > 0:
            self.prob_long = probabilities[:, 0]
            self.prob_short = probabilities[:, 1]
        else:
            self.prob_long = None
            self.prob_short = None
        if ev_probabilities is not None and len(ev_probabilities) > 0:
            self.ev_prob_long = ev_probabilities[:, 0]
            self.ev_prob_short = ev_probabilities[:, 1]
        else:
            self.ev_prob_long = self.prob_long
            self.ev_prob_short = self.prob_short
        
        self.long_tp_mult = long_tp_mult
        self.long_sl_mult = long_sl_mult
        self.short_tp_mult = short_tp_mult
        self.short_sl_mult = short_sl_mult
        self.initial_capital = initial_capital
        
        # Realistic trading costs
        self.spread = spread_pips * 0.0001   # 1 pip = 0.0001 for EURUSD
        self.slippage = slippage_pips * 0.0001
        self.swap_per_bar = (swap_pips_per_day * 0.0001) / 24.0  # spread across 24 1H bars
        
        # Continuous Action Space [-1.0, 1.0] for dynamic risk sizing
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: emb_size + 1 (local_ratio) + 15 (last 5 trades) + 2 (direction, unrealized) + 8 (scaled probs/edge + MTF regimes) = emb_size + 26
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 26,), dtype=np.float32)
        
        self.reset()
        
    def _get_fill_price(self, price, direction):
        """Apply spread and slippage to get a realistic fill price."""
        half_spread = self.spread / 2.0
        slip = self.slippage * self.np_random.random()
        if direction > 0:  # buying
            return price + half_spread + slip
        else:  # selling
            return price - half_spread - slip
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize starting point during training so the agent sees diverse market regimes
        if not self.is_eval and len(self.df) > 1000:
            self.current_step = int(self.np_random.integers(0, len(self.df) - 1000))
        else:
            self.current_step = 0
            
        self.portfolio_value = float(self.initial_capital)
        self.capital = float(self.initial_capital)
        self.position = 0.0  # qty (positive = long, negative = short)
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.current_risk_multiplier = 0.0
        self.trade_initial_risk = 1.0
        self.trade_accumulated_swap = 0.0
        self.bars_held = 0
        
        self.current_sl_price = 0.0
        self.tp_price = 0.0
        
        # Cooldown state tracking
        self.cooldown_remaining = 0
        self.last_sl_direction = 0.0
        
        # Differential Sharpe Ratio trackers
        self.dsr_A = 0.0
        self.dsr_B = 1e-4
        self.dsr_eta = 0.01
        
        self.consecutive_losses = 0
        self.bars_flat = 0
        self.trade_history = [(0.0, 0.0, 0.0)] * 5
        self.portfolio_history = [self.initial_capital] * 5
        
        self.last_portfolio_value = float(self.initial_capital)
        
        # Trade statistics for tracking and metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.timeout_trades = 0
        self.realized_pnl = 0.0
        
        # Diagnostic & telemetry tracking
        self.trade_mfe = 0.0
        self.trade_entry_friction = 0.0
        self.loss_mfe_history = []
        self.hold_bars_wins = []
        self.hold_bars_losses = []
        self.risk_mult_history = []
        self.trade_r_history = []
        
        self.exit_tp_count = 0
        self.exit_tp_pnl = 0.0
        self.exit_be_count = 0
        self.exit_be_pnl = 0.0
        self.exit_sl_count = 0
        self.exit_sl_pnl = 0.0
        self.exit_timeout_count = 0
        self.exit_timeout_pnl = 0.0
        
        self.long_trades = 0
        self.long_wins = 0
        self.long_pnl = 0.0
        self.long_gross_profits = 0.0
        self.long_gross_losses = 0.0
        
        self.short_trades = 0
        self.short_wins = 0
        self.short_pnl = 0.0
        self.short_gross_profits = 0.0
        self.short_gross_losses = 0.0
        
        self.total_gross_pnl = 0.0
        self.total_net_pnl = 0.0
        self.total_friction = 0.0
        
        # Funnel diagnostics
        self.funnel_flat_bars = 0
        self.funnel_raw_signals = 0
        self.funnel_blocked_cooldown = 0
        self.funnel_blocked_trend = 0
        self.funnel_blocked_prob_edge = 0
        self.funnel_blocked_volatility = 0
        self.funnel_executed_trades = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        emb = self.embeddings[self.current_step]
        
        past_val = self.portfolio_history[0] if len(self.portfolio_history) >= 5 else self.initial_capital
        if past_val <= 0:
            past_val = 1e-6
        local_ratio = self.portfolio_value / past_val
        
        hist = []
        for action, reward, reason in self.trade_history[-5:]:
            hist.extend([float(action), float(reward), float(reason)])
            
        price = self.close_prices[self.current_step]
        
        current_direction = 1.0 if self.position > 0 else (-1.0 if self.position < 0 else 0.0)
        unrealized_return = 0.0
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:
                unrealized_return = (price - self.entry_price) / self.entry_price
            else:
                unrealized_return = (self.entry_price - price) / self.entry_price
                
        # Scale up unrealized return for network visibility
        unrealized_return *= 100.0
        
        # Probabilities & Multi-Timeframe Market Regimes (scaled to O(1) for MlpPolicy visibility)
        p_long = float(self.prob_long[self.current_step]) if self.prob_long is not None else 0.5
        p_short = float(self.prob_short[self.current_step]) if self.prob_short is not None else 0.5
        p_long_scaled = (p_long - 0.50) * 25.0
        p_short_scaled = (p_short - 0.50) * 25.0
        p_edge_scaled = (p_long - p_short) * 25.0
        macro_dist = float(self.dist_sma_50[self.current_step]) if self.dist_sma_50 is not None else 0.0
        macro_dist_4h = float(self.dist_sma_50_4h[self.current_step]) if self.dist_sma_50_4h is not None else macro_dist
        macro_dist_1d = float(self.dist_sma_50_1d[self.current_step]) if self.dist_sma_50_1d is not None else macro_dist
        atr_ratio = float(self.atr_ratio[self.current_step]) if self.atr_ratio is not None else 1.0
        rsi_val = float((self.rsi[self.current_step] - 50.0) / 50.0) if self.rsi is not None else 0.0
            
        obs = np.concatenate([
            emb, 
            [local_ratio], 
            hist, 
            [current_direction, unrealized_return],
            [p_long_scaled, p_short_scaled, p_edge_scaled, macro_dist, macro_dist_4h, macro_dist_1d, atr_ratio, rsi_val]
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
        
    def _check_tp_sl_intrabar(self, open_price, high, low):
        """Check TP/SL using Open/High/Low of the bar (intra-bar simulation with gap awareness).
        Returns: (trade_closed, trade_reward, exit_price, exit_reason)
        exit_reason: 1 (TP), -1 (SL), 2 (BE), 0 (None/Timeout)
        """
        if self.position == 0 or self.entry_atr <= 0:
            return False, 0.0, 0.0, 0
            
        if self.position > 0:  # Long
            # 1. Gap Check on Open (conservative gap slippage)
            if open_price <= self.current_sl_price:
                exit_price = self._get_fill_price(open_price, -1)
                reward = -1.0 * self.long_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            # 2. SL checked during bar
            elif low <= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, -1)
                is_be = self.breakeven_sl and (self.current_sl_price > self.entry_price)
                reason = 2 if is_be else -1
                reward = -1.0 * self.long_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, reason
            elif high >= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, -1)
                reward = self.long_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
            else:
                # Ratchet Break-Even Stop Loss if favorable price move >= be_trigger_mult * ATR
                if self.breakeven_sl and high >= (self.entry_price + self.be_trigger_mult * self.entry_atr):
                    be_price = self.entry_price + (0.1 * self.entry_atr)
                    if be_price > self.current_sl_price:
                        self.current_sl_price = be_price
                
        else:  # Short
            # 1. Gap Check on Open (conservative gap slippage)
            if open_price >= self.current_sl_price:
                exit_price = self._get_fill_price(open_price, 1)
                reward = -1.0 * self.short_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            # 2. SL checked during bar
            elif high >= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, 1)
                is_be = self.breakeven_sl and (self.current_sl_price < self.entry_price)
                reason = 2 if is_be else -1
                reward = -1.0 * self.short_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, reason
            elif low <= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, 1)
                reward = self.short_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
            else:
                # Ratchet Break-Even Stop Loss if favorable price move >= be_trigger_mult * ATR
                if self.breakeven_sl and low <= (self.entry_price - self.be_trigger_mult * self.entry_atr):
                    be_price = self.entry_price - (0.1 * self.entry_atr)
                    if be_price < self.current_sl_price:
                        self.current_sl_price = be_price
                
        return False, 0.0, 0.0, 0
        
    def _close_position(self, exit_price):
        """Close position and update capital with realized PnL."""
        pnl = self.position * (exit_price - self.entry_price)
        self.capital += pnl
        self.realized_pnl += pnl
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.current_risk_multiplier = 0.0
        self.bars_held = 0
        return pnl
        
    def _open_position(self, price, atr, direction, risk_multiplier=1.0):
        """Open a new position with spread/slippage applied and dynamic risk size."""
        fill_price = self._get_fill_price(price, direction)
        
        # Base risk is 1%, scaled dynamically by conviction (up to max_risk_multiplier)
        risk_amount = self.portfolio_value * 0.01 * risk_multiplier
        sl_mult = self.long_sl_mult if direction > 0 else self.short_sl_mult
        risk_per_unit = atr * sl_mult
        ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
        max_qty_allowed = (self.portfolio_value * 50 * 0.98) / fill_price
        target_qty = min(ideal_qty, max_qty_allowed) * direction
        
        self.position = target_qty
        self.entry_price = fill_price
        self.entry_atr = atr
        self.current_risk_multiplier = risk_multiplier
        self.trade_initial_risk = max(1.0, risk_amount)
        self.trade_accumulated_swap = 0.0
        self.trade_entry_friction = abs(fill_price - price) * abs(target_qty)
        self.trade_mfe = 0.0
        self.risk_mult_history.append(risk_multiplier)
        self.bars_held = 0
        self.total_trades += 1
        
        # Initialize TP/SL levels
        if direction > 0:
            tp_distance = self.long_tp_mult * atr
            sl_distance = self.long_sl_mult * atr
            self.current_sl_price = fill_price - sl_distance
            self.tp_price = fill_price + tp_distance
        else:
            tp_distance = self.short_tp_mult * atr
            sl_distance = self.short_sl_mult * atr
            self.current_sl_price = fill_price + sl_distance
            self.tp_price = fill_price - tp_distance
        
    def step(self, action):
        if self.current_step >= len(self.df) - 1 or self.portfolio_value <= 0:
            info = {
                'total_trades': self.total_trades,
                'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0,
                'portfolio_value': self.portfolio_value
            }
            return self._get_obs(), 0.0, True, False, info
            
        price = self.close_prices[self.current_step]
        open_price = self.open_prices[self.current_step]
        high = self.high_prices[self.current_step]
        low = self.low_prices[self.current_step]
        atr = self.atr_values[self.current_step]
        
        # Guard against non-positive ATR (aligned with preprocess.py raw ATR)
        atr = max(1e-6, min(atr, price * 0.05))
        
        trade_closed = False
        trade_reward = 0.0
        current_direction = 1.0 if self.position > 0 else (-1.0 if self.position < 0 else 0.0)
        target_direction = 0.0
        
        # Decrement cooldown counter if active
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.last_sl_direction = 0.0
        
        # --- Deduct overnight swap cost for open positions ---
        if self.position != 0:
            self.bars_held += 1
            swap_cost = abs(self.position) * self.swap_per_bar
            self.capital -= swap_cost
            self.trade_accumulated_swap += swap_cost
        
        # --- Check TP/SL using High/Low and fixed entry ATR ---
        if self.position != 0:
            if self.entry_atr > 0:
                if self.position > 0:
                    favorable = (high - self.entry_price) / self.entry_atr
                else:
                    favorable = (self.entry_price - low) / self.entry_atr
                self.trade_mfe = max(self.trade_mfe, float(favorable))
                
            trade_closed, _, exit_price, exit_reason = self._check_tp_sl_intrabar(open_price, high, low)
            
            # --- Check Time Limit (96 Hours) ---
            if not trade_closed and self.bars_held >= self.max_wait_bars:
                trade_closed = True
                exit_price = price
                exit_reason = 0  # Timeout
                self.timeout_trades += 1
                
            if trade_closed:
                held_bars = self.bars_held
                # Friction calculation: entry spread + exit spread + swap
                exit_fill_drag = abs(exit_price - (self.current_sl_price if exit_reason in (-1, 2) else (self.tp_price if exit_reason == 1 else price))) * abs(self.position)
                trade_friction = self.trade_entry_friction + exit_fill_drag + self.trade_accumulated_swap
                self.total_friction += trade_friction
                
                gross_pnl = self._close_position(exit_price)
                net_pnl = gross_pnl - self.trade_accumulated_swap
                self.total_gross_pnl += gross_pnl
                self.total_net_pnl += net_pnl
                
                # Directional breakdown
                if current_direction > 0:
                    self.long_trades += 1
                    if net_pnl > 0: self.long_wins += 1
                    self.long_pnl += net_pnl
                    if gross_pnl > 0: self.long_gross_profits += gross_pnl
                    else: self.long_gross_losses += abs(gross_pnl)
                else:
                    self.short_trades += 1
                    if net_pnl > 0: self.short_wins += 1
                    self.short_pnl += net_pnl
                    if gross_pnl > 0: self.short_gross_profits += gross_pnl
                    else: self.short_gross_losses += abs(gross_pnl)
                    
                # Exit type telemetry
                if exit_reason == 1:
                    self.exit_tp_count += 1
                    self.exit_tp_pnl += net_pnl
                elif exit_reason == 2:
                    self.exit_be_count += 1
                    self.exit_be_pnl += net_pnl
                elif exit_reason == -1:
                    self.exit_sl_count += 1
                    self.exit_sl_pnl += net_pnl
                elif exit_reason == 0:
                    self.exit_timeout_count += 1
                    self.exit_timeout_pnl += net_pnl
                    
                # Loss MFE tracking
                if net_pnl <= 0:
                    self.loss_mfe_history.append(self.trade_mfe)
                    self.hold_bars_losses.append(held_bars)
                else:
                    self.hold_bars_wins.append(held_bars)
                    
                self.trade_mfe = 0.0
                
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                    # Trigger revenge-trade prevention cooldown in the losing direction
                    self.cooldown_remaining = self.cooldown_bars
                    self.last_sl_direction = current_direction
                    
                # Realized risk-normalized net PnL reward and R-tracking
                r_mult = float(net_pnl / max(1.0, self.trade_initial_risk))
                self.trade_r_history.append(r_mult)
                trade_reward = float(np.clip(r_mult, -3.0, 3.0))
                    
        if trade_closed:
            last_dir = 1.0 if current_direction > 0 else 2.0
            self.trade_history.append((last_dir, trade_reward, exit_reason))
            self.trade_history.pop(0)
            self.portfolio_history.append(self.portfolio_value)
            self.portfolio_history.pop(0)
            current_direction = 0.0
            
        # --- Handle agent action ---
        risk_multiplier = 0.0
        if self.position == 0 and not trade_closed:
            self.bars_flat += 1
            self.funnel_flat_bars += 1
            
            raw_action = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            raw_action = float(np.clip(raw_action, -1.0, 1.0))
            
            if abs(raw_action) > self.deadzone:
                self.funnel_raw_signals += 1
                desired_direction = 1.0 if raw_action > 0 else -1.0
                
                # Check 1: Cooldown lockout against revenge trading in the same direction
                is_cooled_down = (self.cooldown_remaining == 0) or (desired_direction != self.last_sl_direction)
                
                # Check 2: Multi-Timeframe Macro Trend Filter (1H + 4H + 1D SMA-50 alignment)
                trend_confirmed = True
                if self.trend_filter and self.dist_sma_50 is not None:
                    macro_dist = float(self.dist_sma_50[self.current_step])
                    macro_4h = float(self.dist_sma_50_4h[self.current_step]) if self.dist_sma_50_4h is not None else 0.0
                    macro_1d = float(self.dist_sma_50_1d[self.current_step]) if self.dist_sma_50_1d is not None else 0.0
                    if desired_direction > 0:
                        if macro_dist < -0.5 or macro_1d < -0.75 or (macro_1d < -0.25 and macro_4h < -0.25):
                            trend_confirmed = False  # Avoid longs in 1H/4H/1D macro downtrends
                    elif desired_direction < 0:
                        if macro_dist > 0.5 or macro_1d > 0.75 or (macro_1d > 0.25 and macro_4h > 0.25):
                            trend_confirmed = False  # Avoid shorts in 1H/4H/1D macro uptrends
                
                # Check 3: Backbone Probability Gate & Expected Value (EV) Gate
                prob_confirmed = True
                if self.prob_long is not None and self.prob_short is not None:
                    p_l = float(self.prob_long[self.current_step])
                    p_s = float(self.prob_short[self.current_step])
                    
                    cost_in_r = (self.spread + self.slippage) / max(1e-5, atr)
                    ev_p_l = float(self.ev_prob_long[self.current_step]) if self.ev_prob_long is not None else p_l
                    ev_p_s = float(self.ev_prob_short[self.current_step]) if self.ev_prob_short is not None else p_s
                    ev_long = ev_p_l * self.long_tp_mult - (1.0 - ev_p_l) * self.long_sl_mult - cost_in_r
                    ev_short = ev_p_s * self.short_tp_mult - (1.0 - ev_p_s) * self.short_sl_mult - cost_in_r
                    min_own_conf = max(self.min_confidence, 0.50 + 0.35 * self.min_edge)

                    if desired_direction > 0:
                        if self.use_ev_gate:
                            if ev_long < self.min_ev_r or p_l < self.min_confidence or p_l <= p_s:
                                prob_confirmed = False
                        else:
                            if p_l < min_own_conf or (p_l - p_s) < self.min_edge:
                                prob_confirmed = False
                    elif desired_direction < 0:
                        if self.use_ev_gate:
                            if ev_short < self.min_ev_r or p_s < self.min_confidence or p_s <= p_l:
                                prob_confirmed = False
                        else:
                            if p_s < min_own_conf or (p_s - p_l) < self.min_edge:
                                prob_confirmed = False
                
                # Check 4: Volatility Regime Filter (avoid dead zones and volatile blowouts)
                vol_confirmed = True
                if self.volatility_filter and self.atr_ratio is not None:
                    curr_atr_ratio = float(self.atr_ratio[self.current_step])
                    if curr_atr_ratio < self.min_atr_ratio or curr_atr_ratio > self.max_atr_ratio:
                        vol_confirmed = False
                
                if not is_cooled_down:
                    self.funnel_blocked_cooldown += 1
                elif not trend_confirmed:
                    self.funnel_blocked_trend += 1
                elif not prob_confirmed:
                    self.funnel_blocked_prob_edge += 1
                elif not vol_confirmed:
                    self.funnel_blocked_volatility += 1
                else:
                    self.funnel_executed_trades += 1
                    target_direction = desired_direction
                    conviction = (abs(raw_action) - self.deadzone) / (1.0 - self.deadzone + 1e-8)
                    conviction = float(np.clip(conviction, 0.0, 1.0))
                    risk_multiplier = 1.0 + conviction * (self.max_risk_multiplier - 1.0)
                    self._open_position(price, atr, target_direction, risk_multiplier)
            else:
                target_direction = 0.0
                
            # Decay portfolio history toward current value
            alpha = 0.01
            for i in range(len(self.portfolio_history)):
                self.portfolio_history[i] = (1 - alpha) * self.portfolio_history[i] + alpha * self.portfolio_value
        else:
            self.bars_flat = 0

        # Update portfolio value (mark-to-market)
        self.portfolio_value = self.capital + (self.position * (price - self.entry_price) if self.position != 0 else 0)
        
        # --- Reward Calculation based on reward_mode ---
        if self.reward_mode in ('asymmetric_rr', 'net_rr'):
            # Realized risk-normalized net PnL (deducting spread, slippage, and swap)
            reward = trade_reward
            if self.position == 0 and not trade_closed and self.flat_penalty > 0:
                reward -= self.flat_penalty
                
        elif self.reward_mode == 'dsr':
            # Moody & Saffell Differential Sharpe Ratio (dense step-by-step return optimization)
            R_t = (self.portfolio_value - self.last_portfolio_value) / (self.last_portfolio_value + 1e-8)
            delta_A = R_t - self.dsr_A
            delta_B = (R_t ** 2) - self.dsr_B
            denom = ((self.dsr_B - (self.dsr_A ** 2)) ** 1.5) + 1e-6
            dsr = (self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B) / denom
            self.dsr_A += self.dsr_eta * delta_A
            self.dsr_B += self.dsr_eta * delta_B
            reward = float(np.clip(dsr * 10.0, -5.0, 5.0))
            if self.position == 0 and not trade_closed and self.flat_penalty > 0:
                reward -= self.flat_penalty
                
        elif self.reward_mode == 'pnl_dense':
            # Step-wise normalized PnL
            step_return = (self.portfolio_value - self.last_portfolio_value) / self.initial_capital * 100.0
            reward = float(step_return) + (trade_reward * 0.5)
            if self.position == 0 and not trade_closed and self.flat_penalty > 0:
                reward -= self.flat_penalty
        else:
            reward = trade_reward

        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        
        # Bankruptcy check (portfolio drops below 10% of capital)
        is_bankrupt = self.portfolio_value < (self.initial_capital * 0.1)
        if is_bankrupt:
            reward -= 50.0  # Terminal penalty
            
        terminated = is_bankrupt or (self.current_step >= len(self.df) - 1)
        
        info = {
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0,
            'portfolio_value': self.portfolio_value,
            'pnl': self.portfolio_value - self.initial_capital
        }
        
        return self._get_obs(), float(reward), terminated, False, info

    def print_funnel_report(self):
        s = self.funnel_raw_signals
        pct = lambda x: (x / s * 100.0) if s > 0 else 0.0
        print("\n  " + "="*68)
        print(f"  [ENTRY FUNNEL] Flat Bars: {self.funnel_flat_bars:,} | Raw Signals (|a| > dz): {s:,} ({(s / max(1, self.funnel_flat_bars)*100):.1f}%)")
        print(f"    ├─ Blocked by Cooldown:        {self.funnel_blocked_cooldown:,} ({pct(self.funnel_blocked_cooldown):.1f}%)")
        print(f"    ├─ Blocked by Macro Trend:    {self.funnel_blocked_trend:,} ({pct(self.funnel_blocked_trend):.1f}%)")
        print(f"    ├─ Blocked by Volatility:     {self.funnel_blocked_volatility:,} ({pct(self.funnel_blocked_volatility):.1f}%)")
        print(f"    ├─ Blocked by Prob/Edge Gate: {self.funnel_blocked_prob_edge:,} ({pct(self.funnel_blocked_prob_edge):.1f}%)")
        print(f"    └─ EXECUTED TRADES:           {self.funnel_executed_trades:,} ({pct(self.funnel_executed_trades):.1f}% conversion)")
        print("  " + "="*68)

    def print_exit_and_cost_report(self):
        total = max(1, self.total_trades)
        long_pf = (self.long_gross_profits / max(1e-4, self.long_gross_losses)) if self.long_trades > 0 else 0.0
        short_pf = (self.short_gross_profits / max(1e-4, self.short_gross_losses)) if self.short_trades > 0 else 0.0
        
        avg_loss_mfe = float(np.mean(self.loss_mfe_history)) if len(self.loss_mfe_history) > 0 else 0.0
        losses_above_1_2 = (sum(1 for m in self.loss_mfe_history if m >= 1.2) / max(1, len(self.loss_mfe_history))) * 100.0 if len(self.loss_mfe_history) > 0 else 0.0
        
        avg_hold_win = float(np.mean(self.hold_bars_wins)) if len(self.hold_bars_wins) > 0 else 0.0
        avg_hold_loss = float(np.mean(self.hold_bars_losses)) if len(self.hold_bars_losses) > 0 else 0.0
        avg_risk_mult = float(np.mean(self.risk_mult_history)) if len(self.risk_mult_history) > 0 else 1.0

        print("  [EXIT TELEMETRY]")
        print(f"    ├─ Full TP (+3.0x ATR):     {self.exit_tp_count} trades ({(self.exit_tp_count/total*100):.1f}%) | Net PnL: ${self.exit_tp_pnl:+.2f}")
        print(f"    ├─ Break-Even (+0.1x ATR):  {self.exit_be_count} trades ({(self.exit_be_count/total*100):.1f}%) | Net PnL: ${self.exit_be_pnl:+.2f}")
        print(f"    ├─ Full SL (-2.0x ATR):     {self.exit_sl_count} trades ({(self.exit_sl_count/total*100):.1f}%) | Net PnL: ${self.exit_sl_pnl:+.2f}")
        print(f"    └─ Timeouts (96h):          {self.exit_timeout_count} trades ({(self.exit_timeout_count/total*100):.1f}%) | Net PnL: ${self.exit_timeout_pnl:+.2f}")
        
        print("  [DIRECTIONAL SPLIT]")
        print(f"    ├─ Longs:  {self.long_trades} trades | WinRate: {(self.long_wins/max(1, self.long_trades)*100):.1f}% | Net PnL: ${self.long_pnl:+.2f} | PF: {long_pf:.2f}")
        print(f"    └─ Shorts: {self.short_trades} trades | WinRate: {(self.short_wins/max(1, self.short_trades)*100):.1f}% | Net PnL: ${self.short_pnl:+.2f} | PF: {short_pf:.2f}")

        print("  [LOSS MFE ANALYSIS]")
        print(f"    └─ Avg peak profit before Full SL: {avg_loss_mfe:+.2f}x ATR ({losses_above_1_2:.1f}% reached >= +1.2x ATR | BE Trigger: {self.be_trigger_mult:.1f}x ATR)")

        cost_drag_pct = (self.total_friction / max(1e-4, self.total_gross_pnl)) * 100.0 if self.total_gross_pnl > 0 else 0.0
        print("  [COST & DURATION ATTRIBUTION]")
        print(f"    ├─ Gross PnL: ${self.total_gross_pnl:+.2f} | Friction: -${self.total_friction:.2f} | Net PnL: ${self.total_net_pnl:+.2f} (Cost Drag: {cost_drag_pct:.1f}%)")
        print(f"    └─ Avg Hold (Wins): {avg_hold_win:.1f}h | Avg Hold (Losses): {avg_hold_loss:.1f}h | Avg Risk Multiplier: {avg_risk_mult:.2f}x")

        print("  [HONEST UNCOMPOUNDED EXPECTANCY]")
        if len(self.trade_r_history) > 0:
            mean_r = float(np.mean(self.trade_r_history))
            std_r = float(np.std(self.trade_r_history))
            t_stat = float((mean_r / (std_r + 1e-8)) * np.sqrt(len(self.trade_r_history)))
            print(f"    ├─ Expectancy per Trade: {mean_r:+.3f} R | Std: {std_r:.3f} R | t-stat: {t_stat:.2f} (N={len(self.trade_r_history)})")
        else:
            print(f"    ├─ Expectancy per Trade: +0.000 R | Std: 0.000 R | t-stat: 0.00 (N=0)")
        print(f"    ├─ Disaggregated Exits: Full TP: {(self.exit_tp_count/total*100):.1f}% | BE Scratch: {(self.exit_be_count/total*100):.1f}% | Full SL: {(self.exit_sl_count/total*100):.1f}% | Timeout: {(self.exit_timeout_count/total*100):.1f}%")
        print(f"    └─ True Win Rate (excl BE): {(self.exit_tp_count/total*100):.1f}% (TPs only vs {(self.winning_trades/total*100):.1f}% nominal wins)")
        print("  " + "="*68 + "\n")
