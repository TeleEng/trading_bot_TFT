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
                 breakeven_sl=True,
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
        self.reward_mode = reward_mode
        self.flat_penalty = flat_penalty
        self.max_risk_multiplier = max_risk_multiplier
        self.cooldown_bars = cooldown_bars
        self.trend_filter = trend_filter
        self.volatility_filter = volatility_filter
        self.min_atr_ratio = min_atr_ratio
        self.max_atr_ratio = max_atr_ratio
        self.breakeven_sl = breakeven_sl
        self.max_wait_bars = max_wait_bars
        
        # Extract numpy arrays for O(1) lookups during RL steps
        self.close_prices = self.df['Close'].values
        self.high_prices = self.df['High'].values if 'High' in self.df.columns else self.close_prices
        self.low_prices = self.df['Low'].values if 'Low' in self.df.columns else self.close_prices
        self.atr_values = self.df['ATR'].values if 'ATR' in self.df.columns else self.close_prices * 0.005
        self.dist_sma_50 = self.df['Dist_SMA_50'].values if 'Dist_SMA_50' in self.df.columns else None
        self.atr_ratio = self.df['ATR_Ratio'].values if 'ATR_Ratio' in self.df.columns else None
        self.rsi = self.df['RSI'].values if 'RSI' in self.df.columns else None
        
        # Extract probabilities if available
        if probabilities is not None and len(probabilities) > 0:
            self.prob_long = probabilities[:, 0]
            self.prob_short = probabilities[:, 1]
        else:
            self.prob_long = None
            self.prob_short = None
        
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
        
        # Observation: emb_size + 1 (local_ratio) + 15 (last 5 trades) + 2 (direction, unrealized) + 5 (probs + regimes) = emb_size + 23
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 256
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 23,), dtype=np.float32)
        
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
        
        # Probabilities & Market Regimes
        p_long = float(self.prob_long[self.current_step]) if self.prob_long is not None else 0.5
        p_short = float(self.prob_short[self.current_step]) if self.prob_short is not None else 0.5
        macro_dist = float(self.dist_sma_50[self.current_step]) if self.dist_sma_50 is not None else 0.0
        atr_ratio = float(self.atr_ratio[self.current_step]) if self.atr_ratio is not None else 1.0
        rsi_val = float((self.rsi[self.current_step] - 50.0) / 50.0) if self.rsi is not None else 0.0
            
        obs = np.concatenate([
            emb, 
            [local_ratio], 
            hist, 
            [current_direction, unrealized_return],
            [p_long, p_short, macro_dist, atr_ratio, rsi_val]
        ]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
        
    def _check_tp_sl_intrabar(self, high, low):
        """Check TP/SL using High/Low of the bar (intra-bar simulation).
        Returns: (trade_closed, trade_reward, exit_price, exit_reason)
        exit_reason: 1 (TP), -1 (SL), 0 (None/Timeout)
        """
        if self.position == 0 or self.entry_atr <= 0:
            return False, 0.0, 0.0, 0
            
        if self.position > 0:  # Long
            # SL checked first (conservative execution)
            if low <= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, -1)
                reward = -1.0 * self.long_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            elif high >= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, -1)
                reward = self.long_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
            else:
                # Ratchet Break-Even Stop Loss if favorable price move >= 1.5x ATR
                if self.breakeven_sl and high >= (self.entry_price + 1.5 * self.entry_atr):
                    be_price = self.entry_price + (0.1 * self.entry_atr)
                    if be_price > self.current_sl_price:
                        self.current_sl_price = be_price
                
        else:  # Short
            # SL checked first (conservative execution)
            if high >= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, 1)
                reward = -1.0 * self.short_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            elif low <= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, 1)
                reward = self.short_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
            else:
                # Ratchet Break-Even Stop Loss if favorable price move >= 1.5x ATR
                if self.breakeven_sl and low <= (self.entry_price - 1.5 * self.entry_atr):
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
        high = self.high_prices[self.current_step]
        low = self.low_prices[self.current_step]
        atr = self.atr_values[self.current_step]
        
        # Cap ATR
        atr = max(price * 0.001, min(atr, price * 0.02))
        
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
            trade_closed, _, exit_price, exit_reason = self._check_tp_sl_intrabar(high, low)
            
            # --- Check Time Limit (96 Hours) ---
            if not trade_closed and self.bars_held >= self.max_wait_bars:
                trade_closed = True
                exit_price = price
                exit_reason = 0  # Timeout
                self.timeout_trades += 1
                
            if trade_closed:
                gross_pnl = self._close_position(exit_price)
                net_pnl = gross_pnl - self.trade_accumulated_swap
                
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                    # Trigger revenge-trade prevention cooldown in the losing direction
                    self.cooldown_remaining = self.cooldown_bars
                    self.last_sl_direction = current_direction
                    
                # Realized risk-normalized net PnL reward
                trade_reward = float(np.clip(net_pnl / self.trade_initial_risk, -3.0, 3.0))
                    
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
            
            raw_action = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            raw_action = float(np.clip(raw_action, -1.0, 1.0))
            
            if abs(raw_action) > self.deadzone:
                desired_direction = 1.0 if raw_action > 0 else -1.0
                
                # Check 1: Cooldown lockout against revenge trading in the same direction
                is_cooled_down = (self.cooldown_remaining == 0) or (desired_direction != self.last_sl_direction)
                
                # Check 2: Macro Trend Filter (SMA-50 alignment)
                trend_confirmed = True
                if self.trend_filter and self.dist_sma_50 is not None:
                    macro_dist = self.dist_sma_50[self.current_step]
                    if desired_direction > 0 and macro_dist < -0.5:
                        trend_confirmed = False  # Avoid longs in strong macro downtrend
                    elif desired_direction < 0 and macro_dist > 0.5:
                        trend_confirmed = False  # Avoid shorts in strong macro uptrend
                
                # Check 3: Backbone Probability Gate & Relative Probability Edge
                prob_confirmed = True
                if self.prob_long is not None and self.prob_short is not None:
                    p_l = float(self.prob_long[self.current_step])
                    p_s = float(self.prob_short[self.current_step])
                    if desired_direction > 0:
                        if p_l < self.min_confidence or (p_l - p_s) < self.min_edge:
                            prob_confirmed = False
                    elif desired_direction < 0:
                        if p_s < self.min_confidence or (p_s - p_l) < self.min_edge:
                            prob_confirmed = False
                
                # Check 4: Volatility Regime Filter (avoid dead zones and volatile blowouts)
                vol_confirmed = True
                if self.volatility_filter and self.atr_ratio is not None:
                    curr_atr_ratio = float(self.atr_ratio[self.current_step])
                    if curr_atr_ratio < self.min_atr_ratio or curr_atr_ratio > self.max_atr_ratio:
                        vol_confirmed = False
                
                if is_cooled_down and trend_confirmed and prob_confirmed and vol_confirmed:
                    target_direction = desired_direction
                    conviction = (abs(raw_action) - self.deadzone) / (1.0 - self.deadzone + 1e-8)
                    conviction = float(np.clip(conviction, 0.0, 1.0))
                    risk_multiplier = 1.0 + conviction * (self.max_risk_multiplier - 1.0)
                    self._open_position(price, atr, target_direction, risk_multiplier)
                else:
                    target_direction = 0.0
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
