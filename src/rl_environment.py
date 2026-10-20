import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingRLEnv(gym.Env):
    """
    High-performance Trading RL Environment with continuous action space,
    dynamic position sizing, and decoupled risk-reward reward formulations.
    """
    def __init__(self, embeddings, df, 
                 long_tp_mult=2.5, long_sl_mult=1.0, 
                 short_tp_mult=4.0, short_sl_mult=1.0, 
                 initial_capital=10000,
                 spread_pips=1.0, slippage_pips=0.3, swap_pips_per_day=0.5, 
                 is_eval=False,
                 deadzone=0.10,
                 reward_mode='asymmetric_rr',
                 flat_penalty=0.005,
                 max_risk_multiplier=5.0):
        super(TradingRLEnv, self).__init__()
        self.is_eval = is_eval
        self.embeddings = embeddings
        self.df = df
        
        # Hyperparameters
        self.deadzone = deadzone
        self.reward_mode = reward_mode
        self.flat_penalty = flat_penalty
        self.max_risk_multiplier = max_risk_multiplier
        
        # Extract numpy arrays for O(1) lookups during RL steps (massive speedup over pandas .iloc)
        self.close_prices = self.df['Close'].values
        self.high_prices = self.df['High'].values if 'High' in self.df.columns else self.close_prices
        self.low_prices = self.df['Low'].values if 'Low' in self.df.columns else self.close_prices
        self.atr_values = self.df['ATR'].values if 'ATR' in self.df.columns else self.close_prices * 0.005
        
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
        
        # Observation: emb_size + 1 (local_ratio) + 15 (last 5 trades: action, reward, exit_reason) + 1 (direction) + 1 (unrealized_return)
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 256
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 18,), dtype=np.float32)
        
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
        self.bars_held = 0
        
        self.current_sl_price = 0.0
        self.tp_price = 0.0
        
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
            
        obs = np.concatenate([emb, [local_ratio], hist, [current_direction, unrealized_return]]).astype(np.float32)
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
            # SL checked first (conservative)
            if low <= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, -1)
                # Long SL: -1.0x risk
                reward = -1.0 * self.long_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            elif high >= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, -1)
                # Long TP: +2.5x risk
                reward = self.long_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
                
        else:  # Short
            # SL checked first (conservative)
            if high >= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, 1)
                # Short SL: -1.0x risk
                reward = -1.0 * self.short_sl_mult * self.current_risk_multiplier
                return True, reward, exit_price, -1
            elif low <= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, 1)
                # Short TP: +4.0x risk
                reward = self.short_tp_mult * self.current_risk_multiplier
                return True, reward, exit_price, 1
                
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
        
        # Base risk is 1%, scaled dynamically by conviction
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
        
        # --- Deduct overnight swap cost for open positions ---
        if self.position != 0:
            self.bars_held += 1
            swap_cost = abs(self.position) * self.swap_per_bar
            self.capital -= swap_cost
        
        # --- Check TP/SL using High/Low and fixed entry ATR ---
        if self.position != 0:
            trade_closed, trade_reward, exit_price, exit_reason = self._check_tp_sl_intrabar(high, low)
            
            # --- Check Time Limit (4 Days = 96 Hours) ---
            if not trade_closed and self.bars_held >= 96:
                trade_closed = True
                exit_price = price
                exit_reason = 0  # Timeout
                self.timeout_trades += 1
                
                # Proportional timeout reward based on distance to SL/TP
                sl_mult = self.long_sl_mult if self.position > 0 else self.short_sl_mult
                denom = (self.entry_atr * sl_mult) + 1e-8
                if self.position > 0:
                    pnl_ratio = (exit_price - self.entry_price) / denom
                else:
                    pnl_ratio = (self.entry_price - exit_price) / denom
                trade_reward = float(np.clip(pnl_ratio, -1.0, 1.0)) * self.current_risk_multiplier
                
            if trade_closed:
                if exit_reason == 1:
                    self.winning_trades += 1
                elif exit_reason == -1:
                    self.losing_trades += 1
                self._close_position(exit_price)
                    
        if trade_closed:
            last_dir = 1.0 if current_direction > 0 else 2.0
            self.trade_history.append((last_dir, trade_reward, exit_reason))
            self.trade_history.pop(0)
            self.portfolio_history.append(self.portfolio_value)
            self.portfolio_history.pop(0)
            current_direction = 0.0
            
        # --- Handle agent action ---
        # The agent can act when there is NO open position and it didn't close one this bar.
        risk_multiplier = 0.0
        if self.position == 0 and not trade_closed:
            self.bars_flat += 1
            
            raw_action = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
            raw_action = float(np.clip(raw_action, -1.0, 1.0))
            
            if abs(raw_action) > self.deadzone:
                target_direction = 1.0 if raw_action > 0 else -1.0
                conviction = (abs(raw_action) - self.deadzone) / (1.0 - self.deadzone)
                conviction = float(np.clip(conviction, 0.0, 1.0))
                risk_multiplier = 1.0 + conviction * (self.max_risk_multiplier - 1.0)
                self._open_position(price, atr, target_direction, risk_multiplier)
            else:
                target_direction = 0.0
                
            # Decay portfolio history toward current value so local_ratio returns to 1.0
            alpha = 0.01
            for i in range(len(self.portfolio_history)):
                self.portfolio_history[i] = (1 - alpha) * self.portfolio_history[i] + alpha * self.portfolio_value
        else:
            self.bars_flat = 0

        # Update portfolio value (mark-to-market)
        self.portfolio_value = self.capital + (self.position * (price - self.entry_price) if self.position != 0 else 0)
        
        # --- Reward Calculation based on reward_mode ---
        if self.reward_mode == 'asymmetric_rr':
            # Clean realized risk-reward upon trade exit, zero upfront entry penalty
            reward = trade_reward
            
            # Modest flat penalty to encourage seeking opportunities
            if self.position == 0 and not trade_closed:
                # Progressive flat penalty if sitting idle for long stretches (> 48 hours)
                idling_factor = 1.0 + (0.02 * min(self.bars_flat, 200))
                reward -= (self.flat_penalty * idling_factor)
                
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
            if self.position == 0 and not trade_closed:
                reward -= self.flat_penalty
                
        elif self.reward_mode == 'pnl_dense':
            # Step-wise normalized PnL
            step_return = (self.portfolio_value - self.last_portfolio_value) / self.initial_capital * 100.0
            reward = float(step_return) + (trade_reward * 0.5)
            if self.position == 0 and not trade_closed:
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
