import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingRLEnv(gym.Env):
    def __init__(self, embeddings, df, long_tp_mult=2.5, long_sl_mult=1.0, short_tp_mult=4.0, short_sl_mult=1.0, initial_capital=10000,
                 spread_pips=1.0, slippage_pips=0.3, swap_pips_per_day=0.5, is_eval=False):
        super(TradingRLEnv, self).__init__()
        self.is_eval = is_eval
        self.embeddings = embeddings
        self.df = df
        
        self.long_tp_mult = long_tp_mult
        self.long_sl_mult = long_sl_mult
        self.short_tp_mult = short_tp_mult
        self.short_sl_mult = short_sl_mult
        self.initial_capital = initial_capital
        
        # Realistic trading costs
        self.spread = spread_pips * 0.0001   # 1 pip = 0.0001 for EURUSD
        self.slippage = slippage_pips * 0.0001
        self.swap_per_bar = (swap_pips_per_day * 0.0001) / 24.0  # spread across 24 1H bars
        
        # 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Observation: emb_size + 1 (local_ratio) + 15 (last 5 trades: action, reward, exit_reason) + 1 (direction) + 1 (unrealized_return)
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 256
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 18,), dtype=np.float32)
        
        self.reset()
        
    def _get_fill_price(self, price, direction):
        """Apply spread and slippage to get a realistic fill price.
        Buying (direction > 0): you pay the ASK = mid + half_spread + slippage
        Selling (direction < 0): you get the BID = mid - half_spread - slippage
        """
        half_spread = self.spread / 2.0
        slip = self.slippage * self.np_random.random()  # random 0 to max slippage
        if direction > 0:  # buying
            return price + half_spread + slip
        else:  # selling
            return price - half_spread - slip
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize starting point during training so the agent sees the whole time series even if it crashes early
        if not self.is_eval and len(self.df) > 1000:
            self.current_step = self.np_random.integers(0, len(self.df) - 1000)
        else:
            self.current_step = 0
            
        self.portfolio_value = self.initial_capital
        self.capital = self.initial_capital
        self.position = 0.0  # qty (positive = long, negative = short)
        self.entry_price = 0.0
        self.entry_atr = 0.0  # ATR at trade entry, used for fixed TP/SL
        self.bars_held = 0    # track how long current position is held
        
        self.current_sl_price = 0.0
        self.tp_price = 0.0
        
        # Differential Sharpe Ratio trackers
        self.dsr_A = 0.0
        self.dsr_B = 0.0
        self.dsr_eta = 0.01
        
        self.consecutive_losses = 0
        self.bars_flat = 0
        self.trade_history = [(0.0, 0.0, 0.0)] * 5
        self.portfolio_history = [self.initial_capital] * 5
        
        self.last_portfolio_value = self.initial_capital
        
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
            
        current_row = self.df.iloc[self.current_step]
        price = current_row['Close']
        
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
        # Handle NaNs safely
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
        
    def _check_tp_sl_intrabar(self, high, low):
        """Check TP/SL using High/Low of the bar (intra-bar simulation).
        Uses dynamic SL (moves to 25% TP if 50% TP is hit).
        Returns: (trade_closed, trade_reward, exit_price, exit_reason)
        exit_reason: 1 (TP), -1 (SL), 0 (None/Timeout)
        """
        if self.position == 0 or self.entry_atr <= 0:
            return False, 0.0, 0.0, 0
            
        if self.position > 0:  # Long
            # SL checked first (conservative: assume adverse move happens first)
            if low <= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, -1)  # selling to close
                return True, -0.286, exit_price, -1  # Remainder of -1.0 (-1.0 - -0.714)
            elif high >= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, -1)
                return True, 3.214, exit_price, 1  # 2.5 + 0.714 refunded
                
        else:  # Short
            # SL checked first (conservative)
            if high >= self.current_sl_price:
                exit_price = self._get_fill_price(self.current_sl_price, 1)  # buying to close
                return True, -0.200, exit_price, -1  # Remainder of -1.0 (-1.0 - -0.800)
            elif low <= self.tp_price:
                exit_price = self._get_fill_price(self.tp_price, 1)
                return True, 4.800, exit_price, 1  # 4.0 + 0.800 refunded
                
        return False, 0.0, 0.0, 0
        
    def _close_position(self, exit_price):
        """Close position and update capital with PnL."""
        pnl = self.position * (exit_price - self.entry_price)
        self.capital += pnl
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.bars_held = 0
        return pnl
        
    def _open_position(self, price, atr, direction):
        """Open a new position with spread/slippage applied."""
        fill_price = self._get_fill_price(price, direction)
        
        risk_amount = self.portfolio_value * 0.02
        sl_mult = self.long_sl_mult if direction > 0 else self.short_sl_mult
        risk_per_unit = atr * sl_mult
        ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
        max_qty_allowed = (self.portfolio_value * 50 * 0.98) / fill_price
        target_qty = min(ideal_qty, max_qty_allowed) * direction
        
        self.position = target_qty
        self.entry_price = fill_price
        self.entry_atr = atr  # Lock in ATR at entry for fixed TP/SL
        self.bars_held = 0
        
        # Initialize dynamic TP/SL levels
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
            return self._get_obs(), 0.0, True, False, {}
            
        current_row = self.df.iloc[self.current_step]
        price = current_row['Close']
        high = current_row.get('High', price)
        low = current_row.get('Low', price)
        atr = current_row.get('ATR', price * 0.005)
        
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
                exit_reason = 0 # Timeout
                # Make timeout identical to SL remainder
                trade_reward = -0.286 if self.position > 0 else -0.200
                    
            if trade_closed:
                self._close_position(exit_price)
                    
        if trade_closed:
            last_dir = 1.0 if current_direction > 0 else 2.0
            self.trade_history.append((last_dir, trade_reward, exit_reason))
            self.trade_history.pop(0)
            self.portfolio_history.append(self.portfolio_value)
            self.portfolio_history.pop(0)
            current_direction = 0.0
            
        # --- Handle agent action ---
        # The agent can ONLY act if there is NO open position.
        # Once in a trade, it MUST hold until TP or SL is hit.
        if self.position == 0:
            self.bars_flat += 1
            if action == 1: target_direction = 1.0
            elif action == 2: target_direction = -1.0
            
            if target_direction != 0:
                self._open_position(price, atr, target_direction)
                
            # Dampen portfolio history towards current value so local_ratio decays back to 1.0
            # This prevents the agent from sitting flat forever just because it had a good streak.
            alpha = 0.01 # 1% decay per hour flat
            for i in range(len(self.portfolio_history)):
                self.portfolio_history[i] = (1 - alpha) * self.portfolio_history[i] + alpha * self.portfolio_value
        else:
            self.bars_flat = 0

        # Update portfolio value (mark-to-market at mid-price)
        self.portfolio_value = self.capital + (self.position * (price - self.entry_price) if self.position != 0 else 0)
        # Apply upfront penalty including expected loss (Credit Assignment)
        action_penalty = 0.0
        if target_direction != 0:
            if target_direction > 0:
                action_penalty = -0.779  # -0.065 (cost) + -0.714 (expected SL hit)
            else:
                action_penalty = -0.865  # -0.065 (cost) + -0.800 (expected SL hit)
            
        # Agent's only goal is to maximize pure R:R and avoid the flat/action penalties
        reward = trade_reward + action_penalty
            
        # Apply flat penalty of 1/12th the original action penalty per hour
        if self.position == 0:
            reward -= 0.0042
            
        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        
        # Check for bankruptcy (portfolio drops below 10% of initial capital)
        is_bankrupt = self.portfolio_value < (self.initial_capital * 0.1)
        
        if is_bankrupt:
            reward -= 100.0  # Massive terminal penalty for "RL Suicide" prevention
            
        terminated = is_bankrupt or self.current_step >= len(self.df) - 1
        
        return self._get_obs(), float(reward), terminated, False, {}
