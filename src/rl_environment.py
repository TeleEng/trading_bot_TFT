import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingRLEnv(gym.Env):
    def __init__(self, embeddings, df, tp_mult=3.0, sl_mult=0.75, initial_capital=10000,
                 spread_pips=1.0, slippage_pips=0.3, swap_pips_per_day=0.5):
        super(TradingRLEnv, self).__init__()
        self.embeddings = embeddings
        self.df = df
        
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.initial_capital = initial_capital
        
        # Realistic trading costs
        self.spread = spread_pips * 0.0001   # 1 pip = 0.0001 for EURUSD
        self.slippage = slippage_pips * 0.0001
        self.swap_per_bar = (swap_pips_per_day * 0.0001) / 24.0  # spread across 24 1H bars
        
        # 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Observation: 128 (emb) + 1 (local_ratio) + 10 (last 5 actions+rewards) + 1 (direction) + 1 (unrealized_return) = 141
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 128
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 13,), dtype=np.float32)
        
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
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.capital = self.initial_capital
        self.position = 0.0  # qty (positive = long, negative = short)
        self.entry_price = 0.0
        self.entry_atr = 0.0  # ATR at trade entry, used for fixed TP/SL
        self.bars_held = 0    # track how long current position is held
        
        self.consecutive_losses = 0
        self.trade_history = [(0.0, 0.0)] * 5
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
        for action, reward in self.trade_history[-5:]:
            hist.extend([float(action), float(reward)])
            
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
        Uses the ENTRY ATR (self.entry_atr) for fixed TP/SL levels.
        Returns: (trade_closed, trade_reward, exit_price)
        """
        if self.position == 0 or self.entry_atr <= 0:
            return False, 0.0, 0.0
            
        tp_distance = self.tp_mult * self.entry_atr
        sl_distance = self.sl_mult * self.entry_atr
        
        if self.position > 0:  # Long
            tp_price = self.entry_price + tp_distance
            sl_price = self.entry_price - sl_distance
            
            # SL checked first (conservative: assume adverse move happens first)
            if low <= sl_price:
                exit_price = self._get_fill_price(sl_price, -1)  # selling to close
                return True, -1.0, exit_price
            elif high >= tp_price:
                exit_price = self._get_fill_price(tp_price, -1)
                return True, 1.0, exit_price
                
        else:  # Short
            tp_price = self.entry_price - tp_distance
            sl_price = self.entry_price + sl_distance
            
            # SL checked first (conservative)
            if high >= sl_price:
                exit_price = self._get_fill_price(sl_price, 1)  # buying to close
                return True, -1.0, exit_price
            elif low <= tp_price:
                exit_price = self._get_fill_price(tp_price, 1)
                return True, 1.0, exit_price
                
        return False, 0.0, 0.0
        
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
        risk_per_unit = atr * self.sl_mult
        ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
        max_qty_allowed = (self.portfolio_value * 50 * 0.98) / fill_price
        target_qty = min(ideal_qty, max_qty_allowed) * direction
        
        self.position = target_qty
        self.entry_price = fill_price
        self.entry_atr = atr  # Lock in ATR at entry for fixed TP/SL
        self.bars_held = 0
        
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
        
        # --- Deduct overnight swap cost for open positions ---
        if self.position != 0:
            self.bars_held += 1
            swap_cost = abs(self.position) * self.swap_per_bar
            self.capital -= swap_cost
        
        # --- Check TP/SL using High/Low and fixed entry ATR ---
        if self.position != 0:
            trade_closed, trade_reward, exit_price = self._check_tp_sl_intrabar(high, low)
            
            if trade_closed:
                self._close_position(exit_price)
                if trade_reward > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    
        if trade_closed:
            last_dir = 1.0 if current_direction > 0 else 2.0
            self.trade_history.append((last_dir, trade_reward))
            self.trade_history.pop(0)
            self.portfolio_history.append(self.portfolio_value)
            self.portfolio_history.pop(0)
            current_direction = 0.0
            
        # --- Handle agent action ---
        # The agent can ONLY act if there is NO open position.
        # Once in a trade, it MUST hold until TP or SL is hit.
        action_reward = 0.0
        
        if self.position == 0:
            target_direction = 0.0
            if action == 1: target_direction = 1.0
            elif action == 2: target_direction = -1.0
            
            if target_direction != 0:
                self._open_position(price, atr, target_direction)
                # Small penalty for entering a trade to discourage excessive trading
                action_reward = -0.05

        # Update portfolio value (mark-to-market at mid-price)
        self.portfolio_value = self.capital + (self.position * (price - self.entry_price) if self.position != 0 else 0)
        
        step_reward = (self.portfolio_value - self.last_portfolio_value) / self.initial_capital
        reward = step_reward * 100 + action_reward + (trade_reward * 5.0)
        
        if self.consecutive_losses >= 3:
            reward -= 2.0
            
        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        
        terminated = self.portfolio_value <= 0 or self.current_step >= len(self.df) - 1
        
        return self._get_obs(), float(reward), terminated, False, {}
