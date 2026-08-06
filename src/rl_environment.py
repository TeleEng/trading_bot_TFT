import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingRLEnv(gym.Env):
    def __init__(self, embeddings, df, tp_mult=1.5, sl_mult=0.75, initial_capital=10000):
        super(TradingRLEnv, self).__init__()
        self.embeddings = embeddings
        self.df = df
        
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.initial_capital = initial_capital
        
        # 0: Flat, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Observation: 128 (emb) + 1 (local_ratio) + 10 (last 5 actions+rewards) + 1 (direction) + 1 (unrealized_return) = 141
        emb_size = embeddings.shape[1] if len(embeddings) > 0 else 128
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(emb_size + 13,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.capital = self.initial_capital # Free cash essentially
        self.position = 0.0 # qty
        self.entry_price = 0.0
        
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
                
        # Scale up unrealized return for network visibility (similar to action_reward scaling)
        unrealized_return *= 100.0
            
        obs = np.concatenate([emb, [local_ratio], hist, [current_direction, unrealized_return]]).astype(np.float32)
        # Handle NaNs safely
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
        
    def step(self, action):
        if self.current_step >= len(self.df) - 1 or self.portfolio_value <= 0:
            return self._get_obs(), 0.0, True, False, {}
            
        current_row = self.df.iloc[self.current_step]
        price = current_row['Close']
        atr = current_row.get('ATR', price * 0.005)
        
        # Cap ATR
        atr = max(price * 0.001, min(atr, price * 0.02))
        
        trade_closed = False
        trade_reward = 0.0
        current_direction = 1.0 if self.position > 0 else (-1.0 if self.position < 0 else 0.0)
        
        # Check TP / SL
        if self.position != 0:
            tp_pct = (self.tp_mult * atr) / self.entry_price
            sl_pct = -((self.sl_mult * atr) / self.entry_price)
            
            unrealized_return = (price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - price) / self.entry_price
            
            if unrealized_return >= tp_pct:
                trade_closed = True
                trade_reward = 1.0
                self.capital += self.position * (price - self.entry_price)
                self.consecutive_losses = 0
            elif unrealized_return <= sl_pct:
                trade_closed = True
                trade_reward = -1.0
                self.capital += self.position * (price - self.entry_price)
                self.consecutive_losses += 1
                
        if trade_closed:
            last_dir = 1.0 if current_direction > 0 else 2.0
            self.trade_history.append((last_dir, trade_reward))
            self.trade_history.pop(0)
            self.portfolio_history.append(self.portfolio_value)
            self.portfolio_history.pop(0)
            self.position = 0.0
            current_direction = 0.0
            
        target_direction = 0.0
        if action == 1: target_direction = 1.0
        elif action == 2: target_direction = -1.0
        
        action_reward = 0.0
        if target_direction != current_direction and not trade_closed:
            if self.position != 0:
                unrealized_return = (price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - price) / self.entry_price
                if unrealized_return > 0:
                    action_reward = unrealized_return * 100
                    self.consecutive_losses = 0
                else:
                    action_reward = unrealized_return * 100
                    self.consecutive_losses += 1
                    
                self.capital += self.position * (price - self.entry_price)
                last_dir = 1.0 if current_direction > 0 else 2.0
                self.trade_history.append((last_dir, action_reward))
                self.trade_history.pop(0)
                self.portfolio_history.append(self.portfolio_value)
                self.portfolio_history.pop(0)
                self.position = 0.0
                
            if target_direction != 0:
                risk_amount = self.portfolio_value * 0.02
                risk_per_unit = atr * self.sl_mult
                ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
                max_qty_allowed = (self.portfolio_value * 50 * 0.98) / price
                target_qty = min(ideal_qty, max_qty_allowed) * target_direction
                
                self.position = target_qty
                self.entry_price = price
                
        elif trade_closed and target_direction != 0:
            risk_amount = self.portfolio_value * 0.02
            risk_per_unit = atr * self.sl_mult
            ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
            max_qty_allowed = (self.portfolio_value * 50 * 0.98) / price
            target_qty = min(ideal_qty, max_qty_allowed) * target_direction
            
            self.position = target_qty
            self.entry_price = price

        self.portfolio_value = self.capital + (self.position * (price - self.entry_price) if self.position != 0 else 0)
        
        step_reward = (self.portfolio_value - self.last_portfolio_value) / self.initial_capital
        reward = step_reward * 100 + action_reward + (trade_reward * 5.0)
        
        if self.consecutive_losses >= 3:
            reward -= 2.0
            
        self.last_portfolio_value = self.portfolio_value
        self.current_step += 1
        
        terminated = self.portfolio_value <= 0 or self.current_step >= len(self.df) - 1
        
        return self._get_obs(), float(reward), terminated, False, {}
