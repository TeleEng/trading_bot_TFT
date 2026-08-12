import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add the src directory to the path so we can import rl_environment
sys.path.append(os.path.dirname(__file__))
from rl_environment import TradingRLEnv

class TestTradingRLEnv(unittest.TestCase):
    def setUp(self):
        # Create a mock 10-step dataframe
        data = {
            'Close': [1.1000] * 10,
            'High': [1.1010] * 10,
            'Low': [1.0990] * 10,
            'ATR': [0.0020] * 10,
        }
        self.df = pd.DataFrame(data)
        self.embeddings = np.zeros((10, 10))  # 10 steps, emb size 10
        
        self.env = TradingRLEnv(
            embeddings=self.embeddings,
            df=self.df,
            long_tp_mult=2.5,
            long_sl_mult=1.0,
            short_tp_mult=4.0,
            short_sl_mult=1.0,
            initial_capital=10000,
            spread_pips=1.0,
            slippage_pips=0.0, # no random slippage for deterministic testing
            swap_pips_per_day=0.5
        )
        self.env.reset()

    def test_spread_application(self):
        # 1.1000 with 1.0 pip spread (0.0001) means half_spread = 0.00005
        # Buy price should be 1.10005, Sell price should be 1.09995
        buy_price = self.env._get_fill_price(1.1000, 1)
        sell_price = self.env._get_fill_price(1.1000, -1)
        self.assertAlmostEqual(buy_price, 1.10005, places=5)
        self.assertAlmostEqual(sell_price, 1.09995, places=5)

    def test_position_sizing(self):
        # Risk amount = 10000 * 0.02 = 200
        # Risk per unit = ATR (0.0020) * 1.0 = 0.0020
        # Ideal qty = 200 / 0.0020 = 100000.0
        self.env._open_position(1.1000, 0.0020, 1) # Long
        self.assertAlmostEqual(self.env.position, 100000.0, places=2)
        
    def test_overnight_swap_deduction(self):
        self.env._open_position(1.1000, 0.0020, 1)
        initial_cap = self.env.capital
        self.env.step(0) # hold for 1 bar
        
        # Swap cost per bar = (0.5 * 0.0001) / 24.0 = 2.0833e-6 per unit
        # Qty = 100000.0 -> Total swap cost = 0.20833
        expected_cap = initial_cap - (100000.0 * 2.083333e-6)
        self.assertAlmostEqual(self.env.capital, expected_cap, places=2)
        
    def test_agent_action_lockdown(self):
        self.env.step(1) # Enter long on step 0
        pos_before = self.env.position
        self.assertTrue(pos_before > 0)
        
        # Try to reverse (Short) or Close
        # Action 2 (Short) should be ignored because a position is open
        self.env.step(2)
        self.assertEqual(self.env.position, pos_before) # Still long, not reversed

    def test_fixed_sl_hit(self):
        # Trigger entry long
        self.env.step(1) 
        # Entry price = 1.10005
        # SL = entry - 0.0020 = 1.09805
        
        # Modify the next row in the dataframe to hit SL
        self.env.df.at[self.env.current_step, 'Low'] = 1.0970
        
        # Take step 1 (which will read row 1)
        obs, reward, done, trunc, info = self.env.step(0)
        
        # Should close the trade
        self.assertEqual(self.env.position, 0)
        # trade history should show -0.286 (remainder of -1.0 after upfront penalty)
        self.assertAlmostEqual(self.env.trade_history[-1][1], -0.286, places=3)

    def test_fixed_tp_hit(self):
        # Trigger entry long
        self.env.step(1) 
        # Entry = 1.10005
        # TP = entry + (2.5 * 0.0020) = 1.10505
        
        # Modify the next row in the dataframe to hit TP
        # We must also raise the Low so it doesn't hit SL or secured SL!
        self.env.df.at[self.env.current_step, 'Low'] = 1.1020
        self.env.df.at[self.env.current_step, 'High'] = 1.1060
        
        self.env.step(0)
        
        self.assertEqual(self.env.position, 0)
        self.assertAlmostEqual(self.env.trade_history[-1][1], 3.214, places=3)
        
    def test_timeout_penalty(self):
        self.env.step(1) # Enter long
        
        # Fast forward 96 bars
        self.env.bars_held = 95
        
        # Step 0 to trigger timeout
        self.env.step(0)
        
        # Trade should be closed due to timeout
        self.assertEqual(self.env.position, 0)
        
        # Reward should be -0.286 for timeout
        self.assertAlmostEqual(self.env.trade_history[-1][1], -0.286, places=3)
        # Exit reason should be 0
        self.assertEqual(self.env.trade_history[-1][2], 0)

    def test_exact_pnl_with_spread(self):
        # Initial capital = 10000.0
        self.env.step(1)
        # Entry price = 1.1000 + 0.00005 = 1.10005
        # Risk amount = 200.0
        # Risk per unit = 0.0020
        # Qty = 100000.0
        
        # TP price = 1.10005 + (2.5 * 0.0020) = 1.10505
        # Hit TP!
        self.env.df.at[self.env.current_step, 'Low'] = 1.1000
        self.env.df.at[self.env.current_step, 'High'] = 1.1060
        self.env.step(0)
        
        # Exit fill price = TP price - 0.00005 = 1.10500
        # Expected PnL = 100000.0 * (1.10500 - 1.10005) = 495.0
        # Swap cost for 1 bar holding = (0.5 * 0.0001) / 24 * 100000 = 0.208333
        # Total profit = 495.0 - 0.208333 = 494.791667
        
        self.assertAlmostEqual(self.env.capital, 10494.79, places=2)
        self.assertAlmostEqual(self.env.portfolio_value, 10494.79, places=2)

    def test_short_sl_hit(self):
        # Trigger entry short
        self.env.step(2) 
        # Entry price = 1.09995 (Spread = 1.0 pip)
        # SL = entry + 0.0020 = 1.10195
        
        # Modify next row to hit SL
        self.env.df.at[self.env.current_step, 'High'] = 1.1030
        
        self.env.step(0)
        
        self.assertEqual(self.env.position, 0)
        self.assertAlmostEqual(self.env.trade_history[-1][1], -0.200, places=3)
        
    def test_short_tp_hit(self):
        self.env.step(2)
        # Entry = 1.09995
        # TP = entry - (4.0 * 0.0020) = 1.09195
        
        # Hit TP but not SL
        self.env.df.at[self.env.current_step, 'High'] = 1.1010
        self.env.df.at[self.env.current_step, 'Low'] = 1.0900
        
        self.env.step(0)
        
        self.assertEqual(self.env.position, 0)
        self.assertAlmostEqual(self.env.trade_history[-1][1], 4.800, places=3)
        
    def test_intrabar_collision_long(self):
        self.env.step(1)
        # Entry = 1.10005
        # TP = 1.10505, SL = 1.09805
        
        # Huge candle hitting BOTH TP and SL in the same bar
        self.env.df.at[self.env.current_step, 'High'] = 1.1060
        self.env.df.at[self.env.current_step, 'Low'] = 1.0970
        
        self.env.step(0)
        
        # Should conservatively hit SL first!
        self.assertEqual(self.env.position, 0)
        self.assertAlmostEqual(self.env.trade_history[-1][1], -0.286, places=3)
        self.assertEqual(self.env.trade_history[-1][2], -1)

    def test_intrabar_collision_short(self):
        self.env.step(2)
        # Entry = 1.09995
        # TP = 1.09195, SL = 1.10195
        
        # Huge candle hitting BOTH
        self.env.df.at[self.env.current_step, 'High'] = 1.1030
        self.env.df.at[self.env.current_step, 'Low'] = 1.0900
        
        self.env.step(0)
        
        # Should conservatively hit SL first!
        self.assertEqual(self.env.position, 0)
        self.assertAlmostEqual(self.env.trade_history[-1][1], -0.200, places=3)
        self.assertEqual(self.env.trade_history[-1][2], -1)

if __name__ == '__main__':
    unittest.main()
