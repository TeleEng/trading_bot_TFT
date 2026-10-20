import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add the src directory to the path
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
            slippage_pips=0.0,  # deterministic fill
            swap_pips_per_day=0.5,
            deadzone=0.10,
            reward_mode='asymmetric_rr',
            flat_penalty=0.005,
            max_risk_multiplier=5.0
        )
        self.env.reset()

    def test_spread_application(self):
        buy_price = self.env._get_fill_price(1.1000, 1)
        sell_price = self.env._get_fill_price(1.1000, -1)
        self.assertAlmostEqual(buy_price, 1.10005, places=5)
        self.assertAlmostEqual(sell_price, 1.09995, places=5)

    def test_position_sizing(self):
        # Risk amount = 10000 * 0.01 * 2.0 = 200
        # Risk per unit = ATR (0.0020) * 1.0 = 0.0020
        # Ideal qty = 200 / 0.0020 = 100000.0
        self.env._open_position(1.1000, 0.0020, 1, 2.0)
        self.assertAlmostEqual(self.env.position, 100000.0, places=2)
        
    def test_overnight_swap_deduction(self):
        self.env._open_position(1.1000, 0.0020, 1, 2.0)
        initial_cap = self.env.capital
        self.env.step([0.0])  # hold for 1 bar
        expected_cap = initial_cap - (100000.0 * 2.083333e-6)
        self.assertAlmostEqual(self.env.capital, expected_cap, places=2)
        
    def test_deadzone_flat(self):
        # Action inside deadzone [-0.10, 0.10] must result in 0 trades / flat
        obs, reward, done, trunc, info = self.env.step([0.05])
        self.assertEqual(self.env.position, 0.0)
        self.assertEqual(self.env.total_trades, 0)
        self.assertAlmostEqual(reward, -0.005, places=4)

    def test_zero_upfront_penalty(self):
        # Opening a trade should NOT hit the agent with an upfront penalty
        obs, reward, done, trunc, info = self.env.step([0.55])
        self.assertTrue(self.env.position > 0)
        self.assertEqual(self.env.total_trades, 1)
        # Entry reward should be 0.0 (no arbitrary penalty at step 0)
        self.assertEqual(reward, 0.0)

    def test_agent_action_lockdown(self):
        self.env.step([0.55])  # Enter long
        pos_before = self.env.position
        self.assertTrue(pos_before > 0)
        
        # Action to reverse should be locked out while position is open
        self.env.step([-0.55])
        self.assertEqual(self.env.position, pos_before)

    def test_fixed_sl_hit(self):
        # Enter Long with conviction: raw action 0.325 -> conviction = 0.25 -> risk_multiplier = 2.0
        self.env.step([0.325])
        self.assertAlmostEqual(self.env.current_risk_multiplier, 2.0, places=2)
        
        # Modify next row in dataframe to hit SL
        self.env.df.at[self.env.current_step, 'Low'] = 1.0970
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        # Position closed
        self.assertEqual(self.env.position, 0.0)
        # SL reward: -1.0 * long_sl_mult * risk_mult = -1.0 * 1.0 * 2.0 = -2.0
        self.assertAlmostEqual(reward, -2.0, places=2)
        self.assertEqual(self.env.losing_trades, 1)

    def test_fixed_tp_hit(self):
        # Enter Long with risk_multiplier = 2.0
        self.env.step([0.325])
        
        # Modify next row to hit TP (1.10005 + 2.5 * 0.0020 = 1.10505)
        self.env.df.at[self.env.current_step, 'Low'] = 1.1010
        self.env.df.at[self.env.current_step, 'High'] = 1.1060
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # TP reward: 2.5 * risk_mult = 2.5 * 2.0 = 5.0
        self.assertAlmostEqual(reward, 5.0, places=2)
        self.assertEqual(self.env.winning_trades, 1)

    def test_short_sl_hit(self):
        # Enter Short with risk_multiplier = 2.0
        self.env.step([-0.325])
        self.assertTrue(self.env.position < 0)
        
        # Force SL hit on short (price rises)
        self.env.df.at[self.env.current_step, 'High'] = 1.1030
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # Short SL reward: -1.0 * 2.0 = -2.0
        self.assertAlmostEqual(reward, -2.0, places=2)

    def test_short_tp_hit(self):
        # Enter Short with risk_multiplier = 2.0
        self.env.step([-0.325])
        
        # Force TP hit on short (price drops to 1.0910)
        self.env.df.at[self.env.current_step, 'High'] = 1.1010
        self.env.df.at[self.env.current_step, 'Low'] = 1.0900
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # Short TP reward: 4.0 * 2.0 = 8.0
        self.assertAlmostEqual(reward, 8.0, places=2)

    def test_intrabar_collision_conservative(self):
        # Enter Long
        self.env.step([0.325])
        
        # Huge bar touching both TP and SL
        self.env.df.at[self.env.current_step, 'High'] = 1.1070
        self.env.df.at[self.env.current_step, 'Low'] = 1.0960
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        # SL must trigger first (conservative execution)
        self.assertEqual(self.env.position, 0.0)
        self.assertAlmostEqual(reward, -2.0, places=2)
        self.assertEqual(self.env.trade_history[-1][2], -1)


if __name__ == '__main__':
    unittest.main()
