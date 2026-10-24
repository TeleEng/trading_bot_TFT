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
            'Dist_SMA_50': [0.1] * 10,  # neutral/mildly bullish
            'ATR_Ratio': [1.0] * 10,
            'RSI': [55.0] * 10
        }
        self.df = pd.DataFrame(data)
        self.embeddings = np.zeros((10, 10))  # 10 steps, emb size 10
        self.probs = np.zeros((10, 2))
        self.probs[:, 0] = 0.55  # Clear long edge
        self.probs[:, 1] = 0.45
        
        self.env = TradingRLEnv(
            embeddings=self.embeddings,
            df=self.df,
            probabilities=self.probs,
            long_tp_mult=3.0,
            long_sl_mult=2.0,
            short_tp_mult=3.0,
            short_sl_mult=2.0,
            initial_capital=10000,
            spread_pips=1.0,
            slippage_pips=0.0,  # deterministic fill
            swap_pips_per_day=0.5,
            deadzone=0.20,
            min_confidence=0.50,
            min_edge=0.01,
            reward_mode='net_rr',
            flat_penalty=0.0,
            max_risk_multiplier=1.5,
            cooldown_bars=4,
            trend_filter=True,
            volatility_filter=True,
            breakeven_sl=True,
            max_wait_bars=96
        )
        self.env.reset()

    def test_spread_application(self):
        buy_price = self.env._get_fill_price(1.1000, 1)
        sell_price = self.env._get_fill_price(1.1000, -1)
        self.assertAlmostEqual(buy_price, 1.10005, places=5)
        self.assertAlmostEqual(sell_price, 1.09995, places=5)

    def test_position_sizing(self):
        # Risk amount = 10000 * 0.01 * 1.5 = 150
        # Risk per unit = ATR (0.0020) * 2.0 = 0.0040
        # Ideal qty = 150 / 0.0040 = 37500.0
        self.env._open_position(1.1000, 0.0020, 1, 1.5)
        self.assertAlmostEqual(self.env.position, 37500.0, places=2)
        
    def test_overnight_swap_deduction(self):
        self.env._open_position(1.1000, 0.0020, 1, 1.5)
        initial_cap = self.env.capital
        self.env.step([0.0])  # hold for 1 bar
        expected_cap = initial_cap - (37500.0 * 2.083333e-6)
        self.assertAlmostEqual(self.env.capital, expected_cap, places=2)
        
    def test_deadzone_flat(self):
        # Action inside deadzone [-0.20, 0.20] must result in 0 trades / flat
        obs, reward, done, trunc, info = self.env.step([0.10])
        self.assertEqual(self.env.position, 0.0)
        self.assertEqual(self.env.total_trades, 0)
        # With zero flat penalty, sitting flat rewards 0.0
        self.assertEqual(reward, 0.0)

    def test_zero_upfront_penalty(self):
        # Opening a trade should NOT hit the agent with an upfront penalty
        obs, reward, done, trunc, info = self.env.step([0.60])
        self.assertTrue(self.env.position > 0)
        self.assertEqual(self.env.total_trades, 1)
        self.assertEqual(reward, 0.0)

    def test_agent_action_lockdown(self):
        self.env.step([0.55])  # Enter long
        pos_before = self.env.position
        self.assertTrue(pos_before > 0)
        
        # Action to reverse should be locked out while position is open
        self.env.step([-0.55])
        self.assertEqual(self.env.position, pos_before)

    def test_fixed_sl_hit(self):
        # Enter Long with conviction: raw action 1.0 -> conviction = 1.0 -> risk_multiplier = 1.5
        self.env.step([1.0])
        self.assertAlmostEqual(self.env.current_risk_multiplier, 1.5, places=2)
        
        # Modify next row in dataframe to hit SL
        self.env.df.at[self.env.current_step, 'Low'] = 1.0970
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        # Position closed
        self.assertEqual(self.env.position, 0.0)
        # SL reward: -1.0 * long_sl_mult * risk_mult = -1.0 * 1.0 * 1.5 = -1.5
        self.assertAlmostEqual(reward, -1.5, places=2)
        self.assertEqual(self.env.losing_trades, 1)

    def test_fixed_tp_hit(self):
        # Enter Long with risk_multiplier = 1.5
        self.env.step([1.0])
        
        # Modify next row to hit TP (1.10005 + 1.5 * 0.0020 = 1.10305)
        self.env.df.at[self.env.current_step, 'Low'] = 1.1010
        self.env.df.at[self.env.current_step, 'High'] = 1.1040
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # TP reward: 1.5 * risk_mult = 1.5 * 1.5 = 2.25
        self.assertAlmostEqual(reward, 2.25, places=2)
        self.assertEqual(self.env.winning_trades, 1)

    def test_short_sl_hit(self):
        # Enter Short with risk_multiplier = 1.5
        self.env.step([-1.0])
        self.assertTrue(self.env.position < 0)
        
        # Force SL hit on short (price rises)
        self.env.df.at[self.env.current_step, 'High'] = 1.1030
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # Short SL reward: -1.0 * 1.5 = -1.5
        self.assertAlmostEqual(reward, -1.5, places=2)

    def test_short_tp_hit(self):
        # Enter Short with risk_multiplier = 1.5
        self.env.step([-1.0])
        
        # Force TP hit on short (price drops to 1.0960)
        self.env.df.at[self.env.current_step, 'High'] = 1.1000
        self.env.df.at[self.env.current_step, 'Low'] = 1.0960
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        self.assertEqual(self.env.position, 0.0)
        # Short TP reward: 1.5 * 1.5 = 2.25
        self.assertAlmostEqual(reward, 2.25, places=2)

    def test_intrabar_collision_conservative(self):
        # Enter Long
        self.env.step([1.0])
        
        # Huge bar touching both TP and SL
        self.env.df.at[self.env.current_step, 'High'] = 1.1070
        self.env.df.at[self.env.current_step, 'Low'] = 1.0960
        obs, reward, done, trunc, info = self.env.step([0.0])
        
        # SL must trigger first (conservative execution)
        self.assertEqual(self.env.position, 0.0)
        self.assertAlmostEqual(reward, -1.5, places=2)
        self.assertEqual(self.env.trade_history[-1][2], -1)


    def test_breakeven_sl_ratchet(self):
        # Enter Long
        self.env.step([1.0])
        initial_sl = self.env.current_sl_price
        # Move price up by 1.6x ATR without hitting TP
        self.env.df.at[self.env.current_step, 'High'] = self.env.entry_price + (1.6 * self.env.entry_atr)
        self.env.df.at[self.env.current_step, 'Low'] = self.env.entry_price
        self.env.step([0.0])
        expected_be_sl = self.env.entry_price + (0.1 * self.env.entry_atr)
        self.assertGreater(self.env.current_sl_price, initial_sl)
        self.assertAlmostEqual(self.env.current_sl_price, expected_be_sl, places=5)

    def test_probability_edge_filter(self):
        # Probabilities without sufficient edge (p_long 0.505, p_short 0.500 -> diff 0.005 < 0.01)
        self.env.prob_long = np.ones(10) * 0.505
        self.env.prob_short = np.ones(10) * 0.500
        obs, reward, done, trunc, info = self.env.step([0.80])
        # Should be filtered out
        self.assertEqual(self.env.position, 0.0)
        self.assertEqual(self.env.total_trades, 0)

    def test_volatility_filter_low_and_high(self):
        # ATR_Ratio too low (< 0.70)
        self.env.df.at[self.env.current_step, 'ATR_Ratio'] = 0.50
        self.env.atr_ratio = self.env.df['ATR_Ratio'].values
        obs, reward, done, trunc, info = self.env.step([0.80])
        self.assertEqual(self.env.position, 0.0)
        self.assertEqual(self.env.total_trades, 0)


if __name__ == '__main__':
    unittest.main()
