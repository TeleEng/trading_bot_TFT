import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from performance import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):
    
    def test_no_trades(self):
        # Scenario 1: Agent sits flat for 10 hours.
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='h'),
            'price': [1.1000] * 10,
            'position': [0.0] * 10,
            'portfolio_value': [10000.0] * 10
        }
        df = pd.DataFrame(data)
        metrics = PerformanceMetrics.calculate_metrics(df, initial_capital=10000.0)
        
        self.assertEqual(metrics['Total Trades'], 0)
        self.assertEqual(metrics['Total Return (%)'], 0.0)
        self.assertEqual(metrics['Win Rate (%)'], 0.0)
        self.assertEqual(metrics['Max Drawdown (%)'], 0.0)
        
    def test_single_massive_loss(self):
        # Scenario 2: Agent enters long and loses 50% of the portfolio.
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=3, freq='h'),
            'price': [1.1000, 1.0500, 1.0000],
            'position': [0.0, 100000.0, 0.0], # flat -> long -> closed
            'portfolio_value': [10000.0, 5000.0, 5000.0] # capital dropped to 5000 when closed
        }
        df = pd.DataFrame(data)
        metrics = PerformanceMetrics.calculate_metrics(df, initial_capital=10000.0)
        
        self.assertEqual(metrics['Total Trades'], 1)
        self.assertEqual(metrics['Total Return (%)'], -50.0)
        self.assertEqual(metrics['Win Rate (%)'], 0.0)
        self.assertEqual(metrics['Max Drawdown (%)'], -50.0)

    def test_open_position_at_end(self):
        # Scenario 3: Agent is in a floating loss when the backtest ends.
        # It shouldn't count as a closed trade for Win Rate, but it should affect Total Return.
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=3, freq='h'),
            'price': [1.1000, 1.1000, 1.0900],
            'position': [0.0, 100000.0, 100000.0], # flat -> long -> holding long
            'portfolio_value': [10000.0, 9995.0, 8995.0] # pays $5 spread, then market drops $1000
        }
        df = pd.DataFrame(data)
        metrics = PerformanceMetrics.calculate_metrics(df, initial_capital=10000.0)
        
        # Trade is NOT closed, so total trades should be 0!
        self.assertEqual(metrics['Total Trades'], 0)
        self.assertEqual(metrics['Win Rate (%)'], 0.0)
        
        # But return and drawdown MUST reflect the floating loss!
        self.assertEqual(metrics['Total Return (%)'], -10.05) # (8995 - 10000) / 10000 = -0.1005
        self.assertEqual(metrics['Max Drawdown (%)'], -10.05)

    def test_flipping_position(self):
        # Scenario 4: Agent flips from Long directly to Short without a 0.0 position gap.
        # This tests if the win rate calculation correctly splits the flip into two separate trades.
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='h'),
            'price': [1.1000, 1.1050, 1.1050, 1.0950, 1.0950],
            'position': [0.0, 100000.0, 100000.0, -100000.0, 0.0], 
            'portfolio_value': [10000.0, 10000.0, 10500.0, 10500.0, 11500.0] 
            # Entry 1 (Long): starts at 10000, closes at 10500 (PnL +500)
            # Entry 2 (Short, flip): starts at 10500, closes at 11500 (PnL +1000)
        }
        df = pd.DataFrame(data)
        metrics = PerformanceMetrics.calculate_metrics(df, initial_capital=10000.0)
        
        self.assertEqual(metrics['Total Trades'], 2)
        self.assertEqual(metrics['Win Rate (%)'], 100.0)
        self.assertEqual(metrics['Total Return (%)'], 15.0) # (11500 - 10000) / 10000
        self.assertEqual(metrics['Max Drawdown (%)'], 0.0) # Never went below peak

    def test_interleaved_win_loss_drawdown(self):
        # Scenario 5: Win, then massive loss, then small win. Ensure drawdown calculates from peak.
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=7, freq='h'),
            'price': [1.10, 1.10, 1.11, 1.11, 1.05, 1.05, 1.06],
            'position': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            'portfolio_value': [100.0, 100.0, 110.0, 110.0, 50.0, 50.0, 60.0]
            # Trade 1: Entry 100 -> Exit 110 (Win)
            # Trade 2: Entry 110 -> Exit 50 (Loss)
            # Trade 3: Entry 50 -> Exit 60 (Win)
            # Peak is 110.0. Trough is 50.0. Drawdown = (50 - 110) / 110 = -54.5454%
        }
        df = pd.DataFrame(data)
        metrics = PerformanceMetrics.calculate_metrics(df, initial_capital=100.0)
        
        self.assertEqual(metrics['Total Trades'], 3)
        self.assertEqual(metrics['Total Return (%)'], -40.0)
        self.assertAlmostEqual(metrics['Max Drawdown (%)'], -54.5454, places=2)
        self.assertAlmostEqual(metrics['Win Rate (%)'], 66.6666, places=2) # 2 wins, 1 loss

if __name__ == '__main__':
    unittest.main()
