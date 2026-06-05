import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceMetrics:
    """Calculate and display trading performance metrics."""

    @staticmethod
    def calculate_metrics(backtest_results, initial_capital=10000):
        """Calculate key performance metrics."""
        if backtest_results.empty or len(backtest_results) == 0:
            return {
                'Total Return (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Win Rate (%)': 0.0,
                'Final Value': initial_capital,
                'Total Trades': 0,
            }

        if 'portfolio_value' not in backtest_results.columns:
            return {
                'Total Return (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Win Rate (%)': 0.0,
                'Final Value': initial_capital,
                'Total Trades': 0,
            }

        portfolio_values = backtest_results['portfolio_value'].values
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital

        # FIX: Sharpe Ratio Annualization - Corrected for hourly trading intervals (252 days * 24 hours)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.sqrt(252 * 24) * returns.mean() / (returns.std() + 1e-10)

        # Max Drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # FIX: Win Rate & Trade Count - Now strictly calculates complete round-trip trades
        trades = []
        entry_price = 0.0
        entry_pos = 0.0
        
        for _, row in backtest_results.iterrows():
            pos = row.get('position', 0)
            price = row.get('price', 0)
            
            if pos != 0 and entry_pos == 0:
                # Open position
                entry_pos = pos
                entry_price = price
            elif pos == 0 and entry_pos != 0:
                # Close position
                pnl = (price - entry_price) / entry_price if entry_pos > 0 else (entry_price - price) / entry_price
                trades.append(pnl)
                entry_pos = 0.0
            elif pos != 0 and entry_pos != 0 and np.sign(pos) != np.sign(entry_pos):
                # Flipped position (e.g., long to short)
                pnl = (price - entry_price) / entry_price if entry_pos > 0 else (entry_price - price) / entry_price
                trades.append(pnl)
                entry_pos = pos
                entry_price = price

        winning_trades = sum(1 for t in trades if t > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'Total Return (%)': total_return * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Win Rate (%)': win_rate * 100,
            'Final Value': portfolio_values[-1],
            'Total Trades': total_trades,
        }

    @staticmethod
    def plot_results(backtest_results, save_path=None):
        """Plot equity curve and price with trading signals."""
        if backtest_results.empty or len(backtest_results) == 0:
            print("No backtest results to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Equity Curve
        if 'portfolio_value' in backtest_results.columns:
            ax1.plot(backtest_results['timestamp'], backtest_results['portfolio_value'],
                     label='Portfolio Value', linewidth=2, color='blue')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.set_title('Equity Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Price with Trading Signals
        if 'price' in backtest_results.columns:
            ax2.plot(backtest_results['timestamp'], backtest_results['price'],
                     label='Asset Price', linewidth=1, color='black')

            # Mark buy/sell signals based on position changes
            position_diff = backtest_results['position'].diff()
            buy_signals = backtest_results[position_diff > 0]
            sell_signals = backtest_results[position_diff < 0]

            if not buy_signals.empty:
                ax2.scatter(buy_signals['timestamp'], buy_signals['price'],
                           color='green', marker='^', label='Buy Signal', s=50)
            if not sell_signals.empty:
                ax2.scatter(sell_signals['timestamp'], sell_signals['price'],
                           color='red', marker='v', label='Sell Signal', s=50)

            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Price ($)')
            ax2.set_title('Trading Signals')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100)
        plt.show()

    @staticmethod
    def print_report(metrics):
        """Print formatted performance report."""
        print("\n" + "="*50)
        print("BACKTEST PERFORMANCE REPORT")
        print("="*50)
        for key, value in metrics.items():
            if '%' in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        print("="*50 + "\n")