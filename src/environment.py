import numpy as np
import pandas as pd

class TradingEnvironment:
    """Custom environment for simulating trading with a portfolio."""

    def __init__(self, initial_capital=10000, leverage=50):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.positions = {}  # {ticker: quantity}
        self.entry_prices = {} # {ticker: avg_entry_price}
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.portfolio_history = [initial_capital]

    def get_portfolio_value(self, prices):
        """Calculate current portfolio value given current prices."""
        cash = self.capital
        for ticker, qty in self.positions.items():
            if ticker in prices:
                cash += qty * prices[ticker]
        return cash

    def get_fee_rate(self, ticker):
        """Dynamic fee rate. Can be configured per ticker."""
        return 0.0001       # 0.01% representing a typical 1 pip Forex spread on EURUSD

    def execute_trade(self, ticker, quantity, price, timestamp):
        """Execute a trade (buy/long if positive quantity, sell/short if negative)."""
        if quantity == 0:
            return False

        cost = abs(quantity * price)
        fee = cost * self.get_fee_rate(ticker)

        # Margin check: if increasing absolute position size, ensure we have enough equity
        old_qty = self.positions.get(ticker, 0.0)
        new_qty = old_qty + quantity
        
        if abs(new_qty) > abs(old_qty):
            margin_required = cost / self.leverage
            if (margin_required + fee) > self.portfolio_value:
                return False  # Insufficient margin

        # Prevent trading if bankrupt
        if self.portfolio_value <= 0:
            return False

        # FIX: Execution logic properly manages bidirectional capital flow
        if quantity > 0:  # Buy / Go Long or Cover Short
            self.capital -= (cost + fee)
        elif quantity < 0:  # Sell / Go Short or Close Long
            self.capital += (cost - fee)

        # Update position
        old_qty = self.positions.get(ticker, 0.0)
        new_qty = old_qty + quantity
        self.positions[ticker] = new_qty

        # FIX: Update entry price for accurate PnL & Stop-loss tracking
        if new_qty == 0:
            self.entry_prices[ticker] = 0.0
        elif (old_qty >= 0 and quantity > 0) or (old_qty <= 0 and quantity < 0):
            # Adding to a position in the same direction: Calculate Volume Weighted Average Price (VWAP)
            old_value = abs(old_qty) * self.entry_prices.get(ticker, price)
            new_value = abs(quantity) * price
            self.entry_prices[ticker] = (old_value + new_value) / abs(new_qty)
        elif (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
            # Flipped position entirely (e.g., long reversed to short)
            self.entry_prices[ticker] = price
        # Else: Partially closing a position; the average entry price remains the same.

        self.trade_history.append({
            'timestamp': timestamp,
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'fee': fee,
        })
        return True

    def update_portfolio_value(self, prices):
        """Update portfolio value at the end of each time step."""
        self.portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_history.append(self.portfolio_value)

    def close_all_positions(self, prices, timestamp):
        """Close all open positions at market prices."""
        for ticker in list(self.positions.keys()):
            qty = self.positions[ticker]
            if qty != 0:
                self.execute_trade(ticker, -qty, prices[ticker], timestamp)

    def reset(self):
        """Reset the environment."""
        self.capital = self.initial_capital
        self.positions = {}
        self.entry_prices = {}
        self.portfolio_value = self.initial_capital
        self.trade_history = []
        self.portfolio_history = [self.initial_capital]