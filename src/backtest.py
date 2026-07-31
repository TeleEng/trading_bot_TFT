import pandas as pd
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")

class Backtester:
    """Run backtest using the Multi-Class model and trading environment."""

    def __init__(self, model, environment, threshold=0.35, risk_percentage=0.2):
        self.model = model
        self.environment = environment
        
        # 3-class baseline is 0.33. 0.35 threshold ensures slight conviction.
        self.threshold = threshold 
        self.risk_percentage = risk_percentage 
        
        # FIX: Align Backtester with the Preprocessor's ATR Logic
        self.tp_mult = 1.5
        self.sl_mult = 0.75

        self.cooldown_periods = 2 # Reduced cooldown for faster 6h strategy
        self._cooldown_counter = 0

    def run(self, data_source, asset_price_col='Close'):
        print(f"Starting Multi-Class Dynamic ATR Backtest...")
        
        if isinstance(data_source, str):
            df = pd.read_csv(data_source, index_col=0, parse_dates=True)
            ticker = MAIN_ASSET
            features_df = df.drop(columns=['target'], errors='ignore')
            print("Precomputing batched GPU predictions for speed...")
            all_probs = self.model.predict_batch(features_df)
        else:
            # Tuple of DataFrames for MTF
            df_1h, df_4h, df_1d, df_1w = data_source
            df = df_1h
            ticker = MAIN_ASSET
            features_df = (df_1h.drop(columns=['target'], errors='ignore'), df_4h, df_1d, df_1w)
            print("Precomputing MTF batched GPU predictions with Voting...")
            all_probs = self.model.predict_batch_classified(features_df)

        results = []
        self.environment.reset()
        self._cooldown_counter = 0
        
        seq_len = self.model.input_chunk_length
        current_sl_threshold = 0.0
        signal_history = []

        for i in range(seq_len, len(df)):
            current_row = df.iloc[i]
            current_timestamp = df.index[i]
            price = current_row[asset_price_col]
            
            # Extract ATR for dynamic logic (fallback to 0.5% move if missing)
            atr = current_row.get('ATR', price * 0.005) 
            
            # Cap ATR to prevent extremely large or small position sizes
            atr_min = price * 0.001  # Minimum 0.1% move
            atr_max = price * 0.02   # Maximum 2.0% move
            atr = max(atr_min, min(atr, atr_max))

            self.environment.update_portfolio_value({ticker: price})
            port_value = self.environment.portfolio_value
            current_pos = self.environment.positions.get(ticker, 0.0)
            entry_price = self.environment.entry_prices.get(ticker, 0.0)

            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1

            closed_due_to_risk = False

            # 1. Manage Open Risk
            if current_pos != 0:
                # Convert the ATR dollar move into a percentage for the trailing math
                tp_pct = (self.tp_mult * atr) / entry_price
                base_sl_pct = -((self.sl_mult * atr) / entry_price)
                
                unrealized_return = (price - entry_price) / entry_price if current_pos > 0 else (entry_price - price) / entry_price
                
                # Rule 2: Break-even Plus
                if unrealized_return >= 0.70 * tp_pct:
                    locked_in_profit = 0.50 * tp_pct
                    if locked_in_profit > current_sl_threshold:
                        current_sl_threshold = locked_in_profit
                
                if unrealized_return <= current_sl_threshold or unrealized_return >= tp_pct:
                    self.environment.execute_trade(ticker, -current_pos, price, current_timestamp)
                    self._cooldown_counter = self.cooldown_periods
                    closed_due_to_risk = True
                    current_pos = 0.0 
                    current_sl_threshold = base_sl_pct 

            # 2. Make Predictions & Issue Orders
            if not closed_due_to_risk and self._cooldown_counter == 0:
                # O(1) lookup instead of O(N) GPU pass
                probs = all_probs[i - seq_len]
                p_flat, p_up, p_down = probs[0], probs[1], probs[2]

                # Raw signal prediction
                raw_signal = 0.0
                if p_up > self.threshold and p_up > p_down:
                    raw_signal = 1.0
                elif p_down > self.threshold and p_down > p_up:
                    raw_signal = -1.0
                
                # Append to rolling buffer (max 5)
                signal_history.append(raw_signal)
                if len(signal_history) > 5:
                    signal_history.pop(0)

                current_direction = 1.0 if current_pos > 0 else (-1.0 if current_pos < 0 else 0.0)
                
                # Default target direction remains the current direction
                target_direction = current_direction

                # Evaluate Signal Buffer for 3-out-of-5 confirmation
                if len(signal_history) == 5:
                    up_votes = signal_history.count(1.0)
                    down_votes = signal_history.count(-1.0)
                    
                    if up_votes >= 3:
                        target_direction = 1.0
                    elif down_votes >= 3:
                        target_direction = -1.0

                # Note: We intentionally ignore raw_signal == 0.0 when in a trade, letting SL/TP handle exits.

                # Execute difference
                if target_direction != current_direction:
                    # We only close to Flat if SL/TP hits, otherwise we swap Long/Short
                    risk_amount = self.environment.portfolio_value * self.risk_percentage
                    risk_per_unit = atr * self.sl_mult
                    ideal_qty = (risk_amount / risk_per_unit) if risk_per_unit > 0 else 0.0
                    
                    # Cap by margin constraint (leave 2% buffer for fees/slippage)
                    max_qty_allowed = (self.environment.portfolio_value * self.environment.leverage * 0.98) / price
                    target_qty = min(ideal_qty, max_qty_allowed) * target_direction
                    
                    trade_size = target_qty - current_pos
                    success = self.environment.execute_trade(ticker, trade_size, price, current_timestamp)
                    if success:
                        current_pos = target_qty

            results.append({
                'timestamp': current_timestamp,
                'price': price,
                'position': current_pos,
                'portfolio_value': port_value,
                'prob_flat': probs[0],
                'prob_up': probs[1],
                'prob_down': probs[2]
            })

        return pd.DataFrame(results)