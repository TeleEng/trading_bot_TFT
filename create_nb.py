import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Trading Bot Full Pipeline
This notebook orchestrates the end-to-end pipeline:
1. Download Data
2. Preprocess Data
3. Split Data
4. Train Model
5. Evaluate Model
6. Backtest"""

code1 = """import os
import sys
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# Setup paths (Assuming this notebook is in the notebooks directory)
base_dir = Path(os.getcwd()).parent
sys.path.append(str(base_dir / "src"))

from data_downloader import download_histdata, TICKERS, START_YEAR, END_YEAR, DATA_PATH
from preprocess import process_all_data, OUTPUT_PATH
from model import PricePredictor
from environment import TradingEnvironment
from backtest import Backtester
from performance import PerformanceMetrics
from viz import plot_tsne_and_confusion_matrix

models_dir = base_dir / "models"
results_dir = base_dir / "results"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
"""

text2 = """## Phase 1: Downloading & Preprocessing Data"""

code2 = """# Uncomment to download and preprocess data
# download_histdata(TICKERS, START_YEAR, END_YEAR, DATA_PATH)
# process_all_data()
"""

text3 = """## Phase 2: Splitting Data (60/20/20)"""

code3 = """model = PricePredictor()

load_dotenv(dotenv_path=base_dir / ".env")
MAIN_ASSET = os.getenv("MAIN_ASSET", "EURUSD")
first_ticker_file = base_dir / "data" / "processed" / f"{MAIN_ASSET}_master.csv"

if not first_ticker_file.exists():
    print(f"[ERROR] {first_ticker_file} not found. Make sure data is preprocessed.")
else:
    df = pd.read_csv(first_ticker_file, index_col=0, parse_dates=True).iloc[-10_000:]
    
    val_idx = int(len(df) * 0.6)
    test_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx - model.input_chunk_length : test_idx]
    test_df = df.iloc[test_idx - model.input_chunk_length :] 

    train_file = base_dir / "data" / "processed" / "train_split.csv"
    val_file = base_dir / "data" / "processed" / "val_split.csv"
    test_file = base_dir / "data" / "processed" / "test_split.csv"
    
    train_df.to_csv(train_file)
    val_df.to_csv(val_file)
    test_df.to_csv(test_file)
    print("Data splits created.")
"""

text4 = """## Phase 3: Self-Supervised Contrastive Training"""

code4 = """train_score, val_score = model.train(str(train_file), str(val_file), epochs=10)
print(f"Final InfoNCE Loss - Train: {train_score:.4f} | Val: {val_score:.4f}")

model_path = models_dir / "model.pkl"
model.save(str(model_path))
print(f"Model saved to {model_path}")
"""

text5 = """## Phase 4: Generating t-SNE & Confusion Matrix"""

code5 = """plot_tsne_and_confusion_matrix(model, str(test_file), str(results_dir))
"""

text6 = """## Phase 5: Running Out-of-Sample Backtest"""

code6 = """environment = TradingEnvironment(initial_capital=10000)
backtester = Backtester(model, environment, threshold=0.35, risk_percentage=0.2)

backtest_results = backtester.run(str(test_file))

metrics = PerformanceMetrics.calculate_metrics(
    backtest_results,
    initial_capital=10000
)
PerformanceMetrics.print_report(metrics)

viz_path = results_dir / "backtest_results.png"
PerformanceMetrics.plot_results(backtest_results, save_path=str(viz_path))
print(f"Backtest visualization saved to {viz_path}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code6)
]

output_path = Path("d:/Work/trading_bot/notebooks/main_pipeline.ipynb")
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created at {output_path}")
