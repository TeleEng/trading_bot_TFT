import os
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv
from datetime import datetime

try:
    from histdata import download_hist_data
except ImportError:
    print("[ERROR] Please install the histdata package: pip install histdata")
    exit(1)

# Load environment variables
load_dotenv()

# Configuration from .env
TICKERS_ENV = os.getenv("HISTDATA_TICKERS", "EURUSD,GBPUSD,USDJPY")
TICKERS = [t.strip() for t in TICKERS_ENV.split(",")]
START_YEAR = int(os.getenv("HISTDATA_START_YEAR", 2008))
END_YEAR = int(os.getenv("HISTDATA_END_YEAR", 2026))

# Ensure data directory exists
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw"
DATA_PATH.mkdir(parents=True, exist_ok=True)

def download_histdata(tickers, start_year, end_year, path):
    """
    Downloads 1-minute FX data from HistData.com and saves it directly.
    Resampling is deferred to the preprocessing stage.
    """
    print("Downloading Raw 1-Minute Data from HistData.com")
    print(f"Time Range: {start_year} to {end_year}\n")

    current_date = datetime.now()
    current_year_actual = current_date.year
    current_month_actual = current_date.month

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df_list = []
        
        # HistData expects lowercase, no-symbol tickers (e.g., 'eurusd')
        clean_ticker = ticker.replace("/", "").replace("-", "").replace("_", "").lower()
        
        for year in range(start_year, end_year + 1):
            if year > current_year_actual:
                print(f"  Skipping {year} (Future Year)")
                continue
                
            # If it's the current ongoing year, HistData requires month-by-month downloads
            months_to_fetch = [None] 
            if year == current_year_actual:
                months_to_fetch = list(range(1, current_month_actual + 1))

            for month in months_to_fetch:
                try:
                    if month is None:
                        print(f"  Downloading {year}...")
                    else:
                        print(f"  Downloading {year}-{month:02d}...")
                    
                    # Download GENERIC_ASCII 1-Minute data
                    zip_path = download_hist_data(
                        year=str(year), 
                        month=str(month) if month else None,
                        pair=clean_ticker, 
                        time_frame='M1',
                        platform='ASCII',
                        output_directory=str(path)
                    )
                    
                    if zip_path:
                        with ZipFile(zip_path, 'r') as z:
                            csv_name = z.namelist()[0]
                            with z.open(csv_name) as f:
                                # GENERIC ASCII format: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
                                df_year = pd.read_csv(
                                    f, 
                                    sep=';', 
                                    names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                                    header=None
                                )
                                
                                # Parse dates and set index
                                df_year['datetime'] = pd.to_datetime(df_year['datetime'], format='%Y%m%d %H%M%S')
                                df_year.set_index('datetime', inplace=True)
                                df_list.append(df_year)
                                
                        # Clean up the zip file from the raw folder after reading into memory
                        os.remove(zip_path)
                        
                except Exception as e:
                    msg = f"Failed to fetch {ticker} for {year}" + (f"-{month:02d}" if month else "")
                    print(f"[WARNING] {msg}: {e}")
                    if 'zip_path' in locals() and zip_path and os.path.exists(zip_path):
                        os.remove(zip_path)

        if df_list:
            print("  Combining 1-Minute data...")
            # Combine all years into one massive DataFrame
            full_df = pd.concat(df_list)
            full_df.sort_index(inplace=True)
            full_df.index.name = 'Date'
            
            # Save the raw 1-minute data directly to the raw directory
            out_file = path / f"{ticker.upper()}.csv"
            full_df.to_csv(out_file)
            print(f"[OK] Saved {len(full_df)} raw 1-minute candles to {out_file}\n")
        else:
            print(f"[WARNING] No data could be processed for {ticker}.\n")

if __name__ == "__main__":
    download_histdata(TICKERS, START_YEAR, END_YEAR, DATA_PATH)