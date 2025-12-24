# market.py - Download market data using yfinance

import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# Stock sectors for testing
STOCK_SECTORS = {
    "TECH": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "AMD", "INTC", "ORCL", "CRM", "ADBE", "NFLX", "CSCO", "AVGO", "QCOM"],
    "FINANCE": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "COF", "USB", "PNC", "TFC", "SPGI", "CB"],
    "HEALTHCARE": ["JNJ", "UNH", "PFE", "CVS", "ABBV", "MRK", "TMO", "ABT", "MDT", "AMGN", "DHR", "BMY", "GILD", "LLY", "ISRG"],
    "ENERGY": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "KMI", "WMB", "HES", "DVN", "HAL", "BKR"],
    "CONSUMER": ["AMZN", "WMT", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "TJX", "DIS", "CMCSA"],
    "INDUSTRIAL": ["BA", "CAT", "GE", "MMM", "UPS", "RTX", "LMT", "HON", "DE", "NOC", "UNP", "FDX", "EMR", "ETN", "ITW"]
}

class MarketDownloader:
    """Efficient market data downloader using yfinance"""
    
    def __init__(self, start_date: str = None, end_date: str = None, data_dir: str = "data"):
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = data_dir
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"[MarketDownloader] Initialized")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Data directory: {self.data_dir}")
    
    def download_ticker(self, ticker: str, retry_count: int = 3) -> Optional[pd.DataFrame]:
        """Download single ticker with retry logic"""
        for attempt in range(retry_count):
            try:
                print(f"[{ticker}] Downloading... (attempt {attempt + 1})")
                
                # Download using yfinance
                data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,  # Adjust for splits/dividends
                    prepost=False,
                    threads=True
                )
                
                if data.empty:
                    print(f"[{ticker}] No data returned")
                    return None
                
                # Clean column names if multi-level
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Ensure we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    print(f"[{ticker}] Missing required columns")
                    return None
                
                # Save to CSV
                output_path = os.path.join(self.data_dir, f"{ticker}.csv")
                data.to_csv(output_path)
                
                print(f"[{ticker}] Success: {len(data)} rows saved")
                return data
                
            except Exception as e:
                print(f"[{ticker}] Error on attempt {attempt + 1}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(1)  # Brief pause before retry
                    
        print(f"[{ticker}] Failed after {retry_count} attempts")
        return None
    
    def download_batch(self, tickers: List[str], max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """Download multiple tickers in parallel"""
        print(f"\n[Batch] Downloading {len(tickers)} tickers with {max_workers} workers")
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.download_ticker, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[ticker] = data
                    else:
                        failed.append(ticker)
                except Exception as e:
                    print(f"[{ticker}] Unexpected error: {str(e)}")
                    failed.append(ticker)
        
        print(f"\n[Batch] Complete: {len(results)} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed tickers: {', '.join(failed)}")
            
        return results
    
    def download_sector(self, sector: str, max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """Download all tickers in a sector"""
        if sector not in STOCK_SECTORS:
            print(f"[Error] Unknown sector: {sector}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"Downloading {sector} sector")
        print(f"{'='*60}")
        
        tickers = STOCK_SECTORS[sector]
        return self.download_batch(tickers, max_workers)
    
    def download_all_sectors(self, sectors: List[str] = None, max_workers: int = 10) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Download multiple sectors"""
        if sectors is None:
            sectors = list(STOCK_SECTORS.keys())
        
        all_data = {}
        
        for sector in sectors:
            sector_data = self.download_sector(sector, max_workers)
            all_data[sector] = sector_data
            
            # Brief pause between sectors to avoid rate limiting
            if sector != sectors[-1]:
                time.sleep(2)
        
        return all_data
    
    def get_available_tickers(self) -> List[str]:
        """Get list of successfully downloaded tickers"""
        files = [f.replace('.csv', '') for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return sorted(files)


def download_market_data(sectors: List[str] = None, 
                        start_date: str = None,
                        end_date: str = None,
                        max_workers: int = 10):
    """Main function to download market data"""
    
    print("\n" + "="*60)
    print("MARKET DATA DOWNLOAD")
    print("="*60)
    
    downloader = MarketDownloader(start_date, end_date)
    
    # Download specified sectors
    if sectors is None:
        sectors = ["TECH", "FINANCE"]  # Default sectors
    
    downloader.download_all_sectors(sectors, max_workers)
    
    # Summary
    available = downloader.get_available_tickers()
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print(f"Total tickers downloaded: {len(available)}")
    print("="*60)
    
    return available


if __name__ == "__main__":
    # Example: Download tech and finance sectors
    download_market_data(sectors=["TECH", "FINANCE"], start_date="2020-01-01")