# indicators.py - Calculate technical indicators using TA-Lib with parallel processing

import pandas as pd
import numpy as np
import talib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class IndicatorsCalculator:
    """Calculate technical indicators using TA-Lib"""
    
    def __init__(self, input_dir: str = "data", output_dir: str = "data"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a dataframe"""
        
        # Extract OHLCV
        open_price = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        volume = df['Volume'].values.astype(float)
        
        # Create new dataframe with original data
        result = df.copy()
        
        # Price and returns
        result['Returns'] = result['Close'].pct_change()
        result['LogReturns'] = np.log(result['Close'] / result['Close'].shift(1))
        
        # === Moving Averages ===
        for period in [5, 10, 20, 50, 100, 200]:
            result[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
            result[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
        
        result['WMA_20'] = talib.WMA(close, timeperiod=20)
        result['DEMA_20'] = talib.DEMA(close, timeperiod=20)
        result['TEMA_20'] = talib.TEMA(close, timeperiod=20)
        result['KAMA_20'] = talib.KAMA(close, timeperiod=20)
        result['KAMA_30'] = talib.KAMA(close, timeperiod=30)
        result['T3'] = talib.T3(close, timeperiod=5, vfactor=0)
        
        # MESA Adaptive Moving Average
        mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
        result['MAMA'] = mama
        result['FAMA'] = fama
        
        # === Bollinger Bands ===
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        result['BB_upper'] = upper
        result['BB_middle'] = middle
        result['BB_lower'] = lower
        result['BB_width'] = upper - lower
        result['BB_percent'] = (close - lower) / (upper - lower)
        
        # Additional BB with different parameters
        upper10, middle10, lower10 = talib.BBANDS(close, timeperiod=10, nbdevup=2, nbdevdn=2)
        result['BB_upper_10'] = upper10
        result['BB_middle_10'] = middle10
        result['BB_lower_10'] = lower10
        
        # === Momentum Indicators ===
        result['RSI_7'] = talib.RSI(close, timeperiod=7)
        result['RSI_14'] = talib.RSI(close, timeperiod=14)
        result['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        result['STOCH_K'] = slowk
        result['STOCH_D'] = slowd
        
        # Stochastic RSI
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        result['STOCHRSI_K'] = fastk
        result['STOCHRSI_D'] = fastd
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        result['MACD'] = macd
        result['MACD_signal'] = macdsignal
        result['MACD_hist'] = macdhist
        
        # Additional momentum indicators
        result['ROC_5'] = talib.ROC(close, timeperiod=5)
        result['ROC_10'] = talib.ROC(close, timeperiod=10)
        result['ROC_20'] = talib.ROC(close, timeperiod=20)
        
        result['MOM_10'] = talib.MOM(close, timeperiod=10)
        result['CMO_14'] = talib.CMO(close, timeperiod=14)
        result['CCI_14'] = talib.CCI(high, low, close, timeperiod=14)
        result['CCI_20'] = talib.CCI(high, low, close, timeperiod=20)
        
        result['WILLR_14'] = talib.WILLR(high, low, close, timeperiod=14)
        result['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        result['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        result['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        result['TRIX'] = talib.TRIX(close, timeperiod=15)
        result['BOP'] = talib.BOP(open_price, high, low, close)
        
        # === Volatility Indicators ===
        result['ATR_7'] = talib.ATR(high, low, close, timeperiod=7)
        result['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        result['ATR_21'] = talib.ATR(high, low, close, timeperiod=21)
        result['NATR_14'] = talib.NATR(high, low, close, timeperiod=14)
        result['TRANGE'] = talib.TRANGE(high, low, close)
        
        # Standard deviation
        result['STDDEV_10'] = talib.STDDEV(close, timeperiod=10, nbdev=1)
        result['STDDEV_20'] = talib.STDDEV(close, timeperiod=20, nbdev=1)
        result['VAR_20'] = talib.VAR(close, timeperiod=20, nbdev=1)
        
        # === Volume Indicators ===
        result['OBV'] = talib.OBV(close, volume)
        result['AD'] = talib.AD(high, low, close, volume)
        result['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        result['MFI_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Volume moving averages
        result['Volume_SMA_20'] = talib.SMA(volume, timeperiod=20)
        result['Volume_ratio'] = volume / result['Volume_SMA_20']
        result['VROC'] = talib.ROC(volume, timeperiod=14)
        
        # On Balance Volume with SMA
        result['OBV_SMA_20'] = talib.SMA(result['OBV'].values, timeperiod=20)
        
        # === Trend Indicators ===
        result['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)
        result['ADXR_14'] = talib.ADXR(high, low, close, timeperiod=14)
        result['DI_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        result['DI_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        result['DX'] = talib.DX(high, low, close, timeperiod=14)
        
        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
        result['AROON_DOWN'] = aroon_down
        result['AROON_UP'] = aroon_up
        result['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        
        result['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # === Pattern Recognition ===
        # Candlestick patterns
        result['CDL_DOJI'] = talib.CDLDOJI(open_price, high, low, close)
        result['CDL_ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
        result['CDL_HAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
        result['CDL_SHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        result['CDL_MORNINGSTAR'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        result['CDL_EVENINGSTAR'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        result['CDL_HARAMI'] = talib.CDLHARAMI(open_price, high, low, close)
        result['CDL_MARUBOZU'] = talib.CDLMARUBOZU(open_price, high, low, close)
        result['CDL_SPINNINGTOP'] = talib.CDLSPINNINGTOP(open_price, high, low, close)
        result['CDL_DRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open_price, high, low, close)
        result['CDL_GRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(open_price, high, low, close)
        result['CDL_HANGINGMAN'] = talib.CDLHANGINGMAN(open_price, high, low, close)
        result['CDL_INVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open_price, high, low, close)
        result['CDL_DARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open_price, high, low, close)
        result['CDL_PIERCING'] = talib.CDLPIERCING(open_price, high, low, close)
        result['CDL_BELTHOLD'] = talib.CDLBELTHOLD(open_price, high, low, close)
        result['CDL_KICKING'] = talib.CDLKICKING(open_price, high, low, close)
        result['CDL_THRUSTING'] = talib.CDLTHRUSTING(open_price, high, low, close)
        
        # === Cycle Indicators ===
        result['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        result['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        result['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
        
        inphase, quadrature = talib.HT_PHASOR(close)
        result['HT_INPHASE'] = inphase
        result['HT_QUADRATURE'] = quadrature
        
        sine, leadsine = talib.HT_SINE(close)
        result['HT_SINE'] = sine
        result['HT_LEADSINE'] = leadsine
        
        # === Price Transform ===
        result['AVGPRICE'] = talib.AVGPRICE(open_price, high, low, close)
        result['MEDPRICE'] = talib.MEDPRICE(high, low)
        result['TYPPRICE'] = talib.TYPPRICE(high, low, close)
        result['WCLPRICE'] = talib.WCLPRICE(high, low, close)
        
        # === Custom Indicators ===
        # Price position in range
        min_20 = talib.MIN(low, timeperiod=20)
        max_20 = talib.MAX(high, timeperiod=20)
        result['PRICE_POSITION'] = (close - min_20) / (max_20 - min_20)
        
        # Efficiency Ratio
        direction = abs(close[-1] - close[0]) if len(close) > 0 else 0
        volatility = np.sum(np.abs(np.diff(close)))
        result['EFFICIENCY_RATIO'] = direction / volatility if volatility != 0 else 0
        
        # Gap indicators
        result['GAP_UP'] = (low > np.roll(high, 1)).astype(int)
        result['GAP_DOWN'] = (high < np.roll(low, 1)).astype(int)
        
        # Pivot points
        pivot = (high + low + close) / 3
        result['PIVOT'] = pivot
        result['PIVOT_R1'] = 2 * pivot - low
        result['PIVOT_S1'] = 2 * pivot - high
        result['PIVOT_R2'] = pivot + (high - low)
        result['PIVOT_S2'] = pivot - (high - low)
        
        # VWAP
        typical_price = (high + low + close) / 3
        result['VWAP'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Price-Volume Trend
        price_change = close - np.roll(close, 1)
        result['PVT'] = (price_change / np.roll(close, 1) * volume).cumsum()
        
        # Golden/Death Cross
        result['GOLDEN_CROSS'] = (
            (result['SMA_50'] > result['SMA_200']) & 
            (result['SMA_50'].shift(1) <= result['SMA_200'].shift(1))
        ).astype(int)
        
        result['DEATH_CROSS'] = (
            (result['SMA_50'] < result['SMA_200']) & 
            (result['SMA_50'].shift(1) >= result['SMA_200'].shift(1))
        ).astype(int)
        
        # Additional calculated indicators for strategy
        result['ATR_14_SMA_20'] = talib.SMA(result['ATR_14'].values, timeperiod=20)
        result['BB_width_SMA_20'] = talib.SMA(result['BB_width'].values, timeperiod=20)
        result['VOLATILITY_RATIO'] = result['ATR_14'] / close
        
        return result
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN values from indicator calculations"""
        # Find the maximum lookback period needed
        # Most indicators need at most 200 periods (SMA_200)
        min_required_rows = 200
        
        # Drop NaN values
        df_clean = df.dropna()
        
        # Ensure we have enough data
        if len(df_clean) < min_required_rows:
            print(f"Warning: Only {len(df_clean)} rows after cleaning (minimum recommended: {min_required_rows})")
        
        return df_clean
    
    def process_ticker(self, ticker: str) -> Tuple[str, bool, int]:
        """Process a single ticker file"""
        input_path = os.path.join(self.input_dir, f"{ticker}.csv")
        
        if not os.path.exists(input_path):
            return ticker, False, 0
        
        try:
            # Load data
            df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
            original_rows = len(df)
            
            # Calculate indicators
            df_with_indicators = self.calculate_all_indicators(df)
            
            # Clean (remove NaN rows)
            df_clean = self.clean_dataframe(df_with_indicators)
            final_rows = len(df_clean)
            
            # Save back to same file
            output_path = os.path.join(self.output_dir, f"{ticker}.csv")
            df_clean.to_csv(output_path)
            
            print(f"[{ticker}] Success: {original_rows} → {final_rows} rows (removed {original_rows - final_rows} NaN rows)")
            return ticker, True, final_rows
            
        except Exception as e:
            print(f"[{ticker}] Error: {str(e)}")
            return ticker, False, 0
    
    def process_all_tickers(self, max_workers: int = None) -> Dict[str, int]:
        """Process all tickers in parallel"""
        # Get all CSV files
        csv_files = [f.replace('.csv', '') for f in os.listdir(self.input_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in input directory")
            return {}
        
        print(f"\n{'='*60}")
        print(f"Processing {len(csv_files)} tickers with TA-Lib indicators")
        print(f"{'='*60}")
        
        # Use all CPU cores if not specified (Windows has a limit of 61)
        if max_workers is None:
            max_workers = min(os.cpu_count(), 61)
        
        results = {}
        failed = []
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.process_ticker, ticker): ticker 
                for ticker in csv_files
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker, success, rows = future.result()
                    if success:
                        results[ticker] = rows
                    else:
                        failed.append(ticker)
                except Exception as e:
                    print(f"[{ticker}] Unexpected error: {str(e)}")
                    failed.append(ticker)
        
        # Summary
        print(f"\n{'='*60}")
        print("INDICATOR CALCULATION COMPLETE")
        print(f"Successfully processed: {len(results)} tickers")
        if failed:
            print(f"Failed: {len(failed)} tickers - {', '.join(failed)}")
        print(f"{'='*60}\n")
        
        return results


def calculate_indicators_parallel(input_dir: str = "data", 
                                output_dir: str = "data",
                                max_workers: int = None):
    """Main function to calculate indicators for all tickers"""
    
    calculator = IndicatorsCalculator(input_dir, output_dir)
    results = calculator.process_all_tickers(max_workers)
    
    return results


if __name__ == "__main__":
    # Example usage
    calculate_indicators_parallel()