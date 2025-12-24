# massive_scale_tester.py - Test millions of strategies efficiently

import numpy as np
import pandas as pd
import cupy as cp
from numba import jit, prange, cuda
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import time
import os
import h5py
from dataclasses import dataclass

# Import grammar from optimized_evolution
from optimized_evolution import TRADING_GRAMMAR, Strategy, FastBacktester


class MassiveScalePreprocessor:
    """Prepare data for massive scale testing"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.h5_file = "massive_scale_data.h5"
    
    def prepare_hdf5_data(self, tickers: List[str] = None):
        """Convert CSV data to HDF5 for fast access"""
        print("Preparing data for massive scale testing...")
        
        # Get all available tickers if not specified
        if tickers is None:
            csv_files = [f.replace('.csv', '') for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            tickers = csv_files[:20]  # Use first 20 for testing
        
        with h5py.File(self.h5_file, 'w') as f:
            for ticker in tickers:
                csv_path = os.path.join(self.data_dir, f"{ticker}.csv")
                if not os.path.exists(csv_path):
                    continue
                
                try:
                    # Load data
                    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                    
                    # Create group for ticker
                    grp = f.create_group(ticker)
                    
                    # Store OHLCV
                    grp.create_dataset('close', data=df['Close'].values)
                    grp.create_dataset('high', data=df['High'].values)
                    grp.create_dataset('low', data=df['Low'].values)
                    grp.create_dataset('volume', data=df['Volume'].values)
                    
                    # Store key indicators
                    indicators = ['RSI_14', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_10', 'EMA_20',
                                 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 'ATR_14',
                                 'ADX_14', 'Volume_SMA_20', 'OBV', 'STOCH_K', 'STOCH_D',
                                 'CCI_14', 'MFI_14']
                    
                    ind_grp = grp.create_group('indicators')
                    for ind in indicators:
                        if ind in df.columns:
                            ind_grp.create_dataset(ind, data=df[ind].values)
                    
                    print(f"  {ticker}: {len(df)} rows prepared")
                    
                except Exception as e:
                    print(f"  {ticker}: Error - {str(e)}")
        
        print(f"Data prepared and saved to {self.h5_file}")


class GPUBacktester:
    """GPU-accelerated backtesting using CuPy"""
    
    def __init__(self, h5_file: str = "massive_scale_data.h5"):
        self.h5_file = h5_file
        self.gpu_available = self._check_gpu()
        
        if self.gpu_available:
            device = cp.cuda.Device(0)
            props = device.attributes
            print(f"GPU detected: Device {device.id}")
            print(f"GPU Memory: {device.mem_info[1] / 1024**3:.1f} GB")
        else:
            print("No GPU available, using CPU")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            cp.cuda.Device(0)
            return True
        except:
            return False
    
    def load_ticker_data_gpu(self, ticker: str) -> Dict:
        """Load ticker data to GPU memory"""
        data = {}
        
        with h5py.File(self.h5_file, 'r') as f:
            if ticker not in f:
                return None
            
            grp = f[ticker]
            
            if self.gpu_available:
                # Load to GPU
                data['close'] = cp.asarray(grp['close'][:])
                data['high'] = cp.asarray(grp['high'][:])
                data['low'] = cp.asarray(grp['low'][:])
                data['volume'] = cp.asarray(grp['volume'][:])
                
                # Load indicators
                data['indicators'] = {}
                for ind_name in grp['indicators']:
                    data['indicators'][ind_name] = cp.asarray(grp['indicators'][ind_name][:])
            else:
                # Load to CPU
                data['close'] = grp['close'][:]
                data['high'] = grp['high'][:]
                data['low'] = grp['low'][:]
                data['volume'] = grp['volume'][:]
                
                data['indicators'] = {}
                for ind_name in grp['indicators']:
                    data['indicators'][ind_name] = grp['indicators'][ind_name][:]
        
        return data
    
    def generate_all_signals_gpu(self, data: Dict, entry_conditions: List[Tuple], 
                                exit_conditions: List[Tuple]) -> Tuple:
        """Generate all possible entry/exit signals on GPU"""
        n_points = len(data['close'])
        n_entries = len(entry_conditions)
        n_exits = len(exit_conditions)
        
        if self.gpu_available:
            # Allocate GPU arrays
            entry_signals = cp.zeros((n_entries, n_points), dtype=cp.bool_)
            exit_signals = cp.zeros((n_exits, n_points), dtype=cp.bool_)
            
            # Generate entry signals
            for i, (left, op, right) in enumerate(entry_conditions):
                if left in data['indicators']:
                    left_vals = data['indicators'][left]
                elif left == 'Close':
                    left_vals = data['close']
                elif left == 'Volume':
                    left_vals = data['volume']
                else:
                    continue
                
                if isinstance(right, str) and right in data['indicators']:
                    right_vals = data['indicators'][right]
                else:
                    right_vals = float(right)
                
                # Apply operator (simplified for GPU)
                if op == '>':
                    entry_signals[i] = left_vals > right_vals
                elif op == '<':
                    entry_signals[i] = left_vals < right_vals
                elif op == 'crosses_above':
                    if isinstance(right_vals, cp.ndarray):
                        entry_signals[i] = (left_vals > right_vals) & (cp.roll(left_vals, 1) <= cp.roll(right_vals, 1))
                    else:
                        entry_signals[i] = (left_vals > right_vals) & (cp.roll(left_vals, 1) <= right_vals)
            
            # Similar for exit signals (simplified)
            for i, (left, op, right) in enumerate(exit_conditions):
                if left not in ['profit', 'loss', 'days_held', 'trailing_stop']:
                    # Similar logic as entry signals
                    pass
            
            return entry_signals, exit_signals
        else:
            # CPU version
            return self._generate_all_signals_cpu(data, entry_conditions, exit_conditions)
    
    def batch_backtest_gpu(self, strategies: np.ndarray, ticker: str) -> np.ndarray:
        """Backtest a batch of strategies on GPU"""
        # Load data
        data = self.load_ticker_data_gpu(ticker)
        if data is None:
            return np.zeros((len(strategies), 4))  # Return empty results
        
        # For GPU version, we'd implement the full backtesting logic
        # For now, fallback to CPU
        return self._batch_backtest_cpu(strategies, ticker)
    
    def _batch_backtest_cpu(self, strategies: np.ndarray, ticker: str) -> np.ndarray:
        """CPU fallback for batch backtesting"""
        results = np.zeros((len(strategies), 4))  # return, trades, win_rate, sharpe
        
        # Load data as DataFrame
        with h5py.File(self.h5_file, 'r') as f:
            if ticker not in f:
                return results
            
            # Reconstruct DataFrame
            grp = f[ticker]
            data_dict = {
                'Close': grp['close'][:],
                'High': grp['high'][:],
                'Low': grp['low'][:],
                'Volume': grp['volume'][:]
            }
            
            # Add indicators
            for ind_name in grp['indicators']:
                data_dict[ind_name] = grp['indicators'][ind_name][:]
            
            df = pd.DataFrame(data_dict)
        
        # Create backtester
        backtester = FastBacktester(df)
        
        # Test each strategy
        for i, strategy_genes in enumerate(strategies):
            strategy = Strategy.from_chromosome(strategy_genes.tolist())
            metrics = backtester.backtest_strategy(strategy)
            
            results[i, 0] = metrics['total_return']
            results[i, 1] = metrics['num_trades']
            results[i, 2] = metrics['win_rate']
            results[i, 3] = metrics['sharpe']
        
        return results


class ExhaustiveStrategyTester:
    """Test all possible strategy combinations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.preprocessor = MassiveScalePreprocessor(data_dir)
        self.gpu_tester = GPUBacktester()
    
    def generate_all_strategies(self) -> np.ndarray:
        """Generate all possible strategy combinations"""
        n_entries = len(TRADING_GRAMMAR['entry_signals'])
        n_exits = len(TRADING_GRAMMAR['exit_signals'])
        n_filters = len(TRADING_GRAMMAR['filters'])
        n_directions = 2  # Long or short
        
        total = n_entries * n_exits * n_filters * n_directions
        print(f"Total possible strategies: {total:,}")
        
        # Generate using itertools
        strategies = []
        for entry, exit, filter, direction in itertools.product(
            range(n_entries), range(n_exits), range(n_filters), range(n_directions)
        ):
            # Create a simple chromosome representation
            chromosome = [entry, exit, filter, direction] + [0] * 46  # Pad to 50
            strategies.append(chromosome)
        
        return np.array(strategies, dtype=np.int32)
    
    def test_all_combinations(self, ticker: str = None, batch_size: int = 10000):
        """Test all strategy combinations"""
        print("\nTesting all possible strategy combinations...")
        
        # Prepare data
        if not os.path.exists("massive_scale_data.h5"):
            self.preprocessor.prepare_hdf5_data()
        
        # Get a ticker to test
        if ticker is None:
            with h5py.File("massive_scale_data.h5", 'r') as f:
                ticker = list(f.keys())[0]
        
        print(f"Testing on: {ticker}")
        
        # Generate all strategies
        all_strategies = self.generate_all_strategies()
        
        # Test in batches
        all_results = []
        start_time = time.time()
        
        for i in range(0, len(all_strategies), batch_size):
            batch = all_strategies[i:i+batch_size]
            
            # Test batch
            if self.gpu_tester.gpu_available:
                results = self.gpu_tester.batch_backtest_gpu(batch, ticker)
            else:
                results = self.gpu_tester._batch_backtest_cpu(batch, ticker)
            
            all_results.append(results)
            
            # Progress
            tested = min(i + batch_size, len(all_strategies))
            elapsed = time.time() - start_time
            rate = tested / elapsed
            eta = (len(all_strategies) - tested) / rate
            
            print(f"Progress: {tested:,}/{len(all_strategies):,} | "
                  f"Rate: {rate:,.0f} strategies/sec | "
                  f"ETA: {eta:.0f}s")
        
        # Combine results
        all_results = np.vstack(all_results)
        
        # Calculate fitness
        fitness = (
            all_results[:, 0] * 100 +      # Return
            all_results[:, 2] * 50 +       # Win rate
            np.clip(all_results[:, 3], -2, 2) * 25  # Sharpe
        )
        
        # Find top strategies
        top_indices = np.argsort(fitness)[-20:][::-1]
        
        print(f"\n{'='*60}")
        print("TOP STRATEGIES FROM EXHAUSTIVE SEARCH")
        print(f"{'='*60}")
        
        for rank, idx in enumerate(top_indices[:10]):
            strategy = Strategy.from_chromosome(all_strategies[idx].tolist())
            result = all_results[idx]
            
            print(f"\n{rank+1}. {strategy.describe()}")
            print(f"   Return: {result[0]:.1%}")
            print(f"   Trades: {result[1]:.0f}")
            print(f"   Win Rate: {result[2]:.1%}")
            print(f"   Sharpe: {result[3]:.2f}")
            print(f"   Fitness: {fitness[idx]:.1f}")
        
        return top_indices, all_strategies, all_results


class ParallelMassiveTester:
    """Test millions of random strategies in parallel"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.n_cores = min(mp.cpu_count(), 61)
    
    @staticmethod
    def test_strategy_batch(args):
        """Test a batch of strategies (for parallel processing)"""
        strategies, ticker, h5_file = args
        
        # Create local tester
        gpu_tester = GPUBacktester(h5_file)
        
        # Test strategies
        if gpu_tester.gpu_available:
            results = gpu_tester.batch_backtest_gpu(strategies, ticker)
        else:
            results = gpu_tester._batch_backtest_cpu(strategies, ticker)
        
        return results
    
    def test_millions(self, n_strategies: int = 1000000, 
                     ticker: str = None,
                     chunk_size: int = 10000):
        """Test millions of random strategies"""
        print(f"\nTesting {n_strategies:,} random strategies...")
        
        # Prepare data
        preprocessor = MassiveScalePreprocessor(self.data_dir)
        if not os.path.exists("massive_scale_data.h5"):
            preprocessor.prepare_hdf5_data()
        
        # Get ticker
        if ticker is None:
            with h5py.File("massive_scale_data.h5", 'r') as f:
                ticker = list(f.keys())[0]
        
        print(f"Testing on: {ticker}")
        print(f"Using {self.n_cores} CPU cores")
        
        # Generate random strategies
        print("Generating random strategies...")
        strategies = np.random.randint(0, 256, size=(n_strategies, 50), dtype=np.int32)
        
        # Split into chunks for parallel processing
        chunks = []
        for i in range(0, n_strategies, chunk_size):
            chunk = strategies[i:i+chunk_size]
            chunks.append((chunk, ticker, "massive_scale_data.h5"))
        
        # Process in parallel
        print(f"Processing {len(chunks)} chunks...")
        start_time = time.time()
        
        all_results = []
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            futures = [executor.submit(self.test_strategy_batch, chunk) for chunk in chunks]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                all_results.append(result)
                
                # Progress update
                progress = (i + 1) / len(chunks) * 100
                elapsed = time.time() - start_time
                rate = (i + 1) * chunk_size / elapsed
                
                print(f"Progress: {progress:.1f}% | Rate: {rate:,.0f} strategies/sec", end='\r')
        
        print()  # New line after progress
        
        # Combine results
        all_results = np.vstack(all_results)
        
        # Calculate statistics
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f} seconds")
        print(f"Average rate: {n_strategies/total_time:,.0f} strategies/second")
        
        # Find best strategies
        fitness = (
            all_results[:, 0] * 100 +      # Return
            all_results[:, 2] * 50 +       # Win rate
            np.clip(all_results[:, 3], -2, 2) * 25  # Sharpe
        )
        
        top_indices = np.argsort(fitness)[-20:][::-1]
        
        print(f"\n{'='*60}")
        print("TOP STRATEGIES FROM RANDOM SEARCH")
        print(f"{'='*60}")
        
        for rank, idx in enumerate(top_indices[:10]):
            strategy = Strategy.from_chromosome(strategies[idx].tolist())
            result = all_results[idx]
            
            print(f"\n{rank+1}. {strategy.describe()}")
            print(f"   Return: {result[0]:.1%}")
            print(f"   Trades: {result[1]:.0f}")
            print(f"   Win Rate: {result[2]:.1%}")
            print(f"   Sharpe: {result[3]:.2f}")
            print(f"   Fitness: {fitness[idx]:.1f}")
        
        return strategies[top_indices], all_results[top_indices]


if __name__ == "__main__":
    # Example 1: Test all combinations
    print("Example 1: Exhaustive search")
    exhaustive = ExhaustiveStrategyTester()
    exhaustive.test_all_combinations()
    
    # Example 2: Test millions of random strategies
    print("\n\nExample 2: Random search (1 million strategies)")
    massive = ParallelMassiveTester()
    massive.test_millions(n_strategies=1000000)