# optimized_evolution.py - Fast grammatical evolution for trading strategies

import numpy as np
import pandas as pd
from numba import jit, prange
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import random
import hashlib
from market import STOCK_SECTORS

# Simplified but effective trading grammar
TRADING_GRAMMAR = {
    # Entry signals that actually work
    'entry_signals': [
        # Momentum - adjusted for actual RSI range
        ('RSI_14', '<', 35),  # Oversold (more realistic)
        ('RSI_14', '>', 65),  # Overbought (more realistic)
        ('RSI_14', '<', 40),
        ('RSI_14', '>', 60),
        ('RSI_14', '<', 45),  # Even more signals
        ('RSI_14', '>', 55),
        ('RSI_14', 'crosses_above', 40),
        ('RSI_14', 'crosses_below', 60),
        
        # Moving average signals
        ('Close', 'crosses_above', 'SMA_20'),
        ('Close', 'crosses_below', 'SMA_20'),
        ('SMA_10', 'crosses_above', 'SMA_50'),
        ('SMA_50', 'crosses_above', 'SMA_200'),
        ('EMA_10', 'crosses_above', 'EMA_20'),
        
        # Bollinger Bands
        ('Close', '<', 'BB_lower'),
        ('Close', '>', 'BB_upper'),
        ('Close', 'crosses_above', 'BB_lower'),
        ('Close', 'crosses_below', 'BB_upper'),
        
        # MACD
        ('MACD', 'crosses_above', 'MACD_signal'),
        ('MACD', 'crosses_below', 'MACD_signal'),
        ('MACD_hist', '>', 0),
        ('MACD_hist', '<', 0),
        
        # Volume
        ('Volume', '>', 'Volume_SMA_20'),
        ('OBV', 'crosses_above', 'OBV_SMA_20'),
        
        # Stochastic
        ('STOCH_K', '<', 20),
        ('STOCH_K', '>', 80),
        ('STOCH_K', 'crosses_above', 'STOCH_D'),
        
        # CCI - adjusted for actual range (-340 to 325)
        ('CCI_14', '<', -100),
        ('CCI_14', '>', 100),
        ('CCI_14', '<', -50),
        ('CCI_14', '>', 50),
        
        # Patterns
        ('CDL_HAMMER', '>', 0),
        ('CDL_ENGULFING', '>', 0),
        ('CDL_DOJI', '>', 0),
        ('CDL_MORNINGSTAR', '>', 0),
    ],
    
    # Exit signals
    'exit_signals': [
        # Fixed targets
        ('profit', '>', 1),  # 1% profit (more realistic)
        ('profit', '>', 2),  # 2% profit
        ('profit', '>', 3),  # 3% profit
        ('profit', '>', 5),  # 5% profit
        ('profit', '>', 10), # 10% profit
        ('loss', '>', 1),    # 1% stop loss
        ('loss', '>', 2),    # 2% stop loss
        ('loss', '>', 3),    # 3% stop loss
        ('loss', '>', 5),    # 5% stop loss
        
        # Time-based
        ('days_held', '>', 3),
        ('days_held', '>', 5),
        ('days_held', '>', 10),
        ('days_held', '>', 20),
        
        # Indicator-based - adjusted thresholds
        ('RSI_14', '>', 65),
        ('RSI_14', '<', 35),
        ('RSI_14', '>', 60),
        ('RSI_14', '<', 40),
        ('Close', 'crosses_below', 'SMA_20'),
        ('Close', 'crosses_above', 'SMA_20'),
        ('MACD', 'crosses_below', 'MACD_signal'),
        
        # Trailing stops
        ('trailing_stop', '=', 2),
        ('trailing_stop', '=', 3),
        ('trailing_stop', '=', 5),
        ('trailing_stop', '=', 10),
    ],
    
    # Filters (additional conditions)
    'filters': [
        None,  # No filter
        ('ADX_14', '>', 20),  # Trending market
        ('ADX_14', '>', 25),
        ('Volume', '>', 'Volume_SMA_20'),  # High volume
        ('ATR_14', '>', 'ATR_14_SMA_20'),  # High volatility
        ('BB_width', '>', 'BB_width_SMA_20'),
        ('MFI_14', '<', 80),  # Not overbought (volume)
        ('MFI_14', '>', 20),  # Not oversold (volume)
    ]
}

@dataclass
class Strategy:
    """Compact strategy representation"""
    chromosome: List[int]
    entry_idx: int
    exit_idx: int
    filter_idx: int
    direction: int  # 1 for long, -1 for short
    
    @classmethod
    def from_chromosome(cls, chromosome: List[int]) -> 'Strategy':
        # Map chromosome to strategy components
        entry_idx = chromosome[0] % len(TRADING_GRAMMAR['entry_signals'])
        exit_idx = chromosome[1] % len(TRADING_GRAMMAR['exit_signals'])
        filter_idx = chromosome[2] % len(TRADING_GRAMMAR['filters'])
        direction = 1 if chromosome[3] % 2 == 0 else -1
        
        return cls(
            chromosome=chromosome,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            filter_idx=filter_idx,
            direction=direction
        )
    
    def describe(self) -> str:
        """Human-readable strategy description"""
        entry = TRADING_GRAMMAR['entry_signals'][self.entry_idx]
        exit = TRADING_GRAMMAR['exit_signals'][self.exit_idx]
        filter_cond = TRADING_GRAMMAR['filters'][self.filter_idx]
        
        desc = f"{'LONG' if self.direction == 1 else 'SHORT'} when {entry[0]} {entry[1]} {entry[2]}"
        desc += f", EXIT when {exit[0]} {exit[1]} {exit[2]}"
        if filter_cond:
            desc += f", FILTER: {filter_cond[0]} {filter_cond[1]} {filter_cond[2]}"
        
        return desc
    
    def get_hash(self) -> str:
        """Unique hash for strategy"""
        return hashlib.md5(self.describe().encode()).hexdigest()[:8]


class FastBacktester:
    """Optimized backtester using vectorized operations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.n_points = len(data)
        
        # Pre-extract arrays for speed
        self.dates = data.index.values
        self.close = data['Close'].values
        self.high = data['High'].values
        self.low = data['Low'].values
        self.volume = data['Volume'].values
        
        # Pre-calculate indicators we'll need
        self._prepare_indicators()
    
    def _prepare_indicators(self):
        """Pre-calculate indicator arrays for fast access"""
        self.indicators = {}
        
        # Direct copy of existing indicators
        for col in self.data.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                self.indicators[col] = self.data[col].values
        
        # Add any missing calculated indicators
        if 'Volume_SMA_20' not in self.indicators and 'Volume' in self.data.columns:
            self.indicators['Volume_SMA_20'] = pd.Series(self.volume).rolling(20).mean().values
        
        if 'OBV_SMA_20' not in self.indicators and 'OBV' in self.indicators:
            self.indicators['OBV_SMA_20'] = pd.Series(self.indicators['OBV']).rolling(20).mean().values
        
        if 'ATR_14_SMA_20' not in self.indicators and 'ATR_14' in self.indicators:
            self.indicators['ATR_14_SMA_20'] = pd.Series(self.indicators['ATR_14']).rolling(20).mean().values
        
        if 'BB_width_SMA_20' not in self.indicators and 'BB_width' in self.indicators:
            self.indicators['BB_width_SMA_20'] = pd.Series(self.indicators['BB_width']).rolling(20).mean().values
    
    def evaluate_condition(self, condition: Tuple) -> np.ndarray:
        """Vectorized condition evaluation"""
        left, operator, right = condition
        
        # Special runtime conditions
        if left in ['profit', 'loss', 'days_held', 'trailing_stop']:
            return np.zeros(self.n_points, dtype=bool)
        
        # Get left values
        if left in self.indicators:
            left_vals = self.indicators[left]
        elif left == 'Close':
            left_vals = self.close
        elif left == 'Volume':
            left_vals = self.volume
        else:
            return np.zeros(self.n_points, dtype=bool)
        
        # Get right values
        if isinstance(right, (int, float)):
            right_vals = float(right)
        elif right in self.indicators:
            right_vals = self.indicators[right]
        else:
            return np.zeros(self.n_points, dtype=bool)
        
        # Apply operator
        if operator == '>':
            return left_vals > right_vals
        elif operator == '<':
            return left_vals < right_vals
        elif operator == '>=':
            return left_vals >= right_vals
        elif operator == '<=':
            return left_vals <= right_vals
        elif operator == '==' or operator == '=':
            return left_vals == right_vals
        elif operator == 'crosses_above':
            if isinstance(right_vals, np.ndarray):
                prev_left = np.roll(left_vals, 1)
                prev_right = np.roll(right_vals, 1)
                return (left_vals > right_vals) & (prev_left <= prev_right)
            else:
                prev = np.roll(left_vals, 1)
                return (left_vals > right_vals) & (prev <= right_vals)
        elif operator == 'crosses_below':
            if isinstance(right_vals, np.ndarray):
                prev_left = np.roll(left_vals, 1)
                prev_right = np.roll(right_vals, 1)
                return (left_vals < right_vals) & (prev_left >= prev_right)
            else:
                prev = np.roll(left_vals, 1)
                return (left_vals < right_vals) & (prev >= right_vals)
        
        return np.zeros(self.n_points, dtype=bool)
    
    def _run_backtest_numba(self, close: np.ndarray, entry_signals: np.ndarray, 
                           exit_signals: np.ndarray, direction: int,
                           exit_type: str, exit_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Backtest core (without numba for now)"""
        n = len(close)
        trades = []
        trade_days = []
        
        position = 0
        entry_price = 0.0
        entry_idx = 0
        highest_price = 0.0
        
        for i in range(n):
            current_price = close[i]
            
            # Check exit conditions
            if position != 0:
                exit_triggered = False
                days_held = i - entry_idx
                
                # Check exit signal
                if exit_signals[i]:
                    exit_triggered = True
                
                # Check profit/loss exits
                if not exit_triggered:
                    if position == 1:  # Long position
                        current_return = (current_price - entry_price) / entry_price * 100
                        
                        if exit_type == 'profit' and current_return > exit_value:
                            exit_triggered = True
                        elif exit_type == 'loss' and current_return < -exit_value:
                            exit_triggered = True
                        elif exit_type == 'trailing_stop':
                            if current_price > highest_price:
                                highest_price = current_price
                            if current_price < highest_price * (1 - exit_value/100):
                                exit_triggered = True
                                
                    else:  # Short position
                        current_return = (entry_price - current_price) / entry_price * 100
                        
                        if exit_type == 'profit' and current_return > exit_value:
                            exit_triggered = True
                        elif exit_type == 'loss' and current_return < -exit_value:
                            exit_triggered = True
                
                # Check time exit
                if not exit_triggered and exit_type == 'days_held' and days_held > exit_value:
                    exit_triggered = True
                
                # Execute exit
                if exit_triggered:
                    if position == 1:
                        trade_return = (current_price - entry_price) / entry_price
                    else:
                        trade_return = (entry_price - current_price) / entry_price
                    
                    trades.append(trade_return)
                    trade_days.append(days_held)
                    position = 0
            
            # Check entry conditions
            if position == 0 and entry_signals[i]:
                position = direction
                entry_price = current_price
                entry_idx = i
                highest_price = current_price
        
        # Convert to arrays
        if trades:
            return np.array(trades), np.array(trade_days)
        else:
            return np.array([]), np.array([])
    
    def backtest_strategy(self, strategy: Strategy) -> Dict:
        """Run backtest for a single strategy"""
        # Get strategy components
        entry_condition = TRADING_GRAMMAR['entry_signals'][strategy.entry_idx]
        exit_condition = TRADING_GRAMMAR['exit_signals'][strategy.exit_idx]
        filter_condition = TRADING_GRAMMAR['filters'][strategy.filter_idx]
        
        # Generate entry signals
        entry_signals = self.evaluate_condition(entry_condition)
        
        # Apply filter if exists
        if filter_condition:
            filter_mask = self.evaluate_condition(filter_condition)
            entry_signals = entry_signals & filter_mask
        
        # Generate exit signals (for indicator-based exits)
        exit_left = exit_condition[0]
        if exit_left not in ['profit', 'loss', 'days_held', 'trailing_stop']:
            exit_signals = self.evaluate_condition(exit_condition)
        else:
            exit_signals = np.zeros(self.n_points, dtype=bool)
        
        # Determine exit type for special handling
        exit_type = exit_condition[0]
        
        # Only convert to float if it's actually a number
        try:
            exit_value = float(exit_condition[2])
        except (ValueError, TypeError):
            # If it's not a number (like 'SMA_20'), use 0 as default
            # These exits are handled by the evaluate_condition above
            exit_value = 0.0
        
        # Run the backtest
        trades, trade_days = self._run_backtest_numba(
            self.close, entry_signals, exit_signals, 
            strategy.direction, exit_type, exit_value
        )
        
        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'fitness': 0.0
            }
        
        # Basic metrics
        total_return = np.sum(trades)
        num_trades = len(trades)
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades <= 0]
        win_rate = len(winning_trades) / num_trades
        avg_trade = np.mean(trades)
        
        # Risk metrics
        if np.std(trades) > 0:
            sharpe = np.sqrt(252 / np.mean(trade_days)) * avg_trade / np.std(trades)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Profit factor
        if len(losing_trades) > 0 and np.sum(np.abs(losing_trades)) > 0:
            profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))
        else:
            profit_factor = 10.0 if len(winning_trades) > 0 else 0.0
        
        # Calculate fitness (balanced scoring)
        fitness = (
            total_return * 100 +                    # Return component
            win_rate * 50 +                         # Win rate component  
            min(num_trades / 10, 10) * 10 +        # Activity component (cap at 100 trades)
            max(-10, min(10, sharpe)) * 5 +        # Risk-adjusted return
            (1 + max_drawdown) * 20 +              # Drawdown penalty
            min(profit_factor, 3) * 10             # Profit factor bonus
        )
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'fitness': max(0, fitness)
        }


class GrammaticalEvolution:
    """Grammatical Evolution engine"""
    
    def __init__(self, population_size: int = 1000, chromosome_length: int = 50):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        self.elite_size = max(50, population_size // 10)
        
        # Tracking
        self.generation = 0
        self.best_strategies = []
        self.strategy_cache = {}
    
    def create_population(self) -> List[List[int]]:
        """Create initial random population"""
        population = []
        
        for _ in range(self.population_size):
            # Random chromosome
            chromosome = [random.randint(0, 255) for _ in range(self.chromosome_length)]
            population.append(chromosome)
        
        return population
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Mutate chromosome"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:
                    # Small change
                    mutated[i] = (mutated[i] + random.randint(-10, 10)) % 256
                else:
                    # Random value
                    mutated[i] = random.randint(0, 255)
        
        return mutated
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Two-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def tournament_selection(self, population: List[List[int]], 
                           fitnesses: List[float], 
                           tournament_size: int = 3) -> List[int]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    def evolve_population(self, population: List[List[int]], 
                         fitnesses: List[float]) -> List[List[int]]:
        """Create next generation"""
        # Sort by fitness
        sorted_pairs = sorted(zip(fitnesses, population), reverse=True)
        sorted_pop = [p for f, p in sorted_pairs]
        
        # New population
        new_population = []
        
        # Keep elite
        new_population.extend(sorted_pop[:self.elite_size])
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact size
        return new_population[:self.population_size]


def evaluate_population_batch(args):
    """Evaluate a batch of strategies on multiple stocks"""
    chromosomes, stock_files, batch_id = args
    
    results = []
    
    # Load stock data
    stock_data = {}
    for stock_file in stock_files[:5]:  # Test on up to 5 stocks
        try:
            df = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
            if len(df) > 100:  # Minimum data requirement
                stock_data[stock_file] = df
        except:
            continue
    
    if not stock_data:
        return [(0.0, {}) for _ in chromosomes]
    
    # Evaluate each strategy
    for chromosome in chromosomes:
        strategy = Strategy.from_chromosome(chromosome)
        
        total_fitness = 0
        total_metrics = {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'sharpe': 0
        }
        
        # Test on each stock
        for stock_file, data in stock_data.items():
            try:
                backtester = FastBacktester(data)
                metrics = backtester.backtest_strategy(strategy)
                
                total_fitness += metrics['fitness']
                for key in total_metrics:
                    total_metrics[key] += metrics.get(key, 0)
            except:
                continue
        
        # Average metrics
        n_stocks = len(stock_data)
        avg_fitness = total_fitness / n_stocks
        
        for key in total_metrics:
            total_metrics[key] /= n_stocks
        
        results.append((avg_fitness, total_metrics))
    
    return results


def run_evolution(sector: str = "TECH", 
                 generations: int = 50,
                 population_size: int = 1000,
                 data_dir: str = "data") -> List[Tuple[Strategy, Dict]]:
    """Run grammatical evolution for a sector - FIXED VERSION"""
    
    print(f"\n{'='*60}")
    print(f"GRAMMATICAL EVOLUTION - {sector} SECTOR")
    print(f"{'='*60}")
    print(f"Population: {population_size}")
    print(f"Generations: {generations}")
    
    # Get stock files for sector
    sector_tickers = STOCK_SECTORS.get(sector, [])
    stock_files = []
    
    for ticker in sector_tickers:
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        if os.path.exists(file_path):
            stock_files.append(file_path)
    
    print(f"Stocks available: {len(stock_files)}")
    
    if not stock_files:
        print("No data files found!")
        return []
    
    # Initialize evolution
    evolution = GrammaticalEvolution(population_size)
    population = evolution.create_population()
    
    # Track all unique strategies and their best performance
    all_strategies_performance = {}  # hash -> (strategy, metrics, fitness)
    
    for gen in range(generations):
        start_time = time.time()
        
        # Prepare batches for parallel evaluation
        n_workers = min(mp.cpu_count(), 61)
        batch_size = len(population) // n_workers
        batches = []
        
        for i in range(n_workers):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < n_workers - 1 else len(population)
            batch = population[start_idx:end_idx]
            batches.append((batch, stock_files, i))
        
        # Parallel evaluation
        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(evaluate_population_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Extract fitnesses and track unique strategies
        fitnesses = []
        for i, (fitness, metrics) in enumerate(all_results):
            fitnesses.append(fitness)
            
            # Create strategy and get its hash
            strategy = Strategy.from_chromosome(population[i])
            strategy_hash = strategy.get_hash()
            
            # Track if this is the best performance for this strategy
            if strategy_hash not in all_strategies_performance or fitness > all_strategies_performance[strategy_hash][2]:
                all_strategies_performance[strategy_hash] = (strategy, metrics.copy(), fitness)
        
        # Track best in this generation
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_strategy = Strategy.from_chromosome(population[best_idx])
        best_metrics = all_results[best_idx][1]
        
        # Stats
        avg_fitness = np.mean(fitnesses)
        elapsed = time.time() - start_time
        
        print(f"Gen {gen+1:3d} | Best: {best_fitness:6.1f} | Avg: {avg_fitness:6.1f} | "
              f"Time: {elapsed:4.1f}s | Return: {best_metrics['total_return']:.1%} | "
              f"Trades: {best_metrics['num_trades']:.0f} | "
              f"Unique strategies: {len(all_strategies_performance)}")
        
        # Evolve
        if gen < generations - 1:
            population = evolution.evolve_population(population, fitnesses)
    
    # Sort all unique strategies by fitness
    unique_strategies = list(all_strategies_performance.values())
    unique_strategies.sort(key=lambda x: x[2], reverse=True)
    
    # Print top unique strategies
    print(f"\n{'='*60}")
    print(f"TOP 10 UNIQUE STRATEGIES (from {len(unique_strategies)} total unique)")
    print(f"{'='*60}")
    
    seen_descriptions = set()
    displayed = 0
    
    for strategy, metrics, fitness in unique_strategies:
        desc = strategy.describe()
        
        # Skip if we've seen this exact description
        if desc in seen_descriptions:
            continue
            
        seen_descriptions.add(desc)
        displayed += 1
        
        print(f"\n{displayed}. {desc}")
        print(f"   Fitness: {fitness:.1f}")
        print(f"   Return: {metrics['total_return']:.1%}")
        print(f"   Trades: {metrics['num_trades']:.0f}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Sharpe: {metrics['sharpe']:.2f}")
        
        if displayed >= 10:
            break
    
    # Return unique strategies
    return [(s, m) for s, m, f in unique_strategies[:20]]


if __name__ == "__main__":
    # Test run
    run_evolution("TECH", generations=20, population_size=500)