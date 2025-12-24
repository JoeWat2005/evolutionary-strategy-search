# main.py - Main interface for the complete trading strategy system

import argparse
import time
from datetime import datetime
import os
import sys

# Import our modules
from market import download_market_data, STOCK_SECTORS
from indicators import calculate_indicators_parallel
from optimized_evolution import run_evolution
from massive_scale_tester import ExhaustiveStrategyTester, ParallelMassiveTester, MassiveScalePreprocessor


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")


def prepare_data(args):
    """Download and prepare market data"""
    print_header("DATA PREPARATION")
    
    # Step 1: Download data
    if not args.skip_download:
        print("\n[1/2] Downloading market data...")
        download_market_data(
            sectors=args.sectors,
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=10
        )
    else:
        print("\n[1/2] Skipping download (using existing data)")
    
    # Step 2: Calculate indicators
    print("\n[2/2] Calculating technical indicators...")
    results = calculate_indicators_parallel(
        input_dir="data",
        output_dir="data",
        max_workers=args.workers
    )
    
    print(f"\nData preparation complete! {len(results)} files ready.")


def run_grammatical_evolution(args):
    """Run grammatical evolution"""
    print_header("GRAMMATICAL EVOLUTION")
    
    # Check if data exists
    data_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    if not data_files:
        print("No data files found! Run with --prepare first.")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Run evolution for each sector
    for sector in args.sectors:
        results = run_evolution(
            sector=sector,
            generations=args.generations,
            population_size=args.population,
            data_dir="data"
        )
        
        # Save results
        if results:
            output_file = f"results_{sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Top Strategies for {sector} Sector\n")
                f.write("="*60 + "\n\n")
                
                for i, (strategy, metrics) in enumerate(results[:10]):
                    f.write(f"{i+1}. {strategy.describe()}\n")
                    f.write(f"   Return: {metrics['total_return']:.1%}\n")
                    f.write(f"   Trades: {metrics['num_trades']:.0f}\n")
                    f.write(f"   Win Rate: {metrics['win_rate']:.1%}\n")
                    f.write(f"   Sharpe: {metrics['sharpe']:.2f}\n\n")
            
            print(f"Results saved to: {output_file}")


def run_exhaustive_search(args):
    """Test all possible strategy combinations"""
    print_header("EXHAUSTIVE STRATEGY SEARCH")
    
    # Prepare data for massive scale
    preprocessor = MassiveScalePreprocessor("data")
    if not os.path.exists("massive_scale_data.h5"):
        print("Preparing data for massive scale testing...")
        preprocessor.prepare_hdf5_data()
    
    # Run exhaustive search
    tester = ExhaustiveStrategyTester("data")
    tester.test_all_combinations(ticker=args.ticker, batch_size=args.batch_size)


def run_massive_random(args):
    """Test millions of random strategies"""
    print_header(f"MASSIVE RANDOM SEARCH ({args.n_strategies:,} strategies)")
    
    # Prepare data
    preprocessor = MassiveScalePreprocessor("data")
    if not os.path.exists("massive_scale_data.h5"):
        print("Preparing data for massive scale testing...")
        preprocessor.prepare_hdf5_data()
    
    # Run massive test
    tester = ParallelMassiveTester("data")
    tester.test_millions(
        n_strategies=args.n_strategies,
        ticker=args.ticker,
        chunk_size=args.batch_size
    )


def main():
    parser = argparse.ArgumentParser(
        description='Complete Trading Strategy Testing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data for TECH and FINANCE sectors
  python main.py --prepare --sectors TECH FINANCE
  
  # Run grammatical evolution
  python main.py --evolution --sectors TECH --generations 50
  
  # Test all possible combinations
  python main.py --exhaustive --ticker AAPL
  
  # Test 10 million random strategies
  python main.py --massive --strategies 10000000
  
  # Full pipeline
  python main.py --prepare --evolution --sectors TECH FINANCE
        """
    )
    
    # Mode selection
    parser.add_argument('--prepare', action='store_true',
                       help='Download and prepare market data')
    parser.add_argument('--evolution', action='store_true',
                       help='Run grammatical evolution')
    parser.add_argument('--exhaustive', action='store_true',
                       help='Test all possible strategy combinations')
    parser.add_argument('--massive', action='store_true',
                       help='Test millions of random strategies')
    
    # Data preparation options
    parser.add_argument('--sectors', nargs='+', 
                       choices=list(STOCK_SECTORS.keys()) + ['ALL'],
                       default=['TECH'],
                       help='Market sectors (default: TECH)')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading, only calculate indicators')
    
    # Evolution options
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations (default: 50)')
    parser.add_argument('--population', type=int, default=1000,
                       help='Population size (default: 1000)')
    
    # Testing options
    parser.add_argument('--ticker', type=str, default=None,
                       help='Specific ticker for testing')
    parser.add_argument('--strategies', type=int, default=1000000,
                       dest='n_strategies',
                       help='Number of strategies for massive test')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size for parallel processing')
    
    # General options
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Handle 'ALL' sectors
    if 'ALL' in args.sectors:
        args.sectors = list(STOCK_SECTORS.keys())
    
    # Validate modes
    modes_selected = sum([args.prepare, args.evolution, args.exhaustive, args.massive])
    if modes_selected == 0:
        print("Error: Select at least one mode: --prepare, --evolution, --exhaustive, or --massive")
        parser.print_help()
        sys.exit(1)
    
    # Print configuration
    print_header("TRADING STRATEGY SYSTEM")
    print(f"Start time: {datetime.now()}")
    print(f"Modes: ", end='')
    modes = []
    if args.prepare: modes.append("Prepare Data")
    if args.evolution: modes.append("Grammatical Evolution")
    if args.exhaustive: modes.append("Exhaustive Search")
    if args.massive: modes.append(f"Massive Random ({args.n_strategies:,})")
    print(", ".join(modes))
    
    start_time = time.time()
    
    # Execute modes in order
    try:
        if args.prepare:
            prepare_data(args)
        
        if args.evolution:
            run_grammatical_evolution(args)
        
        if args.exhaustive:
            run_exhaustive_search(args)
        
        if args.massive:
            run_massive_random(args)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    elapsed = time.time() - start_time
    print_header("COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()