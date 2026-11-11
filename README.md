> [!WARNING]
> **Deprecated: no longer maintained.** In November 2026 this prototype was merged into my University of Birmingham
> final-year project, *Adversarial Grammatical Evolution for Robust Quantitative Trading Strategies*, on the School of
> Computer Science GitLab: <https://git.cs.bham.ac.uk/projects-2026-27/jxw1627> (University login required).
>
> **How it was merged.** On 4 Nov 2026 this repository was imported into the project with `git subtree` as
> `legacy/evolutionary-strategy-search/`. Its seven commits are part of the project's history and its files are
> unchanged. It is kept there as a design reference and is not run. The project's own search engine, Forge in the
> Crucible research platform, replaces this prototype's search and backtester. Forge runs grammatical evolution over
> BNF grammars beside tree GP and random search, and this prototype's rule templates shaped one of those grammars. The
> project's `docs/history/README.md` describes how its earlier projects were consolidated.

# Evolutionary strategy search

A Python prototype that builds rule-based trading strategies from technical indicators, searches them with grammatical evolution, exhaustive enumeration or large-scale random sampling, and backtests them on daily historical prices.

**Status:** deprecated research prototype, kept for reference (see the notice above).

For research and education only. This is not financial advice and not a production trading system.

## Features

- Downloads daily OHLCV prices, adjusted for splits and dividends, through yfinance for six built-in sectors of 15 US stocks each: `TECH`, `FINANCE`, `HEALTHCARE`, `ENERGY`, `CONSUMER` and `INDUSTRIAL`.
- Adds more than 100 indicator columns per ticker, mostly from TA-Lib: moving averages, Bollinger Bands, RSI, MACD, stochastics, ATR, volume indicators, ADX, Hilbert transform cycles and candlestick patterns.
- Describes each strategy as an entry signal, an exit rule (profit target, stop loss, holding period, indicator signal or trailing stop), an optional filter and a direction (long or short).
- Three search modes: grammatical evolution, an exhaustive search over every rule combination, and large-scale random sampling.
- Reports total return, number of trades, win rate and a Sharpe-style ratio for the best strategies.
- Spreads work across processes, up to 61 at a time (the Windows limit for process pools).

## Tech stack

- Python 3.11
- pandas and NumPy
- TA-Lib for indicators
- yfinance for price data
- h5py for the HDF5 cache used by the exhaustive and random modes
- Numba and CuPy are imported, but the backtest loop runs in plain Python and the GPU path falls back to the CPU
- `concurrent.futures` process and thread pools

## Getting started

### Prerequisites

- Python 3.11, as stated in `todo.txt`.
- TA-Lib and CuPy may need extra system setup on some platforms; the earlier README flagged this for Windows.
- CuPy must be installed for `main.py` to start, even without a GPU: `massive_scale_tester.py` imports it unconditionally. `requirements.txt` pins `cupy-cuda12x`, the CUDA 12 build.

### Installation

```bash
git clone https://github.com/JoeWat2005/evolutionary-strategy-search.git
cd evolutionary-strategy-search

python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

pip install -r requirements.txt
```

### Usage

Everything runs through `main.py`. Choose at least one mode. Modes can be combined, and they run in this order: prepare, evolution, exhaustive, massive.

1. Download prices and calculate indicators:

   ```bash
   python main.py --prepare --sectors TECH FINANCE
   ```

2. Run grammatical evolution on a sector. The top 10 strategies for each sector are written to `results_<SECTOR>_<timestamp>.txt`.

   ```bash
   python main.py --evolution --sectors TECH --generations 50
   ```

3. Backtest every rule combination on one ticker:

   ```bash
   python main.py --exhaustive --ticker AAPL
   ```

4. Backtest a large number of random strategies on one ticker:

   ```bash
   python main.py --massive --strategies 10000000 --ticker AAPL
   ```

5. Prepare data and run evolution in one go:

   ```bash
   python main.py --prepare --evolution --sectors TECH FINANCE
   ```

The exhaustive and random modes print their top 10 strategies to the console instead of writing a file.

| Option | Default | Used by | Description |
|---|---|---|---|
| `--prepare` | | | Download prices and calculate indicators |
| `--evolution` | | | Run grammatical evolution |
| `--exhaustive` | | | Test every strategy combination |
| `--massive` | | | Test random strategies |
| `--sectors` | `TECH` | prepare, evolution | One or more sectors, or `ALL` |
| `--start-date` | `2020-01-01` | prepare | Download start date (YYYY-MM-DD) |
| `--end-date` | today | prepare | Download end date (YYYY-MM-DD) |
| `--skip-download` | off | prepare | Reuse the CSV files in `data/` and only calculate indicators |
| `--generations` | `50` | evolution | Number of generations |
| `--population` | `1000` | evolution | Population size |
| `--ticker` | first ticker in the HDF5 cache | exhaustive, massive | Ticker to test |
| `--strategies` | `1000000` | massive | Number of random strategies |
| `--batch-size` | `10000` | exhaustive, massive | Strategies per batch or chunk |
| `--workers` | CPU count, capped at 61 | prepare | Worker processes for the indicator step |

## How it works

1. **Data** (`market.py`): downloads daily bars for each ticker with `yf.download` (auto-adjusted) in a thread pool and saves them to `data/<TICKER>.csv`.
2. **Indicators** (`indicators.py`): reads each CSV, adds the indicator columns, drops rows with missing values (the longest look-back is the 200-period SMA) and overwrites the same file. Tickers are processed in parallel.
3. **Strategy grammar** (`optimized_evolution.py`): `TRADING_GRAMMAR` lists the entry signals, exit rules and filters. A strategy is a chromosome of integers; the first four genes pick the entry signal, exit rule, filter and direction by modulo.
4. **Backtest** (`FastBacktester`): steps through closing prices one bar at a time, holding at most one position, and calculates total return (the sum of per-trade returns), trade count, win rate, a Sharpe-style ratio annualised on 252 days, maximum drawdown, profit factor and a combined fitness score. No commissions, spreads or slippage are modelled.
5. **Search**:
   - *Evolution*: tournament selection, two-point crossover, mutation and elitism. Each strategy is scored on up to five tickers from the sector and its fitness is averaged.
   - *Exhaustive* (`massive_scale_tester.py`): enumerates every entry, exit, filter and direction combination and backtests each one on a single ticker.
   - *Random*: samples random chromosomes and backtests them in parallel chunks on a single ticker.

   Both of these modes first copy up to 20 ticker CSVs into `massive_scale_data.h5` for faster loading, and rank results by a simpler score built from return, win rate and the Sharpe-style ratio.

### Limitations

- The execution model is simplified: no realistic slippage, spreads or liquidity, and minimal position sizing and risk control.
- Strategies are scored on the full price history; there is no out-of-sample or walk-forward validation. Overfitting risk is high when brute-forcing large strategy spaces.
- The "GPU" backtest path is a stub that falls back to the CPU.
- High CPU usage is expected.

`todo.txt` records these limitations and the improvements that were planned for a next version.

## Project structure

```
.
├── main.py                   # command-line entry point
├── market.py                 # yfinance downloader and sector ticker lists
├── indicators.py             # indicator pipeline (TA-Lib)
├── optimized_evolution.py    # strategy grammar, backtester and grammatical evolution
├── massive_scale_tester.py   # HDF5 cache, exhaustive and random search
├── requirements.txt          # pinned dependencies
└── todo.txt                  # prototype notes, limitations and planned work
```

Created at run time: `data/` (price CSVs), `massive_scale_data.h5` and `results_*.txt`.

## Related repositories

- [quantforge](https://github.com/JoeWat2005/quantforge): local quantitative research lab with a Parquet price warehouse, walk-forward experiments and Monte Carlo simulation. It was merged into the same final-year project.
- [multi-tenant-tb](https://github.com/JoeWat2005/multi-tenant-tb): multi-tenant crypto trading bot platform with Claude-driven decisions and Alpaca paper trading.

## Licence

No licence is granted. Do not reuse or redistribute.
