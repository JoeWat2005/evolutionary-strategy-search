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

# Evolutionary Strategy Search

This repository is a **prototype trading strategy search and backtesting system** built in Python.
It explores large strategy spaces using **grammatical evolution, exhaustive search, and massive random testing**
over historical OHLCV market data.

> ⚠️ This is an experimental research prototype, not a production trading system.

---

## Requirements

- **Python 3.11**
- Tested on Linux / macOS (Windows may require extra setup for TA-Lib / CuPy)

---

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/JoeWat2005/evolutionary-strategy-search.git
cd evolutionary-strategy-search

python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

All interaction is done via **CLI arguments**.  
Inputs are parsed using `argparse`.

### 1. Prepare market data

```bash
python main.py --prepare --sectors TECH FINANCE
```

Options:
- `--sectors` : Market sectors (e.g. TECH, FINANCE, HEALTHCARE, ALL)
- `--start-date` / `--end-date`
- `--skip-download` : Reuse existing CSVs

---

### 2. Run grammatical evolution

```bash
python main.py --evolution --sectors TECH --generations 50
```

Results are written to `results_*.txt`.

---

### 3. Exhaustive strategy search

```bash
python main.py --exhaustive --ticker AAPL
```

---

### 4. Massive random search

```bash
python main.py --massive --strategies 10000000 --ticker AAPL
```

---

## Project Structure

```
.
├── main.py
├── market.py
├── indicators.py
├── optimized_evolution.py
├── massive_scale_tester.py
├── requirements.txt
└── data/
```

---

## Notes

- Execution model is simplified (no realistic slippage, spreads, or liquidity).
- GPU paths exist but currently fall back to CPU.
- Overfitting risk is high when brute-forcing large strategy spaces.
- High CPU usage is expected.

---

## License

No license is granted.  
Do not reuse or redistribute.
