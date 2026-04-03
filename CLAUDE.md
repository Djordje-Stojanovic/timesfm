> **PROTECTED: "The Algorithm" section below must NEVER be edited, shortened, or reworded. It is a direct transcription of Elon Musk's engineering principles and is non-negotiable. If you feel the urge to "clean it up" — don't.**

# TimesFM Stock Prediction

Quick test of Google's TimesFM 2.5 (200M param time-series foundation model) for predicting NVIDIA next-day close price using 5 years of daily OHLC data.

## Stack

- **Python 3.11** (PyTorch compatibility sweet spot)
- **PyTorch** (CUDA, GPU-accelerated inference)
- **TimesFM 2.5** (200M param, `google/timesfm-2.5-200m-pytorch`)
- **yfinance** (stock market OHLC data)
- **matplotlib** (chart generation for run reports)
- **numpy** (array operations, log transforms)
- **jax + jaxlib** (XReg covariate regression)
- **uv** (package management)

## Project Structure

```
C:\AI\TimeSeries\
├── CLAUDE.md                # This file — project bible
├── LEARNINGS.md             # Hard-won lessons (append-only)
├── README.md                # TimesFM docs (cleaned from upstream)
├── forecast_personal.py     # Personal finance forecast (savings, portfolio, investments)
├── predict.py               # Generalized stock prediction (any ticker + peers, XReg)
├── predict_mag7.py          # Mag7 + AVGO multi-strategy prediction
├── predict_nvidia.py        # Quick NVDA next-day close prediction
├── data/
│   └── personal_finance.py  # Personal financial data (income, portfolio, investments)
├── runs/                    # Timestamped forecast runs with charts + reports
│   └── YYYYMMDD_HHMMSS/
│       ├── report.md        # Full run report with tables + methodology
│       ├── 00_dashboard.png # 4-panel overview chart
│       ├── 01_*.png         # Individual forecast charts
│       └── ...
├── .claude/
│   └── settings.local.json  # Claude Code permissions
├── src/timesfm/             # TimesFM library (upstream, don't edit)
├── pyproject.toml           # TimesFM package definition
├── requirements.txt         # Upstream requirements
└── .venv/                   # Python 3.11 virtual environment
```

## Commands

```bash
# Activate environment
source .venv/Scripts/activate    # Windows/Git Bash

# Personal finance forecast (savings, portfolio, investments — generates charts + report)
python forecast_personal.py

# Stock prediction (any ticker with peer covariates)
python predict.py --target AXP --peers SPGI,MCO,FICO,V,MA
python predict.py --target TSLA --peers RIVN,LCID,NIO,GM,F

# Quick single-stock predictions
python predict_mag7.py          # Mag7 + AVGO
python predict_nvidia.py        # Quick NVDA test

# Install (first time)
uv venv --python 3.11
source .venv/Scripts/activate
uv pip install -e ".[torch]"
uv pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[xreg]"
uv pip install "jax[cpu]>=0.4.30"
uv pip install yfinance matplotlib
```

## Run System

Every forecast generates a timestamped run in `runs/YYYYMMDD_HHMMSS/`:
- `report.md` — full results with tables, methodology, and chart references
- `00_dashboard.png` — 4-panel overview
- Individual charts per forecast series

Runs are committed to git as a historical record of predictions.

## The Algorithm (Run Before Every Task)

Non-negotiable. This is the decision sequence for every feature, fix, and refactor:

### 1. Question the Requirement
- What user value does this create right now?
- What breaks if we skip it? If "nothing" — skip it.
- The best part is no part. The best process is no process.
- If a feature doesn't serve a real need today, delete it from the plan.

### 2. Delete
- Remove unnecessary steps, layers, abstractions, files, dependencies.
- If you haven't added something back at least 10% of the time, you're not deleting enough.
- No dead code. No unused imports. No "just in case" logic.
- No comments explaining what — write code that explains itself.
- Three similar lines > premature abstraction.

### 3. Simplify
- Only after deleting. Don't optimize what shouldn't exist.
- Fewer files, fewer moving parts. One file > two. Inline > abstracted.
- If it takes more than 30 seconds to understand, it's too complex.
- Solve today's problem, not tomorrow's imagined one.

### 4. Accelerate
- Optimize only what survived deletion and simplification.
- Ship in hours, not days. Smallest safe change that unlocks progress.
- Working code now beats perfect code later.
- Don't batch — ship each change individually so you know what broke.

### 5. Automate (Last)
- Only automate proven stable workflows.
- Never automate confusion. Never automate a process that shouldn't exist.
- Don't build tooling for something you've only done once.

### 6. Ship Prod-Ready Code
- Every change must build and run. No broken commits.
- No half-finished features, no placeholder junk on main.
- If it's not ready, it doesn't get committed.

## Key Technical Notes

### Data Preparation (CRITICAL)
- **Stocks (daily):** Use **log-returns** `r_t = ln(P_t / P_(t-1))`, NOT raw prices. Raw prices cause mean-reversion to historical average. Reconstruct via `P * exp(cumsum(returns))`.
- **Monthly financial data (savings, portfolio):** Raw values work for moderate-range data. Use log-returns for portfolio value (strong trend).
- **Always-positive series** (EUR amounts): Set `infer_is_positive=True`, `force_flip_invariance=False`.
- **Series that can be negative** (returns %): Set `infer_is_positive=False`, `force_flip_invariance=True`.

### Model Config
- **max_context**: Match data length. Use nearest multiple of 32 (patch_len). 1024 for daily stocks, 64 for monthly data.
- **max_horizon**: Must be multiple of 128 (output_patch_len). Use 128 minimum.
- **Quantile output**: Shape `(batch, horizon, 10)` — index 0=mean, 1=P10, 2=P20, ..., 5=P50(median), ..., 9=P90.
- **XReg covariates**: `forecast_with_covariates()` — dynamic covariates must span context + horizon.

### Architecture
- **TimesFM is univariate** — each series forecast independently. Use XReg for cross-series signal.
- **Two XReg modes**: `"xreg + timesfm"` (adjust residuals) vs `"timesfm + xreg"` (covariates explain main signal).
- **XReg requires**: `return_backcast=True`, `pip install timesfm[xreg]`, compatible jax+jaxlib.

### Windows
- **CUDA PyTorch**: `uv pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`
- **UTF-8 output**: `sys.stdout.reconfigure(encoding="utf-8")` + run with `PYTHONIOENCODING=utf-8`

## Best Practices

- Don't edit anything in `src/timesfm/` — treat as upstream library.
- Keep prediction logic in a single file (`predict_nvidia.py`).
- Log every learning to LEARNINGS.md.
- Commit after each working change.
- Simple > clever. This is a test, not a product.

## Git

- `master` branch, simple commits.
- Commit format: `<type>: <what and why>`
- Types: `feat`, `fix`, `docs`
