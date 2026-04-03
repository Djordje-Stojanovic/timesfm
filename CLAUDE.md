> **PROTECTED: "The Algorithm" section below must NEVER be edited, shortened, or reworded. It is a direct transcription of Elon Musk's engineering principles and is non-negotiable. If you feel the urge to "clean it up" — don't.**

# TimesFM Stock Prediction

Quick test of Google's TimesFM 2.5 (200M param time-series foundation model) for predicting NVIDIA next-day close price using 5 years of daily OHLC data.

## Stack

- **Python 3.11** (PyTorch compatibility sweet spot)
- **PyTorch** (CUDA, GPU-accelerated inference)
- **TimesFM 2.5** (200M param, `google/timesfm-2.5-200m-pytorch`)
- **yfinance** (NVIDIA historical OHLC data)
- **numpy** (array operations, log transforms)
- **uv** (package management)

## Project Structure

```
C:\AI\TimeSeries\
├── CLAUDE.md                # This file — project bible
├── LEARNINGS.md             # Hard-won lessons (append-only)
├── README.md                # TimesFM docs (cleaned from upstream)
├── predict_nvidia.py        # Main prediction script
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

# Run prediction
python predict_nvidia.py

# Install (first time)
uv venv --python 3.11
source .venv/Scripts/activate
uv pip install -e ".[torch]"
uv pip install yfinance
```

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

- **Log-transform prices** before feeding to TimesFM, `exp()` the output. Critical for financial data — stabilizes scale, makes returns additive.
- **Context**: Last 1024 trading days (~4 years of daily data).
- **Horizon**: 1 (next day close price).
- **ForecastConfig flags**: `normalize_inputs=True`, `use_continuous_quantile_head=True`, `force_flip_invariance=True`, `fix_quantile_crossing=True`.
- **Output**: `point_forecast` shape `(batch, horizon)`, `quantile_forecast` shape `(batch, horizon, 10)` — index 0 is mean, 1-9 are P10 through P90.
- **max_horizon must be multiple of 128** (output_patch_len). Use 128 minimum even for horizon=1.
- **TimesFM is univariate** — OHLC columns are forecast independently as separate batch elements.
- **Windows CUDA**: If `torch.cuda.is_available()` returns False, install CUDA-enabled PyTorch: `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`
- **TimesFM is mediocre on raw stock data** out of the box. This is a quick feasibility test, not production. Fine-tuning on financial data significantly improves results.

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
