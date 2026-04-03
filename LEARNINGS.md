# LEARNINGS.md — Hard-Won Lessons

Append-only log of things learned the hard way. Each entry: date, what happened, the fix, and why.

---

## Log-Returns vs Log-Prices for Stock Prediction (2026-04-03)

**Problem:** TimesFM standalone predicted AVGO at $55 when actual price was $314 (-82%). NVDA predicted $22 vs actual $177 (-87%). Every stock showed massive negative predictions.

**Root cause:** We fed log-transformed prices (e.g., log($314) = 5.75) as input. Stocks with strong uptrends (AVGO went from $40 to $314 in 4 years) have a series that goes from 3.6 to 6.0 in log space. After the model's internal normalization, the recent values sit at the extreme high end. TimesFM interprets this as "way above average" and predicts mean-reversion back to the historical average: exp(mean(log_prices)) = exp(4.6) ≈ $100.

**Fix:** Use **daily log-returns** `r_t = ln(P_t / P_(t-1))` instead of raw log-prices. Log-returns are stationary, centered around 0, and have no trend — exactly what the model expects. Reconstruct price via `P_future = P_last * exp(cumsum(predicted_returns))`.

**Result:** After fix, AVGO standalone predicted $314.92 (+0.12%) — reasonable. XReg with Mag7 covariates predicted $313.65 (-0.29%) with tighter confidence intervals (3.6% vs 6.3%).

**How to apply:** NEVER feed raw price levels or log-prices to TimesFM for stocks. Always use returns or differenced data. This applies to any time series with strong trends.

---

## XReg Requires jax + jaxlib (Not Just jax) (2026-04-03)

**Problem:** `uv pip install -e ".[xreg]"` installed `jax==0.2.22` (ancient) without jaxlib. Import failed with `ModuleNotFoundError: No module named 'jaxlib'`.

**Fix:** Install compatible versions: `uv pip install "jax[cpu]>=0.4.30"` which pulls matching jaxlib automatically. The `.[xreg]` pyproject.toml dependency pins jax too low.

**How to apply:** After installing `.[xreg]`, always verify with `python -c "from timesfm.utils import xreg_lib; print('ok')"`.

---

## PyTorch Installs CPU-Only by Default on Windows (2026-04-03)

**Problem:** `uv pip install -e ".[torch]"` installed `torch==2.11.0` (CPU-only). `torch.cuda.is_available()` returned False despite having an RTX 5070.

**Fix:** Force CUDA build: `uv pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`. The `+cu128` suffix is key — without it uv resolves to the generic (CPU) wheel.

**How to apply:** After any torch install, always verify: `python -c "import torch; print(torch.cuda.is_available())"`. RTX 5070 (sm_120/Blackwell) requires CUDA 12.8+.

---

## GitHub Public Forks Cannot Upload LFS Objects (2026-04-03)

**Problem:** `git push --force` failed with "cannot upload new objects to public fork" because the repo had .gitattributes tracking .png/.gif files via Git LFS.

**Fix:** Remove `.gitattributes` and the LFS-tracked files from the index: `git rm .gitattributes` + `git rm --cached` the tracked files. Then `git lfs uninstall`.

**How to apply:** When squashing a forked repo with LFS, remove LFS tracking first.

---

## XReg Tightens Confidence Intervals (2026-04-03)

**Observation:** XReg with Mag7 covariates produced a tighter 80% confidence interval (3.6% of price) compared to standalone (6.3%). This makes sense — if correlated stocks confirm the direction, uncertainty should decrease.

**Standalone Day +1:** $314.92, P10=$305.62, P90=$325.35 (80% CI = 6.3%)
**XReg Day +1:** $313.65, P10=$308.21, P90=$319.47 (80% CI = 3.6%)

**How to apply:** XReg covariates are most valuable for reducing uncertainty, not necessarily for improving point forecast accuracy. AVGO-NVDA correlation of 0.61 is the strongest in the Mag7 set — semiconductor peers provide the best signal.

---

## Windows stdout Encoding Breaks on Unicode (2026-04-03)

**Problem:** Python 3.11 on Windows uses cp1252 for stdout by default. Arrow characters (→, ▲, ▼) cause `UnicodeEncodeError`.

**Fix:** Add at script top: `sys.stdout.reconfigure(encoding="utf-8")` and run with `PYTHONIOENCODING=utf-8 python -X utf8`.

**How to apply:** Every Python script on Windows that prints non-ASCII should set UTF-8 explicitly.

---

## Monthly Data Needs Different Config Than Daily Stock Data (2026-04-03)

**Problem:** First run of personal finance forecast produced flat predictions (1,412 EUR for every month, no uncertainty). The model predicted a single constant for all 9 future months.

**Root cause:** Three compounding issues:
1. `CONTEXT_LEN=128` but only 54 data points — excessive padding confused the model
2. `infer_is_positive=False` for EUR amounts that are always positive — model wasted capacity on impossible negative values
3. `force_flip_invariance=True` for positive-only series — averaging f(x) and -f(-x) cancels signal

**Fix:**
- Set `CONTEXT_LEN=64` (nearest multiple of patch_len=32 above our 54 points)
- For positive series (savings, investments): `infer_is_positive=True`, `force_flip_invariance=False`
- For series that can be negative (return %): `infer_is_positive=False`, `force_flip_invariance=True`
- Recompile model between series with different settings

**How to apply:** Always match ForecastConfig to the data characteristics. Monthly data with 50-100 points needs different settings than daily stock data with 1000+ points. The `infer_is_positive` and `force_flip_invariance` flags have a big impact on short series.

---

## Portfolio Log-Returns Extrapolate Aggressively (2026-04-03)

**Observation:** Portfolio value forecast using log-returns predicted +119% growth in 9 months (115K -> 253K EUR). This is because the recent monthly log-returns average ~2.5%/month, and the model extrapolates this linearly. For a portfolio that has been growing steadily, this is the model's best guess — but it ignores mean reversion, market conditions, and the fact that past returns don't predict future returns.

**How to apply:** Log-return forecasts for portfolios should be treated as "if current trend continues" scenarios, not reliable predictions. The uncertainty bands (P10-P90) are more useful than the point forecast. For financial planning, use the P10 (bearish) scenario as the conservative estimate.
