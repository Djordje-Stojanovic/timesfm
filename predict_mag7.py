"""
TimesFM 2.5 — Mag7 + Broadcom Multi-Strategy Prediction

Uses DAILY LOG-RETURNS (not raw prices) as input — critical for stocks with
strong trends. Raw log-prices cause mean-reversion artifacts where the model
predicts a crash to the historical average.

Strategy 1: AVGO standalone (baseline) — log-returns
Strategy 2: All Mag7 + AVGO batch (sector comparison)
Strategy 3: AVGO with Mag7 as XReg covariates (cross-stock signal)
Strategy 4: Multi-horizon forecast (1, 5, 10, 20 trading days)

Downloads 5 years of daily OHLC for: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AVGO
"""

import os
import sys
import time
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import yfinance as yf

import timesfm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET = "AVGO"
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
ALL_TICKERS = MAG7 + [TARGET]
CONTEXT_LEN = 1024
HORIZONS = [1, 5, 10, 20]
MAX_HORIZON = 128  # must be multiple of output_patch_len

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def fmt_bytes(b: int) -> str:
    if b < 1024**2:
        return f"{b / 1024:.0f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "GPU: N/A (CPU mode)"
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return (
        f"VRAM: {fmt_bytes(allocated)} allocated / "
        f"{fmt_bytes(reserved)} reserved / "
        f"{fmt_bytes(total)} total"
    )


def section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def returns_to_price(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Convert predicted log-returns back to price levels."""
    cumulative = np.cumsum(log_returns)
    return last_price * np.exp(cumulative)


def print_prediction(label: str, last: float, pred: float, q_prices: np.ndarray, indent: int = 4) -> None:
    pad = " " * indent
    pct = (pred / last - 1) * 100
    direction = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"
    print(f"{pad}{label}")
    print(f"{pad}  Last:      ${last:>10.2f}")
    print(f"{pad}  Predicted: ${pred:>10.2f}  {direction} {pct:+.2f}%")
    print(f"{pad}  P10:       ${float(q_prices[0]):>10.2f}  (bearish)")
    print(f"{pad}  P50:       ${float(q_prices[1]):>10.2f}  (median)")
    print(f"{pad}  P90:       ${float(q_prices[2]):>10.2f}  (bullish)")
    spread = float(q_prices[2]) - float(q_prices[0])
    print(f"{pad}  80% CI:    ±${spread / 2:.2f}  ({spread / last * 100:.1f}% of price)")


def quantiles_to_prices(last_price: float, q_log_returns: np.ndarray, horizon_idx: int) -> np.ndarray:
    """Convert quantile log-returns at a specific horizon to prices.

    For multi-step horizons, we sum log-returns from step 0 to horizon_idx.
    q_log_returns shape: (horizon, 10)
    Returns: array of [P10_price, P50_price, P90_price]
    """
    if horizon_idx == 0:
        cumulative_q = q_log_returns[0, :]
    else:
        # For multi-step: sum the point forecast returns for steps 0..horizon_idx-1,
        # then add the quantile spread at the final step
        # This is an approximation — proper multi-step quantiles would require simulation
        cumulative_q = np.sum(q_log_returns[:horizon_idx + 1, :], axis=0)

    prices = last_price * np.exp(cumulative_q)
    return np.array([prices[1], prices[5], prices[9]])  # P10, P50, P90


# ===========================================================================
# PHASE 1: Data Download
# ===========================================================================
section("PHASE 1: DATA DOWNLOAD")
t_start = time.perf_counter()
t0 = time.perf_counter()

print(f"\n  Downloading {len(ALL_TICKERS)} tickers: {', '.join(ALL_TICKERS)}")
print(f"  Period: 5 years daily OHLC")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

data = {}
for ticker_name in ALL_TICKERS:
    t_tick = time.perf_counter()
    ticker = yf.Ticker(ticker_name)
    df = ticker.history(period="5y", interval="1d")
    elapsed = time.perf_counter() - t_tick

    data[ticker_name] = df
    n_rows = len(df)
    date_start = df.index[0].strftime("%Y-%m-%d")
    date_end = df.index[-1].strftime("%Y-%m-%d")
    last_close = df["Close"].iloc[-1]
    pct_5y = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    nan_count = df["Close"].isna().sum()

    print(
        f"  {ticker_name:>5}  |  {n_rows:>5} days  |  "
        f"{date_start} -> {date_end}  |  "
        f"${last_close:>8.2f}  |  5y: {pct_5y:>+7.1f}%  |  "
        f"NaN: {nan_count}  |  {fmt_time(elapsed)}"
    )

t_download = time.perf_counter() - t0
print(f"\n  Total download: {fmt_time(t_download)}")

# Data quality summary
print(f"\n  Data Quality:")
min_rows = min(len(df) for df in data.values())
max_rows = max(len(df) for df in data.values())
print(f"    Row range: {min_rows} - {max_rows} trading days")
print(f"    Context window: last {CONTEXT_LEN} days ({CONTEXT_LEN / 252:.1f} years)")
print(f"    Total data points: {sum(len(df) * 4 for df in data.values()):,} (OHLC x tickers)")

# Correlation matrix on close returns
subsection("CLOSE PRICE RETURN CORRELATIONS (90-day)")
returns_90d = {}
for name, df in data.items():
    r = np.diff(np.log(df["Close"].values[-91:]))
    returns_90d[name] = r

# Print correlation header
header = "        " + "  ".join(f"{t:>6}" for t in ALL_TICKERS)
print(header)
for i, t1 in enumerate(ALL_TICKERS):
    row = f"  {t1:>5} "
    for j, t2 in enumerate(ALL_TICKERS):
        corr = np.corrcoef(returns_90d[t1], returns_90d[t2])[0, 1]
        if i == j:
            row += f"  {'1.00':>6}"
        else:
            row += f"  {corr:>6.2f}"
    print(row)

# Highlight AVGO correlations
avgo_corrs = []
for t in MAG7:
    c = np.corrcoef(returns_90d[TARGET], returns_90d[t])[0, 1]
    avgo_corrs.append((t, c))
avgo_corrs.sort(key=lambda x: -x[1])
print(f"\n  {TARGET} strongest correlations:")
for t, c in avgo_corrs:
    bar = "█" * int(abs(c) * 30)
    print(f"    {t:>5}: {c:+.3f}  {bar}")


# ===========================================================================
# PHASE 2: MODEL LOADING
# ===========================================================================
section("PHASE 2: MODEL LOADING")
t0 = time.perf_counter()

print(f"\n  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {gpu_stats()}")

print(f"\n  Loading TimesFM 2.5 200M (PyTorch)...")
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

t_load = time.perf_counter() - t0
print(f"  Model loaded in {fmt_time(t_load)}")
if torch.cuda.is_available():
    print(f"  {gpu_stats()}")

print(f"\n  Compiling with:")
print(f"    max_context={CONTEXT_LEN}, max_horizon={MAX_HORIZON}")
print(f"    normalize_inputs=True, quantile_head=True")
print(f"    flip_invariance=True, fix_quantile_crossing=True")
print(f"    infer_is_positive=False (returns can be negative)")
print(f"    return_backcast=True (for XReg)")

t0 = time.perf_counter()
model.compile(
    timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=MAX_HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
        return_backcast=True,
    )
)
t_compile = time.perf_counter() - t0
print(f"  Compiled in {fmt_time(t_compile)}")
if torch.cuda.is_available():
    print(f"  {gpu_stats()}")


# ===========================================================================
# PHASE 3: PREPARE LOG-RETURN DATA
# ===========================================================================
section("PHASE 3: DATA PREPARATION (LOG-RETURNS)")
print(f"\n  Using daily log-returns: r_t = ln(P_t / P_(t-1))")
print(f"  This removes trend and makes the series stationary.")
print(f"  Prices reconstructed via: P_future = P_last * exp(cumsum(r_predicted))")
print()

log_returns = {}
last_prices = {}
for name, df in data.items():
    prices = df["Close"].values
    lr = np.diff(np.log(prices))  # daily log-returns, length = n-1
    ctx = lr[-CONTEXT_LEN:] if len(lr) > CONTEXT_LEN else lr
    log_returns[name] = ctx
    last_prices[name] = float(prices[-1])

    mean_r = ctx.mean() * 252  # annualized
    std_r = ctx.std() * np.sqrt(252)  # annualized vol
    sharpe = mean_r / std_r if std_r > 0 else 0
    print(
        f"  {name:>5}: {len(ctx)} returns  |  "
        f"Ann. return: {mean_r * 100:>+6.1f}%  |  "
        f"Ann. vol: {std_r * 100:>5.1f}%  |  "
        f"Sharpe: {sharpe:>5.2f}  |  "
        f"Last: ${last_prices[name]:.2f}"
    )


# ===========================================================================
# STRATEGY 1: AVGO Standalone (Baseline)
# ===========================================================================
section("STRATEGY 1: AVGO STANDALONE (LOG-RETURNS)")
print(f"\n  Forecasting {TARGET} daily log-returns from its own history.")
print(f"  Context: {len(log_returns[TARGET])} daily returns")
print(f"  Then reconstructing price from last known close.")

t0 = time.perf_counter()
point_1, quant_1 = model.forecast(horizon=HORIZONS[-1], inputs=[log_returns[TARGET]])
t_forecast_1 = time.perf_counter() - t0

last_avgo = last_prices[TARGET]

for h in HORIZONS:
    # Sum log-returns from day 1 to day h to get cumulative return
    cum_return = float(np.sum(point_1[0, :h]))
    pred_price = last_avgo * np.exp(cum_return)

    # For quantiles at horizon h, sum point returns for days 1..h-1,
    # then add quantile at day h
    if h == 1:
        q_cum = quant_1[0, 0, :]
    else:
        point_sum = float(np.sum(point_1[0, :h - 1]))
        q_cum = point_sum + quant_1[0, h - 1, :]

    q_prices = last_avgo * np.exp(q_cum)
    q_arr = np.array([float(q_prices[1]), float(q_prices[5]), float(q_prices[9])])

    label = f"Day +{h} ({h} trading day{'s' if h > 1 else ''} ahead)"
    print_prediction(label, last_avgo, pred_price, q_arr)

print(f"\n  Inference time: {fmt_time(t_forecast_1)}")
if torch.cuda.is_available():
    print(f"  {gpu_stats()}")


# ===========================================================================
# STRATEGY 2: Full Mag7 + AVGO Batch (Sector View)
# ===========================================================================
section("STRATEGY 2: MAG7 + AVGO BATCH (NEXT-DAY SECTOR VIEW)")
print(f"\n  Forecasting all 8 stocks independently (next day).")
print(f"  Each stock uses its own 1024-day log-return history.")
print(f"  No cross-stock info — purely independent forecasts.")

inputs_batch = [log_returns[t] for t in ALL_TICKERS]

t0 = time.perf_counter()
point_2, quant_2 = model.forecast(horizon=1, inputs=inputs_batch)
t_forecast_2 = time.perf_counter() - t0

print(f"\n  {'Ticker':>6}  {'Last':>10}  {'Predicted':>10}  {'Change':>8}  "
      f"{'P10':>10}  {'P90':>10}  {'80% CI':>10}")
print(f"  {'=' * 6}  {'=' * 10}  {'=' * 10}  {'=' * 8}  "
      f"{'=' * 10}  {'=' * 10}  {'=' * 10}")

for i, name in enumerate(ALL_TICKERS):
    last = last_prices[name]
    pred_return = float(point_2[i, 0])
    pred_price = last * np.exp(pred_return)
    pct = (pred_price / last - 1) * 100

    q_returns = quant_2[i, 0, :]
    p10 = last * np.exp(float(q_returns[1]))
    p90 = last * np.exp(float(q_returns[9]))
    ci_pct = (p90 - p10) / last * 100

    direction = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"
    marker = "  << TARGET" if name == TARGET else ""
    print(
        f"  {name:>6}  ${last:>9.2f}  ${pred_price:>9.2f}  {direction}{pct:>+6.2f}%  "
        f"${p10:>9.2f}  ${p90:>9.2f}  {ci_pct:>8.1f}%{marker}"
    )

# Sector consensus
all_pcts = []
for i, name in enumerate(ALL_TICKERS):
    last = last_prices[name]
    pred_return = float(point_2[i, 0])
    pct = (np.exp(pred_return) - 1) * 100
    all_pcts.append(pct)

bullish = sum(1 for p in all_pcts if p > 0.05)
bearish = sum(1 for p in all_pcts if p < -0.05)
flat = len(all_pcts) - bullish - bearish
avg_pct = np.mean(all_pcts)
print(f"\n  Sector consensus: {bullish} bullish / {flat} flat / {bearish} bearish  |  Avg: {avg_pct:+.2f}%")
print(f"  Batch inference ({len(ALL_TICKERS)} series): {fmt_time(t_forecast_2)}")


# ===========================================================================
# STRATEGY 3: AVGO with Mag7 Covariates (XReg)
# ===========================================================================
section("STRATEGY 3: AVGO WITH MAG7 COVARIATES (XReg)")
print(f"\n  Using Mag7 log-return series as dynamic numerical covariates for {TARGET}.")
print(f"  Mode: 'xreg + timesfm' (XReg adjusts TimesFM baseline residuals)")
print(f"  This tests if correlated Mag7 movements improve AVGO prediction.")

# Step 1: Forecast each Mag7 stock to get future covariate returns
print(f"\n  Step 1: Forecasting Mag7 future returns for covariate horizon...")
t0 = time.perf_counter()
mag7_inputs = [log_returns[t] for t in MAG7]
mag7_point, _ = model.forecast(horizon=HORIZONS[-1], inputs=mag7_inputs)
t_mag7 = time.perf_counter() - t0
print(f"  Mag7 forecasts done in {fmt_time(t_mag7)}")

for j, name in enumerate(MAG7):
    cum_ret = float(np.sum(mag7_point[j, :HORIZONS[-1]])) * 100
    print(f"    {name}: predicted {HORIZONS[-1]}-day cumulative return: {cum_ret:+.2f}%")

# Step 2: Build dynamic covariates
print(f"\n  Step 2: Building covariate matrices...")
horizon = HORIZONS[-1]
ctx_len = len(log_returns[TARGET])

dynamic_numerical_covariates = {}
for j, mag7_name in enumerate(MAG7):
    mag7_ctx = log_returns[mag7_name]
    # Align lengths
    if len(mag7_ctx) > ctx_len:
        mag7_ctx = mag7_ctx[-ctx_len:]
    elif len(mag7_ctx) < ctx_len:
        pad = np.zeros(ctx_len - len(mag7_ctx))
        mag7_ctx = np.concatenate([pad, mag7_ctx])

    # Horizon portion: forecasted returns
    mag7_future = mag7_point[j, :horizon]
    full_cov = np.concatenate([mag7_ctx, mag7_future]).tolist()
    dynamic_numerical_covariates[mag7_name] = [full_cov]

    print(f"    {mag7_name}: {len(mag7_ctx)} ctx + {horizon} horizon = {len(full_cov)} total")

# Step 3: Run forecast_with_covariates
print(f"\n  Step 3: Running XReg forecast...")
t0 = time.perf_counter()

try:
    target_input = log_returns[TARGET].tolist()
    point_3, quant_3 = model.forecast_with_covariates(
        inputs=[target_input],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True,
        ridge=0.1,
        force_on_cpu=True,
    )

    t_xreg = time.perf_counter() - t0
    print(f"  XReg forecast done in {fmt_time(t_xreg)}")

    subsection(f"{TARGET}: Standalone vs XReg (Mag7 Covariates)")

    print(f"\n  {'Horizon':>10}  {'Standalone':>12}  {'Chg':>7}  {'XReg':>12}  {'Chg':>7}  {'Delta':>10}")
    print(f"  {'=' * 10}  {'=' * 12}  {'=' * 7}  {'=' * 12}  {'=' * 7}  {'=' * 10}")

    for h in HORIZONS:
        # Standalone
        cum_s = float(np.sum(point_1[0, :h]))
        price_s = last_avgo * np.exp(cum_s)
        pct_s = (price_s / last_avgo - 1) * 100

        # XReg
        cum_x = float(np.sum(point_3[0][:h]))
        price_x = last_avgo * np.exp(cum_x)
        pct_x = (price_x / last_avgo - 1) * 100

        delta = price_x - price_s
        print(
            f"  Day +{h:>4}  ${price_s:>10.2f}  {pct_s:>+6.2f}%  "
            f"${price_x:>10.2f}  {pct_x:>+6.2f}%  ${delta:>+8.2f}"
        )

    # Detailed day +1
    print()
    cum_x_1 = float(point_3[0][0])
    pred_x_1 = last_avgo * np.exp(cum_x_1)
    q_x_1 = quant_3[0][0, :]
    q_prices_1 = np.array([
        last_avgo * np.exp(float(q_x_1[1])),
        last_avgo * np.exp(float(q_x_1[5])),
        last_avgo * np.exp(float(q_x_1[9])),
    ])
    print_prediction(f"{TARGET} Day +1 (XReg with Mag7)", last_avgo, pred_x_1, q_prices_1)

    xreg_success = True
except Exception as e:
    t_xreg = time.perf_counter() - t0
    print(f"\n  XReg FAILED after {fmt_time(t_xreg)}: {e}")
    print(f"  Falling back to standalone predictions only.")
    xreg_success = False


# ===========================================================================
# STRATEGY 4: Multi-Horizon Forecast (AVGO)
# ===========================================================================
section("STRATEGY 4: MULTI-HORIZON TRAJECTORY")
print(f"\n  {TARGET} price trajectory over next 20 trading days.")
print(f"  Point forecast + P10/P30/P50/P70/P90 uncertainty bands.")
print(f"  Based on standalone log-return forecast (Strategy 1).")

print(f"\n  {'Day':>5}  {'Price':>10}  {'Change':>8}  {'P10':>10}  {'P30':>10}  "
      f"{'P50':>10}  {'P70':>10}  {'P90':>10}  {'80% CI':>8}")
print(f"  {'=' * 5}  {'=' * 10}  {'=' * 8}  {'=' * 10}  {'=' * 10}  "
      f"{'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 8}")

# Reconstruct prices from cumulative log-returns
for h in range(1, HORIZONS[-1] + 1):
    cum_return = float(np.sum(point_1[0, :h]))
    pred_price = last_avgo * np.exp(cum_return)
    pct = (pred_price / last_avgo - 1) * 100

    # Quantile prices at horizon h
    if h == 1:
        q_cum = quant_1[0, 0, :]
    else:
        point_sum = float(np.sum(point_1[0, :h - 1]))
        q_cum = point_sum + quant_1[0, h - 1, :]

    q_prices = last_avgo * np.exp(q_cum)
    p10 = float(q_prices[1])
    p30 = float(q_prices[3])
    p50 = float(q_prices[5])
    p70 = float(q_prices[7])
    p90 = float(q_prices[9])
    ci_width = (p90 - p10) / last_avgo * 100

    direction = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"

    # Print all days for first 5, then every 5th
    if h <= 5 or h in HORIZONS or h % 5 == 0:
        print(
            f"  +{h:>4}  ${pred_price:>9.2f}  {direction}{pct:>+6.2f}%  "
            f"${p10:>9.2f}  ${p30:>9.2f}  ${p50:>9.2f}  ${p70:>9.2f}  ${p90:>9.2f}  "
            f"{ci_width:>6.1f}%"
        )

# Uncertainty growth
print(f"\n  Uncertainty Growth (80% confidence interval width):")
for h in HORIZONS:
    if h == 1:
        q_cum = quant_1[0, 0, :]
    else:
        point_sum = float(np.sum(point_1[0, :h - 1]))
        q_cum = point_sum + quant_1[0, h - 1, :]
    q_prices = last_avgo * np.exp(q_cum)
    width_80 = float(q_prices[9] - q_prices[1])
    width_pct = width_80 / last_avgo * 100
    bars = "█" * max(1, int(width_pct * 3))
    print(f"    Day +{h:>2}: +/-${width_80 / 2:>7.2f} ({width_pct:>5.1f}% of price)  {bars}")


# ===========================================================================
# SUMMARY
# ===========================================================================
section("FINAL SUMMARY")
t_total = time.perf_counter() - t_start

print(f"\n  Target: {TARGET} (Broadcom Inc.)")
print(f"  Last close: ${last_avgo:.2f} ({data[TARGET].index[-1].strftime('%Y-%m-%d')})")
print(f"  Model: TimesFM 2.5 200M (PyTorch)")
print(f"  Input: Daily log-returns, {CONTEXT_LEN}-day context (~{CONTEXT_LEN / 252:.1f} years)")

print(f"\n  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  NEXT-DAY {TARGET} PREDICTIONS                            │")
print(f"  ├─────────────────────┬───────────┬─────────┬─────────────┤")
print(f"  │  Strategy           │   Price   │  Change │  80% CI     │")
print(f"  ├─────────────────────┼───────────┼─────────┼─────────────┤")

# Strategy 1: Standalone
cum_s1 = float(point_1[0, 0])
price_s1 = last_avgo * np.exp(cum_s1)
pct_s1 = (price_s1 / last_avgo - 1) * 100
q_s1 = quant_1[0, 0, :]
p10_s1 = last_avgo * np.exp(float(q_s1[1]))
p90_s1 = last_avgo * np.exp(float(q_s1[9]))
dir_s1 = "▲" if pct_s1 > 0.05 else "▼" if pct_s1 < -0.05 else "─"
print(f"  │  Standalone         │ ${price_s1:>8.2f} │ {dir_s1}{pct_s1:>+5.2f}% │ ${p10_s1:.0f}-${p90_s1:.0f} │")

# Strategy 2: Batch
idx_avgo = ALL_TICKERS.index(TARGET)
cum_b = float(point_2[idx_avgo, 0])
price_b = last_avgo * np.exp(cum_b)
pct_b = (price_b / last_avgo - 1) * 100
q_b = quant_2[idx_avgo, 0, :]
p10_b = last_avgo * np.exp(float(q_b[1]))
p90_b = last_avgo * np.exp(float(q_b[9]))
dir_b = "▲" if pct_b > 0.05 else "▼" if pct_b < -0.05 else "─"
print(f"  │  Batch (indep.)     │ ${price_b:>8.2f} │ {dir_b}{pct_b:>+5.2f}% │ ${p10_b:.0f}-${p90_b:.0f} │")

# Strategy 3: XReg
if xreg_success:
    cum_x = float(point_3[0][0])
    price_x = last_avgo * np.exp(cum_x)
    pct_x = (price_x / last_avgo - 1) * 100
    q_x = quant_3[0][0, :]
    p10_x = last_avgo * np.exp(float(q_x[1]))
    p90_x = last_avgo * np.exp(float(q_x[9]))
    dir_x = "▲" if pct_x > 0.05 else "▼" if pct_x < -0.05 else "─"
    print(f"  │  XReg (Mag7)        │ ${price_x:>8.2f} │ {dir_x}{pct_x:>+5.2f}% │ ${p10_x:.0f}-${p90_x:.0f} │")
else:
    print(f"  │  XReg (Mag7)        │    N/A    │   N/A   │    N/A      │")

print(f"  └─────────────────────┴───────────┴─────────┴─────────────┘")

print(f"\n  Timing Breakdown:")
print(f"    Phase 1 - Data download:      {fmt_time(t_download):>8}")
print(f"    Phase 2 - Model load:         {fmt_time(t_load):>8}")
print(f"    Phase 2 - Model compile:      {fmt_time(t_compile):>8}")
print(f"    Strategy 1 (standalone):      {fmt_time(t_forecast_1):>8}")
print(f"    Strategy 2 (batch x8):        {fmt_time(t_forecast_2):>8}")
try:
    print(f"    Strategy 3 (mag7 forecast):   {fmt_time(t_mag7):>8}")
    print(f"    Strategy 3 (xreg regression): {fmt_time(t_xreg):>8}")
except Exception:
    pass
print(f"    ──────────────────────────────────")
print(f"    Total wall time:              {fmt_time(t_total):>8}")

if torch.cuda.is_available():
    print(f"\n  GPU Performance:")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    {gpu_stats()}")
    print(f"    Peak VRAM: {fmt_bytes(torch.cuda.max_memory_allocated())}")

print(f"""
  METHODOLOGY NOTES:
  - Input: daily log-returns r_t = ln(P_t / P_(t-1)), NOT raw prices
  - Why: raw prices cause mean-reversion artifacts in trending stocks
  - Price reconstruction: P_future = P_last * exp(sum(predicted_returns))
  - XReg uses Mag7 returns as covariates + ridge regression (lambda=0.1)
  - Quantiles are approximate for multi-step horizons (no MC simulation)
  - TimesFM is a general-purpose foundation model, NOT fine-tuned for stocks
  - Treat all predictions as experimental. This is NOT financial advice.
""")
