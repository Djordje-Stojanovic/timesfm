"""
TimesFM 2.5 — Generalized Multi-Strategy Stock Prediction

Usage:
    python predict.py                          # defaults (AVGO + Mag7)
    python predict.py --target AXP --peers SPGI,MCO,FICO,V,MA

Strategies:
    1. Target standalone (baseline from own log-returns)
    2. All peers + target batch (sector comparison)
    3. Target with peers as XReg covariates (cross-stock signal)
    4. Multi-horizon OHLC forecast (1, 5, 10, 20 trading days)
"""

import argparse
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
# CLI Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TimesFM stock prediction")
parser.add_argument("--target", default="AXP", help="Target ticker to predict")
parser.add_argument("--peers", default="SPGI,MCO,FICO,V,MA", help="Comma-separated peer tickers")
parser.add_argument("--context", type=int, default=1024, help="Context length in trading days")
parser.add_argument("--period", default="5y", help="yfinance download period")
args = parser.parse_args()

TARGET = args.target
PEERS = [t.strip() for t in args.peers.split(",")]
ALL_TICKERS = PEERS + [TARGET]
CONTEXT_LEN = args.context
HORIZONS = [1, 5, 10, 20]
MAX_HORIZON = 128
OHLC_COLS = ["Open", "High", "Low", "Close"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_time(s: float) -> str:
    return f"{s * 1000:.0f}ms" if s < 1 else (f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min")

def fmt_bytes(b: int) -> str:
    if b < 1024**2: return f"{b / 1024:.0f} KB"
    if b < 1024**3: return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"

def gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "GPU: N/A (CPU mode)"
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    t = torch.cuda.get_device_properties(0).total_memory
    return f"VRAM: {fmt_bytes(a)} alloc / {fmt_bytes(r)} reserved / {fmt_bytes(t)} total"

def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# ===========================================================================
# PHASE 1: Data Download
# ===========================================================================
section(f"PHASE 1: DATA DOWNLOAD ({len(ALL_TICKERS)} tickers)")
t_start = time.perf_counter()
t0 = time.perf_counter()

print(f"\n  Target: {TARGET}")
print(f"  Peers:  {', '.join(PEERS)}")
print(f"  Period: {args.period} daily OHLC")
print(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

data = {}
for ticker_name in ALL_TICKERS:
    t_tick = time.perf_counter()
    ticker = yf.Ticker(ticker_name)
    df = ticker.history(period=args.period, interval="1d")
    elapsed = time.perf_counter() - t_tick
    data[ticker_name] = df

    n = len(df)
    d0 = df.index[0].strftime("%Y-%m-%d")
    d1 = df.index[-1].strftime("%Y-%m-%d")
    last = df["Close"].iloc[-1]
    pct_5y = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(
        f"  {ticker_name:>5}  |  {n:>5} days  |  {d0} -> {d1}  |  "
        f"${last:>8.2f}  |  ret: {pct_5y:>+7.1f}%  |  {fmt_time(elapsed)}"
    )

t_download = time.perf_counter() - t0
print(f"\n  Download: {fmt_time(t_download)}  |  "
      f"{sum(len(d) * 4 for d in data.values()):,} OHLC data points")

# Correlations
subsection("90-DAY RETURN CORRELATIONS")
returns_90d = {}
for name, df in data.items():
    returns_90d[name] = np.diff(np.log(df["Close"].values[-91:]))

header = "        " + "  ".join(f"{t:>6}" for t in ALL_TICKERS)
print(header)
for t1 in ALL_TICKERS:
    row = f"  {t1:>5} "
    for t2 in ALL_TICKERS:
        c = np.corrcoef(returns_90d[t1], returns_90d[t2])[0, 1]
        row += f"  {c:>6.2f}" if t1 != t2 else f"  {'1.00':>6}"
    print(row)

target_corrs = [(t, np.corrcoef(returns_90d[TARGET], returns_90d[t])[0, 1]) for t in PEERS]
target_corrs.sort(key=lambda x: -x[1])
print(f"\n  {TARGET} strongest correlations:")
for t, c in target_corrs:
    print(f"    {t:>5}: {c:+.3f}  {'█' * int(abs(c) * 30)}")


# ===========================================================================
# PHASE 2: Model Loading
# ===========================================================================
section("PHASE 2: MODEL LOADING")
t0 = time.perf_counter()

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"\n  Device: {device_name}  |  PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {gpu_stats()}")

print(f"  Loading TimesFM 2.5 200M...")
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
t_load = time.perf_counter() - t0
print(f"  Loaded in {fmt_time(t_load)}  |  {gpu_stats()}")

print(f"  Compiling (ctx={CONTEXT_LEN}, horizon={MAX_HORIZON}, returns_mode, xreg_ready)...")
t0 = time.perf_counter()
model.compile(timesfm.ForecastConfig(
    max_context=CONTEXT_LEN, max_horizon=MAX_HORIZON,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=False,
    fix_quantile_crossing=True, return_backcast=True,
))
t_compile = time.perf_counter() - t0
print(f"  Compiled in {fmt_time(t_compile)}")


# ===========================================================================
# PHASE 3: Prepare Log-Return Data
# ===========================================================================
section("PHASE 3: DATA PREPARATION (LOG-RETURNS)")
print(f"\n  r_t = ln(P_t / P_(t-1))  |  Removes trend, makes series stationary")
print()

log_returns = {}  # ticker -> {col -> array}
last_prices = {}  # ticker -> {col -> float}

for name, df in data.items():
    log_returns[name] = {}
    last_prices[name] = {}
    parts = []
    for col in OHLC_COLS:
        prices = df[col].values
        lr = np.diff(np.log(prices))
        ctx = lr[-CONTEXT_LEN:] if len(lr) > CONTEXT_LEN else lr
        log_returns[name][col] = ctx
        last_prices[name][col] = float(prices[-1])
        ann_ret = ctx.mean() * 252
        ann_vol = ctx.std() * np.sqrt(252)
        parts.append(f"{col[0]}:{ann_ret * 100:+.0f}%/{ann_vol * 100:.0f}%v")
    print(f"  {name:>5}: {len(ctx)} returns  |  {' '.join(parts)}  |  Close=${last_prices[name]['Close']:.2f}")


# ===========================================================================
# STRATEGY 1: Target Standalone OHLC
# ===========================================================================
section(f"STRATEGY 1: {TARGET} STANDALONE — ALL OHLC")
print(f"\n  Forecasting {TARGET} Open/High/Low/Close from its own history.")

ohlc_inputs = [log_returns[TARGET][col] for col in OHLC_COLS]

t0 = time.perf_counter()
point_ohlc, quant_ohlc = model.forecast(horizon=HORIZONS[-1], inputs=ohlc_inputs)
t_s1 = time.perf_counter() - t0

for h in HORIZONS:
    subsection(f"Day +{h}")
    print(f"  {'':>6}  {'Last':>10}  {'Predicted':>10}  {'Change':>8}  {'P10':>10}  {'P50':>10}  {'P90':>10}  {'80% CI':>8}")
    print(f"  {'':>6}  {'=' * 10}  {'=' * 10}  {'=' * 8}  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 8}")
    for i, col in enumerate(OHLC_COLS):
        last = last_prices[TARGET][col]
        cum = float(np.sum(point_ohlc[i, :h]))
        pred = last * np.exp(cum)
        pct = (pred / last - 1) * 100
        # Quantiles
        if h == 1:
            q_cum = quant_ohlc[i, 0, :]
        else:
            ps = float(np.sum(point_ohlc[i, :h - 1]))
            q_cum = ps + quant_ohlc[i, h - 1, :]
        qp = last * np.exp(q_cum)
        p10, p50, p90 = float(qp[1]), float(qp[5]), float(qp[9])
        ci = (p90 - p10) / last * 100
        d = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"
        print(f"  {col:>6}  ${last:>9.2f}  ${pred:>9.2f}  {d}{pct:>+6.2f}%  ${p10:>9.2f}  ${p50:>9.2f}  ${p90:>9.2f}  {ci:>6.1f}%")

print(f"\n  Inference: {fmt_time(t_s1)}  |  {gpu_stats()}")


# ===========================================================================
# STRATEGY 2: Sector Batch (Close only, next day)
# ===========================================================================
section(f"STRATEGY 2: SECTOR BATCH — NEXT-DAY CLOSE")
print(f"\n  All {len(ALL_TICKERS)} tickers independently, next-day close.")

batch_inputs = [log_returns[t]["Close"] for t in ALL_TICKERS]
t0 = time.perf_counter()
point_batch, quant_batch = model.forecast(horizon=1, inputs=batch_inputs)
t_s2 = time.perf_counter() - t0

print(f"\n  {'Ticker':>6}  {'Last':>10}  {'Predicted':>10}  {'Change':>8}  {'P10':>10}  {'P90':>10}  {'80% CI':>8}")
print(f"  {'=' * 6}  {'=' * 10}  {'=' * 10}  {'=' * 8}  {'=' * 10}  {'=' * 10}  {'=' * 8}")

all_pcts = []
for i, name in enumerate(ALL_TICKERS):
    last = last_prices[name]["Close"]
    pred = last * np.exp(float(point_batch[i, 0]))
    pct = (pred / last - 1) * 100
    all_pcts.append(pct)
    q = quant_batch[i, 0, :]
    p10 = last * np.exp(float(q[1]))
    p90 = last * np.exp(float(q[9]))
    ci = (p90 - p10) / last * 100
    d = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"
    tag = "  << TARGET" if name == TARGET else ""
    print(f"  {name:>6}  ${last:>9.2f}  ${pred:>9.2f}  {d}{pct:>+6.2f}%  ${p10:>9.2f}  ${p90:>9.2f}  {ci:>6.1f}%{tag}")

bull = sum(1 for p in all_pcts if p > 0.05)
bear = sum(1 for p in all_pcts if p < -0.05)
print(f"\n  Sector: {bull} bullish / {len(all_pcts) - bull - bear} flat / {bear} bearish  |  Avg: {np.mean(all_pcts):+.2f}%")
print(f"  Inference ({len(ALL_TICKERS)} series): {fmt_time(t_s2)}")


# ===========================================================================
# STRATEGY 3: XReg — Target OHLC with Peer Close as Covariates
# ===========================================================================
section(f"STRATEGY 3: {TARGET} OHLC WITH PEER COVARIATES (XReg)")
print(f"\n  Using {len(PEERS)} peer close-return series as covariates for each OHLC column.")
print(f"  Mode: 'xreg + timesfm'  |  Ridge: 0.1  |  force_on_cpu: True")

# Forecast peer close returns for horizon
print(f"\n  Step 1: Forecasting peer returns for covariate horizon...")
t0 = time.perf_counter()
peer_inputs = [log_returns[t]["Close"] for t in PEERS]
peer_point, _ = model.forecast(horizon=HORIZONS[-1], inputs=peer_inputs)
t_peer = time.perf_counter() - t0
print(f"  Peer forecasts: {fmt_time(t_peer)}")

for j, name in enumerate(PEERS):
    cr = float(np.sum(peer_point[j, :HORIZONS[-1]])) * 100
    print(f"    {name}: {HORIZONS[-1]}-day cum. return: {cr:+.2f}%")

# Build covariates and run XReg for each OHLC column
print(f"\n  Step 2: Running XReg for each OHLC column...")
xreg_results = {}  # col -> (point, quant)
t0_xreg_total = time.perf_counter()

for col in OHLC_COLS:
    ctx_len = len(log_returns[TARGET][col])
    horizon = HORIZONS[-1]

    dyn_covs = {}
    for j, peer in enumerate(PEERS):
        peer_ctx = log_returns[peer]["Close"]
        if len(peer_ctx) > ctx_len:
            peer_ctx = peer_ctx[-ctx_len:]
        elif len(peer_ctx) < ctx_len:
            peer_ctx = np.concatenate([np.zeros(ctx_len - len(peer_ctx)), peer_ctx])
        peer_future = peer_point[j, :horizon]
        dyn_covs[peer] = [np.concatenate([peer_ctx, peer_future]).tolist()]

    try:
        target_input = log_returns[TARGET][col].tolist()
        pt, qt = model.forecast_with_covariates(
            inputs=[target_input],
            dynamic_numerical_covariates=dyn_covs,
            xreg_mode="xreg + timesfm",
            normalize_xreg_target_per_input=True,
            ridge=0.1,
            force_on_cpu=True,
        )
        xreg_results[col] = (pt, qt)
        print(f"    {col}: OK")
    except Exception as e:
        print(f"    {col}: FAILED ({e})")

t_s3 = time.perf_counter() - t0_xreg_total

if xreg_results:
    for h in HORIZONS:
        subsection(f"Day +{h} — Standalone vs XReg")
        print(f"  {'':>6}  {'Last':>10}  {'Standalone':>10}  {'Chg':>7}  {'XReg':>10}  {'Chg':>7}  {'XReg P10':>10}  {'XReg P90':>10}  {'Delta':>8}")
        print(f"  {'':>6}  {'=' * 10}  {'=' * 10}  {'=' * 7}  {'=' * 10}  {'=' * 7}  {'=' * 10}  {'=' * 10}  {'=' * 8}")
        for i, col in enumerate(OHLC_COLS):
            last = last_prices[TARGET][col]
            # Standalone
            cum_s = float(np.sum(point_ohlc[i, :h]))
            price_s = last * np.exp(cum_s)
            pct_s = (price_s / last - 1) * 100

            if col in xreg_results:
                pt_x, qt_x = xreg_results[col]
                cum_x = float(np.sum(pt_x[0][:h]))
                price_x = last * np.exp(cum_x)
                pct_x = (price_x / last - 1) * 100
                delta = price_x - price_s

                if h == 1:
                    q_cum = qt_x[0][0, :]
                else:
                    ps = float(np.sum(pt_x[0][:h - 1]))
                    q_cum = ps + qt_x[0][h - 1, :]
                qp = last * np.exp(q_cum)
                p10_x, p90_x = float(qp[1]), float(qp[9])

                d_s = "▲" if pct_s > 0.05 else "▼" if pct_s < -0.05 else "─"
                d_x = "▲" if pct_x > 0.05 else "▼" if pct_x < -0.05 else "─"
                print(
                    f"  {col:>6}  ${last:>9.2f}  ${price_s:>9.2f}  {d_s}{pct_s:>+5.2f}%  "
                    f"${price_x:>9.2f}  {d_x}{pct_x:>+5.2f}%  "
                    f"${p10_x:>9.2f}  ${p90_x:>9.2f}  ${delta:>+7.2f}"
                )
            else:
                print(f"  {col:>6}  ${last:>9.2f}  ${price_s:>9.2f}  {pct_s:>+5.2f}%  {'N/A':>10}")

print(f"\n  XReg total: {fmt_time(t_s3)} (peers: {fmt_time(t_peer)}, regression: {fmt_time(t_s3 - t_peer + 0.001)})  |  {gpu_stats()}")


# ===========================================================================
# STRATEGY 4: Multi-Horizon Trajectory (Close)
# ===========================================================================
section(f"STRATEGY 4: {TARGET} CLOSE — 20-DAY TRAJECTORY")
print(f"\n  Standalone close price trajectory with uncertainty bands.")

close_idx = OHLC_COLS.index("Close")
last_c = last_prices[TARGET]["Close"]

print(f"\n  {'Day':>5}  {'Price':>10}  {'Change':>8}  {'P10':>10}  {'P30':>10}  {'P50':>10}  {'P70':>10}  {'P90':>10}  {'80% CI':>8}")
print(f"  {'=' * 5}  {'=' * 10}  {'=' * 8}  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 8}")

for h in range(1, HORIZONS[-1] + 1):
    cum = float(np.sum(point_ohlc[close_idx, :h]))
    pred = last_c * np.exp(cum)
    pct = (pred / last_c - 1) * 100
    if h == 1:
        q_cum = quant_ohlc[close_idx, 0, :]
    else:
        ps = float(np.sum(point_ohlc[close_idx, :h - 1]))
        q_cum = ps + quant_ohlc[close_idx, h - 1, :]
    qp = last_c * np.exp(q_cum)
    d = "▲" if pct > 0.05 else "▼" if pct < -0.05 else "─"
    ci = (float(qp[9]) - float(qp[1])) / last_c * 100

    if h <= 5 or h in HORIZONS or h % 5 == 0:
        print(
            f"  +{h:>4}  ${pred:>9.2f}  {d}{pct:>+6.2f}%  "
            f"${float(qp[1]):>9.2f}  ${float(qp[3]):>9.2f}  ${float(qp[5]):>9.2f}  "
            f"${float(qp[7]):>9.2f}  ${float(qp[9]):>9.2f}  {ci:>6.1f}%"
        )


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
section("FINAL SUMMARY")
t_total = time.perf_counter() - t_start

print(f"\n  Target: {TARGET}  |  Peers: {', '.join(PEERS)}")
print(f"  Last date: {data[TARGET].index[-1].strftime('%Y-%m-%d')}")
print(f"  Model: TimesFM 2.5 200M (PyTorch)  |  Context: {CONTEXT_LEN} days")

# Summary table
print(f"\n  ┌─────────────────────────────────────────────────────────────────────────┐")
print(f"  │  NEXT-DAY {TARGET} PREDICTION — OHLC                                      │")
print(f"  ├────────┬────────────┬─────────────────────────┬─────────────────────────┤")
print(f"  │        │    Last    │      Standalone          │      XReg (peers)       │")
print(f"  │  Col   │   Price    │   Price    Chg    80%CI  │   Price    Chg    80%CI  │")
print(f"  ├────────┼────────────┼─────────────────────────┼─────────────────────────┤")

for i, col in enumerate(OHLC_COLS):
    last = last_prices[TARGET][col]

    # Standalone
    cum_s = float(point_ohlc[i, 0])
    price_s = last * np.exp(cum_s)
    pct_s = (price_s / last - 1) * 100
    qs = quant_ohlc[i, 0, :]
    ci_s = (last * np.exp(float(qs[9])) - last * np.exp(float(qs[1]))) / last * 100
    ds = "▲" if pct_s > 0.05 else "▼" if pct_s < -0.05 else "─"

    # XReg
    if col in xreg_results:
        pt_x, qt_x = xreg_results[col]
        cum_x = float(pt_x[0][0])
        price_x = last * np.exp(cum_x)
        pct_x = (price_x / last - 1) * 100
        qx = qt_x[0][0, :]
        ci_x = (last * np.exp(float(qx[9])) - last * np.exp(float(qx[1]))) / last * 100
        dx = "▲" if pct_x > 0.05 else "▼" if pct_x < -0.05 else "─"
        xreg_str = f"${price_x:>8.2f}  {dx}{pct_x:>+5.2f}%  {ci_x:>4.1f}%"
    else:
        xreg_str = f"{'N/A':>25}"

    print(f"  │ {col:>6} │ ${last:>9.2f} │ ${price_s:>8.2f}  {ds}{pct_s:>+5.2f}%  {ci_s:>4.1f}% │ {xreg_str} │")

print(f"  └────────┴────────────┴─────────────────────────┴─────────────────────────┘")

print(f"\n  Timing:")
print(f"    Data download:     {fmt_time(t_download):>8}  |  Strategy 1 (OHLC):     {fmt_time(t_s1):>8}")
print(f"    Model load:        {fmt_time(t_load):>8}  |  Strategy 2 (batch):    {fmt_time(t_s2):>8}")
print(f"    Model compile:     {fmt_time(t_compile):>8}  |  Strategy 3 (XReg):     {fmt_time(t_s3):>8}")
print(f"    ─────────────────────────────────────────────────────────────")
print(f"    Total:             {fmt_time(t_total):>8}")

if torch.cuda.is_available():
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}  |  {gpu_stats()}  |  Peak: {fmt_bytes(torch.cuda.max_memory_allocated())}")

print(f"\n  NOTE: TimesFM is not fine-tuned for stocks. This is NOT financial advice.\n")
