"""
TimesFM 2.5 — NVIDIA Next-Day Close Price Prediction

Downloads 5 years of NVDA daily OHLC data, log-transforms close prices,
and uses TimesFM 2.5 (200M) to forecast the next trading day's close.
"""

import numpy as np
import torch
import yfinance as yf

import timesfm

CONTEXT_LEN = 1024

# ---------------------------------------------------------------------------
# 1. Download NVIDIA historical data (5 years, daily)
# ---------------------------------------------------------------------------
print("Downloading NVDA data (5 years daily)...")
ticker = yf.Ticker("NVDA")
df = ticker.history(period="5y", interval="1d")
print(f"Downloaded {len(df)} trading days: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Last close: ${df['Close'].iloc[-1]:.2f}")
print()

# ---------------------------------------------------------------------------
# 2. Prepare data: log-transform close prices
# ---------------------------------------------------------------------------
close_prices = df["Close"].values
log_close = np.log(close_prices)

context = log_close[-CONTEXT_LEN:] if len(log_close) > CONTEXT_LEN else log_close
print(f"Using {len(context)} days as context (log-transformed)")

# ---------------------------------------------------------------------------
# 3. Load TimesFM 2.5 200M (PyTorch)
# ---------------------------------------------------------------------------
print("\nLoading TimesFM 2.5 200M...")
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=128,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
)
print("Model loaded and compiled.")

# ---------------------------------------------------------------------------
# 4. Forecast next day close — Close only
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("CLOSE PRICE ONLY")
print("=" * 50)
point_forecast, quantile_forecast = model.forecast(
    horizon=1,
    inputs=[context],
)

pred_log = point_forecast[0, 0]
pred_price = float(np.exp(pred_log))
last_price = float(close_prices[-1])

print(f"Last close:       ${last_price:.2f}")
print(f"Predicted close:  ${pred_price:.2f} ({(pred_price / last_price - 1) * 100:+.2f}%)")

q_log = quantile_forecast[0, 0, :]
q_prices = np.exp(q_log)
print(f"P10 (bearish):    ${float(q_prices[1]):.2f}")
print(f"P50 (median):     ${float(q_prices[5]):.2f}")
print(f"P90 (bullish):    ${float(q_prices[9]):.2f}")
print(f"Range P10-P90:    ${float(q_prices[1]):.2f} — ${float(q_prices[9]):.2f}")

# ---------------------------------------------------------------------------
# 5. Bonus: OHLC as 4 separate series
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("OHLC AS 4 SEPARATE SERIES")
print("=" * 50)
ohlc_cols = ["Open", "High", "Low", "Close"]
ohlc_inputs = []
for col in ohlc_cols:
    series = np.log(df[col].values)
    ohlc_inputs.append(series[-CONTEXT_LEN:] if len(series) > CONTEXT_LEN else series)

point_ohlc, quantile_ohlc = model.forecast(
    horizon=1,
    inputs=ohlc_inputs,
)

for i, col in enumerate(ohlc_cols):
    pred = float(np.exp(point_ohlc[i, 0]))
    last = float(df[col].iloc[-1])
    q = np.exp(quantile_ohlc[i, 0, :])
    print(
        f"{col:>5}: last=${last:.2f}  pred=${pred:.2f} ({(pred / last - 1) * 100:+.2f}%)  "
        f"P10=${float(q[1]):.2f}  P90=${float(q[9]):.2f}"
    )

print("\nDone. Note: TimesFM is not fine-tuned for stocks — treat as a quick feasibility test.")
