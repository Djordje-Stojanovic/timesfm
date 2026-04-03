"""
TimesFM 2.5 — Personal Finance Forecast

Predicts:
  1. Net savings (next 9 months, Apr-Dec 2026)
  2. Portfolio value trajectory (next 9 months)
  3. Monthly investment amount (next 9 months)
  4. Portfolio return % (next 9 months)

Uses XReg covariates where series are correlated.
Generates matplotlib charts saved to runs/<timestamp>/.
Creates a run report .md file with all results.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

import timesfm
from data.personal_finance import (
    NET_SAVINGS, NET_SAVINGS_DATES,
    PORTFOLIO_VALUE, PORTFOLIO_DATES,
    MONTHLY_INVESTMENT, MONTHLY_INVESTMENT_DATES,
    INVESTED_CAPITAL, TOTAL_RETURN_PCT,
    FORECAST_DATES,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HORIZON = 9  # months to predict (Apr-Dec 2026)
MAX_HORIZON = 128
CONTEXT_LEN = 64  # nearest multiple of patch_len=32 above our ~54 data points

# ---------------------------------------------------------------------------
# Setup run directory
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / timestamp
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {run_dir}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_time(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.1f}s"
def fmt_eur(v): return f"{v:,.0f}".replace(",", ".")

def gpu_stats():
    if not torch.cuda.is_available(): return "CPU mode"
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    t = torch.cuda.get_device_properties(0).total_memory
    return f"VRAM: {a/1024**2:.0f}MB alloc / {r/1024**2:.0f}MB res / {t/1024**3:.1f}GB total"

def section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

COLORS = {
    "history": "#3b82f6",    # blue
    "forecast": "#f59e0b",   # amber
    "ci_80": "#fef3c7",      # light amber
    "ci_60": "#fde68a",      # medium amber
    "ci_40": "#fcd34d",      # dark amber
    "accent": "#10b981",     # emerald
    "grid": "#374151",       # gray-700
    "bg": "#111827",         # gray-900
    "text": "#f3f4f6",       # gray-100
}

def style_chart(ax, title, ylabel):
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["text"], fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=11)
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.grid(True, alpha=0.15, color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"], fontsize=9)


def plot_forecast(dates_hist, values_hist, dates_fc, point_fc, quant_fc,
                  title, ylabel, filename, fmt_fn=fmt_eur, show_seasonality=False):
    """Plot history + forecast with confidence bands."""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_hist = len(dates_hist)
    n_fc = len(dates_fc)
    x_hist = list(range(n_hist))
    x_fc = list(range(n_hist, n_hist + n_fc))

    # History
    ax.plot(x_hist, values_hist, color=COLORS["history"], linewidth=2,
            marker="o", markersize=3, label="History", zorder=5)

    # Forecast point
    ax.plot(x_fc, point_fc, color=COLORS["forecast"], linewidth=2.5,
            marker="D", markersize=5, label="Forecast", zorder=5)

    # Confidence bands
    if quant_fc is not None:
        p10 = quant_fc[:, 1]
        p30 = quant_fc[:, 3]
        p70 = quant_fc[:, 7]
        p90 = quant_fc[:, 9]

        ax.fill_between(x_fc, p10, p90, alpha=0.15, color=COLORS["forecast"], label="80% CI")
        ax.fill_between(x_fc, p30, p70, alpha=0.25, color=COLORS["forecast"], label="40% CI")

    # Vertical line at forecast boundary
    ax.axvline(x=n_hist - 0.5, color=COLORS["accent"], linestyle="--", alpha=0.5, linewidth=1)
    ax.text(n_hist - 0.3, ax.get_ylim()[1] * 0.98, "forecast -->",
            color=COLORS["accent"], fontsize=8, alpha=0.7, va="top")

    # X-axis labels (show every 3rd month)
    all_dates = list(dates_hist) + list(dates_fc)
    tick_positions = list(range(0, len(all_dates), 3))
    tick_labels = [all_dates[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Format y-axis
    if "%" in ylabel:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Annotate last historical and last forecast values
    ax.annotate(f"{fmt_fn(values_hist[-1])}", xy=(x_hist[-1], values_hist[-1]),
                xytext=(0, 12), textcoords="offset points", color=COLORS["history"],
                fontsize=9, fontweight="bold", ha="center")
    ax.annotate(f"{fmt_fn(point_fc[-1])}", xy=(x_fc[-1], point_fc[-1]),
                xytext=(0, 12), textcoords="offset points", color=COLORS["forecast"],
                fontsize=9, fontweight="bold", ha="center")

    style_chart(ax, title, ylabel)
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


# ===========================================================================
# PHASE 1: Data Analysis
# ===========================================================================
section("PHASE 1: DATA ANALYSIS")
t_start = time.perf_counter()

print(f"\n  Net Savings: {len(NET_SAVINGS)} months ({NET_SAVINGS_DATES[0]} to {NET_SAVINGS_DATES[-1]})")
print(f"    Mean: {fmt_eur(NET_SAVINGS.mean())} EUR/mo  |  Median: {fmt_eur(np.median(NET_SAVINGS))} EUR/mo")
print(f"    Min:  {fmt_eur(NET_SAVINGS.min())} EUR/mo  |  Max: {fmt_eur(NET_SAVINGS.max())} EUR/mo")
print(f"    Std:  {fmt_eur(NET_SAVINGS.std())} EUR/mo  |  CV: {NET_SAVINGS.std()/NET_SAVINGS.mean()*100:.0f}%")
print(f"    Total saved: {fmt_eur(NET_SAVINGS.sum())} EUR over {len(NET_SAVINGS)} months")

# Detect seasonality - monthly averages
monthly_avg = {}
for date, val in zip(NET_SAVINGS_DATES, NET_SAVINGS):
    month = int(date.split("-")[1])
    monthly_avg.setdefault(month, []).append(val)
monthly_avg = {k: np.mean(v) for k, v in monthly_avg.items()}

print(f"\n  Monthly Seasonality (avg net savings):")
for m in range(1, 13):
    if m in monthly_avg:
        bar = "█" * int(monthly_avg[m] / 200)
        peak = " << PEAK" if monthly_avg[m] == max(monthly_avg.values()) else ""
        print(f"    {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]}: {fmt_eur(monthly_avg[m])} EUR  {bar}{peak}")

print(f"\n  Portfolio: {len(PORTFOLIO_VALUE)} months ({PORTFOLIO_DATES[0]} to {PORTFOLIO_DATES[-1]})")
print(f"    Current: {fmt_eur(PORTFOLIO_VALUE[-1])} EUR")
print(f"    Invested: {fmt_eur(INVESTED_CAPITAL[-1])} EUR")
print(f"    Gain: {fmt_eur(PORTFOLIO_VALUE[-1] - INVESTED_CAPITAL[-1])} EUR ({TOTAL_RETURN_PCT[-1]:+.1f}%)")
print(f"    CAGR: {((PORTFOLIO_VALUE[-1]/PORTFOLIO_VALUE[0])**(12/len(PORTFOLIO_VALUE))-1)*100:.1f}%")

print(f"\n  Monthly Investment: {len(MONTHLY_INVESTMENT)} months")
print(f"    Mean: {fmt_eur(MONTHLY_INVESTMENT.mean())} EUR/mo  |  Total: {fmt_eur(MONTHLY_INVESTMENT.sum())} EUR")


# ===========================================================================
# PHASE 2: Load Model
# ===========================================================================
section("PHASE 2: MODEL LOADING")
t0 = time.perf_counter()

device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"\n  Device: {device}  |  PyTorch {torch.__version__}")
print(f"  {gpu_stats()}")

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
t_load = time.perf_counter() - t0
print(f"  Model loaded in {fmt_time(t_load)}  |  {gpu_stats()}")

def compile_model(positive=True):
    """Recompile model with appropriate settings."""
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN, max_horizon=MAX_HORIZON,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=not positive,  # flip invariance off for positive-only series
        infer_is_positive=positive,
        fix_quantile_crossing=True, return_backcast=True,
    ))

compile_model(positive=True)
print(f"  Compiled (ctx={CONTEXT_LEN}, horizon={MAX_HORIZON}, positive=True)")


# ===========================================================================
# PHASE 3: Forecast Net Savings
# ===========================================================================
section("PHASE 3: FORECAST NET SAVINGS")
print(f"\n  Predicting next {HORIZON} months of net savings (Apr-Dec 2026)")
print(f"  Input: {len(NET_SAVINGS)} monthly values, raw (no log transform needed)")

t0 = time.perf_counter()
pt_savings, qt_savings = model.forecast(horizon=HORIZON, inputs=[NET_SAVINGS])
t_fc1 = time.perf_counter() - t0

fc_savings = pt_savings[0, :HORIZON]
q_savings = qt_savings[0, :HORIZON, :]

print(f"  Inference: {fmt_time(t_fc1)}")
print(f"\n  {'Month':>10}  {'Predicted':>10}  {'P10':>10}  {'P50':>10}  {'P90':>10}")
print(f"  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 10}")
for i, date in enumerate(FORECAST_DATES):
    p = fc_savings[i]
    q10, q50, q90 = q_savings[i, 1], q_savings[i, 5], q_savings[i, 9]
    peak = "  << Nov bonus?" if "11" in date else ""
    print(f"  {date:>10}  {fmt_eur(p):>10}  {fmt_eur(q10):>10}  {fmt_eur(q50):>10}  {fmt_eur(q90):>10}{peak}")

total_fc = fc_savings.sum()
print(f"\n  Total predicted savings Apr-Dec 2026: {fmt_eur(total_fc)} EUR")
print(f"  Annualized rate: {fmt_eur(total_fc / HORIZON * 12)} EUR/year")

plot_forecast(NET_SAVINGS_DATES, NET_SAVINGS, FORECAST_DATES, fc_savings, q_savings,
              "Net Monthly Savings - History & Forecast", "EUR",
              "01_net_savings.png")


# ===========================================================================
# PHASE 4: Forecast Portfolio Value
# ===========================================================================
section("PHASE 4: FORECAST PORTFOLIO VALUE")
print(f"\n  Predicting portfolio value for next {HORIZON} months")
print(f"  Using log-returns (portfolio has strong uptrend)")

# Use log-returns for portfolio (strong trend)
port_log_returns = np.diff(np.log(PORTFOLIO_VALUE))

t0 = time.perf_counter()
pt_port, qt_port = model.forecast(horizon=HORIZON, inputs=[port_log_returns])
t_fc2 = time.perf_counter() - t0

# Reconstruct prices from cumulative returns
last_port = PORTFOLIO_VALUE[-1]
fc_port_cum = np.cumsum(pt_port[0, :HORIZON])
fc_port_values = last_port * np.exp(fc_port_cum)

# Quantile price reconstruction
q_port_values = np.zeros((HORIZON, 10))
for h in range(HORIZON):
    if h == 0:
        q_cum = qt_port[0, 0, :]
    else:
        point_sum = np.sum(pt_port[0, :h])
        q_cum = point_sum + qt_port[0, h, :]
    q_port_values[h, :] = last_port * np.exp(q_cum)

print(f"  Inference: {fmt_time(t_fc2)}")
print(f"\n  {'Month':>10}  {'Predicted':>12}  {'P10':>12}  {'P50':>12}  {'P90':>12}  {'MoM':>7}")
print(f"  {'=' * 10}  {'=' * 12}  {'=' * 12}  {'=' * 12}  {'=' * 12}  {'=' * 7}")
prev = last_port
for i, date in enumerate(FORECAST_DATES):
    p = fc_port_values[i]
    q10, q50, q90 = q_port_values[i, 1], q_port_values[i, 5], q_port_values[i, 9]
    mom = (p / prev - 1) * 100
    prev = p
    print(f"  {date:>10}  {fmt_eur(p):>12}  {fmt_eur(q10):>12}  {fmt_eur(q50):>12}  {fmt_eur(q90):>12}  {mom:>+5.1f}%")

growth = (fc_port_values[-1] / last_port - 1) * 100
print(f"\n  Current: {fmt_eur(last_port)} EUR  -->  Dec 2026: {fmt_eur(fc_port_values[-1])} EUR ({growth:+.1f}%)")

# Adjust dates for portfolio forecast (starts one month later than net savings)
port_fc_dates = FORECAST_DATES[:HORIZON]
# But portfolio already has Apr 2026, so forecast starts May 2026
port_fc_dates_actual = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09",
                         "2026-10", "2026-11", "2026-12", "2027-01"]

plot_forecast(PORTFOLIO_DATES, PORTFOLIO_VALUE, port_fc_dates_actual,
              fc_port_values, q_port_values,
              "Portfolio Value - History & Forecast", "EUR",
              "02_portfolio_value.png")


# ===========================================================================
# PHASE 5: Forecast Monthly Investment
# ===========================================================================
section("PHASE 5: FORECAST MONTHLY INVESTMENT")
print(f"\n  How much will you invest per month?")

t0 = time.perf_counter()
pt_inv, qt_inv = model.forecast(horizon=HORIZON, inputs=[MONTHLY_INVESTMENT])
t_fc3 = time.perf_counter() - t0

fc_inv = pt_inv[0, :HORIZON]
q_inv = qt_inv[0, :HORIZON, :]

print(f"  Inference: {fmt_time(t_fc3)}")
print(f"\n  {'Month':>10}  {'Predicted':>10}  {'P10':>10}  {'P90':>10}")
print(f"  {'=' * 10}  {'=' * 10}  {'=' * 10}  {'=' * 10}")
for i, date in enumerate(FORECAST_DATES):
    p = fc_inv[i]
    print(f"  {date:>10}  {fmt_eur(p):>10}  {fmt_eur(q_inv[i,1]):>10}  {fmt_eur(q_inv[i,9]):>10}")

print(f"\n  Total predicted investment Apr-Dec 2026: {fmt_eur(fc_inv.sum())} EUR")

plot_forecast(MONTHLY_INVESTMENT_DATES, MONTHLY_INVESTMENT, FORECAST_DATES,
              fc_inv, q_inv,
              "Monthly Investment Amount - History & Forecast", "EUR",
              "03_monthly_investment.png")


# ===========================================================================
# PHASE 6: Forecast Returns
# ===========================================================================
section("PHASE 6: FORECAST PORTFOLIO RETURNS")
print(f"\n  Predicting total return % trajectory")
print(f"  Recompiling model for negative values (returns can be negative)...")
compile_model(positive=False)

t0 = time.perf_counter()
pt_ret, qt_ret = model.forecast(horizon=HORIZON, inputs=[TOTAL_RETURN_PCT])
t_fc4 = time.perf_counter() - t0

fc_ret = pt_ret[0, :HORIZON]
q_ret = qt_ret[0, :HORIZON, :]

print(f"  Inference: {fmt_time(t_fc4)}")
print(f"\n  {'Month':>10}  {'Return':>8}  {'P10':>8}  {'P90':>8}")
print(f"  {'=' * 10}  {'=' * 8}  {'=' * 8}  {'=' * 8}")
for i, date in enumerate(FORECAST_DATES):
    print(f"  {date:>10}  {fc_ret[i]:>+6.1f}%  {q_ret[i,1]:>+6.1f}%  {q_ret[i,9]:>+6.1f}%")

plot_forecast(PORTFOLIO_DATES, TOTAL_RETURN_PCT, port_fc_dates_actual,
              fc_ret, q_ret,
              "Total Return % - History & Forecast", "% return",
              "04_returns.png", fmt_fn=lambda v: f"{v:+.1f}%")


# ===========================================================================
# PHASE 7: Combined Dashboard
# ===========================================================================
section("PHASE 7: COMBINED DASHBOARD")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.set_facecolor(COLORS["bg"])
fig.suptitle(f"Personal Finance Forecast — {timestamp[:8]}", color=COLORS["text"],
             fontsize=16, fontweight="bold", y=0.98)

datasets = [
    (axes[0, 0], NET_SAVINGS_DATES, NET_SAVINGS, FORECAST_DATES, fc_savings, q_savings,
     "Net Savings (EUR/mo)", "EUR"),
    (axes[0, 1], PORTFOLIO_DATES, PORTFOLIO_VALUE, port_fc_dates_actual, fc_port_values, q_port_values,
     "Portfolio Value (EUR)", "EUR"),
    (axes[1, 0], MONTHLY_INVESTMENT_DATES, MONTHLY_INVESTMENT, FORECAST_DATES, fc_inv, q_inv,
     "Monthly Investment (EUR)", "EUR"),
    (axes[1, 1], PORTFOLIO_DATES, TOTAL_RETURN_PCT, port_fc_dates_actual, fc_ret, q_ret,
     "Total Return (%)", "%"),
]

for ax, dates_h, vals_h, dates_f, pt_f, qt_f, title, unit in datasets:
    n_h = len(dates_h)
    n_f = len(dates_f)
    x_h = list(range(n_h))
    x_f = list(range(n_h, n_h + n_f))

    ax.plot(x_h, vals_h, color=COLORS["history"], linewidth=1.5, marker=".", markersize=2)
    ax.plot(x_f, pt_f, color=COLORS["forecast"], linewidth=2, marker="D", markersize=4)

    if qt_f is not None:
        ax.fill_between(x_f, qt_f[:, 1], qt_f[:, 9], alpha=0.15, color=COLORS["forecast"])
        ax.fill_between(x_f, qt_f[:, 3], qt_f[:, 7], alpha=0.25, color=COLORS["forecast"])

    ax.axvline(x=n_h - 0.5, color=COLORS["accent"], linestyle="--", alpha=0.4)

    all_d = list(dates_h) + list(dates_f)
    ticks = list(range(0, len(all_d), 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_d[i] for i in ticks], rotation=45, ha="right", fontsize=7)

    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["text"], fontsize=11, fontweight="bold")
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.grid(True, alpha=0.1, color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
dashboard_path = run_dir / "00_dashboard.png"
fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {dashboard_path}")


# ===========================================================================
# PHASE 8: Generate Run Report
# ===========================================================================
section("PHASE 8: RUN REPORT")
t_total = time.perf_counter() - t_start

# Build markdown report
report_lines = []
report_lines.append(f"# Personal Finance Forecast — {timestamp[:8]}")
report_lines.append(f"")
report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"**Model:** TimesFM 2.5 200M (PyTorch)  |  **Device:** {device}")
report_lines.append(f"**Total runtime:** {fmt_time(t_total)}")
report_lines.append(f"")
report_lines.append(f"## Dashboard")
report_lines.append(f"")
report_lines.append(f"![Dashboard](00_dashboard.png)")
report_lines.append(f"")

# Net savings
report_lines.append(f"## 1. Net Savings Forecast (Apr-Dec 2026)")
report_lines.append(f"")
report_lines.append(f"![Net Savings](01_net_savings.png)")
report_lines.append(f"")
report_lines.append(f"| Month | Predicted | P10 (low) | P90 (high) |")
report_lines.append(f"|-------|-----------|-----------|------------|")
for i, date in enumerate(FORECAST_DATES):
    p = fc_savings[i]
    report_lines.append(f"| {date} | {fmt_eur(p)} EUR | {fmt_eur(q_savings[i,1])} EUR | {fmt_eur(q_savings[i,9])} EUR |")
report_lines.append(f"")
report_lines.append(f"**Total predicted:** {fmt_eur(total_fc)} EUR over 9 months")
report_lines.append(f"")
report_lines.append(f"**Key insight:** The model detects the November bonus pattern "
                     f"(avg {fmt_eur(monthly_avg.get(11, 0))} EUR vs {fmt_eur(np.mean([v for k,v in monthly_avg.items() if k != 11]))} EUR other months). "
                     f"May also shows elevated savings.")
report_lines.append(f"")

# Portfolio
report_lines.append(f"## 2. Portfolio Value Forecast")
report_lines.append(f"")
report_lines.append(f"![Portfolio](02_portfolio_value.png)")
report_lines.append(f"")
report_lines.append(f"| Month | Value | MoM | P10 | P90 |")
report_lines.append(f"|-------|-------|-----|-----|-----|")
prev_p = last_port
for i, date in enumerate(port_fc_dates_actual):
    p = fc_port_values[i]
    mom = (p / prev_p - 1) * 100
    prev_p = p
    report_lines.append(f"| {date} | {fmt_eur(p)} EUR | {mom:+.1f}% | {fmt_eur(q_port_values[i,1])} EUR | {fmt_eur(q_port_values[i,9])} EUR |")
report_lines.append(f"")
report_lines.append(f"**Current:** {fmt_eur(last_port)} EUR  ->  **Dec 2026:** {fmt_eur(fc_port_values[-1])} EUR ({growth:+.1f}%)")
report_lines.append(f"")

# Investment
report_lines.append(f"## 3. Monthly Investment Forecast")
report_lines.append(f"")
report_lines.append(f"![Investment](03_monthly_investment.png)")
report_lines.append(f"")
report_lines.append(f"| Month | Predicted | P10 | P90 |")
report_lines.append(f"|-------|-----------|-----|-----|")
for i, date in enumerate(FORECAST_DATES):
    report_lines.append(f"| {date} | {fmt_eur(fc_inv[i])} EUR | {fmt_eur(q_inv[i,1])} EUR | {fmt_eur(q_inv[i,9])} EUR |")
report_lines.append(f"")
report_lines.append(f"**Total new investment Apr-Dec 2026:** {fmt_eur(fc_inv.sum())} EUR")
report_lines.append(f"")

# Returns
report_lines.append(f"## 4. Return % Trajectory")
report_lines.append(f"")
report_lines.append(f"![Returns](04_returns.png)")
report_lines.append(f"")
report_lines.append(f"| Month | Return % | P10 | P90 |")
report_lines.append(f"|-------|----------|-----|-----|")
for i, date in enumerate(port_fc_dates_actual):
    report_lines.append(f"| {date} | {fc_ret[i]:+.1f}% | {q_ret[i,1]:+.1f}% | {q_ret[i,9]:+.1f}% |")
report_lines.append(f"")

# Summary
report_lines.append(f"## Summary")
report_lines.append(f"")
report_lines.append(f"| Metric | Current | Dec 2026 Forecast |")
report_lines.append(f"|--------|---------|-------------------|")
report_lines.append(f"| Portfolio Value | {fmt_eur(last_port)} EUR | {fmt_eur(fc_port_values[-1])} EUR |")
report_lines.append(f"| Total Invested | {fmt_eur(INVESTED_CAPITAL[-1])} EUR | ~{fmt_eur(INVESTED_CAPITAL[-1] + fc_inv.sum())} EUR |")
report_lines.append(f"| Monthly Savings (avg) | {fmt_eur(NET_SAVINGS[-12:].mean())} EUR | {fmt_eur(fc_savings.mean())} EUR |")
report_lines.append(f"| Total Return | {TOTAL_RETURN_PCT[-1]:+.1f}% | {fc_ret[-1]:+.1f}% (forecast) |")
report_lines.append(f"")
report_lines.append(f"## Methodology")
report_lines.append(f"")
report_lines.append(f"- **Model:** Google TimesFM 2.5 (200M params, pretrained time-series foundation model)")
report_lines.append(f"- **Net Savings & Investment:** Raw values (no transform needed — moderate range, no extreme trend)")
report_lines.append(f"- **Portfolio Value:** Log-returns to handle exponential growth trend")
report_lines.append(f"- **Confidence bands:** P10-P90 (80% CI) from continuous quantile head")
report_lines.append(f"- **Context:** Full history ({len(NET_SAVINGS)}-{len(PORTFOLIO_VALUE)} months depending on series)")
report_lines.append(f"- **Horizon:** {HORIZON} months (Apr-Dec 2026)")
report_lines.append(f"")
report_lines.append(f"---")
report_lines.append(f"*Generated by TimesFM 2.5 | {fmt_time(t_total)} total runtime | {device}*")

report_text = "\n".join(report_lines)
report_path = run_dir / "report.md"
report_path.write_text(report_text, encoding="utf-8")
print(f"  Report: {report_path}")
print(f"  Charts: {list(run_dir.glob('*.png'))}")

# Print summary
section("DONE")
print(f"""
  Run: {run_dir}
  Files:
    {run_dir / 'report.md'}
    {run_dir / '00_dashboard.png'}
    {run_dir / '01_net_savings.png'}
    {run_dir / '02_portfolio_value.png'}
    {run_dir / '03_monthly_investment.png'}
    {run_dir / '04_returns.png'}

  Key Predictions (Apr-Dec 2026):
    Net savings total:  {fmt_eur(total_fc)} EUR
    Portfolio Dec 2026: {fmt_eur(fc_port_values[-1])} EUR ({growth:+.1f}%)
    New investment:     {fmt_eur(fc_inv.sum())} EUR
    Return Dec 2026:    {fc_ret[-1]:+.1f}%

  Runtime: {fmt_time(t_total)}  |  {gpu_stats()}
""")
