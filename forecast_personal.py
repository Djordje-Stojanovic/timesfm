"""
TimesFM 2.5 — Personal Finance Forecast

Predicts:
  1. Net savings (next 9 months, Apr-Dec 2026)
  2. Income forecast (next 9 months)
  3. Savings rate (net/income) forecast
  4. Portfolio value trajectory (next 9 months)
  5. Monthly investment amount (next 9 months)
  6. Portfolio return % (next 9 months)

Generates matplotlib charts saved to runs/<timestamp>/.
Creates a comprehensive run report .md file.
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
    INCOME, INCOME_DATES,
    NET_SAVINGS, NET_SAVINGS_DATES,
    PORTFOLIO_VALUE, PORTFOLIO_DATES,
    MONTHLY_INVESTMENT, MONTHLY_INVESTMENT_DATES,
    INVESTED_CAPITAL, TOTAL_RETURN_PCT,
    FORECAST_DATES,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HORIZON = 9
MAX_HORIZON = 128
CONTEXT_LEN = 64

# ---------------------------------------------------------------------------
# Setup run directory
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_time(s): return f"{s*1000:.0f}ms" if s < 1 else f"{s:.1f}s"
def fmt_eur(v): return f"{v:,.0f}".replace(",", ".")

def gpu_stats():
    if not torch.cuda.is_available(): return "CPU mode"
    a, r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    t = torch.cuda.get_device_properties(0).total_memory
    return f"VRAM: {a/1024**2:.0f}MB / {t/1024**3:.1f}GB"

def section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")

COLORS = {
    "history": "#3b82f6", "forecast": "#f59e0b", "accent": "#10b981",
    "red": "#ef4444", "purple": "#a855f7",
    "grid": "#374151", "bg": "#111827", "text": "#f3f4f6",
}

def style_chart(ax, title, ylabel):
    ax.set_facecolor(COLORS["bg"])
    ax.figure.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.grid(True, alpha=0.12, color=COLORS["grid"])
    for spine in ax.spines.values(): spine.set_color(COLORS["grid"])

def plot_series(ax, dates_h, vals_h, dates_f, pt_f, qt_f, color_h, color_f, label_h="History", label_f="Forecast"):
    n_h, n_f = len(dates_h), len(dates_f)
    x_h, x_f = list(range(n_h)), list(range(n_h, n_h + n_f))
    ax.plot(x_h, vals_h, color=color_h, linewidth=1.8, marker="o", markersize=2.5, label=label_h, zorder=5)
    ax.plot(x_f, pt_f, color=color_f, linewidth=2.5, marker="D", markersize=4, label=label_f, zorder=5)
    if qt_f is not None:
        ax.fill_between(x_f, qt_f[:, 1], qt_f[:, 9], alpha=0.12, color=color_f)
        ax.fill_between(x_f, qt_f[:, 3], qt_f[:, 7], alpha=0.22, color=color_f)
    ax.axvline(x=n_h - 0.5, color=COLORS["accent"], linestyle="--", alpha=0.4, linewidth=0.8)
    all_d = list(dates_h) + list(dates_f)
    step = max(1, len(all_d) // 12)
    ticks = list(range(0, len(all_d), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([all_d[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    return x_h, x_f


# ===========================================================================
# PHASE 1: Compute Derived Series
# ===========================================================================
section("PHASE 1: DATA ANALYSIS")
t_start = time.perf_counter()

# Savings rate: net / income for overlapping months
n_overlap = min(len(INCOME), len(NET_SAVINGS))
savings_rate = NET_SAVINGS[:n_overlap] / INCOME[:n_overlap] * 100
savings_rate_dates = NET_SAVINGS_DATES[:n_overlap]

# Expenses = Income - Net
expenses = INCOME - NET_SAVINGS[:n_overlap]

print(f"  {'Series':<25} {'Months':>6} {'From':>10} {'To':>10} {'Mean':>10} {'Last':>10}")
print(f"  {'=' * 25} {'=' * 6} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
for name, dates, vals, unit in [
    ("Income", INCOME_DATES, INCOME, "EUR"),
    ("Net Savings", NET_SAVINGS_DATES, NET_SAVINGS, "EUR"),
    ("Savings Rate", savings_rate_dates, savings_rate, "%"),
    ("Expenses", INCOME_DATES, expenses, "EUR"),
    ("Portfolio Value", PORTFOLIO_DATES, PORTFOLIO_VALUE, "EUR"),
    ("Monthly Investment", MONTHLY_INVESTMENT_DATES, MONTHLY_INVESTMENT, "EUR"),
    ("Total Return", PORTFOLIO_DATES, TOTAL_RETURN_PCT, "%"),
    ("Invested Capital", PORTFOLIO_DATES, INVESTED_CAPITAL, "EUR"),
]:
    fmt = (lambda v: f"{v:.1f}%") if unit == "%" else (lambda v: fmt_eur(v) + " EUR")
    print(f"  {name:<25} {len(vals):>6} {dates[0]:>10} {dates[-1]:>10} {fmt(vals.mean()):>10} {fmt(vals[-1]):>10}")

# Monthly seasonality
print(f"\n  Monthly Patterns (avg):")
print(f"  {'Mon':>5}  {'Income':>10}  {'Net Sav':>10}  {'Sav Rate':>8}  {'Expenses':>10}  {'Invest':>10}")
print(f"  {'=' * 5}  {'=' * 10}  {'=' * 10}  {'=' * 8}  {'=' * 10}  {'=' * 10}")
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for m in range(1, 13):
    inc_avg = np.mean([INCOME[i] for i, d in enumerate(INCOME_DATES) if int(d.split("-")[1]) == m]) if any(int(d.split("-")[1]) == m for d in INCOME_DATES) else 0
    sav_avg = np.mean([NET_SAVINGS[i] for i, d in enumerate(NET_SAVINGS_DATES) if int(d.split("-")[1]) == m])
    rate_avg = (sav_avg / inc_avg * 100) if inc_avg > 0 else 0
    exp_avg = inc_avg - sav_avg if inc_avg > 0 else 0
    inv_avg = np.mean([MONTHLY_INVESTMENT[i] for i, d in enumerate(MONTHLY_INVESTMENT_DATES) if int(d.split("-")[1]) == m]) if any(int(d.split("-")[1]) == m for d in MONTHLY_INVESTMENT_DATES) else 0
    peak_inc = " <<" if m in [5, 11] else ""
    print(f"  {month_names[m-1]:>5}  {fmt_eur(inc_avg):>10}  {fmt_eur(sav_avg):>10}  {rate_avg:>6.1f}%  {fmt_eur(exp_avg):>10}  {fmt_eur(inv_avg):>10}{peak_inc}")

# Year-over-year growth
print(f"\n  Year-over-Year Net Savings:")
for year in [2022, 2023, 2024, 2025]:
    y_vals = [NET_SAVINGS[i] for i, d in enumerate(NET_SAVINGS_DATES) if d.startswith(str(year))]
    total = sum(y_vals)
    print(f"    {year}: {fmt_eur(total)} EUR ({len(y_vals)} months, avg {fmt_eur(total/len(y_vals))} EUR/mo)")

print(f"\n  Net Worth Summary:")
print(f"    Total saved (all time):       {fmt_eur(NET_SAVINGS.sum())} EUR")
print(f"    Total invested (portfolio):   {fmt_eur(INVESTED_CAPITAL[-1])} EUR")
print(f"    Portfolio value:              {fmt_eur(PORTFOLIO_VALUE[-1])} EUR")
print(f"    Investment gains:             {fmt_eur(PORTFOLIO_VALUE[-1] - INVESTED_CAPITAL[-1])} EUR ({TOTAL_RETURN_PCT[-1]:+.1f}%)")
uninvested = NET_SAVINGS.sum() - INVESTED_CAPITAL[-1]
print(f"    Cash / other (saved - invested): {fmt_eur(uninvested)} EUR")
print(f"    Implied net worth:            {fmt_eur(PORTFOLIO_VALUE[-1] + uninvested)} EUR")


# ===========================================================================
# PHASE 2: Load Model
# ===========================================================================
section("PHASE 2: MODEL")
t0 = time.perf_counter()
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"  Device: {device}  |  {gpu_stats()}")

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
t_load = time.perf_counter() - t0
print(f"  Loaded in {fmt_time(t_load)}  |  {gpu_stats()}")

def compile_pos():
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN, max_horizon=MAX_HORIZON,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=False, infer_is_positive=True,
        fix_quantile_crossing=True, return_backcast=True))

def compile_neg():
    model.compile(timesfm.ForecastConfig(
        max_context=CONTEXT_LEN, max_horizon=MAX_HORIZON,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=False,
        fix_quantile_crossing=True, return_backcast=True))


# ===========================================================================
# PHASE 3: Run All Forecasts
# ===========================================================================
section("PHASE 3: FORECASTING")
results = {}

# --- Income ---
compile_pos()
print(f"\n  [1/6] Income...")
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[INCOME])
results["income"] = (pt[0, :HORIZON], qt[0, :HORIZON, :])
print(f"         {fmt_time(time.perf_counter()-t0)}")

# --- Net Savings ---
print(f"  [2/6] Net Savings...")
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[NET_SAVINGS])
results["savings"] = (pt[0, :HORIZON], qt[0, :HORIZON, :])
print(f"         {fmt_time(time.perf_counter()-t0)}")

# --- Monthly Investment ---
print(f"  [3/6] Monthly Investment...")
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[MONTHLY_INVESTMENT])
results["investment"] = (pt[0, :HORIZON], qt[0, :HORIZON, :])
print(f"         {fmt_time(time.perf_counter()-t0)}")

# --- Portfolio (log-returns) ---
print(f"  [4/6] Portfolio Value (log-returns)...")
port_lr = np.diff(np.log(PORTFOLIO_VALUE))
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[port_lr])
last_port = PORTFOLIO_VALUE[-1]
fc_port = last_port * np.exp(np.cumsum(pt[0, :HORIZON]))
q_port = np.zeros((HORIZON, 10))
for h in range(HORIZON):
    ps = float(np.sum(pt[0, :h])) if h > 0 else 0.0
    q_port[h, :] = last_port * np.exp(ps + qt[0, h, :])
results["portfolio"] = (fc_port, q_port)
print(f"         {fmt_time(time.perf_counter()-t0)}")

# --- Savings Rate (can be negative theoretically, use neg compile) ---
compile_neg()
print(f"  [5/6] Savings Rate...")
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[savings_rate])
results["rate"] = (pt[0, :HORIZON], qt[0, :HORIZON, :])
print(f"         {fmt_time(time.perf_counter()-t0)}")

# --- Returns ---
print(f"  [6/6] Portfolio Returns...")
t0 = time.perf_counter()
pt, qt = model.forecast(horizon=HORIZON, inputs=[TOTAL_RETURN_PCT])
results["returns"] = (pt[0, :HORIZON], qt[0, :HORIZON, :])
print(f"         {fmt_time(time.perf_counter()-t0)}")

# Derived: forecast expenses = forecast income - forecast savings
fc_income = results["income"][0]
fc_savings = results["savings"][0]
fc_expenses = fc_income - fc_savings


# ===========================================================================
# PHASE 4: Print Results
# ===========================================================================
section("PHASE 4: RESULTS")

datasets = [
    ("Income",           FORECAST_DATES, fc_income,           results["income"][1],     "EUR"),
    ("Net Savings",      FORECAST_DATES, fc_savings,           results["savings"][1],    "EUR"),
    ("Savings Rate",     FORECAST_DATES, results["rate"][0],   results["rate"][1],       "%"),
    ("Expenses (derived)", FORECAST_DATES, fc_expenses,        None,                     "EUR"),
    ("Investment",       FORECAST_DATES, results["investment"][0], results["investment"][1], "EUR"),
    ("Portfolio Value",  FORECAST_DATES, fc_port,              q_port,                   "EUR"),
    ("Return %",         FORECAST_DATES, results["returns"][0], results["returns"][1],   "%"),
]

for name, dates, point, quant, unit in datasets:
    print(f"\n  {name}:")
    fmt = (lambda v: f"{v:+.1f}%") if unit == "%" else (lambda v: f"{fmt_eur(v)} EUR")
    for i, d in enumerate(dates):
        p = point[i]
        if quant is not None:
            q10, q90 = quant[i, 1], quant[i, 9]
            print(f"    {d}:  {fmt(p):>14}   (P10: {fmt(q10)}, P90: {fmt(q90)})")
        else:
            print(f"    {d}:  {fmt(p):>14}")
    if unit == "EUR":
        print(f"    Total: {fmt(point.sum())}")


# ===========================================================================
# PHASE 5: Charts
# ===========================================================================
section("PHASE 5: CHARTS")

# Chart 1: Income + Savings + Expenses stacked view
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
fig.set_facecolor(COLORS["bg"])
fig.suptitle("Income, Savings & Expenses", color=COLORS["text"], fontsize=15, fontweight="bold")

# Income
ax = axes[0]
plot_series(ax, INCOME_DATES, INCOME, FORECAST_DATES, fc_income, results["income"][1],
            COLORS["history"], COLORS["forecast"])
style_chart(ax, "Monthly Income", "EUR")
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

# Net Savings
ax = axes[1]
plot_series(ax, NET_SAVINGS_DATES, NET_SAVINGS, FORECAST_DATES, fc_savings, results["savings"][1],
            COLORS["accent"], COLORS["forecast"])
style_chart(ax, "Net Savings", "EUR")
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

# Savings Rate
ax = axes[2]
plot_series(ax, savings_rate_dates, savings_rate, FORECAST_DATES, results["rate"][0], results["rate"][1],
            COLORS["purple"], COLORS["forecast"])
ax.axhline(y=50, color=COLORS["red"], linestyle=":", alpha=0.4, linewidth=0.8)
ax.text(0, 51, "50% target", color=COLORS["red"], fontsize=7, alpha=0.5)
style_chart(ax, "Savings Rate (Net / Income)", "%")
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(run_dir / "01_income_savings.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 01_income_savings.png")

# Chart 2: Portfolio + Returns
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.set_facecolor(COLORS["bg"])
fig.suptitle("Portfolio Performance", color=COLORS["text"], fontsize=15, fontweight="bold")

port_fc_dates = ["2026-05","2026-06","2026-07","2026-08","2026-09","2026-10","2026-11","2026-12","2027-01"]

ax = axes[0]
plot_series(ax, PORTFOLIO_DATES, PORTFOLIO_VALUE, port_fc_dates, fc_port, q_port,
            COLORS["history"], COLORS["forecast"])
# Also plot invested capital
n_inv = len(INVESTED_CAPITAL)
ax.plot(range(n_inv), INVESTED_CAPITAL, color=COLORS["red"], linewidth=1, linestyle="--", alpha=0.6, label="Invested Capital")
style_chart(ax, "Portfolio Value vs Invested Capital", "EUR")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

ax = axes[1]
plot_series(ax, PORTFOLIO_DATES, TOTAL_RETURN_PCT, port_fc_dates, results["returns"][0], results["returns"][1],
            COLORS["accent"], COLORS["forecast"])
ax.axhline(y=0, color=COLORS["red"], linestyle=":", alpha=0.4)
style_chart(ax, "Total Return %", "%")
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(run_dir / "02_portfolio.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 02_portfolio.png")

# Chart 3: Investment
fig, ax = plt.subplots(figsize=(14, 4))
plot_series(ax, MONTHLY_INVESTMENT_DATES, MONTHLY_INVESTMENT, FORECAST_DATES,
            results["investment"][0], results["investment"][1],
            COLORS["history"], COLORS["forecast"])
style_chart(ax, "Monthly Investment Amount", "EUR")
ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)
fig.tight_layout()
fig.savefig(run_dir / "03_investment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 03_investment.png")

# Chart 4: 6-panel dashboard
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.set_facecolor(COLORS["bg"])
fig.suptitle(f"Personal Finance Dashboard — {datetime.now().strftime('%B %Y')}", color=COLORS["text"], fontsize=16, fontweight="bold")

panels = [
    (axes[0,0], INCOME_DATES, INCOME, FORECAST_DATES, fc_income, results["income"][1], "Income (EUR/mo)", COLORS["history"]),
    (axes[0,1], NET_SAVINGS_DATES, NET_SAVINGS, FORECAST_DATES, fc_savings, results["savings"][1], "Net Savings (EUR/mo)", COLORS["accent"]),
    (axes[0,2], savings_rate_dates, savings_rate, FORECAST_DATES, results["rate"][0], results["rate"][1], "Savings Rate (%)", COLORS["purple"]),
    (axes[1,0], PORTFOLIO_DATES, PORTFOLIO_VALUE, port_fc_dates, fc_port, q_port, "Portfolio (EUR)", COLORS["history"]),
    (axes[1,1], MONTHLY_INVESTMENT_DATES, MONTHLY_INVESTMENT, FORECAST_DATES, results["investment"][0], results["investment"][1], "Monthly Invest (EUR)", COLORS["history"]),
    (axes[1,2], PORTFOLIO_DATES, TOTAL_RETURN_PCT, port_fc_dates, results["returns"][0], results["returns"][1], "Total Return (%)", COLORS["accent"]),
]

for ax, dh, vh, df, pf, qf, title, col in panels:
    plot_series(ax, dh, vh, df, pf, qf, col, COLORS["forecast"])
    style_chart(ax, title, "")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(run_dir / "00_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: 00_dashboard.png")


# ===========================================================================
# PHASE 6: Generate Report
# ===========================================================================
section("PHASE 6: REPORT")
t_total = time.perf_counter() - t_start

# Compute key metrics for report
avg_rate_hist = savings_rate.mean()
avg_rate_fc = results["rate"][0].mean()
total_saved_all = NET_SAVINGS.sum()
total_fc_savings = fc_savings.sum()
total_fc_income = fc_income.sum()
total_fc_investment = results["investment"][0].sum()

R = []  # report lines
def r(line=""): R.append(line)

r(f"# Personal Finance Forecast Report")
r(f"")
r(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
r(f"**Model:** TimesFM 2.5 200M | **GPU:** {device} | **Runtime:** {fmt_time(t_total)}")
r(f"**Forecast horizon:** {HORIZON} months (Apr - Dec 2026)")
r(f"")
r(f"---")
r(f"")
r(f"## Dashboard")
r(f"")
r(f"![Dashboard](00_dashboard.png)")
r(f"")

# Current state
r(f"## Current Financial State (as of {NET_SAVINGS_DATES[-1]})")
r(f"")
r(f"| Metric | Value |")
r(f"|--------|-------|")
r(f"| Total saved (Oct 2021 - Mar 2026) | **{fmt_eur(total_saved_all)} EUR** |")
r(f"| Portfolio value | **{fmt_eur(PORTFOLIO_VALUE[-1])} EUR** |")
r(f"| Total invested (cost basis) | {fmt_eur(INVESTED_CAPITAL[-1])} EUR |")
r(f"| Investment gains | {fmt_eur(PORTFOLIO_VALUE[-1] - INVESTED_CAPITAL[-1])} EUR ({TOTAL_RETURN_PCT[-1]:+.1f}%) |")
r(f"| Cash / other | ~{fmt_eur(total_saved_all - INVESTED_CAPITAL[-1])} EUR |")
r(f"| Last monthly income | {fmt_eur(INCOME[-1])} EUR |")
r(f"| Last monthly savings | {fmt_eur(NET_SAVINGS[-1])} EUR |")
r(f"| Average savings rate | {avg_rate_hist:.1f}% |")
r(f"| Average monthly investment | {fmt_eur(MONTHLY_INVESTMENT.mean())} EUR |")
r(f"")

# Seasonality
r(f"## Seasonal Patterns")
r(f"")
r(f"| Month | Avg Income | Avg Savings | Savings Rate | Avg Investment |")
r(f"|-------|------------|-------------|--------------|----------------|")
for m in range(1, 13):
    inc_vals = [INCOME[i] for i, d in enumerate(INCOME_DATES) if int(d.split("-")[1]) == m]
    sav_vals = [NET_SAVINGS[i] for i, d in enumerate(NET_SAVINGS_DATES) if int(d.split("-")[1]) == m]
    inv_vals = [MONTHLY_INVESTMENT[i] for i, d in enumerate(MONTHLY_INVESTMENT_DATES) if int(d.split("-")[1]) == m]
    inc_avg = np.mean(inc_vals) if inc_vals else 0
    sav_avg = np.mean(sav_vals)
    rate = sav_avg / inc_avg * 100 if inc_avg > 0 else 0
    inv_avg = np.mean(inv_vals) if inv_vals else 0
    peak = " **PEAK**" if m in [5, 11] else ""
    r(f"| {month_names[m-1]} | {fmt_eur(inc_avg)} | {fmt_eur(sav_avg)} | {rate:.0f}% | {fmt_eur(inv_avg)} |{peak}")
r(f"")
r(f"**Pattern:** May and November show 2x income spikes (bonuses/variable compensation), driving savings rate above 70%.")
r(f"")

# Income + Savings forecast
r(f"## 1. Income & Savings Forecast")
r(f"")
r(f"![Income & Savings](01_income_savings.png)")
r(f"")
r(f"| Month | Income | Net Savings | Savings Rate | Expenses |")
r(f"|-------|--------|-------------|--------------|----------|")
for i, d in enumerate(FORECAST_DATES):
    inc = fc_income[i]
    sav = fc_savings[i]
    rate = sav / inc * 100 if inc > 0 else 0
    exp = inc - sav
    r(f"| {d} | {fmt_eur(inc)} | {fmt_eur(sav)} | {rate:.0f}% | {fmt_eur(exp)} |")
r(f"")
r(f"| Totals (9 months) | {fmt_eur(total_fc_income)} | {fmt_eur(total_fc_savings)} | {total_fc_savings/total_fc_income*100:.0f}% | {fmt_eur(total_fc_income - total_fc_savings)} |")
r(f"")

# Portfolio
r(f"## 2. Portfolio Forecast")
r(f"")
r(f"![Portfolio](02_portfolio.png)")
r(f"")
r(f"| Month | Value | P10 (bear) | P90 (bull) | MoM |")
r(f"|-------|-------|------------|------------|-----|")
prev = last_port
for i, d in enumerate(port_fc_dates):
    p = fc_port[i]
    mom = (p / prev - 1) * 100
    prev = p
    r(f"| {d} | {fmt_eur(p)} | {fmt_eur(q_port[i,1])} | {fmt_eur(q_port[i,9])} | {mom:+.1f}% |")
growth = (fc_port[-1] / last_port - 1) * 100
r(f"")
r(f"**{fmt_eur(last_port)} EUR -> {fmt_eur(fc_port[-1])} EUR ({growth:+.1f}%)**")
r(f"")
r(f"*Note: Portfolio forecast extrapolates recent ~2.5%/mo growth trend. Use P10 ({fmt_eur(q_port[-1,1])} EUR) as conservative estimate.*")
r(f"")

# Investment
r(f"## 3. Monthly Investment Forecast")
r(f"")
r(f"![Investment](03_investment.png)")
r(f"")
r(f"| Month | Predicted | P10 | P90 |")
r(f"|-------|-----------|-----|-----|")
for i, d in enumerate(FORECAST_DATES):
    inv = results["investment"][0][i]
    r(f"| {d} | {fmt_eur(inv)} | {fmt_eur(results['investment'][1][i,1])} | {fmt_eur(results['investment'][1][i,9])} |")
r(f"")
r(f"**Total new investment:** {fmt_eur(total_fc_investment)} EUR")
r(f"**Projected total invested by Dec 2026:** ~{fmt_eur(INVESTED_CAPITAL[-1] + total_fc_investment)} EUR")
r(f"")

# Returns
r(f"## 4. Return Trajectory")
r(f"")
r(f"| Month | Return % | P10 | P90 |")
r(f"|-------|----------|-----|-----|")
for i, d in enumerate(port_fc_dates):
    ret = results["returns"][0][i]
    r(f"| {d} | {ret:+.1f}% | {results['returns'][1][i,1]:+.1f}% | {results['returns'][1][i,9]:+.1f}% |")
r(f"")

# Summary
r(f"## Summary: Where You'll Be by Dec 2026")
r(f"")
r(f"| Metric | Now | Dec 2026 (forecast) | Change |")
r(f"|--------|-----|---------------------|--------|")
r(f"| Portfolio | {fmt_eur(PORTFOLIO_VALUE[-1])} EUR | {fmt_eur(fc_port[-1])} EUR | {growth:+.1f}% |")
r(f"| Invested Capital | {fmt_eur(INVESTED_CAPITAL[-1])} EUR | ~{fmt_eur(INVESTED_CAPITAL[-1] + total_fc_investment)} EUR | +{fmt_eur(total_fc_investment)} |")
r(f"| Monthly Savings (avg) | {fmt_eur(NET_SAVINGS[-12:].mean())} EUR | {fmt_eur(fc_savings.mean())} EUR | |")
r(f"| Savings Rate | {avg_rate_hist:.0f}% | {avg_rate_fc:.0f}% | |")
r(f"| New Savings (9mo) | | {fmt_eur(total_fc_savings)} EUR | |")
r(f"| New Investment (9mo) | | {fmt_eur(total_fc_investment)} EUR | |")
r(f"")

r(f"## Methodology")
r(f"")
r(f"- **Model:** Google TimesFM 2.5 (200M params, pretrained time-series foundation model)")
r(f"- **Income, Savings, Investment:** Raw values (no transform — moderate range, seasonal)")
r(f"- **Savings Rate:** Raw % values, compiled with negative-capable settings")
r(f"- **Portfolio Value:** Log-returns to handle exponential growth, then reconstructed")
r(f"- **Expenses:** Derived (Income - Net Savings), not independently forecast")
r(f"- **Confidence:** P10-P90 (80% CI) from continuous quantile head")
r(f"- **Context:** Full history ({len(INCOME)}-{len(PORTFOLIO_VALUE)} months depending on series)")
r(f"- **Horizon:** {HORIZON} months")
r(f"- **Caveats:** Portfolio forecast assumes trend continuation. Returns forecast shows mean reversion. Neither accounts for market shocks or life changes.")
r(f"")
r(f"---")
r(f"*Generated by TimesFM 2.5 | {fmt_time(t_total)} | {device} | {gpu_stats()}*")

report_path = run_dir / "report.md"
report_path.write_text("\n".join(R), encoding="utf-8")
print(f"  Report: {report_path}")

# ===========================================================================
# DONE
# ===========================================================================
section("DONE")
print(f"""
  Run directory: {run_dir}/
  Files:
    report.md           — Full forecast report with tables
    00_dashboard.png    — 6-panel overview
    01_income_savings.png — Income + Savings + Savings Rate
    02_portfolio.png    — Portfolio value + Returns
    03_investment.png   — Monthly investment amount

  Key Numbers:
    Income (9mo):       {fmt_eur(total_fc_income)} EUR
    Net Savings (9mo):  {fmt_eur(total_fc_savings)} EUR
    Savings Rate:       {total_fc_savings/total_fc_income*100:.0f}%
    New Investment:     {fmt_eur(total_fc_investment)} EUR
    Portfolio Dec 2026: {fmt_eur(fc_port[-1])} EUR ({growth:+.1f}%)

  Runtime: {fmt_time(t_total)}  |  {gpu_stats()}
""")
