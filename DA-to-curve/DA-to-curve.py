from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_MODEL = "lightgbm"
SPREAD_THRESHOLD_EUR = 5.0
STRONG_THRESHOLD_EUR = 10.0
ROLLING_WINDOW_DAYS = 7
STOP_LOSS_EUR = 15.0
MAX_CONSEC_LOSS_DAYS = 3
MAE_EXIT_THRESHOLD = 30.0

def load_blocks(blocks_path: Path) -> pd.DataFrame:
    df = pd.read_csv(blocks_path, parse_dates=["date"])
    print(f"  Loaded {len(df)} daily rows  "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    return df


def available_models(df: pd.DataFrame) -> List[str]:
    skip = {"actual", "actual_test"}
    models = []
    for c in df.columns:
        if c.startswith("base_"):
            name = c.replace("base_", "")
            if name not in skip:
                models.append(name)
    return models


def build_prompt_curve_proxy(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    df = df.copy()
    df["proxy_base"] = df["base_actual"].shift(1).rolling(window).mean()
    df["proxy_peak"] = df["peak_actual"].shift(1).rolling(window).mean()
    df["proxy_offpeak"] = df["offpeak_actual"].shift(1).rolling(window).mean()
    return df


def compute_spreads(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    df = df.copy()
    for product in ("base", "peak", "offpeak"):
        fcst_col = f"{product}_{model}"
        proxy_col = f"proxy_{product}"
        if fcst_col in df.columns and proxy_col in df.columns:
            df[f"spread_{product}"] = df[fcst_col] - df[proxy_col]
            df[f"spread_pct_{product}"] = (
                df[f"spread_{product}"] / df[proxy_col].replace(0, np.nan) * 100
            )
    return df


def generate_signals(
    df: pd.DataFrame,
    product: str = "base",
    threshold: float = SPREAD_THRESHOLD_EUR,
    strong: float = STRONG_THRESHOLD_EUR,
) -> pd.DataFrame:
    df = df.copy()
    spread_col = f"spread_{product}"
    if spread_col not in df.columns:
        raise ValueError(f"Missing column {spread_col}")

    conditions = [
        df[spread_col] > threshold,
        df[spread_col] < -threshold,
    ]
    choices = [1, -1]
    df["signal"] = np.select(conditions, choices, default=0)

    df["conviction"] = np.where(
        df[spread_col].abs() > strong, "Strong",
        np.where(df[spread_col].abs() > threshold, "Moderate", "Weak"),
    )
    df["action"] = df["signal"].map({1: "BUY", -1: "SELL", 0: "HOLD"})
    return df


def backtest(
    df: pd.DataFrame,
    product: str = "base",
) -> pd.DataFrame:
    df = df.copy()
    actual_col = f"{product}_actual"
    proxy_col = f"proxy_{product}"

    df["pnl_daily"] = df["signal"] * (df[actual_col] - df[proxy_col])

    df["pnl_cumulative"] = df["pnl_daily"].cumsum()

    df["pnl_rolling_7d"] = df["pnl_daily"].rolling(7, min_periods=1).mean()

    df["is_loss"] = df["pnl_daily"] < 0
    loss_streak = df["is_loss"].astype(int)
    df["consec_losses"] = loss_streak.groupby(
        (loss_streak != loss_streak.shift()).cumsum()
    ).cumsum() * loss_streak

    return df


def backtest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    active = df[df["signal"] != 0]
    total_days = len(df)
    active_days = len(active)

    if active_days == 0:
        return {"total_days": total_days, "active_days": 0, "total_pnl": 0.0}

    wins = (active["pnl_daily"] > 0).sum()
    losses = (active["pnl_daily"] < 0).sum()

    return {
        "total_days": total_days,
        "active_days": active_days,
        "participation_rate": f"{active_days / total_days * 100:.1f}%",
        "total_pnl_eur_mwh": round(float(df["pnl_cumulative"].iloc[-1]), 2),
        "avg_daily_pnl": round(float(active["pnl_daily"].mean()), 2),
        "win_rate": f"{wins / active_days * 100:.1f}%",
        "wins": int(wins),
        "losses": int(losses),
        "best_day": round(float(active["pnl_daily"].max()), 2),
        "worst_day": round(float(active["pnl_daily"].min()), 2),
        "sharpe_daily": round(
            float(active["pnl_daily"].mean() / active["pnl_daily"].std())
            if active["pnl_daily"].std() > 0 else 0.0, 3
        ),
        "max_consec_losses": int(df["consec_losses"].max()),
    }


def evaluate_invalidation(
    df: pd.DataFrame,
    model: str,
    product: str = "base",
    lookback: int = 14,
) -> Dict[str, Any]:
    recent = df.tail(lookback).copy()
    fcst_col = f"{product}_{model}"
    actual_col = f"{product}_actual"
    proxy_col = f"proxy_{product}"

    if fcst_col in recent.columns:
        errors = (recent[fcst_col] - recent[actual_col]).abs()
        mae = float(errors.mean())
        half = len(recent) // 2
        mae_first = float(errors.iloc[:half].mean())
        mae_second = float(errors.iloc[half:].mean())
        error_trend = "Increasing" if mae_second > mae_first * 1.2 else "Stable"
    else:
        mae = None
        error_trend = "Unknown"

    max_consec = int(recent["consec_losses"].max()) if "consec_losses" in recent.columns else 0

    extreme = False
    if proxy_col in recent.columns and actual_col in recent.columns:
        ratio = recent[actual_col] / recent[proxy_col].replace(0, np.nan)
        extreme = bool((ratio > 2.0).any() or (ratio < 0.3).any())

    criteria = {
        "block_mae": {
            "metric": f"{product.title()} block MAE (last {lookback}d)",
            "threshold": f"> {MAE_EXIT_THRESHOLD} EUR/MWh",
            "current": f"{mae:.2f} EUR/MWh" if mae else "N/A",
            "triggered": (mae is not None and mae > MAE_EXIT_THRESHOLD),
        },
        "error_trend": {
            "metric": "Error trajectory (2nd half vs 1st half of lookback)",
            "threshold": "Increasing > 20%",
            "current": error_trend,
            "triggered": error_trend == "Increasing",
        },
        "consecutive_losses": {
            "metric": "Consecutive losing days",
            "threshold": f"> {MAX_CONSEC_LOSS_DAYS} days",
            "current": f"{max_consec} days",
            "triggered": max_consec > MAX_CONSEC_LOSS_DAYS,
        },
        "extreme_price_event": {
            "metric": "Actual / Proxy ratio",
            "threshold": "> 2.0x or < 0.3x (supply shock / demand collapse)",
            "current": "Detected" if extreme else "Normal",
            "triggered": extreme,
        },
    }

    any_triggered = any(c["triggered"] for c in criteria.values())

    return {
        "criteria": criteria,
        "invalidation_triggered": any_triggered,
        "recommended_action": "EXIT all positions" if any_triggered else "MAINTAIN positions",
        "lookback_days": lookback,
    }


def build_monthly_curve(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    agg = {}
    for product in ("base", "peak", "offpeak"):
        agg[f"{product}_actual"] = (f"{product}_actual", "mean")
        fcst_col = f"{product}_{model}"
        if fcst_col in df.columns:
            agg[f"{product}_forecast"] = (fcst_col, "mean")
        proxy_col = f"proxy_{product}"
        if proxy_col in df.columns:
            agg[f"{product}_proxy"] = (proxy_col, "mean")

    if "pnl_daily" in df.columns:
        agg["pnl_total"] = ("pnl_daily", "sum")
        agg["pnl_mean"] = ("pnl_daily", "mean")
        agg["signal_avg"] = ("signal", "mean")
        agg["active_days"] = ("signal", lambda x: (x != 0).sum())
        agg["n_days"] = ("signal", "count")

    monthly = df.groupby("month").agg(**agg).reset_index()
    monthly["month"] = monthly["month"].astype(str)

    if "base_forecast" in monthly.columns:
        monthly["spread_base"] = monthly["base_forecast"] - monthly["base_actual"]
        monthly["spread_peak"] = monthly["peak_forecast"] - monthly["peak_actual"]

    return monthly


def plot_all(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    model: str,
    product: str,
    out_dir: Path,
) -> List[Path]:
    saved = []
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, prod, title in zip(
        axes,
        ["base", "peak", "offpeak"],
        ["Baseload", "Peakload", "Offpeak"],
    ):
        ax.plot(df["date"], df[f"{prod}_actual"], label="Actual DA", linewidth=0.8, alpha=0.7)
        fcst_col = f"{prod}_{model}"
        if fcst_col in df.columns:
            ax.plot(df["date"], df[fcst_col], label=f"Forecast ({model})", linewidth=0.8, alpha=0.7)
        ax.plot(df["date"], df[f"proxy_{prod}"], label=f"Proxy ({ROLLING_WINDOW_DAYS}d)", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"{title} Block Prices")
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("DA-to-Curve: Block Prices Overview", fontsize=14, y=1.01)
    plt.tight_layout()
    p = out_dir / "block_prices_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    spread_col = f"spread_{product}"
    ax1.fill_between(
        df["date"], df[spread_col],
        where=df[spread_col] > 0, color="green", alpha=0.3, label="Bullish (FV > Ref)",
    )
    ax1.fill_between(
        df["date"], df[spread_col],
        where=df[spread_col] < 0, color="red", alpha=0.3, label="Bearish (FV < Ref)",
    )
    ax1.axhline(SPREAD_THRESHOLD_EUR, color="green", linestyle=":", alpha=0.5, label=f"+{SPREAD_THRESHOLD_EUR} threshold")
    ax1.axhline(-SPREAD_THRESHOLD_EUR, color="red", linestyle=":", alpha=0.5, label=f"-{SPREAD_THRESHOLD_EUR} threshold")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Spread (EUR/MWh)")
    ax1.set_title(f"{product.title()} Spread: Model Fair Value vs Reference Curve")
    ax1.legend(fontsize=8)

    colors_map = {1: "green", -1: "red", 0: "gray"}
    colors = df["signal"].map(colors_map)
    ax2.bar(df["date"], df["signal"], color=colors, alpha=0.6, width=1)
    ax2.set_ylabel("Signal")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["SELL", "HOLD", "BUY"])
    ax2.set_title("Daily Directional View")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    p = out_dir / "spread_and_signals.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    if "base_forecast" in monthly.columns:
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))

        x = range(len(monthly))

        axes[0].plot(x, monthly["base_actual"], "o-", label="Realised Settlement",
                     linewidth=1.5, markersize=5, color="navy")
        axes[0].plot(x, monthly["base_forecast"], "s-", label=f"Model Fair Value ({model})",
                     linewidth=1.5, markersize=5, color="darkorange")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("EUR/MWh")
        axes[0].set_title("Monthly Baseload: Realised Settlement vs Model Fair Value")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        if "peak_forecast" in monthly.columns:
            axes[1].plot(x, monthly["peak_actual"], "o-", label="Realised Settlement",
                         linewidth=1.5, markersize=5, color="navy")
            axes[1].plot(x, monthly["peak_forecast"], "s-", label=f"Model Fair Value ({model})",
                         linewidth=1.5, markersize=5, color="darkorange")
            axes[1].set_xticks(list(x))
            axes[1].set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=7)
            axes[1].set_ylabel("EUR/MWh")
            axes[1].set_title("Monthly Peakload: Realised Settlement vs Model Fair Value")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

        colors = ["green" if s > 0 else "red" for s in monthly["spread_base"]]
        axes[2].bar(x, monthly["spread_base"], color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("Spread (EUR/MWh)")
        axes[2].set_title("Monthly Forecast Bias (Model FV − Realised)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        p = out_dir / "monthly_fair_value_curve.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def generate_report(
    monthly: pd.DataFrame,
    bt_summary: Dict[str, Any],
    invalidation: Dict[str, Any],
    model: str,
    product: str,
    out_dir: Path,
) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    active_days = bt_summary.get("active_days", 0)
    wins = bt_summary.get("wins", 0)
    dir_acc = f"{wins / active_days * 100:.0f}%" if active_days > 0 else "N/A"

    report = f"""

**Generated:** {now}
**Model:** {model.title()}
**Primary product:** {product.title()}
**Reference curve:** {ROLLING_WINDOW_DAYS}-day trailing average of realised DA blocks
**Signal threshold:** +/-{SPREAD_THRESHOLD_EUR} EUR/MWh

---



1. **Hourly DA forecasts** from the {model.title()} model are aggregated into
   daily block prices — Baseload (hours 1-24), Peakload (hours 9-20, EPEX
   convention), and Offpeak (complement).

2. A **reference curve** is constructed as the {ROLLING_WINDOW_DAYS}-day
   trailing average of recently realised DA block prices. This captures the
   recent price level and is used to measure whether the model forecast
   deviates from the prevailing trend. In production, the reference would be
   replaced by the actual EEX Phelix DE prompt-month forward price (German
   Power Baseload/Peakload Calendar Month Futures, CME/EEX).

3. The **spread** (forecast block minus reference curve) measures the
   model's directional view: positive means the model expects delivery
   prices above recent levels; negative means below.

4. When |spread| exceeds {SPREAD_THRESHOLD_EUR} EUR/MWh, the view is
   classified as **BUY** or **SELL**. Below the threshold it is **HOLD**
   (the model sees no meaningful deviation from recent trend).


| Block    | Delivery hours      | Description                    |
|----------|--------------------|---------------------------------|
| Baseload | 1-24 (00:00-23:00) | Flat average, all hours          |
| Peakload | 9-20 (08:00-19:00) | Business hours, higher demand    |
| Offpeak  | 1-8, 21-24          | Night + evening, lower demand    |


The block-level fair value maps directly to exchange-traded products:

| Fair-Value Block | Tradable Product (CME/EEX)                     |
|------------------|-------------------------------------------------|
| Baseload Month   | German Power Baseload Calendar Month Future      |
| Peakload Month   | German Power Peakload Calendar Month Future      |
| Baseload Week    | German Power Baseload Calendar Week Future       |

A baseload month futures contract settles to the arithmetic average of
hourly DA auction prices over the delivery month (sized MWh/h x 24 x days,
adjusted for DST). Our monthly fair value is therefore directly comparable
to the contract's settlement value.

---


This table shows the model's monthly average forecast vs realised DA
settlement. The **spread** (forecast minus actual) measures forecast bias at
the monthly level — the same granularity as a prompt-month futures contract.

| Month | Actual Base | Fcst Base | Spread | Actual Peak | Fcst Peak |
|-------|------------|----------|--------|------------|----------|

The average absolute spread (baseload) is a measure of how close the model's
monthly view is to realised settlement — this is the residual error a prompt
trader would face when using this forecast as a fair-value anchor.

---


The forecast's directional accuracy is assessed by checking whether the
model correctly predicts the sign of (actual block − reference curve).
This is a **directional** measure, not a P&L claim.

| Metric                     | Value                        |
|----------------------------|------------------------------|
| Out-of-sample days         | {bt_summary['total_days']}   |
| Days with active signal    | {active_days} ({bt_summary.get('participation_rate', 'N/A')}) |
| Directional accuracy       | {dir_acc}                    |
| Max consecutive mispredictions | {bt_summary.get('max_consec_losses', 0)} days |

**Note:** This accuracy is measured against a {ROLLING_WINDOW_DAYS}-day
trailing average, which is a slow-moving, backward-looking reference — not a
real forward price. Against a live EEX forward (which already embeds market
expectations, supply/demand outlook, and risk premia), directional accuracy
would be lower and edge smaller. The metric demonstrates the model carries
signal relative to a naive baseline, not that these hit rates are achievable
in live trading.

---



Each morning before the DA auction closes:

1. Run the forecast model for the next 24 hours.
2. Aggregate into baseload / peakload blocks → this is your **fair value (FV)**.
3. Obtain the current prompt-month forward price from EEX/EPEX (**MKT**).
4. Compute **edge = FV − MKT**.
5. **If edge > +{SPREAD_THRESHOLD_EUR} EUR/MWh** → the model sees the
   month settling higher than the market is pricing → consider **buying**
   the prompt-month forward.
6. **If edge < −{SPREAD_THRESHOLD_EUR} EUR/MWh** → the model sees the
   month settling lower → consider **selling** the prompt-month forward.
7. The position converges as DA prices materialise during the delivery
   month. Settlement = average hourly DA over the month.


- Compare the forecast *peak premium* (peak − base) against its historical
  distribution.
- If the premium widens beyond the 75th percentile → **sell peak / buy base**
  (mean-reversion).
- If it narrows below the 25th percentile → **buy peak / sell base**.


- Average the next 7 days of forecast blocks to build a "week-ahead FV."
- Compare against the week-ahead EEX forward.
- Useful for sizing longer-duration positions with lower transaction costs.


- **Transaction costs:** Typical bid-ask spreads on prompt-month DE baseload
  are ~0.10–0.50 EUR/MWh. Only trade when |edge| comfortably exceeds this.
- **Position frequency:** One position per delivery month is more realistic
  than daily re-entry. Use the latest signal before delivery starts.
- **Conviction scaling:** Scale position size by |edge| / threshold —
  larger deviations warrant larger positions.

---


**Overall recommendation:** {invalidation['recommended_action']}


1. **Model degradation:** If the rolling 14-day block MAE exceeds
   {MAE_EXIT_THRESHOLD} EUR/MWh, the forecast is no longer reliable — stop
   using it for positioning and retrain.

2. **Regime change:** If actual DA prices diverge >2x or <0.3x from the
   reference curve (e.g., supply shock, interconnector outage), the model's
   training regime no longer reflects reality. Exit and wait for a new
   stable regime.

3. **Persistent directional failure:** If the forecast mispredicts direction
   on >{MAX_CONSEC_LOSS_DAYS} consecutive days, reduce conviction and
   reassess whether a structural shift has occurred.

4. **Fundamental shift:** If renewable penetration changes by >30% week-
   over-week (e.g., extended Dunkelflaute or record solar), the supply stack
   has shifted and the model may need recalibration.

---


- **Reference curve ≠ tradable forward.** The {ROLLING_WINDOW_DAYS}-day
  trailing average used here is backward-looking and not itself tradable. In
  production, the benchmark must be an actual EEX/EPEX forward price. The
  signal quality metrics reported above would differ (likely lower accuracy,
  smaller edge) against a real forward.
- **No transaction costs modelled.** Real bid-ask spreads on prompt-month DE
  base are ~0.10–0.50 EUR/MWh.
- **No volume modelling.** We assume flat 1 MW positions; real trading
  requires volume-weighted sizing based on portfolio exposure and market
  liquidity.
- **Model retraining.** The {model.title()} model was trained on data
  through Dec 2023. Periodic retraining (e.g., monthly walk-forward) is
  essential for production use.
- **Single market.** This view covers DE/LU only. Cross-border flows and
  neighbouring zone prices may affect prompt positioning in practice.
"""