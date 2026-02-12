
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

PEAK_HOURS = range(9, 21)
CET_TZ = "Europe/Berlin"

def read_smard_market_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    start_col_candidates = [c for c in df.columns if c.lower() in ("start date", "startdate", "start")]
    if not start_col_candidates:
        raise ValueError(f"Could not find 'Start date' column in {path}. Columns: {df.columns.tolist()}")
    start_col = start_col_candidates[0]

    de_lu_candidates = [c for c in df.columns if c.startswith("Germany/Luxembourg") and "€/MWh" in c]
    if not de_lu_candidates:
        raise ValueError(
            f"Could not find DE/LU price column in {path}. "
            f"Expected 'Germany/Luxembourg [€/MWh] ...'. Columns: {df.columns.tolist()}"
        )
    price_col = de_lu_candidates[0]

    out = df[[start_col, price_col]].copy()
    out.rename(columns={start_col: "timestamp", price_col: "price_eur_mwh"}, inplace=True)

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
    out["price_eur_mwh"] = (
        out["price_eur_mwh"].astype(str).str.strip().replace({"-": None, "": None})
    )
    out["price_eur_mwh"] = pd.to_numeric(out["price_eur_mwh"], errors="coerce")
    out = out.dropna(subset=["timestamp", "price_eur_mwh"])
    return out


def read_training_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "price_eur_mwh"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(CET_TZ)
    return df[["timestamp", "price_eur_mwh"]].copy()


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert(CET_TZ)

    pred_cols = [c for c in df.columns if c.startswith("y_pred__")]
    keep = ["timestamp", "y_true"] + pred_cols
    return df[[c for c in keep if c in df.columns]].copy()

def _assign_delivery_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delivery_hour"] = df["timestamp"].dt.hour + 1
    df["date"] = df["timestamp"].dt.date
    return df


def build_blocks_for_column(
    df: pd.DataFrame,
    price_col: str,
    suffix: str,
) -> pd.DataFrame:
    df = _assign_delivery_info(df)
    sub = df[["date", "delivery_hour", price_col]].dropna(subset=[price_col])

    base = sub.groupby("date")[price_col].mean().rename(f"base_{suffix}")

    peak_mask = sub["delivery_hour"].isin(PEAK_HOURS)
    peak = sub.loc[peak_mask].groupby("date")[price_col].mean().rename(f"peak_{suffix}")

    offpeak = sub.loc[~peak_mask].groupby("date")[price_col].mean().rename(f"offpeak_{suffix}")

    return pd.concat([base, peak, offpeak], axis=1)


def build_qa_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = _assign_delivery_info(df)

    n_hours = df.groupby("date").size().rename("n_hours")
    peak_mask = df["delivery_hour"].isin(PEAK_HOURS)
    n_peak = df.loc[peak_mask].groupby("date").size().rename("n_peak_hours")

    qa = pd.concat([n_hours, n_peak], axis=1).fillna(0).astype(int)
    qa["qa_flag_missing_hours"] = qa["n_hours"] < 23
    return qa

def build_actual_blocks(
    training_ready_path: Path,
) -> pd.DataFrame:
    df = read_training_ready(training_ready_path)
    blocks = build_blocks_for_column(df, "price_eur_mwh", "actual")
    qa = build_qa_counts(df)
    return blocks.join(qa, how="left").reset_index()


def build_forecast_blocks(
    predictions_path: Path,
) -> pd.DataFrame:
    df = read_predictions(predictions_path)

    df_actual = df[["timestamp", "y_true"]].rename(columns={"y_true": "price_eur_mwh"})
    actual_blocks = build_blocks_for_column(df_actual, "price_eur_mwh", "actual_test")

    pred_cols = [c for c in df.columns if c.startswith("y_pred__")]
    model_blocks = []
    for col in pred_cols:
        model_name = col.replace("y_pred__", "").lower()
        blocks = build_blocks_for_column(df, col, model_name)
        model_blocks.append(blocks)

    all_blocks = pd.concat([actual_blocks] + model_blocks, axis=1)
    return all_blocks.reset_index()


def build_combined_blocks(
    training_ready_path: Path,
    predictions_path: Path,
) -> pd.DataFrame:
    actual = build_actual_blocks(training_ready_path)

    forecast = build_forecast_blocks(predictions_path)

    combined = actual.merge(forecast, on="date", how="left", suffixes=("", "_fcst"))

    pred_cols = [c for c in forecast.columns if c.startswith("base_") and c != "base_actual_test"]
    for col in pred_cols:
        model_suffix = col.replace("base_", "")
        if f"peak_{model_suffix}" in combined.columns:
            combined[f"spread_base_{model_suffix}"] = (
                combined[f"base_{model_suffix}"] - combined["base_actual"]
            )
            combined[f"spread_peak_{model_suffix}"] = (
                combined[f"peak_{model_suffix}"] - combined["peak_actual"]
            )

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)

    return combined


def build_raw_smard_blocks(
    input_paths: List[Path],
) -> pd.DataFrame:
    parts = []
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(read_smard_market_csv(p))

    hourly = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    blocks = build_blocks_for_column(hourly, "price_eur_mwh", "actual")
    qa = build_qa_counts(hourly)
    daily = blocks.join(qa, how="left").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build daily Baseload / Peakload block prices for DA-to-curve analysis."
    )
    ap.add_argument(
        "--mode",
        choices=["raw", "pipeline", "combined"],
        default="combined",
        help=(
            "raw: read raw SMARD CSVs; "
            "pipeline: read training_ready.csv only; "
            "combined: read training_ready.csv + predictions.csv (default)"
        ),
    )
    ap.add_argument(
        "--training-ready",
        default="data/processed/training_ready.csv",
        help="Path to cleaned training_ready.csv (for pipeline/combined modes).",
    )
    ap.add_argument(
        "--predictions",
        default="outputs/predictions.csv",
        help="Path to predictions.csv from training (for combined mode).",
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        help="Raw SMARD CSV files (for raw mode only).",
    )
    ap.add_argument(
        "--out",
        default="data/processed/daily_blocks.csv",
        help="Output CSV path for daily blocks.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent

    if args.mode == "raw":
        if not args.inputs:
            raise ValueError("--inputs required in raw mode")
        daily = build_raw_smard_blocks([Path(p) for p in args.inputs])

    elif args.mode == "pipeline":
        tr_path = root / args.training_ready
        daily = build_actual_blocks(tr_path)
        daily["date"] = pd.to_datetime(daily["date"])

    elif args.mode == "combined":
        tr_path = root / args.training_ready
        pred_path = root / args.predictions
        if not tr_path.exists():
            raise FileNotFoundError(f"training_ready not found: {tr_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions not found: {pred_path}")
        daily = build_combined_blocks(tr_path, pred_path)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)

    print(f"Wrote daily blocks ({args.mode} mode) to: {out_path}")
    print(f"  Rows: {len(daily)}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Columns: {daily.columns.tolist()}")
    print("\nSample (last 5 days):")
    print(daily.tail(5).to_string(index=False))

    spread_cols = [c for c in daily.columns if c.startswith("spread_")]
    if spread_cols:
        test_period = daily.dropna(subset=spread_cols, how="all")
        print(f"\n--- Forecast period ({len(test_period)} days) ---")
        for col in spread_cols:
            vals = test_period[col].dropna()
            print(
                f"  {col:40s}  mean={vals.mean():+.2f}  std={vals.std():.2f}  "
                f"min={vals.min():+.2f}  max={vals.max():+.2f}"
            )


if __name__ == "__main__":
    main()
