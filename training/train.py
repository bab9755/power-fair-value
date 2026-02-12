from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.base
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-9, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def rmae(mae_model: float, mae_bench: float) -> float:
    return float(mae_model / mae_bench) if mae_bench > 0 else np.nan


@dataclass
class FeatureConfig:
    price_lags: Tuple[int, ...] = (24, 48, 168)
    roll_windows: Tuple[int, ...] = (24, 168)
    add_hourly_models: bool = True


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def make_lag_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    for lag in cfg.price_lags:
        df[f"price_lag_{lag}"] = df["price_eur_mwh"].shift(lag)

    for w in cfg.roll_windows:
        df[f"price_roll_mean_{w}"] = df["price_eur_mwh"].shift(1).rolling(w).mean()
        df[f"price_roll_median_{w}"] = df["price_eur_mwh"].shift(1).rolling(w).median()
        df[f"price_roll_std_{w}"] = df["price_eur_mwh"].shift(1).rolling(w).std()

    for c in [
        "load_mwh", "residual_load_mwh",
        "gen_total_mwh", "gen_pv_wind_mwh",
        "gen_wind_offshore_mwh", "gen_wind_onshore_mwh",
        "gen_solar_mwh", "gen_other_mwh",
        "price_neighbors_avg_eur_mwh",
    ]:
        if c in df.columns:
            df[f"{c}_lag_24"] = df[c].shift(24)
            df[f"{c}_lag_168"] = df[c].shift(168)

    if "gen_pv_wind_mwh" in df.columns and "load_mwh" in df.columns:
        df["ren_share_load"] = df["gen_pv_wind_mwh"] / (df["load_mwh"] + 1e-6)

    return df


def seasonal_naive(df: pd.DataFrame, horizon_lag: int) -> pd.Series:
    return df["price_eur_mwh"].shift(horizon_lag)
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    ignore = {"timestamp", "price_eur_mwh", "date"}
    return [c for c in df.columns if c not in ignore]

def build_elasticnet() -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=20000, random_state=42))
    ])

def build_lgbm() -> Pipeline:
    if not HAVE_LGBM:
        raise RuntimeError("lightgbm not installed. `pip install lightgbm`")
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        verbose=-1,
    )
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])
    pipe.set_output(transform="pandas")
    return pipe

def fit_predict_hourly(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model: Pipeline,
    feature_cols: List[str],
) -> pd.Series:
    preds = pd.Series(index=df_test.index, dtype=float)
    for h in range(24):
        tr = df_train[df_train["hour"] == h]
        te = df_test[df_test["hour"] == h]
        if len(tr) < 200 or len(te) == 0:
            continue
        Xtr, ytr = tr[feature_cols], tr["price_eur_mwh"]
        Xte = te[feature_cols]
        m = sklearn.base.clone(model)
        m.fit(Xtr, ytr)
        preds.loc[te.index] = m.predict(Xte)
    return preds


def evaluate_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training_ready.csv")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--test_start", default="2024-01-01", help="Chronological split point (inclusive for test).")
    ap.add_argument("--use_lgbm", action="store_true", help="Train LightGBM model (recommended).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    cfg = FeatureConfig()
    df = make_time_features(df)
    df = make_lag_features(df, cfg)

    min_lag = max(cfg.price_lags)
    df = df.iloc[min_lag:].copy()

    test_start = pd.to_datetime(args.test_start, utc=True)
    df_train = df[df["timestamp"] < test_start].copy()
    df_test = df[df["timestamp"] >= test_start].copy()

    if len(df_train) < 5000 or len(df_test) < 1000:
        raise ValueError(f"Split too small. Train={len(df_train)}, Test={len(df_test)}. Adjust --test_start.")

    feature_cols = get_feature_cols(df)

    df_test["y_true"] = df_test["price_eur_mwh"].values
    df_test["y_pred__SeasonalNaive_D1"] = seasonal_naive(df, 24).loc[df_test.index]
    df_test["y_pred__SeasonalNaive_D7"] = seasonal_naive(df, 168).loc[df_test.index]

    en = build_elasticnet()
    df_test["y_pred__ElasticNet"] = fit_predict_hourly(df_train, df_test, en, feature_cols)

    if args.use_lgbm:
        lgbm = build_lgbm()
        df_test["y_pred__LightGBM"] = fit_predict_hourly(df_train, df_test, lgbm, feature_cols)
    else:
        df_test["y_pred__LightGBM"] = np.nan

    models = ["SeasonalNaive_D1", "SeasonalNaive_D7", "ElasticNet"]
    if args.use_lgbm:
        models.append("LightGBM")

    bench_d1 = df_test["y_pred__SeasonalNaive_D1"].values
    bench_d7 = df_test["y_pred__SeasonalNaive_D7"].values
    y = df_test["y_true"].values

    mask_d1 = ~np.isnan(bench_d1)
    mae_d1 = float(mean_absolute_error(y[mask_d1], bench_d1[mask_d1]))
    mask_d7 = ~np.isnan(bench_d7)
    mae_d7 = float(mean_absolute_error(y[mask_d7], bench_d7[mask_d7]))

    metric_rows = []
    for m in models:
        col = f"y_pred__{m}"
        yp = df_test[col].values
        mask = ~np.isnan(yp) & ~np.isnan(y)
        met = evaluate_block(y[mask], yp[mask])
        met["model"] = m
        met["rmae_D1"] = rmae(met["mae"], mae_d1)
        met["rmae_D7"] = rmae(met["mae"], mae_d7)
        metric_rows.append(met)

    metrics = pd.DataFrame(metric_rows).sort_values("mae")
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    hourly_rows = []
    for m in models:
        col = f"y_pred__{m}"
        for h in range(24):
            sub = df_test[df_test["hour"] == h]
            yt = sub["y_true"].values
            yp = sub[col].values
            mask = ~np.isnan(yp) & ~np.isnan(yt)
            if mask.sum() < 50:
                continue
            hourly_rows.append({
                "model": m,
                "hour": h,
                "mae": float(mean_absolute_error(yt[mask], yp[mask])),
                "rmse": rmse(yt[mask], yp[mask]),
                "smape": smape(yt[mask], yp[mask]),
            })
    hourly = pd.DataFrame(hourly_rows)
    hourly_path = os.path.join(args.out_dir, "metrics_by_hour.csv")
    hourly.to_csv(hourly_path, index=False)

    keep_cols = ["timestamp", "y_true"] + [f"y_pred__{m}" for m in models] + ["hour", "dow", "month", "is_weekend"]
    preds_path = os.path.join(args.out_dir, "predictions.csv")
    df_test[keep_cols].to_csv(preds_path, index=False)

    # Plot best-performing model's predictions vs actuals (typically LightGBM)
    try:
        best_model_name = str(metrics.iloc[0]["model"])
        best_col = f"y_pred__{best_model_name}"
        if best_col in df_test.columns:
            plot_df = df_test[["timestamp", "y_true", best_col]].dropna().tail(1000)

            if not plot_df.empty:
                plt.figure(figsize=(12, 4))
                plt.plot(plot_df["timestamp"], plot_df["y_true"], label="Actual", linewidth=1)
                plt.plot(
                    plot_df["timestamp"],
                    plot_df[best_col],
                    label=best_model_name,
                    linewidth=1,
                )
                plt.xlabel("Timestamp")
                plt.ylabel("Price (EUR/MWh)")
                plt.title(f"Test period: actual vs {best_model_name} forecasts")
                plt.legend()
                plt.tight_layout()

                best_plot_path = os.path.join(
                    args.out_dir, f"{best_model_name.lower()}_forecast_vs_actual.png"
                )
                plt.savefig(best_plot_path, dpi=150)
                plt.close()
                print(f"Saved: {best_plot_path}")
    except Exception as e:
        print(f"Warning: failed to generate best model plot: {e}")

    # Additional diagnostic plots focused on LightGBM
    try:
        lg_col = "y_pred__LightGBM"
        if args.use_lgbm and lg_col in df_test.columns:
            # Scatter: actual vs prediction
            valid = df_test[["y_true", lg_col]].dropna()
            if not valid.empty:
                plt.figure(figsize=(5, 5))
                plt.scatter(valid["y_true"], valid[lg_col], s=5, alpha=0.3)
                vmin = float(np.nanmin([valid["y_true"].min(), valid[lg_col].min()]))
                vmax = float(np.nanmax([valid["y_true"].max(), valid[lg_col].max()]))
                plt.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)
                plt.xlabel("Actual price (EUR/MWh)")
                plt.ylabel("LightGBM prediction (EUR/MWh)")
                plt.title("LightGBM: actual vs predicted (test set)")
                plt.tight_layout()
                scatter_path = os.path.join(args.out_dir, "lightgbm_y_true_vs_pred_scatter.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                print(f"Saved: {scatter_path}")

                # Residual histogram
                residuals = valid["y_true"] - valid[lg_col]
                plt.figure(figsize=(8, 4))
                plt.hist(residuals, bins=50, alpha=0.8)
                plt.xlabel("Residual (actual - prediction) [EUR/MWh]")
                plt.ylabel("Count")
                plt.title("LightGBM residual distribution (test set)")
                plt.tight_layout()
                resid_path = os.path.join(args.out_dir, "lightgbm_residual_hist.png")
                plt.savefig(resid_path, dpi=150)
                plt.close()
                print(f"Saved: {resid_path}")

            # MAE by hour-of-day for LightGBM
            if not hourly.empty:
                hourly_lgbm = hourly[hourly["model"] == "LightGBM"]
                if not hourly_lgbm.empty:
                    plt.figure(figsize=(10, 4))
                    plt.bar(hourly_lgbm["hour"], hourly_lgbm["mae"], width=0.8)
                    plt.xlabel("Hour of day")
                    plt.ylabel("MAE (EUR/MWh)")
                    plt.title("LightGBM MAE by delivery hour (test set)")
                    plt.xticks(range(0, 24))
                    plt.tight_layout()
                    by_hour_path = os.path.join(args.out_dir, "lightgbm_mae_by_hour.png")
                    plt.savefig(by_hour_path, dpi=150)
                    plt.close()
                    print(f"Saved: {by_hour_path}")

            # Global feature importance from a single LightGBM fitted on all training data
            if HAVE_LGBM:
                global_lgbm = build_lgbm()
                Xtr_all, ytr_all = df_train[feature_cols], df_train["price_eur_mwh"]
                global_lgbm.fit(Xtr_all, ytr_all)
                booster = global_lgbm.named_steps["model"]
                importances = booster.feature_importances_
                fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
                fi = fi.sort_values("importance", ascending=False)
                top = fi.head(20)
                if not top.empty:
                    plt.figure(figsize=(8, max(4, 0.3 * len(top))))
                    plt.barh(top["feature"][::-1], top["importance"][::-1])
                    plt.xlabel("Importance")
                    plt.title("LightGBM feature importance (global model)")
                    plt.tight_layout()
                    fi_path = os.path.join(args.out_dir, "lightgbm_feature_importance.png")
                    plt.savefig(fi_path, dpi=150)
                    plt.close()
                    print(f"Saved: {fi_path}")
    except Exception as e:
        print(f"Warning: failed to generate LightGBM diagnostic plots: {e}")

    joblib.dump({"feature_cols": feature_cols, "cfg": cfg.__dict__}, os.path.join(args.out_dir, "pipeline_meta.joblib"))

    context_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "context")
    os.makedirs(context_dir, exist_ok=True)
    metrics.to_csv(os.path.join(context_dir, "model_metrics.csv"), index=False)
    hourly.to_csv(os.path.join(context_dir, "model_metrics_by_hour.csv"), index=False)
    df_test[keep_cols].to_csv(os.path.join(context_dir, "model_predictions.csv"), index=False)
    import json as _json
    training_summary = {
        "test_start": str(args.test_start),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "feature_config": cfg.__dict__,
        "models": models,
        "use_lgbm": args.use_lgbm,
        "best_model": metrics.iloc[0]["model"],
        "best_mae": float(metrics.iloc[0]["mae"]),
        "best_rmse": float(metrics.iloc[0]["rmse"]),
        "best_smape": float(metrics.iloc[0]["smape"]),
    }
    with open(os.path.join(context_dir, "training_summary.json"), "w") as _f:
        _json.dump(training_summary, _f, indent=2, default=str)
    print(f"\nContext saved to: {context_dir}/")

    print("\n==============================")
    print("MODEL METRICS (sorted by MAE)")
    print("==============================")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {hourly_path}")
    print(f"Saved: {preds_path}")

    if not HAVE_LGBM and args.use_lgbm:
        print("\nNOTE: LightGBM requested but not installed. Install with: pip install lightgbm")


if __name__ == "__main__":
    main()
