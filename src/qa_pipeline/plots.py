
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def generate_all_plots(
    df: pd.DataFrame, artifacts_dir: Path
) -> list[str]:

    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plot_paths = []

    try:
        path = plot_missingness_over_time(df, plots_dir / "missingness_over_time.png")
        plot_paths.append(str(path))
    except Exception as e:
        print(f"Failed to generate missingness plot: {e}")

    try:
        path = plot_price_distribution(df, plots_dir / "price_distribution.png")
        plot_paths.append(str(path))
    except Exception as e:
        print(f"Failed to generate price distribution plot: {e}")

    try:
        path = plot_hourly_seasonality(df, plots_dir / "hourly_seasonality.png")
        plot_paths.append(str(path))
    except Exception as e:
        print(f"Failed to generate hourly seasonality plot: {e}")

    try:
        path = plot_correlation_heatmap(df, plots_dir / "correlation_heatmap.png")
        plot_paths.append(str(path))
    except Exception as e:
        print(f"Failed to generate correlation heatmap: {e}")

    return plot_paths


def plot_missingness_over_time(df: pd.DataFrame, output_path: Path) -> Path:

    fig, ax = plt.subplots(figsize=(14, 6))

    df_temp = df.copy()
    df_temp["year_month"] = df_temp["timestamp"].dt.to_period("M")

    numeric_cols = [col for col in df.columns if col not in ["timestamp", "year_month"]]

    monthly_missing = (
        df_temp.groupby("year_month")[numeric_cols]
        .apply(lambda x: x.isna().sum())
        .reset_index()
    )
    monthly_missing["year_month"] = monthly_missing["year_month"].astype(str)

    cols_with_missing = [
        col for col in numeric_cols if monthly_missing[col].sum() > 0
    ]

    if cols_with_missing:
        for col in cols_with_missing[:5]:
            ax.plot(monthly_missing["year_month"], monthly_missing[col], marker="o", label=col)

        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Missing Count")
        ax.set_title("Missing Data Over Time (Top 5 Columns)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No missing data detected", ha="center", va="center", fontsize=14)
        ax.set_title("Missing Data Over Time")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_price_distribution(df: pd.DataFrame, output_path: Path) -> Path:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    price = df["price_eur_mwh"].dropna()

    axes[0].hist(price, bins=100, edgecolor="black", alpha=0.7)
    axes[0].axvline(price.median(), color="red", linestyle="--", label=f"Median: {price.median():.2f}")
    axes[0].axvline(price.mean(), color="blue", linestyle="--", label=f"Mean: {price.mean():.2f}")
    axes[0].set_xlabel("Price (EUR/MWh)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Price Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(price, vert=True)
    axes[1].set_ylabel("Price (EUR/MWh)")
    axes[1].set_title("Price Box Plot")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_hourly_seasonality(df: pd.DataFrame, output_path: Path) -> Path:

    fig, ax = plt.subplots(figsize=(12, 6))

    df_temp = df.copy()
    df_temp["hour"] = df_temp["timestamp"].dt.hour

    hourly_stats = df_temp.groupby("hour")["price_eur_mwh"].agg(["mean", "std", "median"])

    ax.plot(hourly_stats.index, hourly_stats["mean"], marker="o", linewidth=2, label="Mean")
    ax.plot(hourly_stats.index, hourly_stats["median"], marker="s", linewidth=2, label="Median")
    ax.fill_between(
        hourly_stats.index,
        hourly_stats["mean"] - hourly_stats["std"],
        hourly_stats["mean"] + hourly_stats["std"],
        alpha=0.3,
        label="±1 Std Dev",
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Hourly Price Seasonality")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> Path:

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 15:
        priority_cols = [
            "price_eur_mwh",
            "load_mwh",
            "residual_load_mwh",
            "gen_pv_wind_mwh",
            "gen_solar_mwh",
            "gen_wind_onshore_mwh",
            "gen_wind_offshore_mwh",
        ]
        numeric_cols = [col for col in priority_cols if col in numeric_cols]

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
