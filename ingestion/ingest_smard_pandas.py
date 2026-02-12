from __future__ import annotations

from pathlib import Path

import pandas as pd

def _parse_timestamp(start_col: pd.Series) -> pd.Series:
    return pd.to_datetime(start_col.str.strip(), format="%b %d, %Y %I:%M %p")


def _clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(["-", ""], pd.NA)
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def read_smard_csv(filepath: Path) -> pd.DataFrame:

    print(f"  Reading {filepath.name} ...", end=" ")
    df = pd.read_csv(filepath, sep=";", encoding="utf-8")

    df["timestamp"] = _parse_timestamp(df["Start date"])
    df = df.drop(columns=["Start date", "End date"])

    for col in [c for c in df.columns if c != "timestamp"]:
        df[col] = _clean_numeric(df[col])

    value_cols = [c for c in df.columns if c != "timestamp"]
    df = df.dropna(subset=value_cols, how="all")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    print(f"{len(df):,} rows  |  {df['timestamp'].min():%Y-%m-%d} -> {df['timestamp'].max():%Y-%m-%d}")
    return df

def merge_category_files(data_dir: Path, category: str) -> pd.DataFrame | None:

    cat_dir = data_dir / "raw" / "smard" / category
    if not cat_dir.exists():
        print(f"  [skip] directory not found: {cat_dir}")
        return None

    csv_files = sorted(cat_dir.glob("*.csv"))
    if not csv_files:
        print(f"  [skip] no CSVs in {cat_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"  {category.upper()} – {len(csv_files)} file(s)")
    print(f"{'='*60}")

    dfs = [read_smard_csv(f) for f in csv_files]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    print(f"  -> merged: {len(merged):,} rows  |  "
          f"{merged['timestamp'].min():%Y-%m-%d} -> {merged['timestamp'].max():%Y-%m-%d}")
    return merged


MARKET_RENAME = {
    "Germany/Luxembourg [€/MWh] Calculated resolutions": "price_eur_mwh",
    "∅ DE/LU neighbours [€/MWh] Calculated resolutions": "price_neighbors_avg_eur_mwh",
}

CONSUMPTION_RENAME = {
    "grid load [MWh] Calculated resolutions": "load_mwh",
    "Residual load [MWh] Calculated resolutions": "residual_load_mwh",
}

GENERATION_RENAME = {
    "Total [MWh] Calculated resolutions": "gen_total_mwh",
    "Photovoltaics and wind [MWh] Calculated resolutions": "gen_pv_wind_mwh",
    "Wind offshore [MWh] Calculated resolutions": "gen_wind_offshore_mwh",
    "Wind onshore [MWh] Calculated resolutions": "gen_wind_onshore_mwh",
    "Photovoltaics [MWh] Calculated resolutions": "gen_solar_mwh",
    "Other [MWh] Calculated resolutions": "gen_other_mwh",
}


def select_columns(df: pd.DataFrame, category: str) -> pd.DataFrame:
    rename_map = {
        "market": MARKET_RENAME,
        "consumption": CONSUMPTION_RENAME,
        "generation": GENERATION_RENAME,
    }[category]

    available = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=available)

    keep = ["timestamp"] + list(available.values())
    return df[[c for c in keep if c in df.columns]]


def build_unified_dataset(data_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  SMARD Data Ingestion Pipeline")
    print("=" * 60)

    categories = ["market", "consumption", "generation"]
    frames: dict[str, pd.DataFrame] = {}

    for cat in categories:
        raw = merge_category_files(data_dir, cat)
        if raw is not None:
            frames[cat] = select_columns(raw, cat)

    if not frames:
        raise ValueError("No data found in any category!")

    keys = list(frames.keys())
    unified = frames[keys[0]]
    for key in keys[1:]:
        unified = unified.merge(frames[key], on="timestamp", how="outer")

    unified = unified.sort_values("timestamp").reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  UNIFIED DATASET")
    print(f"{'='*60}")
    print(f"  Rows    : {len(unified):,}")
    print(f"  Columns : {list(unified.columns)}")
    print(f"  Range   : {unified['timestamp'].min():%Y-%m-%d %H:%M} -> "
          f"{unified['timestamp'].max():%Y-%m-%d %H:%M}")
    print(f"  Nulls   :\n{unified.isnull().sum().to_string()}")
    print(f"  Memory  : {unified.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return unified


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    unified = build_unified_dataset(data_dir)

    out_dir = data_dir / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)

    out_parquet = out_dir / "smard_unified.parquet"
    unified.to_parquet(out_parquet, index=False)
    print(f"\n  Saved parquet -> {out_parquet}")

    out_csv = out_dir / "smard_unified.csv"
    unified.to_csv(out_csv, index=False)
    print(f"  Saved CSV     -> {out_csv}")

    print(f"\n  First rows:\n{unified.head().to_string()}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
