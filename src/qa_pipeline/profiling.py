
from typing import Any

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:

    profile = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }

    if "timestamp" in df.columns:
        ts_col = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            profile["date_range"] = {
                "start": ts_col.min(),
                "end": ts_col.max(),
                "duration_days": (ts_col.max() - ts_col.min()).days,
            }
        else:
            profile["timestamp_type"] = str(ts_col.dtype)

    profile["columns_detail"] = {}
    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                col_profile.update(
                    {
                        "min": float(valid_data.min()),
                        "max": float(valid_data.max()),
                        "mean": float(valid_data.mean()),
                        "median": float(valid_data.median()),
                        "std": float(valid_data.std()),
                        "negative_count": int((valid_data < 0).sum()),
                        "zero_count": int((valid_data == 0).sum()),
                    }
                )

        profile["columns_detail"][col] = col_profile

    return profile


def compare_profiles(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:

    comparison = {
        "row_change": after["shape"]["rows"] - before["shape"]["rows"],
        "column_change": after["shape"]["columns"] - before["shape"]["columns"],
    }

    missing_changes = {}
    for col in set(before["columns_detail"].keys()) & set(
        after["columns_detail"].keys()
    ):
        before_missing = before["columns_detail"][col]["missing_count"]
        after_missing = after["columns_detail"][col]["missing_count"]
        if before_missing != after_missing:
            missing_changes[col] = {
                "before": before_missing,
                "after": after_missing,
                "change": after_missing - before_missing,
            }

    if missing_changes:
        comparison["missing_data_changes"] = missing_changes

    return comparison
