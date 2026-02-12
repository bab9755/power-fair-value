
from typing import Optional

import pandas as pd

from ..domain.patterns import TimeSeriesPatterns


class TimeSeriesTools:

    def __init__(self):

        self.patterns = TimeSeriesPatterns()

    def fill_timestamp_gaps(
        self, df: pd.DataFrame, max_gap_hours: int = 24, method: str = "linear"
    ) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp")

        df_indexed = df.set_index("timestamp")
        ts_series = pd.Series(range(len(df_indexed)), index=df_indexed.index)
        gaps = self.patterns.detect_gaps(ts_series, max_gap_hours=0)

        if len(gaps) == 0:
            return df, {
                "method": "fill_timestamp_gaps",
                "gaps_found": 0,
                "gaps_filled": 0,
                "gaps_too_large": 0,
            }

        start = df["timestamp"].min()
        end = df["timestamp"].max()
        tz = df["timestamp"].dt.tz
        full_range = pd.date_range(start=start, end=end, freq="h", tz=tz)

        original_len = len(df)

        df = df.set_index("timestamp")
        df = df.reindex(full_range)
        df.index.name = "timestamp"

        rows_added = len(df) - original_len

        small_gaps = gaps[gaps["duration_hours"] <= max_gap_hours]
        large_gaps = gaps[gaps["duration_hours"] > max_gap_hours]

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = df[col].interpolate(method=method, limit=max_gap_hours)

        df = df.reset_index()

        summary = {
            "method": "fill_timestamp_gaps",
            "gaps_found": len(gaps),
            "gaps_filled": len(small_gaps),
            "gaps_too_large": len(large_gaps),
            "max_gap_hours": max_gap_hours,
            "interpolation_method": method,
            "rows_added": rows_added,
        }

        if len(large_gaps) > 0:
            summary["large_gaps"] = large_gaps.to_dict("records")

        return df, summary

    def detect_time_series_breaks(
        self, df: pd.DataFrame, threshold_hours: int = 48
    ) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df_indexed = df.set_index("timestamp")
        gaps = self.patterns.detect_gaps(
            df_indexed.index.to_series(), max_gap_hours=threshold_hours
        )

        summary = {
            "method": "detect_time_series_breaks",
            "threshold_hours": threshold_hours,
            "breaks_found": len(gaps),
        }

        if len(gaps) > 0:
            summary["breaks"] = gaps.to_dict("records")
            summary["recommendation"] = (
                f"Found {len(gaps)} breaks > {threshold_hours}h. "
                "Consider manual review or splitting into separate training periods."
            )

        return df, summary

    def validate_hourly_frequency(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp")
        time_diffs = df["timestamp"].diff()

        expected_freq = pd.Timedelta(hours=1)

        exact_hourly = (time_diffs == expected_freq).sum()
        less_than_hourly = (
            (time_diffs < expected_freq) & (time_diffs != pd.Timedelta(0))
        ).sum()
        more_than_hourly = (time_diffs > expected_freq).sum()

        duplicates = df["timestamp"].duplicated().sum()

        issues = []
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps")
        if less_than_hourly > 0:
            issues.append(f"{less_than_hourly} intervals < 1 hour")
        if more_than_hourly > 0:
            issues.append(f"{more_than_hourly} gaps > 1 hour")

        is_valid = len(issues) == 0

        summary = {
            "method": "validate_hourly_frequency",
            "is_valid": is_valid,
            "total_records": len(df),
            "exact_hourly_intervals": int(exact_hourly),
            "duplicate_timestamps": int(duplicates),
            "intervals_less_than_hour": int(less_than_hourly),
            "intervals_more_than_hour": int(more_than_hourly),
        }

        if issues:
            summary["issues"] = issues

        return df, summary

    def handle_dst_transitions(
        self, df: pd.DataFrame, timezone: str = "Europe/Berlin"
    ) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(
                timezone, ambiguous="infer", nonexistent="shift_forward"
            )

        time_diffs = df["timestamp"].diff()

        spring_gaps = time_diffs[time_diffs == pd.Timedelta(hours=2)]

        duplicates = df["timestamp"].duplicated(keep=False)
        fall_duplicates = df[duplicates]

        if len(fall_duplicates) > 0:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df = df.groupby("timestamp", as_index=False)[numeric_cols].mean()

        summary = {
            "method": "handle_dst_transitions",
            "timezone": timezone,
            "spring_gaps_found": len(spring_gaps),
            "fall_duplicates_found": len(fall_duplicates),
            "duplicates_averaged": len(fall_duplicates) > 0,
        }

        return df, summary

    def reindex_to_hourly(
        self, df: pd.DataFrame, fill_method: Optional[str] = None
    ) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        original_len = len(df)

        start = df["timestamp"].min()
        end = df["timestamp"].max()
        tz = df["timestamp"].dt.tz
        full_range = pd.date_range(start=start, end=end, freq="h", tz=tz)

        df = df.set_index("timestamp")
        df = df.reindex(full_range)
        df.index.name = "timestamp"

        new_rows = len(df) - original_len

        if fill_method == "ffill":
            df = df.fillna(method="ffill")
        elif fill_method == "bfill":
            df = df.fillna(method="bfill")
        elif fill_method == "interpolate":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

        df = df.reset_index()

        summary = {
            "method": "reindex_to_hourly",
            "original_length": original_len,
            "new_length": len(df),
            "rows_added": new_rows,
            "fill_method": fill_method,
        }

        return df, summary
