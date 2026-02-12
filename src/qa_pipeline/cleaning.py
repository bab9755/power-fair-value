
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytz
from dateutil import tz

from .config import QAConfig


class CleaningLog:

    def __init__(self):
        self.actions: list[dict[str, Any]] = []

    def add_action(
        self,
        action: str,
        action_type: str,
        params: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:

        self.actions.append(
            {
                "action": action,
                "type": action_type,
                "params": params,
                "before": before,
                "after": after,
            }
        )

    def to_dict(self) -> list[dict[str, Any]]:

        return self.actions


def clean_dataset(
    df: pd.DataFrame, config: QAConfig
) -> tuple[pd.DataFrame, CleaningLog]:

    log = CleaningLog()
    df = df.copy()

    df, log = _clean_timestamps(df, config, log)

    df, log = _handle_missing_timestamps(df, config, log)

    df, log = _coerce_numeric_types(df, config, log)

    df, log = _clean_column_values(df, config, log)

    return df, log


def _clean_timestamps(
    df: pd.DataFrame, config: QAConfig, log: CleaningLog
) -> tuple[pd.DataFrame, CleaningLog]:

    before_rows = len(df)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must have 'timestamp' column")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        log.add_action(
            action="Parse timestamp column to datetime",
            action_type="transform",
            params={"column": "timestamp"},
            before={"dtype": "object"},
            after={"dtype": str(df["timestamp"].dtype)},
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.add_action(
        action="Sort by timestamp ascending",
        action_type="sort",
        params={"column": "timestamp"},
        before={"rows": before_rows},
        after={"rows": len(df)},
    )

    duplicates = df["timestamp"].duplicated()
    dup_count = duplicates.sum()
    if dup_count > 0:
        df = df[~duplicates].reset_index(drop=True)
        log.add_action(
            action="Drop duplicate timestamps",
            action_type="drop",
            params={"duplicates_found": int(dup_count)},
            before={"rows": before_rows},
            after={"rows": len(df)},
        )

    df, log = _apply_timezone_policy(df, config, log)

    return df, log


def _apply_timezone_policy(
    df: pd.DataFrame, config: QAConfig, log: CleaningLog
) -> tuple[pd.DataFrame, CleaningLog]:

    ts_col = df["timestamp"]

    if config.timezone_policy == "europe_berlin":
        berlin_tz = pytz.timezone("Europe/Berlin")

        if ts_col.dt.tz is None:
            try:
                df["timestamp"] = ts_col.dt.tz_localize(berlin_tz, ambiguous='NaT')
                
                nat_count = df["timestamp"].isna().sum()
                if nat_count > 0:
                    df["timestamp"] = ts_col.dt.tz_localize(berlin_tz, ambiguous=True)
                    log.add_action(
                        action="Localize timestamps to Europe/Berlin (DST ambiguous times handled)",
                        action_type="transform",
                        params={
                            "timezone": "Europe/Berlin",
                            "ambiguous": "first_occurrence",
                            "dst_ambiguous_count": int(nat_count),
                        },
                        before={"tz": "naive"},
                        after={"tz": "Europe/Berlin"},
                    )
                else:
                    df["timestamp"] = ts_col.dt.tz_localize(berlin_tz, ambiguous="infer")
                    log.add_action(
                        action="Localize timestamps to Europe/Berlin",
                        action_type="transform",
                        params={"timezone": "Europe/Berlin", "ambiguous": "infer"},
                        before={"tz": "naive"},
                        after={"tz": "Europe/Berlin"},
                    )
            except Exception as e:
                df["timestamp"] = ts_col.dt.tz_localize(berlin_tz, ambiguous=True)
                log.add_action(
                    action="Localize timestamps to Europe/Berlin (fallback to first occurrence)",
                    action_type="transform",
                    params={
                        "timezone": "Europe/Berlin",
                        "ambiguous": True,
                        "note": f"Fallback due to: {str(e)}",
                    },
                    before={"tz": "naive"},
                    after={"tz": "Europe/Berlin"},
                )
        else:
            if str(ts_col.dt.tz) != "Europe/Berlin":
                df["timestamp"] = ts_col.dt.tz_convert(berlin_tz)
                log.add_action(
                    action="Convert timestamps to Europe/Berlin",
                    action_type="transform",
                    params={"from_tz": str(ts_col.dt.tz), "to_tz": "Europe/Berlin"},
                    before={"tz": str(ts_col.dt.tz)},
                    after={"tz": "Europe/Berlin"},
                )

    elif config.timezone_policy == "utc":
        if ts_col.dt.tz is None:
            berlin_tz = pytz.timezone("Europe/Berlin")
            df["timestamp"] = (
                ts_col.dt.tz_localize(berlin_tz, ambiguous="infer").dt.tz_convert(
                    pytz.UTC
                )
            )
            log.add_action(
                action="Localize to Europe/Berlin then convert to UTC",
                action_type="transform",
                params={"assume_tz": "Europe/Berlin", "target_tz": "UTC"},
                before={"tz": "naive"},
                after={"tz": "UTC"},
            )
        else:
            df["timestamp"] = ts_col.dt.tz_convert(pytz.UTC)
            log.add_action(
                action="Convert timestamps to UTC",
                action_type="transform",
                params={"from_tz": str(ts_col.dt.tz), "to_tz": "UTC"},
                before={"tz": str(ts_col.dt.tz)},
                after={"tz": "UTC"},
            )

    return df, log


def _handle_missing_timestamps(
    df: pd.DataFrame, config: QAConfig, log: CleaningLog
) -> tuple[pd.DataFrame, CleaningLog]:

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()

    expected_range = pd.date_range(
        start=min_ts, end=max_ts, freq=config.frequency, tz=df["timestamp"].dt.tz
    )

    actual_timestamps = set(df["timestamp"])
    missing_timestamps = sorted(set(expected_range) - actual_timestamps)

    if missing_timestamps:
        dst_spring_gaps = []
        other_gaps = []

        for ts in missing_timestamps:
            if ts.month == 3 and ts.hour == 2 and 24 <= ts.day <= 31:
                dst_spring_gaps.append(ts)
            else:
                other_gaps.append(ts)

        if config.dst_policy == "allow_missing_spring_hour":
            if dst_spring_gaps:
                log.add_action(
                    action="Identified DST spring-forward gaps (expected)",
                    action_type="identify",
                    params={
                        "policy": "allow_missing_spring_hour",
                        "gaps": [str(ts) for ts in dst_spring_gaps],
                    },
                    before={"missing_count": len(missing_timestamps)},
                    after={"expected_gaps": len(dst_spring_gaps)},
                )

            if other_gaps:
                log.add_action(
                    action="Identified unexpected missing timestamps",
                    action_type="identify",
                    params={
                        "count": len(other_gaps),
                        "first_10": [str(ts) for ts in other_gaps[:10]],
                    },
                    before={"rows": len(df)},
                    after={"unexpected_gaps": len(other_gaps)},
                )

        elif config.dst_policy == "reindex_and_impute":
            df_reindexed = df.set_index("timestamp").reindex(expected_range)
            rows_added = len(df_reindexed) - len(df)

            df_reindexed = df_reindexed.ffill()

            df = df_reindexed.reset_index().rename(columns={"index": "timestamp"})

            log.add_action(
                action="Reindex to full hourly range and impute missing values",
                action_type="impute",
                params={
                    "method": "forward_fill",
                    "gaps_filled": len(missing_timestamps),
                    "dst_gaps": len(dst_spring_gaps),
                    "other_gaps": len(other_gaps),
                },
                before={"rows": len(df) - rows_added},
                after={"rows": len(df)},
            )

    return df, log


def _coerce_numeric_types(
    df: pd.DataFrame, config: QAConfig, log: CleaningLog
) -> tuple[pd.DataFrame, CleaningLog]:

    numeric_columns = [col for col in df.columns if col != "timestamp"]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            invalid_count = df[col].isna().sum() - (
                df[col].isna().sum() if col in df.columns else 0
            )

            log.add_action(
                action=f"Coerce column '{col}' to numeric",
                action_type="transform",
                params={"column": col, "errors": "coerce"},
                before={"dtype": str(before_dtype)},
                after={"dtype": str(df[col].dtype), "invalid_coerced": int(invalid_count)},
            )

    return df, log


def _clean_column_values(
    df: pd.DataFrame, config: QAConfig, log: CleaningLog
) -> tuple[pd.DataFrame, CleaningLog]:

    numeric_columns = [col for col in df.columns if col != "timestamp"]

    for col in numeric_columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            log.add_action(
                action=f"Replace inf values with NaN in '{col}'",
                action_type="transform",
                params={"column": col, "inf_count": int(inf_count)},
                before={"inf_count": int(inf_count)},
                after={"inf_count": 0},
            )

    if config.consistency.forbid_negative_generation:
        gen_load_cols = [
            col
            for col in numeric_columns
            if (col.startswith("gen_") or col == "load_mwh")
            and col != "residual_load_mwh"
        ]

        for col in gen_load_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                log.add_action(
                    action=f"Found negative values in '{col}' (flagged, not corrected)",
                    action_type="flag",
                    params={
                        "column": col,
                        "negative_count": int(negative_count),
                        "policy": "forbid_negative_generation=True",
                    },
                    before={"negative_count": int(negative_count)},
                    after={"action_taken": "none (flagged only)"},
                )

    return df, log
