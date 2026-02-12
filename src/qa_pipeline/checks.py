
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from .config import QAConfig

warnings.filterwarnings('ignore', message='Converting to PeriodArray/Index')


class CheckResult:

    def __init__(
        self,
        id: str,
        name: str,
        status: Literal["PASS", "WARN", "FAIL"],
        metrics: dict[str, Any],
        thresholds: dict[str, Any],
        message: str,
        recommendation: str = "",
    ):
        self.id = id
        self.name = name
        self.status = status
        self.metrics = metrics
        self.thresholds = thresholds
        self.message = message
        self.recommendation = recommendation

    def to_dict(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "message": self.message,
            "recommendation": self.recommendation,
        }


def run_all_checks(df: pd.DataFrame, config: QAConfig) -> list[CheckResult]:

    checks = [
        check_schema_and_types(df, config),
        check_time_series_integrity(df, config),
        check_missingness(df, config),
        check_value_sanity(df, config),
        check_cross_field_consistency(df, config),
        check_target_readiness(df, config),
    ]

    return checks


def check_schema_and_types(df: pd.DataFrame, config: QAConfig) -> CheckResult:

    missing_cols = set(config.required_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(config.required_columns)

    timestamp_ok = pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    numeric_cols = [col for col in df.columns if col != "timestamp"]
    non_numeric = [
        col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if missing_cols or not timestamp_ok or non_numeric:
        status = "FAIL"
        message = "Schema validation failed"
    elif extra_cols:
        status = "WARN"
        message = "Schema has extra columns (not required)"
    else:
        status = "PASS"
        message = "Schema and types are valid"

    return CheckResult(
        id="schema_types",
        name="Schema & Types Validation",
        status=status,
        metrics={
            "columns_count": len(df.columns),
            "missing_required": list(missing_cols),
            "extra_columns": list(extra_cols),
            "timestamp_is_datetime": timestamp_ok,
            "non_numeric_columns": non_numeric,
        },
        thresholds={"required_columns": config.required_columns},
        message=message,
        recommendation="Ensure all required columns are present and properly typed"
        if status != "PASS"
        else "",
    )


def check_time_series_integrity(df: pd.DataFrame, config: QAConfig) -> CheckResult:

    ts_col = df["timestamp"]

    is_monotonic = ts_col.is_monotonic_increasing

    dup_count = ts_col.duplicated().sum()

    if len(df) > 1:
        time_diffs = ts_col.diff()[1:]
        expected_delta = pd.Timedelta("1H")

        freq_anomalies = (time_diffs != expected_delta).sum()

        min_ts = ts_col.min()
        max_ts = ts_col.max()
        expected_range = pd.date_range(
            start=min_ts, end=max_ts, freq=config.frequency, tz=ts_col.dt.tz
        )
        missing_count = len(set(expected_range) - set(ts_col))

        missing_timestamps = sorted(set(expected_range) - set(ts_col))
        dst_spring_gaps = sum(
            1
            for ts in missing_timestamps
            if ts.month == 3 and ts.hour == 2 and 24 <= ts.day <= 31
        )
        unexpected_gaps = missing_count - dst_spring_gaps

    else:
        freq_anomalies = 0
        missing_count = 0
        dst_spring_gaps = 0
        unexpected_gaps = 0

    if not is_monotonic or dup_count > 0:
        status = "FAIL"
        message = "Time series integrity violated"
    elif unexpected_gaps > 0:
        if config.dst_policy == "allow_missing_spring_hour":
            status = "FAIL"
            message = "Unexpected missing timestamps found"
        else:
            status = "WARN"
            message = "Missing timestamps found (handled by policy)"
    elif dst_spring_gaps > 0 and config.dst_policy == "allow_missing_spring_hour":
        status = "PASS"
        message = "Time series integrity OK (DST gaps expected)"
    else:
        status = "PASS"
        message = "Time series integrity OK"

    return CheckResult(
        id="time_series_integrity",
        name="Time Series Integrity",
        status=status,
        metrics={
            "is_monotonic_increasing": is_monotonic,
            "duplicate_count": int(dup_count),
            "frequency_anomalies": int(freq_anomalies),
            "missing_timestamps_total": int(missing_count),
            "dst_spring_gaps": int(dst_spring_gaps),
            "unexpected_gaps": int(unexpected_gaps),
        },
        thresholds={
            "expected_frequency": config.frequency,
            "dst_policy": config.dst_policy,
        },
        message=message,
        recommendation="Remove duplicates and ensure monotonic timestamps"
        if status == "FAIL"
        else "",
    )


def check_missingness(df: pd.DataFrame, config: QAConfig) -> CheckResult:

    missing_stats = {}
    fail_cols = []
    warn_cols = []

    for col in df.columns:
        if col == "timestamp":
            continue

        missing_pct = df[col].isna().mean() * 100
        missing_stats[col] = {
            "count": int(df[col].isna().sum()),
            "percent": round(missing_pct, 2),
        }

        if missing_pct > config.missingness.fail_missing_pct:
            fail_cols.append(col)
        elif missing_pct > config.missingness.warn_missing_pct:
            warn_cols.append(col)

    temporal_patterns = {}
    for col in df.columns:
        if col == "timestamp":
            continue

        if df[col].isna().sum() > 0:
            df_temp = df.copy()
            df_temp["year_month"] = df_temp["timestamp"].dt.to_period("M")
            monthly_missing = (
                df_temp.groupby("year_month")[col].apply(lambda x: x.isna().sum())
            )

            high_missing_months = monthly_missing[monthly_missing > 100]
            if len(high_missing_months) > 0:
                top_months_dict = {str(k): int(v) for k, v in high_missing_months.nlargest(3).items()}
                temporal_patterns[col] = {
                    "months_affected": len(high_missing_months),
                    "top_months": top_months_dict,
                }

    if fail_cols:
        status = "FAIL"
        message = f"Excessive missing data in {len(fail_cols)} column(s)"
    elif warn_cols:
        status = "WARN"
        message = f"Elevated missing data in {len(warn_cols)} column(s)"
    else:
        status = "PASS"
        message = "Missing data levels acceptable"

    return CheckResult(
        id="missingness",
        name="Missing Data Analysis",
        status=status,
        metrics={
            "per_column": missing_stats,
            "fail_threshold_exceeded": fail_cols,
            "warn_threshold_exceeded": warn_cols,
            "temporal_clustering": temporal_patterns,
        },
        thresholds={
            "warn_pct": config.missingness.warn_missing_pct,
            "fail_pct": config.missingness.fail_missing_pct,
        },
        message=message,
        recommendation=f"Review columns with high missingness: {', '.join(fail_cols + warn_cols)}"
        if (fail_cols or warn_cols)
        else "",
    )


def check_value_sanity(df: pd.DataFrame, config: QAConfig) -> CheckResult:

    sanity_stats = {}
    issues = []

    if "price_eur_mwh" in df.columns:
        price = df["price_eur_mwh"].dropna()

        if len(price) > 0:
            price_min = price.min()
            price_max = price.max()
            negative_rate = (price < 0).sum() / len(price) * 100

            extreme_threshold = price.quantile(
                config.outliers.price_extreme_percentile / 100
            )
            extreme_spikes = (price > extreme_threshold).sum()

            out_of_bounds = (
                (price < config.outliers.price_hard_min)
                | (price > config.outliers.price_hard_max)
            ).sum()

            sanity_stats["price_eur_mwh"] = {
                "min": float(price_min),
                "max": float(price_max),
                "mean": float(price.mean()),
                "median": float(price.median()),
                "negative_rate_pct": round(negative_rate, 2),
                "extreme_spikes_count": int(extreme_spikes),
                "extreme_threshold": float(extreme_threshold),
                "out_of_bounds_count": int(out_of_bounds),
            }

            if out_of_bounds > 0:
                issues.append(f"Price has {out_of_bounds} values outside hard bounds")

    for col in df.columns:
        if col in ["load_mwh", "residual_load_mwh"] or col.startswith("gen_"):
            values = df[col].dropna()

            if len(values) > 0:
                negative_count = (values < 0).sum()

                sanity_stats[col] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "negative_count": int(negative_count),
                }

                if (
                    col != "residual_load_mwh"
                    and negative_count > 0
                    and config.consistency.forbid_negative_generation
                ):
                    issues.append(f"{col} has {negative_count} negative values")

    if issues:
        status = "WARN"
        message = f"Value sanity issues found: {len(issues)} issue(s)"
    else:
        status = "PASS"
        message = "Value ranges are reasonable"

    return CheckResult(
        id="value_sanity",
        name="Value Sanity Checks",
        status=status,
        metrics=sanity_stats,
        thresholds={
            "price_hard_min": config.outliers.price_hard_min,
            "price_hard_max": config.outliers.price_hard_max,
            "price_extreme_percentile": config.outliers.price_extreme_percentile,
        },
        message=message,
        recommendation="Review flagged columns for data quality issues"
        if issues
        else "",
    )


def check_cross_field_consistency(
    df: pd.DataFrame, config: QAConfig
) -> CheckResult:

    consistency_stats = {}
    issues = []

    renewable_cols = [
        "gen_wind_offshore_mwh",
        "gen_wind_onshore_mwh",
        "gen_solar_mwh",
    ]

    if all(col in df.columns for col in renewable_cols) and "gen_pv_wind_mwh" in df.columns:
        df_check = df.dropna(subset=renewable_cols + ["gen_pv_wind_mwh"])

        if len(df_check) > 0:
            calculated_sum = (
                df_check["gen_wind_offshore_mwh"]
                + df_check["gen_wind_onshore_mwh"]
                + df_check["gen_solar_mwh"]
            )
            reported_sum = df_check["gen_pv_wind_mwh"]

            rel_diff = (
                abs(calculated_sum - reported_sum) / (reported_sum + 1e-6) * 100
            )

            tolerance = config.consistency.renewable_sum_tolerance_pct
            mismatches = (rel_diff > tolerance).sum()
            mismatch_rate = mismatches / len(df_check) * 100

            consistency_stats["renewable_sum"] = {
                "rows_checked": len(df_check),
                "mismatches": int(mismatches),
                "mismatch_rate_pct": round(mismatch_rate, 2),
                "mean_rel_diff_pct": round(rel_diff.mean(), 2),
                "max_rel_diff_pct": round(rel_diff.max(), 2),
            }

            if mismatch_rate > 5:
                issues.append(
                    f"Renewable sum mismatch in {mismatch_rate:.1f}% of rows"
                )

    if "gen_total_mwh" in df.columns and "gen_pv_wind_mwh" in df.columns and "gen_other_mwh" in df.columns:
        df_check = df.dropna(
            subset=["gen_total_mwh", "gen_pv_wind_mwh", "gen_other_mwh"]
        )

        if len(df_check) > 0:
            calculated_total = (
                df_check["gen_pv_wind_mwh"] + df_check["gen_other_mwh"]
            )
            reported_total = df_check["gen_total_mwh"]

            correlation = calculated_total.corr(reported_total)

            rel_diff = (
                abs(calculated_total - reported_total)
                / (reported_total + 1e-6)
                * 100
            )

            consistency_stats["total_gen"] = {
                "rows_checked": len(df_check),
                "correlation": round(correlation, 4),
                "mean_rel_diff_pct": round(rel_diff.mean(), 2),
                "max_rel_diff_pct": round(rel_diff.max(), 2),
            }

            if correlation < 0.95:
                issues.append(
                    f"Low correlation between calculated and reported total generation: {correlation:.3f}"
                )

    if issues:
        status = "WARN"
        message = f"Consistency issues found in cross-field checks"
    else:
        status = "PASS"
        message = "Cross-field consistency OK"

    return CheckResult(
        id="cross_field_consistency",
        name="Cross-Field Consistency",
        status=status,
        metrics=consistency_stats,
        thresholds={
            "renewable_sum_tolerance_pct": config.consistency.renewable_sum_tolerance_pct
        },
        message=message,
        recommendation="Review data source for inconsistent aggregations"
        if issues
        else "",
    )


def check_target_readiness(df: pd.DataFrame, config: QAConfig) -> CheckResult:

    target_col = config.target_column

    if target_col not in df.columns:
        return CheckResult(
            id="target_readiness",
            name="Target Column Readiness",
            status="FAIL",
            metrics={"column_exists": False},
            thresholds={"target_column": target_col},
            message=f"Target column '{target_col}' not found",
            recommendation=f"Ensure '{target_col}' column exists in dataset",
        )

    missing_pct = df[target_col].isna().mean() * 100
    missing_count = df[target_col].isna().sum()

    if missing_pct > config.target_max_missing_pct:
        status = "FAIL"
        message = f"Target column has {missing_pct:.2f}% missing (exceeds {config.target_max_missing_pct}%)"
    else:
        status = "PASS"
        message = f"Target column ready (missing: {missing_pct:.2f}%)"

    return CheckResult(
        id="target_readiness",
        name="Target Column Readiness",
        status=status,
        metrics={
            "column_exists": True,
            "missing_count": int(missing_count),
            "missing_pct": round(missing_pct, 4),
        },
        thresholds={"max_missing_pct": config.target_max_missing_pct},
        message=message,
        recommendation=f"Impute or remove rows with missing {target_col}"
        if status == "FAIL"
        else "",
    )
