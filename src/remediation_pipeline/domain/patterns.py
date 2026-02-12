
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class TimeSeriesPatterns:

    @staticmethod
    def decompose_seasonal(
        series: pd.Series, period: str = "weekly"
    ) -> dict:

        if series.isna().all():
            return {
                "trend": series,
                "seasonal": pd.Series(0, index=series.index),
                "residual": pd.Series(0, index=series.index),
            }

        period_map = {
            "hourly": 24,
            "daily": 24,
            "weekly": 24 * 7,
        }
        freq = period_map.get(period, 24)

        trend = series.rolling(window=freq, center=True, min_periods=1).mean()

        detrended = series - trend

        if not detrended.index.empty and isinstance(detrended.index, pd.DatetimeIndex):
            if period == "weekly":
                seasonal_pattern = (
                    detrended.groupby([detrended.index.dayofweek, detrended.index.hour])
                    .mean()
                )
                seasonal = detrended.index.map(
                    lambda x: seasonal_pattern.get((x.dayofweek, x.hour), 0)
                )
            elif period == "daily":
                seasonal_pattern = detrended.groupby(detrended.index.hour).mean()
                seasonal = detrended.index.map(lambda x: seasonal_pattern.get(x.hour, 0))
            else:
                seasonal = pd.Series(0, index=series.index)

            seasonal = pd.Series(seasonal, index=series.index)
        else:
            seasonal = pd.Series(0, index=series.index)

        residual = series - trend - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }

    @staticmethod
    def extract_weekly_pattern(series: pd.Series) -> pd.DataFrame:

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")

        df = pd.DataFrame({"value": series})
        df["day_of_week"] = df.index.dayofweek
        df["hour_of_day"] = df.index.hour
        df["hour_of_week"] = df["day_of_week"] * 24 + df["hour_of_day"]

        pattern = df.groupby("hour_of_week")["value"].mean()
        return pattern.to_frame()

    @staticmethod
    def fill_with_seasonal_naive(
        series: pd.Series, period: str = "weekly"
    ) -> pd.Series:

        filled = series.copy()

        if period == "weekly":
            lag = 24 * 7
        elif period == "daily":
            lag = 24
        else:
            raise ValueError(f"Unsupported period: {period}")

        missing_mask = filled.isna()
        filled[missing_mask] = filled.shift(lag)[missing_mask]

        filled[missing_mask] = filled.shift(-lag)[missing_mask]

        return filled

    @staticmethod
    def detect_gaps(series: pd.Series, max_gap_hours: int = 24) -> pd.DataFrame:

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")

        time_diffs = series.index.to_series().diff()

        expected_freq = pd.Timedelta(hours=1)

        gaps = time_diffs[time_diffs > expected_freq]

        if len(gaps) == 0:
            return pd.DataFrame(
                columns=["start", "end", "duration_hours", "size"]
            )

        gap_info = []
        for idx, gap_duration in gaps.items():
            hours = gap_duration.total_seconds() / 3600
            if hours <= max_gap_hours:
                continue

            gap_info.append(
                {
                    "start": series.index[series.index.get_loc(idx) - 1],
                    "end": idx,
                    "duration_hours": hours,
                    "size": int(hours),
                }
            )

        return pd.DataFrame(gap_info)

    @staticmethod
    def interpolate_with_constraint(
        series: pd.Series,
        method: str = "linear",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> pd.Series:

        interpolated = series.interpolate(method=method, limit_direction="both")

        if min_value is not None:
            interpolated = interpolated.clip(lower=min_value)
        if max_value is not None:
            interpolated = interpolated.clip(upper=max_value)

        return interpolated

    @staticmethod
    def impute_by_correlation(
        target: pd.Series, reference: pd.Series, min_correlation: float = 0.5
    ) -> tuple[pd.Series, dict]:

        common_mask = target.notna() & reference.notna()
        common_target = target[common_mask]
        common_reference = reference[common_mask]

        if len(common_target) < 10:
            return target, {"error": "Insufficient data for correlation"}

        correlation = common_target.corr(common_reference)

        if abs(correlation) < min_correlation:
            return target, {
                "error": f"Correlation too low: {correlation:.3f}",
                "correlation": float(correlation),
            }

        X = common_reference.values.reshape(-1, 1)
        y = common_target.values
        model = LinearRegression()
        model.fit(X, y)

        imputed = target.copy()
        missing_mask = target.isna() & reference.notna()
        if missing_mask.sum() > 0:
            X_missing = reference[missing_mask].values.reshape(-1, 1)
            predictions = model.predict(X_missing)
            imputed[missing_mask] = predictions

        metadata = {
            "correlation": float(correlation),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "imputed_count": int(missing_mask.sum()),
        }

        return imputed, metadata
