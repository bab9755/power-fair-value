
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PriceContext:

    hour: int
    month: int
    renewable_gen: float
    load: float
    price: float


class ElectricityMarketDomain:

    def __init__(self):

        self.typical_daily_load_pattern = {
            0: 0.75,
            1: 0.70,
            2: 0.68,
            3: 0.67,
            4: 0.68,
            5: 0.72,
            6: 0.82,
            7: 0.91,
            8: 0.96,
            9: 0.98,
            10: 1.00,
            11: 0.99,
            12: 0.97,
            13: 0.95,
            14: 0.94,
            15: 0.93,
            16: 0.94,
            17: 0.98,
            18: 1.00,
            19: 0.99,
            20: 0.96,
            21: 0.92,
            22: 0.87,
            23: 0.80,
        }

        self.seasonal_load_multipliers = {
            1: 1.15,
            2: 1.12,
            3: 1.05,
            4: 0.95,
            5: 0.88,
            6: 0.85,
            7: 0.87,
            8: 0.88,
            9: 0.92,
            10: 0.98,
            11: 1.08,
            12: 1.13,
        }

    def is_negative_price_valid(
        self, hour: int, renewable_gen: float, load: float, month: int
    ) -> tuple[bool, str]:

        renewable_load_ratio = renewable_gen / load if load > 0 else 0

        if renewable_load_ratio > 0.8:
            return (
                True,
                f"High renewable penetration ({renewable_load_ratio:.1%}) can cause negative prices",
            )

        if hour in range(0, 6) and renewable_gen > load * 0.6:
            return (
                True,
                f"Nighttime with excess renewable generation (hour={hour}, ratio={renewable_load_ratio:.1%})",
            )

        if month in [4, 5, 9, 10] and renewable_load_ratio > 0.7:
            return (
                True,
                f"Shoulder season with high renewables (month={month}, ratio={renewable_load_ratio:.1%})",
            )

        return (False, "Negative price without clear renewable surplus")

    def get_expected_solar_range(
        self, hour: int, month: int
    ) -> tuple[float, float, str]:

        daylight_hours = self._get_daylight_hours(month)
        sunrise, sunset = daylight_hours

        if hour < sunrise or hour >= sunset:
            return (0.0, 0.0, f"Nighttime (sunrise={sunrise}, sunset={sunset})")

        if sunrise <= hour < sunset:
            hours_from_sunrise = hour - sunrise
            total_daylight = sunset - sunrise
            progress = hours_from_sunrise / total_daylight

            intensity = np.sin(progress * np.pi)

            seasonal_multiplier = self._get_solar_seasonal_multiplier(month)

            return (
                intensity * seasonal_multiplier * 0.5,  
                intensity * seasonal_multiplier * 1.0,  
                f"Daytime hour {hour} (intensity={intensity:.2f}, season_mult={seasonal_multiplier:.2f})",
            )

        return (0.0, 0.0, "Outside daylight hours")

    def _get_daylight_hours(self, month: int) -> tuple[int, int]:

        daylight_patterns = {
            1: (8, 17),
            2: (7, 18),
            3: (6, 19),
            4: (6, 20),
            5: (5, 21),
            6: (4, 22),
            7: (5, 21),
            8: (6, 21),
            9: (7, 19),
            10: (7, 18),
            11: (7, 17),
            12: (8, 16),
        }
        return daylight_patterns[month]

    def _get_solar_seasonal_multiplier(self, month: int) -> float:

        seasonal_solar = {
            1: 0.3,
            2: 0.4,
            3: 0.6,
            4: 0.8,
            5: 0.95,
            6: 1.0,
            7: 1.0,
            8: 0.95,
            9: 0.8,
            10: 0.6,
            11: 0.4,
            12: 0.3,
        }
        return seasonal_solar[month]

    def get_expected_load_pattern(self, hour: int, month: int) -> float:

        hourly = self.typical_daily_load_pattern[hour]
        seasonal = self.seasonal_load_multipliers[month]
        return hourly * seasonal

    def validate_generation_consistency(
        self, row: pd.Series
    ) -> tuple[bool, Optional[str]]:

        errors = []

        if all(
            col in row
            for col in [
                "gen_pv_wind_mwh",
                "gen_solar_mwh",
                "gen_wind_offshore_mwh",
                "gen_wind_onshore_mwh",
            ]
        ):
            pv_wind = row["gen_pv_wind_mwh"]
            components = (
                row["gen_solar_mwh"]
                + row["gen_wind_offshore_mwh"]
                + row["gen_wind_onshore_mwh"]
            )

            if pd.notna(pv_wind) and pd.notna(components):
                diff_pct = abs(pv_wind - components) / max(pv_wind, 1.0) * 100
                if diff_pct > 2.0:
                    errors.append(
                        f"gen_pv_wind ({pv_wind:.1f}) != sum of components ({components:.1f}), diff={diff_pct:.1f}%"
                    )

        if all(col in row for col in ["gen_total_mwh", "gen_pv_wind_mwh", "gen_other_mwh"]):
            total = row["gen_total_mwh"]
            pv_wind = row["gen_pv_wind_mwh"]
            other = row["gen_other_mwh"]

            if all(pd.notna([total, pv_wind, other])):
                expected = pv_wind + other
                diff_pct = abs(total - expected) / max(total, 1.0) * 100
                if diff_pct > 2.0:
                    errors.append(
                        f"gen_total ({total:.1f}) != pv_wind + other ({expected:.1f}), diff={diff_pct:.1f}%"
                    )

        gen_columns = [
            col for col in row.index if col.startswith("gen_") and col.endswith("_mwh")
        ]
        for col in gen_columns:
            if pd.notna(row[col]) and row[col] < -0.1:
                errors.append(f"{col} is negative: {row[col]:.2f}")

        if errors:
            return (False, "; ".join(errors))

        return (True, None)

    def correlate_price_with_fundamentals(self, df: pd.DataFrame) -> dict:

        insights = {}

        if "price_eur_mwh" not in df.columns:
            return {"error": "price_eur_mwh column not found"}

        price = df["price_eur_mwh"].dropna()

        if "load_mwh" in df.columns:
            load = df["load_mwh"].dropna()
            common_idx = price.index.intersection(load.index)
            if len(common_idx) > 10:
                corr = price.loc[common_idx].corr(load.loc[common_idx])
                insights["price_load_correlation"] = float(corr)

        if "gen_pv_wind_mwh" in df.columns:
            renewable = df["gen_pv_wind_mwh"].dropna()
            common_idx = price.index.intersection(renewable.index)
            if len(common_idx) > 10:
                corr = price.loc[common_idx].corr(renewable.loc[common_idx])
                insights["price_renewable_correlation"] = float(corr)

        if "residual_load_mwh" in df.columns:
            residual = df["residual_load_mwh"].dropna()
            common_idx = price.index.intersection(residual.index)
            if len(common_idx) > 10:
                corr = price.loc[common_idx].corr(residual.loc[common_idx])
                insights["price_residual_load_correlation"] = float(corr)

        return insights

    def detect_anomalous_patterns(self, df: pd.DataFrame) -> list[dict]:

        anomalies = []

        if "timestamp" not in df.columns:
            return anomalies

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month

        if "gen_solar_mwh" in df.columns:
            summer_midday = df[
                (df["month"].isin([6, 7, 8]))
                & (df["hour"].isin([11, 12, 13, 14]))
            ]
            zero_solar = summer_midday[
                summer_midday["gen_solar_mwh"].fillna(0) < 0.1
            ]
            if len(zero_solar) > 0:
                anomalies.append(
                    {
                        "type": "suspicious_zero_solar",
                        "count": len(zero_solar),
                        "description": "Zero solar generation during summer midday",
                        "timestamps": zero_solar["timestamp"].head(5).tolist(),
                    }
                )

        if "load_mwh" in df.columns:
            median_load = df["load_mwh"].median()
            extreme_load = df[df["load_mwh"] > median_load * 2]
            if len(extreme_load) > 0:
                anomalies.append(
                    {
                        "type": "extreme_load",
                        "count": len(extreme_load),
                        "description": f"Load exceeds 2x median ({median_load:.1f} MWh)",
                        "max_value": float(extreme_load["load_mwh"].max()),
                    }
                )

        return anomalies
