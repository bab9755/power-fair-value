
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..domain.patterns import TimeSeriesPatterns


class MissingDataTools:

    def __init__(self):

        self.patterns = TimeSeriesPatterns()

    def impute_by_correlation(
        self, df: pd.DataFrame, target_column: str, reference_column: str
    ) -> tuple[pd.DataFrame, dict]:

        if target_column not in df.columns:
            return df, {"error": f"Column '{target_column}' not found"}

        if reference_column not in df.columns:
            return df, {"error": f"Reference column '{reference_column}' not found"}

        before_missing = df[target_column].isna().sum()

        target = df[target_column]
        reference = df[reference_column]

        imputed, metadata = self.patterns.impute_by_correlation(
            target, reference, min_correlation=0.5
        )

        df = df.copy()
        df[target_column] = imputed

        after_missing = df[target_column].isna().sum()

        summary = {
            "method": "correlation_regression",
            "target_column": target_column,
            "reference_column": reference_column,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "imputed_count": int(before_missing - after_missing),
            **metadata,
        }

        return df, summary

    def impute_by_seasonal_pattern(
        self, df: pd.DataFrame, column: str, period: str = "weekly"
    ) -> tuple[pd.DataFrame, dict]:

        if column not in df.columns:
            return df, {"error": f"Column '{column}' not found"}

        before_missing = df[column].isna().sum()

        decomposition = self.patterns.decompose_seasonal(df[column], period=period)

        df = df.copy()
        missing_mask = df[column].isna()

        expected = decomposition["trend"] + decomposition["seasonal"]
        df.loc[missing_mask, column] = expected[missing_mask]

        df[column] = df[column].fillna(method="ffill").fillna(method="bfill")

        after_missing = df[column].isna().sum()

        summary = {
            "method": "seasonal_decomposition",
            "column": column,
            "period": period,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "imputed_count": int(before_missing - after_missing),
        }

        return df, summary

    def impute_by_calculation(
        self, df: pd.DataFrame, target_column: str, component_columns: list[str]
    ) -> tuple[pd.DataFrame, dict]:

        if target_column not in df.columns:
            return df, {"error": f"Target column '{target_column}' not found"}

        missing_components = [
            col for col in component_columns if col not in df.columns
        ]
        if missing_components:
            return df, {
                "error": f"Component columns not found: {missing_components}"
            }

        before_missing = df[target_column].isna().sum()

        df = df.copy()

        missing_target = df[target_column].isna()
        components_exist = df[component_columns].notna().all(axis=1)
        can_calculate = missing_target & components_exist

        calculated_values = df.loc[can_calculate, component_columns].sum(axis=1)
        df.loc[can_calculate, target_column] = calculated_values

        after_missing = df[target_column].isna().sum()

        summary = {
            "method": "calculate_from_components",
            "target_column": target_column,
            "component_columns": component_columns,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "calculated_count": int(can_calculate.sum()),
        }

        return df, summary

    def impute_by_interpolation(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "linear",
        limit: Optional[int] = None,
    ) -> tuple[pd.DataFrame, dict]:

        if column not in df.columns:
            return df, {"error": f"Column '{column}' not found"}

        before_missing = df[column].isna().sum()

        df = df.copy()
        df[column] = df[column].interpolate(
            method=method, limit=limit, limit_direction="both"
        )

        after_missing = df[column].isna().sum()

        summary = {
            "method": f"interpolation_{method}",
            "column": column,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "imputed_count": int(before_missing - after_missing),
            "limit": limit,
        }

        return df, summary

    def impute_by_forward_fill(
        self, df: pd.DataFrame, column: str, limit: Optional[int] = None
    ) -> tuple[pd.DataFrame, dict]:

        if column not in df.columns:
            return df, {"error": f"Column '{column}' not found"}

        before_missing = df[column].isna().sum()

        df = df.copy()
        df[column] = df[column].fillna(method="ffill", limit=limit)

        after_missing = df[column].isna().sum()

        summary = {
            "method": "forward_fill",
            "column": column,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "imputed_count": int(before_missing - after_missing),
            "limit": limit,
        }

        return df, summary

    def impute_by_seasonal_naive(
        self, df: pd.DataFrame, column: str, period: str = "weekly"
    ) -> tuple[pd.DataFrame, dict]:

        if column not in df.columns:
            return df, {"error": f"Column '{column}' not found"}

        before_missing = df[column].isna().sum()

        df = df.copy()
        filled = self.patterns.fill_with_seasonal_naive(df[column], period=period)
        df[column] = filled

        after_missing = df[column].isna().sum()

        summary = {
            "method": f"seasonal_naive_{period}",
            "column": column,
            "before_missing": int(before_missing),
            "after_missing": int(after_missing),
            "imputed_count": int(before_missing - after_missing),
        }

        return df, summary
