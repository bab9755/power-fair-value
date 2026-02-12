
from typing import Any

import pandas as pd

from .config import RemediationConfig
from .issue_parser import Issue, IssueCategory
from .tools.missing_data import MissingDataTools
from .tools.time_series import TimeSeriesTools
from .tools.validation import ValidationTools


class FixStrategy:

    def __init__(
        self,
        config: RemediationConfig,
        missing_tools: MissingDataTools,
        ts_tools: TimeSeriesTools,
        validation_tools: ValidationTools,
    ):

        self.config = config
        self.missing_tools = missing_tools
        self.ts_tools = ts_tools
        self.validation_tools = validation_tools

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        raise NotImplementedError


class ImputeByCorrelationStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        column = issue.metadata.get("column")
        if not column:
            return df, {"error": "No column specified in issue metadata"}

        strategy_config = self.config.missing_data.strategies.get(column, {})
        reference_column = strategy_config.get("reference", "price_eur_mwh")

        df_before = df.copy()
        df_fixed, summary = self.missing_tools.impute_by_correlation(
            df, column, reference_column
        )

        validation = self.validation_tools.validate_fix(df_before, df_fixed, column)

        return df_fixed, {"fix_summary": summary, "validation": validation}


class ImputeByCalculationStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        column = issue.metadata.get("column")
        if not column:
            return df, {"error": "No column specified in issue metadata"}

        strategy_config = self.config.missing_data.strategies.get(column, {})
        components = strategy_config.get("components", [])

        if not components:
            return df, {"error": f"No components specified for column {column}"}

        df_before = df.copy()
        df_fixed, summary = self.missing_tools.impute_by_calculation(
            df, column, components
        )

        validation = self.validation_tools.validate_fix(df_before, df_fixed, column)

        return df_fixed, {"fix_summary": summary, "validation": validation}


class ImputeBySeasonalPatternStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        column = issue.metadata.get("column")
        if not column:
            return df, {"error": "No column specified in issue metadata"}

        strategy_config = self.config.missing_data.strategies.get(column, {})
        period = strategy_config.get("period", "weekly")

        df_before = df.copy()
        df_fixed, summary = self.missing_tools.impute_by_seasonal_pattern(
            df, column, period
        )

        validation = self.validation_tools.validate_fix(df_before, df_fixed, column)

        return df_fixed, {"fix_summary": summary, "validation": validation}


class ImputeByInterpolationStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        column = issue.metadata.get("column")
        if not column:
            return df, {"error": "No column specified in issue metadata"}

        method = self.config.time_series.interpolation_method
        max_gap = self.config.time_series.max_gap_hours

        df_before = df.copy()
        df_fixed, summary = self.missing_tools.impute_by_interpolation(
            df, column, method=method, limit=max_gap
        )

        validation = self.validation_tools.validate_fix(df_before, df_fixed, column)

        return df_fixed, {"fix_summary": summary, "validation": validation}


class ImputeBySeasonalNaiveStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        column = issue.metadata.get("column")
        if not column:
            return df, {"error": "No column specified in issue metadata"}

        strategy_config = self.config.missing_data.strategies.get(column, {})
        period = strategy_config.get("period", "weekly")

        df_before = df.copy()
        df_fixed, summary = self.missing_tools.impute_by_seasonal_naive(
            df, column, period
        )

        validation = self.validation_tools.validate_fix(df_before, df_fixed, column)

        return df_fixed, {"fix_summary": summary, "validation": validation}


class FillTimestampGapsStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        max_gap = self.config.time_series.max_gap_hours
        method = self.config.time_series.interpolation_method

        df_fixed, summary = self.ts_tools.fill_timestamp_gaps(
            df, max_gap_hours=max_gap, method=method
        )

        return df_fixed, {"fix_summary": summary}


class HandleDSTTransitionsStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        timezone = issue.metadata.get("timezone", "Europe/Berlin")

        df_fixed, summary = self.ts_tools.handle_dst_transitions(df, timezone=timezone)

        return df_fixed, {"fix_summary": summary}


class RemoveDuplicatesStrategy(FixStrategy):

    def apply(self, df: pd.DataFrame, issue: Issue) -> tuple[pd.DataFrame, dict]:

        if "timestamp" not in df.columns:
            return df, {"error": "No timestamp column found"}

        before_len = len(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        agg_dict = {col: "mean" for col in numeric_cols}

        df_fixed = df.groupby("timestamp", as_index=False).agg(
            {**agg_dict, **{col: "first" for col in df.columns if col not in numeric_cols and col != "timestamp"}}
        )

        after_len = len(df_fixed)
        removed = before_len - after_len

        summary = {
            "method": "remove_duplicates",
            "before_length": before_len,
            "after_length": after_len,
            "duplicates_removed": removed,
        }

        return df_fixed, {"fix_summary": summary}


class StrategyFactory:

    def __init__(
        self,
        config: RemediationConfig,
        missing_tools: MissingDataTools,
        ts_tools: TimeSeriesTools,
        validation_tools: ValidationTools,
    ):

        self.config = config
        self.missing_tools = missing_tools
        self.ts_tools = ts_tools
        self.validation_tools = validation_tools

        self.strategies = {
            "impute_by_correlation": ImputeByCorrelationStrategy,
            "impute_by_calculation": ImputeByCalculationStrategy,
            "impute_by_seasonal_pattern": ImputeBySeasonalPatternStrategy,
            "impute_by_interpolation": ImputeByInterpolationStrategy,
            "impute_by_seasonal_naive": ImputeBySeasonalNaiveStrategy,
            "fill_timestamp_gaps": FillTimestampGapsStrategy,
            "handle_dst_transitions": HandleDSTTransitionsStrategy,
            "remove_duplicates": RemoveDuplicatesStrategy,
        }

    def get_strategy(self, strategy_name: str) -> FixStrategy:

        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy_class = self.strategies[strategy_name]
        return strategy_class(
            self.config,
            self.missing_tools,
            self.ts_tools,
            self.validation_tools,
        )

    def apply_strategy(
        self, df: pd.DataFrame, issue: Issue
    ) -> tuple[pd.DataFrame, dict]:

        if not issue.fix_strategy:
            return df, {"error": "No fix strategy specified for issue"}

        strategy = self.get_strategy(issue.fix_strategy)
        return strategy.apply(df, issue)
