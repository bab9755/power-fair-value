
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.remediation_pipeline.config import RemediationConfig
from src.remediation_pipeline.domain.electricity import ElectricityMarketDomain
from src.remediation_pipeline.domain.patterns import TimeSeriesPatterns
from src.remediation_pipeline.issue_parser import Issue, IssueCategory, IssueParser, IssueSeverity
from src.remediation_pipeline.tools.missing_data import MissingDataTools
from src.remediation_pipeline.tools.time_series import TimeSeriesTools
from src.remediation_pipeline.tools.validation import ValidationTools


@pytest.fixture
def sample_df():

    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price_eur_mwh": np.random.randn(100) * 10 + 50,
            "price_neighbors_avg_eur_mwh": np.random.randn(100) * 10 + 50,
            "load_mwh": np.random.randn(100) * 1000 + 50000,
            "gen_total_mwh": np.random.randn(100) * 1000 + 45000,
            "gen_pv_wind_mwh": np.random.randn(100) * 500 + 20000,
            "gen_other_mwh": np.random.randn(100) * 500 + 25000,
            "gen_solar_mwh": np.random.randn(100) * 200 + 5000,
            "gen_wind_onshore_mwh": np.random.randn(100) * 200 + 10000,
            "gen_wind_offshore_mwh": np.random.randn(100) * 200 + 5000,
            "residual_load_mwh": np.random.randn(100) * 1000 + 30000,
        }
    )
    return df


@pytest.fixture
def sample_df_with_missing(sample_df):

    df = sample_df.copy()
    df.loc[10:15, "price_neighbors_avg_eur_mwh"] = np.nan
    df.loc[20:22, "load_mwh"] = np.nan
    df.loc[30:35, "gen_other_mwh"] = np.nan
    return df


class TestElectricityMarketDomain:

    def test_negative_price_validation(self):

        domain = ElectricityMarketDomain()

        is_valid, reason = domain.is_negative_price_valid(
            hour=2, renewable_gen=40000, load=45000, month=6
        )
        assert is_valid

        is_valid, reason = domain.is_negative_price_valid(
            hour=12, renewable_gen=5000, load=50000, month=6
        )
        assert not is_valid

    def test_solar_generation_range(self):

        domain = ElectricityMarketDomain()

        min_val, max_val, desc = domain.get_expected_solar_range(hour=2, month=6)
        assert min_val == 0.0
        assert max_val == 0.0

        min_val, max_val, desc = domain.get_expected_solar_range(hour=12, month=6)
        assert min_val > 0
        assert max_val > min_val

    def test_generation_consistency_validation(self):

        domain = ElectricityMarketDomain()

        row = pd.Series(
            {
                "gen_pv_wind_mwh": 20000,
                "gen_solar_mwh": 5000,
                "gen_wind_onshore_mwh": 10000,
                "gen_wind_offshore_mwh": 5000,
                "gen_total_mwh": 45000,
                "gen_other_mwh": 25000,
            }
        )
        is_valid, error = domain.validate_generation_consistency(row)
        assert is_valid

        row["gen_pv_wind_mwh"] = 30000
        is_valid, error = domain.validate_generation_consistency(row)
        assert not is_valid


class TestTimeSeriesPatterns:

    def test_seasonal_decomposition(self):

        patterns = TimeSeriesPatterns()

        dates = pd.date_range("2024-01-01", periods=7 * 24, freq="h")
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + 50
        series = pd.Series(values, index=dates)

        decomp = patterns.decompose_seasonal(series, period="daily")

        assert "trend" in decomp
        assert "seasonal" in decomp
        assert "residual" in decomp

    def test_gap_detection(self):

        patterns = TimeSeriesPatterns()

        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        dates = dates.drop(dates[10:15])
        series = pd.Series(np.random.randn(len(dates)), index=dates)

        gaps = patterns.detect_gaps(series, max_gap_hours=3)

        assert len(gaps) > 0
        assert gaps.iloc[0]["duration_hours"] >= 5

    def test_correlation_imputation(self):

        patterns = TimeSeriesPatterns()

        target = pd.Series(np.random.randn(100) * 10 + 50)
        reference = target * 0.9 + np.random.randn(100) * 2

        target.iloc[10:20] = np.nan

        imputed, metadata = patterns.impute_by_correlation(target, reference)

        assert imputed.isna().sum() < target.isna().sum()
        assert "correlation" in metadata


class TestMissingDataTools:

    def test_impute_by_correlation(self, sample_df_with_missing):

        tools = MissingDataTools()

        df_fixed, summary = tools.impute_by_correlation(
            sample_df_with_missing,
            "price_neighbors_avg_eur_mwh",
            "price_eur_mwh",
        )

        assert summary["after_missing"] < summary["before_missing"]
        assert summary["imputed_count"] > 0

    def test_impute_by_calculation(self, sample_df_with_missing):

        tools = MissingDataTools()

        df = sample_df_with_missing.copy()
        df.loc[40:45, "gen_total_mwh"] = np.nan

        df_fixed, summary = tools.impute_by_calculation(
            df, "gen_total_mwh", ["gen_pv_wind_mwh", "gen_other_mwh"]
        )

        assert summary["after_missing"] < summary["before_missing"]
        assert summary["calculated_count"] > 0

    def test_impute_by_interpolation(self, sample_df_with_missing):

        tools = MissingDataTools()

        df_fixed, summary = tools.impute_by_interpolation(
            sample_df_with_missing, "load_mwh", method="linear"
        )

        assert summary["after_missing"] == 0
        assert summary["imputed_count"] > 0


class TestTimeSeriesTools:

    def test_fill_timestamp_gaps(self, sample_df):

        tools = TimeSeriesTools()

        df = sample_df.iloc[[i for i in range(len(sample_df)) if i not in range(10, 15)]]

        df_fixed, summary = tools.fill_timestamp_gaps(df, max_gap_hours=10)

        assert len(df_fixed) > len(df)
        assert summary["gaps_found"] > 0

    def test_validate_hourly_frequency(self, sample_df):

        tools = TimeSeriesTools()

        df_fixed, summary = tools.validate_hourly_frequency(sample_df)

        assert summary["is_valid"]
        assert summary["duplicate_timestamps"] == 0


class TestValidationTools:

    def test_validate_fix(self, sample_df, sample_df_with_missing):

        tools = ValidationTools()

        df_fixed = sample_df_with_missing.copy()
        df_fixed["load_mwh"] = df_fixed["load_mwh"].fillna(df_fixed["load_mwh"].mean())

        result = tools.validate_fix(sample_df_with_missing, df_fixed, "load_mwh")

        assert result["validation_passed"]
        assert result["missing_reduced_by"] > 0

    def test_cross_validate_generation(self, sample_df):

        tools = ValidationTools()

        result = tools.cross_validate_generation(sample_df)

        assert "consistency_score" in result
        assert result["consistency_score"] >= 0

    def test_validate_no_duplicates(self, sample_df):

        tools = ValidationTools()

        result = tools.validate_no_duplicates(sample_df, "timestamp")

        assert result["validation_passed"]
        assert result["duplicate_count"] == 0


class TestIssueParser:

    def test_parse_check(self):

        parser = IssueParser()

        check = {
            "id": "missingness_threshold_price",
            "name": "Price Missing Data",
            "status": "WARN",
            "message": "15% missing data",
            "metadata": {"column": "price_eur_mwh", "missing_pct": 15.0},
        }

        issue = parser._parse_check(check)

        assert issue is not None
        assert issue.category == IssueCategory.MISSING_DATA
        assert issue.fixable
        assert issue.fix_strategy is not None

    def test_prioritize_issues(self):

        parser = IssueParser()

        issues = [
            Issue(
                id="1",
                name="Low",
                category=IssueCategory.MISSING_DATA,
                severity=IssueSeverity.LOW,
                status="WARN",
                description="Low severity",
                metadata={},
                fixable=True,
            ),
            Issue(
                id="2",
                name="Critical",
                category=IssueCategory.MISSING_DATA,
                severity=IssueSeverity.CRITICAL,
                status="FAIL",
                description="Critical severity",
                metadata={},
                fixable=True,
            ),
            Issue(
                id="3",
                name="High",
                category=IssueCategory.TIME_SERIES_GAP,
                severity=IssueSeverity.HIGH,
                status="FAIL",
                description="High severity",
                metadata={},
                fixable=True,
            ),
        ]

        prioritized = parser.prioritize_issues(issues)

        assert prioritized[0].severity == IssueSeverity.CRITICAL
        assert prioritized[1].severity == IssueSeverity.HIGH
        assert prioritized[2].severity == IssueSeverity.LOW


class TestRemediationConfig:

    def test_default_config(self):

        config = RemediationConfig()

        assert config.missing_data.threshold_interpolate == 5.0
        assert config.time_series.max_gap_hours == 24
        assert config.validation.max_iterations == 3

    def test_config_to_yaml(self, tmp_path):

        config = RemediationConfig()
        yaml_path = tmp_path / "config.yaml"

        config.to_yaml(yaml_path)

        assert yaml_path.exists()

        loaded_config = RemediationConfig.from_yaml(yaml_path)
        assert loaded_config.missing_data.threshold_interpolate == 5.0
