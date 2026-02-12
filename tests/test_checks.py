
import pandas as pd
import pytest

from src.qa_pipeline.checks import (
    check_missingness,
    check_schema_and_types,
    check_target_readiness,
    check_time_series_integrity,
    check_value_sanity,
)
from src.qa_pipeline.config import QAConfig


@pytest.fixture
def valid_data():

    dates = pd.date_range("2024-01-01", periods=100, freq="H", tz="Europe/Berlin")
    data = {
        "timestamp": dates,
        "price_eur_mwh": range(100),
        "price_neighbors_avg_eur_mwh": range(100),
        "load_mwh": range(100),
        "residual_load_mwh": range(100),
        "gen_total_mwh": range(100),
        "gen_pv_wind_mwh": range(100),
        "gen_wind_offshore_mwh": range(100),
        "gen_wind_onshore_mwh": range(100),
        "gen_solar_mwh": range(100),
        "gen_other_mwh": range(100),
    }
    return pd.DataFrame(data)


def test_schema_check_valid(valid_data):

    config = QAConfig()
    result = check_schema_and_types(valid_data, config)

    assert result.status == "PASS"
    assert result.metrics["timestamp_is_datetime"] is True
    assert len(result.metrics["missing_required"]) == 0


def test_schema_check_missing_column():

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="H"),
            "price_eur_mwh": range(10),
        }
    )

    config = QAConfig()
    result = check_schema_and_types(df, config)

    assert result.status == "FAIL"
    assert len(result.metrics["missing_required"]) > 0


def test_time_series_integrity_valid(valid_data):

    config = QAConfig()
    result = check_time_series_integrity(valid_data, config)

    assert result.status == "PASS"
    assert result.metrics["is_monotonic_increasing"] is True
    assert result.metrics["duplicate_count"] == 0


def test_time_series_integrity_duplicates():

    dates = pd.date_range("2024-01-01", periods=10, freq="H", tz="Europe/Berlin")
    dates = list(dates) + [dates[0]]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price_eur_mwh": range(len(dates)),
            "price_neighbors_avg_eur_mwh": range(len(dates)),
            "load_mwh": range(len(dates)),
            "residual_load_mwh": range(len(dates)),
            "gen_total_mwh": range(len(dates)),
            "gen_pv_wind_mwh": range(len(dates)),
            "gen_wind_offshore_mwh": range(len(dates)),
            "gen_wind_onshore_mwh": range(len(dates)),
            "gen_solar_mwh": range(len(dates)),
            "gen_other_mwh": range(len(dates)),
        }
    )

    config = QAConfig()
    result = check_time_series_integrity(df, config)

    assert result.status == "FAIL"
    assert result.metrics["duplicate_count"] > 0


def test_missingness_check_low(valid_data):

    config = QAConfig()
    result = check_missingness(valid_data, config)

    assert result.status == "PASS"


def test_missingness_check_high():

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="H", tz="Europe/Berlin"),
            "price_eur_mwh": [None] * 50 + list(range(50)),
            "price_neighbors_avg_eur_mwh": range(100),
            "load_mwh": range(100),
            "residual_load_mwh": range(100),
            "gen_total_mwh": range(100),
            "gen_pv_wind_mwh": range(100),
            "gen_wind_offshore_mwh": range(100),
            "gen_wind_onshore_mwh": range(100),
            "gen_solar_mwh": range(100),
            "gen_other_mwh": range(100),
        }
    )

    config = QAConfig()
    result = check_missingness(df, config)

    assert result.status in ["WARN", "FAIL"]
    assert "price_eur_mwh" in result.metrics["per_column"]
    assert result.metrics["per_column"]["price_eur_mwh"]["percent"] == 50.0


def test_value_sanity_check(valid_data):

    config = QAConfig()
    result = check_value_sanity(valid_data, config)

    assert result.status in ["PASS", "WARN"]
    assert "price_eur_mwh" in result.metrics


def test_target_readiness_pass(valid_data):

    config = QAConfig(target_column="price_eur_mwh", target_max_missing_pct=0.0)
    result = check_target_readiness(valid_data, config)

    assert result.status == "PASS"
    assert result.metrics["missing_pct"] == 0.0


def test_target_readiness_fail():

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="H", tz="Europe/Berlin"),
            "price_eur_mwh": [None] * 10 + list(range(90)),
            "price_neighbors_avg_eur_mwh": range(100),
            "load_mwh": range(100),
            "residual_load_mwh": range(100),
            "gen_total_mwh": range(100),
            "gen_pv_wind_mwh": range(100),
            "gen_wind_offshore_mwh": range(100),
            "gen_wind_onshore_mwh": range(100),
            "gen_solar_mwh": range(100),
            "gen_other_mwh": range(100),
        }
    )

    config = QAConfig(target_column="price_eur_mwh", target_max_missing_pct=0.0)
    result = check_target_readiness(df, config)

    assert result.status == "FAIL"
    assert result.metrics["missing_count"] == 10
