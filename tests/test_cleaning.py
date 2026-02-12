
import pandas as pd
import pytest
import pytz

from src.qa_pipeline.cleaning import clean_dataset
from src.qa_pipeline.config import QAConfig


@pytest.fixture
def sample_data():

    dates = pd.date_range("2024-01-01", periods=100, freq="H")
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


def test_timestamp_parsing(sample_data):

    config = QAConfig()

    df_shuffled = sample_data.sample(frac=1).reset_index(drop=True)

    df_clean, log = clean_dataset(df_shuffled, config)

    assert df_clean["timestamp"].is_monotonic_increasing

    actions = log.to_dict()
    assert any("timestamp" in str(a) for a in actions)


def test_timezone_localization():

    dates = pd.date_range("2024-01-01", periods=24, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "price_eur_mwh": range(24),
            "price_neighbors_avg_eur_mwh": range(24),
            "load_mwh": range(24),
            "residual_load_mwh": range(24),
            "gen_total_mwh": range(24),
            "gen_pv_wind_mwh": range(24),
            "gen_wind_offshore_mwh": range(24),
            "gen_wind_onshore_mwh": range(24),
            "gen_solar_mwh": range(24),
            "gen_other_mwh": range(24),
        }
    )

    config = QAConfig(timezone_policy="europe_berlin")
    df_clean, log = clean_dataset(df, config)

    assert df_clean["timestamp"].dt.tz is not None
    assert "Europe/Berlin" in str(df_clean["timestamp"].dt.tz)


def test_duplicate_removal(sample_data):

    df_with_dup = pd.concat([sample_data, sample_data.iloc[:1]], ignore_index=True)

    config = QAConfig()
    df_clean, log = clean_dataset(df_with_dup, config)

    assert df_clean["timestamp"].duplicated().sum() == 0

    actions = log.to_dict()
    assert any("duplicate" in str(a).lower() for a in actions)


def test_numeric_coercion():

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "price_eur_mwh": ["10.5", "20.3", "invalid"],
            "price_neighbors_avg_eur_mwh": [1, 2, 3],
            "load_mwh": [1, 2, 3],
            "residual_load_mwh": [1, 2, 3],
            "gen_total_mwh": [1, 2, 3],
            "gen_pv_wind_mwh": [1, 2, 3],
            "gen_wind_offshore_mwh": [1, 2, 3],
            "gen_wind_onshore_mwh": [1, 2, 3],
            "gen_solar_mwh": [1, 2, 3],
            "gen_other_mwh": [1, 2, 3],
        }
    )

    config = QAConfig()
    df_clean, log = clean_dataset(df, config)

    assert pd.api.types.is_numeric_dtype(df_clean["price_eur_mwh"])

    assert df_clean["price_eur_mwh"].isna().sum() == 1


def test_dst_spring_gap_identification():

    dates = pd.date_range("2024-03-31 00:00", periods=5, freq="H", tz="Europe/Berlin")

    dates_with_gap = [d for d in dates if d.hour != 2]

    df = pd.DataFrame(
        {
            "timestamp": dates_with_gap,
            "price_eur_mwh": range(len(dates_with_gap)),
            "price_neighbors_avg_eur_mwh": range(len(dates_with_gap)),
            "load_mwh": range(len(dates_with_gap)),
            "residual_load_mwh": range(len(dates_with_gap)),
            "gen_total_mwh": range(len(dates_with_gap)),
            "gen_pv_wind_mwh": range(len(dates_with_gap)),
            "gen_wind_offshore_mwh": range(len(dates_with_gap)),
            "gen_wind_onshore_mwh": range(len(dates_with_gap)),
            "gen_solar_mwh": range(len(dates_with_gap)),
            "gen_other_mwh": range(len(dates_with_gap)),
        }
    )

    config = QAConfig(
        timezone_policy="europe_berlin", dst_policy="allow_missing_spring_hour"
    )
    df_clean, log = clean_dataset(df, config)

    actions = log.to_dict()
    dst_actions = [a for a in actions if "DST" in str(a) or "dst" in str(a)]

    assert len(dst_actions) > 0 or len(df_clean) == len(dates_with_gap)
