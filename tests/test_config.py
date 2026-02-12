
import tempfile
from pathlib import Path

import pytest

from src.qa_pipeline.config import QAConfig


def test_default_config():

    config = QAConfig()

    assert config.timezone_policy in ["utc", "europe_berlin"]
    assert config.dst_policy in ["allow_missing_spring_hour", "reindex_and_impute"]
    assert config.frequency == "H"
    assert len(config.required_columns) == 11
    assert "timestamp" in config.required_columns
    assert "price_eur_mwh" in config.required_columns


def test_config_yaml_roundtrip():

    config = QAConfig(
        timezone_policy="utc",
        dst_policy="reindex_and_impute",
        target_max_missing_pct=5.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        config.to_yaml(config_path)

        assert config_path.exists()

        loaded_config = QAConfig.from_yaml(config_path)

        assert loaded_config.timezone_policy == "utc"
        assert loaded_config.dst_policy == "reindex_and_impute"
        assert loaded_config.target_max_missing_pct == 5.0


def test_config_thresholds():

    config = QAConfig()

    assert config.missingness.warn_missing_pct < config.missingness.fail_missing_pct
    assert config.outliers.price_hard_min < config.outliers.price_hard_max
    assert 0 <= config.outliers.price_extreme_percentile <= 100
    assert config.consistency.renewable_sum_tolerance_pct > 0
