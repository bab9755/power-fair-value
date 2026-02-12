
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class MissingnessThresholds(BaseModel):

    warn_missing_pct: float = Field(default=5.0, description="Warn if missing % exceeds this")
    fail_missing_pct: float = Field(default=20.0, description="Fail if missing % exceeds this")


class OutlierThresholds(BaseModel):

    price_hard_min: float = Field(default=-600.0, description="Hard minimum for price (EUR/MWh)")
    price_hard_max: float = Field(default=1500.0, description="Hard maximum for price (EUR/MWh)")
    price_extreme_percentile: float = Field(
        default=99.9, description="Percentile threshold for extreme price spikes"
    )


class ConsistencyRules(BaseModel):

    renewable_sum_tolerance_pct: float = Field(
        default=1.0, description="Allowed % mismatch in renewable generation sums"
    )
    allow_negative_price: bool = Field(default=True, description="Allow negative prices")
    allow_negative_residual_load: bool = Field(
        default=True, description="Allow negative residual load"
    )
    forbid_negative_generation: bool = Field(
        default=True, description="Forbid negative generation values"
    )


class QAConfig(BaseModel):

    timezone_policy: Literal["utc", "europe_berlin"] = Field(
        default="europe_berlin", description="Timezone handling policy"
    )
    dst_policy: Literal["allow_missing_spring_hour", "reindex_and_impute"] = Field(
        default="allow_missing_spring_hour",
        description="How to handle DST transitions",
    )
    frequency: str = Field(default="h", description="Expected time series frequency")

    missingness: MissingnessThresholds = Field(default_factory=MissingnessThresholds)
    outliers: OutlierThresholds = Field(default_factory=OutlierThresholds)
    consistency: ConsistencyRules = Field(default_factory=ConsistencyRules)

    target_column: str = Field(default="price_eur_mwh", description="Target column name")
    target_max_missing_pct: float = Field(
        default=0.0, description="Maximum allowed missing % for target column"
    )

    required_columns: list[str] = Field(
        default=[
            "timestamp",
            "price_eur_mwh",
            "price_neighbors_avg_eur_mwh",
            "load_mwh",
            "residual_load_mwh",
            "gen_total_mwh",
            "gen_pv_wind_mwh",
            "gen_wind_offshore_mwh",
            "gen_wind_onshore_mwh",
            "gen_solar_mwh",
            "gen_other_mwh",
        ]
    )

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic", description="LLM provider for pipeline"
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model name to use"
    )
    llm_temperature: float = Field(default=0.0, description="Temperature for LLM")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "QAConfig":

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> "QAConfig":

        if config_path is None:
            default_paths = [
                Path("config.yaml"),
                Path("qa_config.yaml"),
            ]
            for path in default_paths:
                if path.exists():
                    return cls.from_yaml(path)
            return cls()

        return cls.from_yaml(config_path)

    def to_yaml(self, path: Path | str) -> None:

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            data = self.model_dump()
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
