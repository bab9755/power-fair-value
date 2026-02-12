
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class MissingDataConfig(BaseModel):

    threshold_interpolate: float = Field(
        default=5.0,
        description="Interpolate if less than this % missing",
    )
    threshold_model_based: float = Field(
        default=15.0,
        description="Use ML-based imputation if 5-15% missing",
    )
    threshold_drop: float = Field(
        default=25.0,
        description="Consider dropping column if > 25% missing",
    )

    strategies: dict[str, dict] = Field(
        default={
            "price_neighbors_avg_eur_mwh": {
                "method": "correlation_regression",
                "reference": "price_eur_mwh",
            },
            "gen_total_mwh": {
                "method": "calculate_from_components",
                "components": ["gen_pv_wind_mwh", "gen_other_mwh"],
            },
            "load_mwh": {
                "method": "seasonal_decomposition",
                "period": "weekly",
            },
            "gen_other_mwh": {
                "method": "seasonal_naive",
                "period": "weekly",
            },
        }
    )


class TimeSeriesConfig(BaseModel):

    max_gap_hours: int = Field(
        default=24,
        description="Maximum gap size to interpolate automatically",
    )
    interpolation_method: str = Field(
        default="linear",
        description="Method for interpolation",
    )
    handle_long_gaps: Literal["flag_for_review", "interpolate", "drop"] = Field(
        default="flag_for_review",
        description="How to handle gaps > max_gap_hours",
    )


class OutlierConfig(BaseModel):

    allow_negative_prices: bool = Field(
        default=True,
        description="Allow negative prices (valid in energy markets)",
    )
    winsorize_extremes: bool = Field(
        default=False,
        description="Winsorize extreme values",
    )
    price_cap_percentile: float = Field(
        default=99.9,
        description="Percentile to cap prices at (if winsorizing)",
    )


class ValidationConfig(BaseModel):

    rerun_qa_after_fix: bool = Field(
        default=True,
        description="Re-run QA checks after each fix",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum remediation iterations",
    )
    fail_if_not_ready: bool = Field(
        default=False,
        description="Fail if not training-ready after max iterations",
    )


class RemediationConfig(BaseModel):

    missing_data: MissingDataConfig = Field(default_factory=MissingDataConfig)
    time_series: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    outliers: OutlierConfig = Field(default_factory=OutlierConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    dry_run: bool = Field(
        default=False,
        description="Dry run mode (show what would be fixed)",
    )
    interactive: bool = Field(
        default=False,
        description="Interactive mode (ask before each fix)",
    )
    backup_original: bool = Field(
        default=True,
        description="Create backup of original dataset",
    )

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic",
        description="LLM provider for pipeline",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name to use",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RemediationConfig":

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> "RemediationConfig":

        if config_path is None:
            default_paths = [
                Path("remediation_config.yaml"),
                Path("config/remediation.yaml"),
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
