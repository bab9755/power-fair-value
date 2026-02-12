from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class InterpretationConfig(BaseModel):
    outputs_dir: str = Field(default="outputs", description="Main outputs directory")
    da_to_curve_dir: str = Field(
        default="outputs/da_to_curve", description="DA-to-curve results directory"
    )
    artifacts_dir: str = Field(default="artifacts", description="QA/remediation artifacts directory")
    data_dir: str = Field(default="data/processed", description="Processed data directory")

    report_output: str = Field(
        default="outputs/interpretation_report.md",
        description="Path for the generated interpretation report",
    )

    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic", description="LLM provider"
    )
    llm_model: str = Field(
        default="claude-sonnet-4-5-20250929", description="Model name to use"
    )
    llm_temperature: float = Field(default=0.0, description="Temperature for LLM")
    llm_max_tokens: int = Field(default=4096, description="Max tokens for LLM response")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "InterpretationConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "interpretation" in data:
            data = data["interpretation"]

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> "InterpretationConfig":
        if config_path is None:
            default_paths = [
                Path("interpretation_config.yaml"),
                Path("config.yaml"),
            ]
            for path in default_paths:
                if path.exists():
                    try:
                        return cls.from_yaml(path)
                    except Exception:
                        continue
            return cls()

        return cls.from_yaml(config_path)
