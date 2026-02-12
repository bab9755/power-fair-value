
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_dataset(path: Path | str) -> pd.DataFrame:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_dataset(df: pd.DataFrame, path: Path | str, format: str = "parquet") -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


class ArtifactManager:

    def __init__(self, artifacts_dir: Path | str, run_id: str | None = None):

        self.artifacts_dir = Path(artifacts_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.artifacts_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def get_path(self, filename: str) -> Path:

        return self.run_dir / filename

    def get_plot_path(self, filename: str) -> Path:

        return self.plots_dir / filename

    def save_json(self, data: dict[str, Any], filename: str) -> Path:

        path = self.get_path(filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def append_log(self, entry: dict[str, Any], filename: str = "run_log.jsonl") -> None:

        path = self.get_path(filename)
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_markdown(self, content: str, filename: str) -> Path:

        path = self.get_path(filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def log_step(self, step: str, **kwargs: Any) -> None:

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **kwargs,
        }
        self.append_log(entry)
