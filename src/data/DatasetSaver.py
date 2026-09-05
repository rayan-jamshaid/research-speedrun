from pathlib import Path
from typing import Dict

import pandas as pd


class DatasetSaver:
    """Persist a train/validation/test dataset triplet as CSV files."""

    def __init__(self, output_dir: str = "saved_datasets"):
        self.output_dir = Path(output_dir)

    def save(
        self,
        name: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        val: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Save ``<name>_{train,test,val}.csv`` inside ``output_dir/name``."""
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": dataset_dir / f"{name}_train.csv",
            "test": dataset_dir / f"{name}_test.csv",
            "val": dataset_dir / f"{name}_val.csv",
        }
        train.to_csv(paths["train"], index=False)
        test.to_csv(paths["test"], index=False)
        val.to_csv(paths["val"], index=False)
        return paths

    def load(self, name: str) -> Dict[str, pd.DataFrame]:
        """Load a previously saved train/validation/test dataset triplet."""
        dataset_dir = self.output_dir / name
        paths = {
            "train": dataset_dir / f"{name}_train.csv",
            "test": dataset_dir / f"{name}_test.csv",
            "val": dataset_dir / f"{name}_val.csv",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"No saved dataset triplet exists for '{name}'. Missing: {missing}"
            )
        return {split: pd.read_csv(path) for split, path in paths.items()}
