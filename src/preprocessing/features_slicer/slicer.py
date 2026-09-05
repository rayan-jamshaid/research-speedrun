from collections.abc import Mapping, Sequence
from typing import Optional

import pandas as pd


class FeatureSlicer:
    """Select a named feature set while retaining pipeline-required columns."""

    def __init__(
        self,
        projects: Optional[Mapping[str, Optional[Sequence[str]]]] = None,
        id_column: str = "subject_id",
        target_column: str = "mortality_flag",
        features: Optional[Sequence[str]] = None,
    ):
        self.projects = dict(projects or {})
        self.required_columns = [id_column, target_column]
        self.features = features

    def slice(
        self, df: pd.DataFrame, features: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Return a copy containing required columns and the selected features."""
        selected_columns = self._columns_to_keep(df, features or self.features)
        return df.loc[:, selected_columns].copy()

    def slice_project(self, df: pd.DataFrame, project_name: str) -> pd.DataFrame:
        """Apply a configured feature project; ``None`` means keep all columns."""
        if project_name not in self.projects:
            raise KeyError(
                f"Unknown feature project '{project_name}'. "
                f"Available projects: {list(self.projects)}"
            )
        features = self.projects[project_name]
        if features is None:
            missing = [column for column in self.required_columns if column not in df.columns]
            if missing:
                raise KeyError(f"Columns not found in the dataframe: {missing}")
            return df.copy()
        return self.slice(df, features)

    def slice_pair(
        self,
        training_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        project_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the same project to source and external-validation datasets."""
        return (
            self.slice_project(training_df, project_name),
            self.slice_project(validation_df, project_name),
        )

    def _columns_to_keep(self, df: pd.DataFrame, features: Sequence[str]) -> list[str]:
        if not features:
            raise ValueError("A feature project must contain at least one feature.")
        duplicate_features = [
            feature for index, feature in enumerate(features)
            if feature in features[:index]
        ]
        if duplicate_features:
            raise ValueError(f"Features must be unique: {duplicate_features}")

        columns = self.required_columns + list(features)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in the dataframe: {missing}")
        return list(dict.fromkeys(columns))
