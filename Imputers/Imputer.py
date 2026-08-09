import pandas as pd
import numpy as np

from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer
)

# Required for IterativeImputer
from sklearn.experimental import enable_iterative_imputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge


class Imputation:
    """
    Collection of column-wise imputation methods.

    Each method:
        - Takes a dataframe.
        - Imputes ONE target column.
        - Returns a NEW dataframe.
    """

    @staticmethod
    def mean(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_new = df.copy()

        imputer = SimpleImputer(strategy="mean")

        df_new[[target_column]] = imputer.fit_transform(
            df_new[[target_column]]
        )

        return df_new

    @staticmethod
    def median(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_new = df.copy()

        imputer = SimpleImputer(strategy="median")

        df_new[[target_column]] = imputer.fit_transform(
            df_new[[target_column]]
        )

        return df_new

    @staticmethod
    def mode(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df_new = df.copy()

        imputer = SimpleImputer(strategy="most_frequent")

        df_new[[target_column]] = imputer.fit_transform(
            df_new[[target_column]]
        )

        return df_new

    @staticmethod
    def knn(
        df: pd.DataFrame,
        target_column: str,
        n_neighbors: int = 5,
        weights: str = "uniform"
    ) -> pd.DataFrame:

        df_new = df.copy()

        numeric_cols = df_new.select_dtypes(include=np.number).columns

        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights
        )

        df_numeric = pd.DataFrame(
            imputer.fit_transform(df_new[numeric_cols]),
            columns=numeric_cols,
            index=df_new.index
        )

        df_new[numeric_cols] = df_numeric

        return df_new

    @staticmethod
    def iterative(
        df: pd.DataFrame,
        target_column: str,
        random_state: int = 42,
        max_iter: int = 10
    ) -> pd.DataFrame:

        df_new = df.copy()

        numeric_cols = df_new.select_dtypes(include=np.number).columns

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=random_state,
            max_iter=max_iter
        )

        df_numeric = pd.DataFrame(
            imputer.fit_transform(df_new[numeric_cols]),
            columns=numeric_cols,
            index=df_new.index
        )

        df_new[numeric_cols] = df_numeric

        return df_new

    @staticmethod
    def random_forest(
        df: pd.DataFrame,
        target_column: str,
        random_state: int = 42,
        max_iter: int = 10,
        n_estimators: int = 100
    ) -> pd.DataFrame:

        df_new = df.copy()

        numeric_cols = df_new.select_dtypes(include=np.number).columns

        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

        imputer = IterativeImputer(
            estimator=estimator,
            random_state=random_state,
            max_iter=max_iter
        )

        df_numeric = pd.DataFrame(
            imputer.fit_transform(df_new[numeric_cols]),
            columns=numeric_cols,
            index=df_new.index
        )

        df_new[numeric_cols] = df_numeric

        return df_new