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
    Collection of imputation methods with fit/transform capability to prevent data leakage.
    
    Usage:
        # Create imputer instance and fit on train data
        imputer = Imputation("iterative")
        imputer.fit(df_train)
        
        # Transform train, val, and test using the same fitted imputer
        df_train_imputed = imputer.transform(df_train)
        df_val_imputed = imputer.transform(df_val)
        df_test_imputed = imputer.transform(df_test)
    """

    def __init__(self, method: str = "iterative", random_state: int = 42, max_iter: int = 10, n_neighbors: int = 5):
        """
        Initialize the imputer with a specific method.
        
        Args:
            method: "mean", "median", "mode", "knn", "iterative", or "random_forest"
            random_state: Random state for reproducibility
            max_iter: Max iterations for iterative methods
            n_neighbors: Number of neighbors for KNN
        """
        self.method = method
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.imputer = None
        self.numeric_cols = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Imputation":
        """
        Fit the imputer on training data. Learn patterns only from train data.
        
        Args:
            df: Training dataframe
            
        Returns:
            self (for chaining)
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.numeric_cols = numeric_cols

        if self.method == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.method == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif self.method == "mode":
            self.imputer = SimpleImputer(strategy="most_frequent")
        elif self.method == "knn":
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors, weights="uniform")
        elif self.method == "iterative":
            self.imputer = IterativeImputer(
                estimator=BayesianRidge(),
                random_state=self.random_state,
                max_iter=self.max_iter
            )
        elif self.method == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.imputer = IterativeImputer(
                estimator=estimator,
                random_state=self.random_state,
                max_iter=self.max_iter
            )
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        # Fit on numeric columns only
        self.imputer.fit(df[self.numeric_cols])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataframe using the fitted imputer.
        
        Args:
            df: Dataframe to impute
            
        Returns:
            Imputed dataframe
        """
        if not self._is_fitted:
            raise RuntimeError(f"Imputer not fitted. Call fit() first.")

        if self.numeric_cols is None:
            raise RuntimeError("Numeric columns not set. Call fit() first.")

        df_new = df.copy()

        # Transform numeric columns using fitted imputer
        df_numeric = pd.DataFrame(
            self.imputer.transform(df_new[self.numeric_cols]),
            columns=self.numeric_cols,
            index=df_new.index
        )

        df_new[self.numeric_cols] = df_numeric
        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step (only use this on train data).
        
        Args:
            df: Dataframe to fit and transform
            
        Returns:
            Imputed dataframe
        """
        return self.fit(df).transform(df)

    # Legacy static methods for backwards compatibility
    @staticmethod
    def mean(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Legacy method. Use Imputation('mean').fit_transform() instead."""
        imputer = Imputation("mean")
        return imputer.fit_transform(df)

    @staticmethod
    def median(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Legacy method. Use Imputation('median').fit_transform() instead."""
        imputer = Imputation("median")
        return imputer.fit_transform(df)

    @staticmethod
    def mode(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Legacy method. Use Imputation('mode').fit_transform() instead."""
        imputer = Imputation("mode")
        return imputer.fit_transform(df)

    @staticmethod
    def knn(df: pd.DataFrame, target_column: str, n_neighbors: int = 5, weights: str = "uniform") -> pd.DataFrame:
        """Legacy method. Use Imputation('knn').fit_transform() instead."""
        imputer = Imputation("knn", n_neighbors=n_neighbors)
        return imputer.fit_transform(df)

    @staticmethod
    def iterative(df: pd.DataFrame, target_column: str, random_state: int = 42, max_iter: int = 10) -> pd.DataFrame:
        """Legacy method. Use Imputation('iterative').fit_transform() instead."""
        imputer = Imputation("iterative", random_state=random_state, max_iter=max_iter)
        return imputer.fit_transform(df)

    @staticmethod
    def random_forest(df: pd.DataFrame, target_column: str, random_state: int = 42, max_iter: int = 10, n_estimators: int = 100) -> pd.DataFrame:
        """Legacy method. Use Imputation('random_forest').fit_transform() instead."""
        imputer = Imputation("random_forest", random_state=random_state, max_iter=max_iter)
        return imputer.fit_transform(df)