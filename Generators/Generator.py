import pandas as pd
from imblearn.over_sampling import SMOTE as ImblearnSMOTE


class Generator:
    """
    A utility class for generating balanced datasets using oversampling techniques.

    Methods
    -------
    SMOTE(dataframe, target_column, random_state=42)
        Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance
        the target classes to a 50-50 distribution.

    Notes
    -----
    - This method should ONLY be applied to the training dataset after
      train-test splitting.
    - SMOTE generates synthetic samples for the minority class.
    - Works only with numerical features. Categorical features should be
      encoded before applying SMOTE.
    """

    @staticmethod
    def SMOTE(
        dataframe: pd.DataFrame,
        target_column: str,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Apply SMOTE to balance the target classes.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input dataframe containing features and target.

        target_column : str
            Name of the target column.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            A new balanced dataframe with synthetic samples added.

        Raises
        ------
        ValueError
            If the target column does not exist.
        """
        import numpy as np

        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # Separate features and target
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        # Count minority samples
        min_class_count = y.value_counts().min()

        # Adjust n_neighbors based on minority class size
        # SMOTE requires n_neighbors <= n_samples - 1
        n_neighbors = min(min_class_count - 1, 5)
        if n_neighbors < 1:
            raise ValueError(f"Insufficient minority class samples for SMOTE. Found {min_class_count} samples, need at least 2.")

        # Apply SMOTE with adjusted n_neighbors (parameter is k_neighbors in imblearn)
        smote = ImblearnSMOTE(random_state=random_state, k_neighbors=n_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Reconstruct dataframe
        balanced_df = pd.DataFrame(
            X_resampled,
            columns=X.columns
        )
        balanced_df[target_column] = y_resampled

        return balanced_df

    @staticmethod
    def downsampler(
        dataframe: pd.DataFrame,
        target_column: str,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Downsample the majority class to match the minority class.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input dataframe containing features and target.

        target_column : str
            Name of the target column.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            A balanced dataframe where all target classes have the
            same number of samples as the minority class.

        Raises
        ------
        ValueError
            If the target column does not exist.
        """
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # Find the size of the minority class
        class_counts = dataframe[target_column].value_counts()
        minority_count = class_counts.min()

        # Downsample every class to the minority count
        dfs = [
            group.sample(n=minority_count, random_state=random_state)
            for _, group in dataframe.groupby(target_column)
        ]
        balanced_df = pd.concat(dfs).reset_index(drop=True)

        # Shuffle the final dataframe
        balanced_df = balanced_df.sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

        return balanced_df