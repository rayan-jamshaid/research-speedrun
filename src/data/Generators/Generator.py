import pandas as pd
from imblearn.over_sampling import SMOTE as ImblearnSMOTE
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import TVAESynthesizer


def _has_cuda() -> bool:
    """
    Return True if a CUDA-capable GPU is available via PyTorch.
    Falls back to False if PyTorch is not installed or no GPU is found.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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


    @staticmethod
    def CTGAN(
        dataframe: pd.DataFrame,
        target_column: str,
        epochs: int = 300,
        random_state: int = 42,
        cuda: bool | None = None
    ) -> pd.DataFrame:
        """
        Apply CTGAN to balance the target classes by generating
        synthetic samples for underrepresented classes.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input dataframe containing features and target.

        target_column : str
            Name of the target column.

        epochs : int, default=300
            Number of epochs used to train CTGAN.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            A new balanced dataframe containing the original data
            plus synthetic samples generated by CTGAN.

        Raises
        ------
        ValueError
            If the target column does not exist or there is insufficient
            data to train CTGAN.
        """

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        if dataframe.empty:
            raise ValueError("Cannot train CTGAN on an empty dataframe.")

        df = dataframe.copy()

        # Number of samples required for each class
        class_counts = df[target_column].value_counts()
        max_class_count = class_counts.max()

        # Detect dataframe metadata
        metadata = Metadata.detect_from_dataframe(
            data=df,
            table_name="training_data"
        )

        # Make sure the target is treated as categorical.
        # This is particularly important for binary targets such as
        # mortality_flag = 0/1.
        metadata.update_column(
            column_name=target_column,
            sdtype="categorical"
        )

        # Resolve GPU flag: auto-detect when not explicitly set
        use_cuda = _has_cuda() if cuda is None else cuda

        # Create and train CTGAN
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            verbose=True,
            cuda=use_cuda
        )

        synthesizer.fit(df)

        synthetic_parts = []

        # Generate enough synthetic samples for every class
        # that is smaller than the majority class.
        for class_value, class_count in class_counts.items():

            samples_needed = max_class_count - class_count

            if samples_needed <= 0:
                continue

            condition = Condition(
                num_rows=samples_needed,
                column_values={
                    target_column: class_value
                }
            )

            synthetic_data = synthesizer.sample_from_conditions(
                conditions=[condition]
            )

            synthetic_parts.append(synthetic_data)

        # If the dataset was already balanced, return a copy
        if not synthetic_parts:
            return df.reset_index(drop=True)

        # Combine original + synthetic data
        synthetic_df = pd.concat(
            synthetic_parts,
            ignore_index=True
        )

        balanced_df = pd.concat(
            [df, synthetic_df],
            ignore_index=True
        )

        # Shuffle
        balanced_df = balanced_df.sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

        return balanced_df


    @staticmethod
    def CopulaGAN(
        dataframe: pd.DataFrame,
        target_column: str,
        epochs: int = 300,
        random_state: int = 42,
        cuda: bool | None = None
    ) -> pd.DataFrame:

        df = dataframe.copy()

        metadata = Metadata.detect_from_dataframe(
            data=df
        )

        # Resolve GPU flag: auto-detect when not explicitly set
        use_cuda = _has_cuda() if cuda is None else cuda

        synthesizer = CopulaGANSynthesizer(
            metadata,
            epochs=epochs,
            verbose=True,
            cuda=use_cuda
        )

        synthesizer.fit(df)

        minority_class = df[target_column].value_counts().idxmin()
        majority_count = df[target_column].value_counts().max()
        minority_count = df[target_column].value_counts().min()

        samples_needed = majority_count - minority_count

        if samples_needed <= 0:
            return df

        condition = Condition(
            num_rows=samples_needed,
            column_values={
                target_column: minority_class
            }
        )

        synthetic_data = synthesizer.sample_from_conditions(
            conditions=[condition]
        )

        return pd.concat(
            [df, synthetic_data],
            ignore_index=True
        )




    
    @staticmethod
    def TVAE(
        dataframe: pd.DataFrame,
        target_column: str,
        epochs: int = 300,
        random_state: int = 42,
        cuda: bool | None = None
    ) -> pd.DataFrame:

        df = dataframe.copy()

        metadata = Metadata.detect_from_dataframe(
            data=df
        )

        # Resolve GPU flag: auto-detect when not explicitly set
        use_cuda = _has_cuda() if cuda is None else cuda

        synthesizer = TVAESynthesizer(
            metadata,
            epochs=epochs,
            verbose=True,
            cuda=use_cuda
        )

        synthesizer.fit(df)

        minority_class = df[target_column].value_counts().idxmin()
        majority_count = df[target_column].value_counts().max()
        minority_count = df[target_column].value_counts().min()

        samples_needed = majority_count - minority_count

        if samples_needed <= 0:
            return df

        condition = Condition(
            num_rows=samples_needed,
            column_values={
                target_column: minority_class
            }
        )

        synthetic_data = synthesizer.sample_from_conditions(
            conditions=[condition]
        )

        return pd.concat(
            [df, synthetic_data],
            ignore_index=True
        )