import pandas as pd
import numpy as np
from scipy import stats


class Outlier:
    """
    Utility class for detecting outliers and replacing them with NaN.

    Unlike row-removal methods, these methods preserve all rows and
    replace only the detected outlier values with NaN. The resulting
    missing values can then be handled by an imputation method.
    """

    @staticmethod
    def iqr(
        df: pd.DataFrame,
        columns: list,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Replaces outliers with NaN using the Interquartile Range (IQR) method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        columns : list
            List of numerical columns on which to detect outliers.

        threshold : float
            IQR multiplier. Default is 1.5.

        Returns
        -------
        pd.DataFrame
            Dataframe with detected outlier values replaced by NaN.
            Rows are NOT removed.
        """

        df_cleaned = df.copy()

        for col in columns:

            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)

            iqr_value = q3 - q1

            lower_bound = q1 - threshold * iqr_value
            upper_bound = q3 + threshold * iqr_value

            # Replace outlier values with NaN
            outlier_mask = (
                (df_cleaned[col] < lower_bound) |
                (df_cleaned[col] > upper_bound)
            )

            df_cleaned.loc[outlier_mask, col] = np.nan

        return df_cleaned

    @staticmethod
    def modified_z_score(
        df: pd.DataFrame,
        columns: list,
        threshold: float = 3.5
    ) -> pd.DataFrame:
        """
        Replaces outliers with NaN using the Modified Z-score method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        columns : list
            List of numerical columns on which to detect outliers.

        threshold : float
            Modified Z-score threshold. Default is 3.5.

        Returns
        -------
        pd.DataFrame
            Dataframe with detected outlier values replaced by NaN.
            Rows are NOT removed.
        """

        df_cleaned = df.copy()

        for col in columns:

            series = df_cleaned[col]

            median = series.median()

            mad = stats.median_abs_deviation(
                series,
                scale="normal",
                nan_policy="omit"
            )

            # Skip if MAD cannot be calculated
            if mad == 0 or np.isnan(mad):
                continue

            # Calculate modified Z-score
            mod_z_scores = (series - median) / mad

            # Identify outliers
            outlier_mask = np.abs(mod_z_scores) > threshold

            # Replace only the outlier values with NaN
            df_cleaned.loc[outlier_mask, col] = np.nan

        return df_cleaned