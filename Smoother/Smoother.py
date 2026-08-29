import pandas as pd
from scipy.stats.mstats import winsorize


class Smoother:
    """
    Data smoothing techniques for tabular datasets.

    Methods
    -------
    winsorization(df, limits=(0.01, 0.01), columns=None)
        Applies Winsorization to numerical columns by capping extreme values.
    """

    @staticmethod
    def winsorization(
        df: pd.DataFrame,
        limits: tuple = (0.01, 0.01),
        columns: list = None
    ) -> pd.DataFrame:
        """
        Apply Winsorization smoothing to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        limits : tuple, default=(0.01, 0.01)
            Fraction of data to cap on each side.
            Example:
                (0.01, 0.01) -> cap lowest 1% and highest 1%
                (0.05, 0.05) -> cap lowest 5% and highest 5%

        columns : list, optional
            List of numerical columns to winsorize.
            If None, all numerical columns are used.

        Returns
        -------
        pd.DataFrame
            Winsorized DataFrame.
        """

        new_df = df.copy()

        if columns is None:
            columns = new_df.select_dtypes(include="number").columns

        for col in columns:
            non_nan_series = new_df[col].dropna()
            if len(non_nan_series) > 0:
                new_df.loc[non_nan_series.index, col] = winsorize(non_nan_series, limits=limits)

        return new_df


