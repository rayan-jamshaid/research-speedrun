import pandas as pd


def missing_value_report(df: pd.DataFrame, output_file: str = "missing_value_report.csv"):
    """
    Generates a report of missing values for every feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    output_file : str
        Path where the CSV report will be saved.

    Returns
    -------
    pd.DataFrame
        Missing value report.
    """

    report = pd.DataFrame({
        "Feature": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean() * 100).round(2).values,
        "Non-Missing Count": df.notna().sum().values
    })

    report = report.sort_values(
        by="Missing Percentage",
        ascending=False
    ).reset_index(drop=True)


    return report



df = pd.read_csv("./data/mimic_iv_merged_data_cleaned.csv")

print(missing_value_report(df, output_file="missing_value_report.csv"))