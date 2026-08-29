# Class Structure and Responsibilities

This document outlines the core classes in the pipeline and their primary responsibilities.

## 🧠 Models (`Models/`)

- `BaseModel`: Abstract base class for all machine learning model wrappers. Defines the standard contract for model training, prediction, and evaluation.
- `Models`: Facade class that preserves the original unified interface for initializing and interacting with classification models.
- `ModelsRegression`: Facade class designed specifically for initializing and interacting with regression models.
- `CatBoostModel`: Wrapper responsible for training and evaluating CatBoost models.
- `XGBoostModel`: Wrapper responsible for training and evaluating XGBoost models.
- `XGBoostRandomSearchModel`: Wrapper that handles XGBoost training alongside Random Search hyperparameter tuning.
- `XGBoostOptunaModel`: Wrapper that handles XGBoost training alongside Optuna hyperparameter optimization.
- `RandomForestModel`: Wrapper responsible for training and evaluating Random Forest classifiers.
- `DecisionTreeModel`: Wrapper responsible for training and evaluating Decision Tree classifiers.
- `LogisticRegressionModel`: Wrapper responsible for training and evaluating Logistic Regression models.
- `LightGBMModel`: Wrapper responsible for training and evaluating LightGBM models.

## 📝 Writers (`Writers/`)

- `BaseWriter` / `BaseWriterRegression`: Abstract base classes defining the contract for exporting model evaluation results (for classification and regression tasks, respectively).
- `Writer` / `WriterRegression`: Facade classes that provide backward compatibility with the original interface, orchestrating the export process.
- `MarkdownWriter` / `MarkdownWriterRegression`: Responsible for formatting and saving model metrics and evaluation results as Markdown files.
- `HTMLWriter` / `HTMLWriterRegression`: Responsible for formatting and saving model metrics and evaluation results as HTML files.
- `CSVWriter` / `CSVWriterRegression`: Responsible for formatting and saving model metrics and evaluation results as CSV files.

## ⚙️ Data Processing & Engineering

- `Outlier` (`Outlier/Outlier.py`): Utility class responsible for detecting outliers (using methods like IQR or Modified Z-Score) and replacing them with missing value indicators (NaN).
- `Imputation` (`Imputers/Imputer.py`): Utility class responsible for executing missing data imputation techniques (e.g., Iterative Imputer) to fill in NaN values.
- `Smoother` (`Smoother/Smoother.py`): Class responsible for applying feature smoothing techniques (like Winsorization) to clip extreme values and reduce noise.
- `Generator` (`Generators/Generator.py`): Utility class responsible for resampling datasets to fix class imbalance, utilizing oversampling (SMOTE) or downsampling techniques.
- `FeatureSlicer` (`features_slicer/slicer.py`): Utility class responsible for slicing DataFrames to retain only a strictly defined list of features/columns.