# Imputation Class

The `Imputation` class provides a collection of methods for handling missing values in tabular datasets. Each method accepts a pandas DataFrame and returns a new DataFrame with missing values imputed.

## Features

- Returns a **new DataFrame** (original DataFrame is not modified).
- Supports both **univariate** and **multivariate** imputation methods.
- Uses implementations provided by **scikit-learn**.
- Designed to be used as a preprocessing step in an ML pipeline.

---

# Methods

## 1. Mean Imputation

```python
Imputation.mean(df, target_column)
```

Replaces missing values in the specified column with the **mean** of that column.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Column to impute. |

### Returns

- `pd.DataFrame`

### Algorithm

Uses:

```python
sklearn.impute.SimpleImputer(strategy="mean")
```

### Suitable For

- Continuous numerical features
- Approximately normally distributed variables

---

## 2. Median Imputation

```python
Imputation.median(df, target_column)
```

Replaces missing values in the specified column with the **median**.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Column to impute. |

### Returns

- `pd.DataFrame`

### Algorithm

Uses:

```python
SimpleImputer(strategy="median")
```

### Suitable For

- Numerical variables
- Skewed distributions
- Medical laboratory values

---

## 3. Mode Imputation

```python
Imputation.mode(df, target_column)
```

Replaces missing values with the most frequently occurring value.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Column to impute. |

### Returns

- `pd.DataFrame`

### Algorithm

Uses:

```python
SimpleImputer(strategy="most_frequent")
```

### Suitable For

- Categorical variables
- Binary features
- Ordinal features

---

## 4. KNN Imputation

```python
Imputation.knn(
    df,
    target_column,
    n_neighbors=5,
    weights="uniform"
)
```

Uses the **K-Nearest Neighbors (KNN)** algorithm to estimate missing values based on similar samples.

> Although `target_column` is included for API consistency, KNN imputation operates on **all numeric columns** simultaneously.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Included for interface consistency. |
| `n_neighbors` | `int` | Number of neighbors. |
| `weights` | `str` | `"uniform"` or `"distance"`. |

### Returns

- `pd.DataFrame`

### Algorithm

Uses:

```python
sklearn.impute.KNNImputer
```

### Suitable For

- Correlated numerical features
- Medical datasets
- Laboratory measurements

---

## 5. Iterative Imputation

```python
Imputation.iterative(
    df,
    target_column,
    random_state=42,
    max_iter=10
)
```

Uses **Multivariate Imputation by Chained Equations (MICE)** through `IterativeImputer`.

Each feature with missing values is predicted using the remaining numerical features.

> Although `target_column` is accepted, the algorithm fits using **all numeric columns**.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Included for interface consistency. |
| `random_state` | `int` | Random seed. |
| `max_iter` | `int` | Maximum number of iterations. |

### Returns

- `pd.DataFrame`

### Default Estimator

```python
BayesianRidge()
```

### Algorithm

Uses:

```python
IterativeImputer(estimator=BayesianRidge())
```

### Suitable For

- Correlated numerical variables
- Medical datasets
- Research applications

---

## 6. Random Forest Imputation

```python
Imputation.random_forest(
    df,
    target_column,
    random_state=42,
    max_iter=10,
    n_estimators=100
)
```

Uses a **Random Forest Regressor** inside `IterativeImputer` to estimate missing values.

Unlike linear iterative imputation, this method captures nonlinear relationships among variables.

> Although `target_column` is accepted, Random Forest imputation uses **all numeric features**.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Input dataframe. |
| `target_column` | `str` | Included for interface consistency. |
| `random_state` | `int` | Random seed. |
| `max_iter` | `int` | Number of MICE iterations. |
| `n_estimators` | `int` | Number of trees in the Random Forest. |

### Returns

- `pd.DataFrame`

### Algorithm

Uses:

```python
IterativeImputer(
    estimator=RandomForestRegressor(...)
)
```

### Suitable For

- Complex nonlinear datasets
- Medical research
- High-dimensional tabular data

---

# Notes

- Mean, Median, and Mode perform **univariate** imputation, modifying only the specified target column.
- KNN, Iterative, and Random Forest perform **multivariate** imputation using all numerical features to estimate missing values.
- The original dataframe is never modified; each method returns a new dataframe.
- Non-numeric columns remain unchanged when using multivariate imputers.
- For research reproducibility, `random_state` should be fixed when using stochastic methods.

---

# Example

```python
from preprocessing.imputation import Imputation

df = Imputation.mean(df, "age")

df = Imputation.median(df, "creatinine")

df = Imputation.mode(df, "sex")

df = Imputation.knn(df, "albumin")

df = Imputation.iterative(df, "bilirubin")

df = Imputation.random_forest(df, "platelets")
```
