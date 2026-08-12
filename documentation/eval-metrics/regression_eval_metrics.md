# Regression Evaluation Metrics

For regression:

$$
y_i = \text{actual value}
$$

$$
\hat{y}_i = \text{predicted value}
$$

$$
e_i = y_i - \hat{y}_i
$$

where $e_i$ is the prediction error (residual).

---

## 1. MAE (Mean Absolute Error)

The average absolute difference between the actual and predicted values.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

**Lower is better.**

MAE is easy to interpret because it is expressed in the same units as the target variable.

---

## 2. MSE (Mean Squared Error)

The average of the squared prediction errors.

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

**Lower is better.**

Because the errors are squared, MSE gives greater penalty to large errors and is therefore sensitive to outliers.

---

## 3. RMSE (Root Mean Squared Error)

The square root of the Mean Squared Error.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

**Lower is better.**

RMSE is expressed in the same units as the target variable while still penalizing large errors more strongly than MAE.

---

## 4. Median Absolute Error (MedAE)

The median of the absolute prediction errors.

$$
MedAE =
\operatorname{median}
\left(
|y_i-\hat{y}_i|
\right)
$$

**Lower is better.**

MedAE is more robust to outliers than MAE and RMSE.

---

## 5. MBE (Mean Bias Error)

Measures the average direction and magnitude of prediction bias.

$$
MBE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)
$$

A value close to zero indicates little overall bias.

- **MBE > 0:** Model tends to underpredict.
- **MBE < 0:** Model tends to overpredict.
- **MBE = 0:** No average systematic bias.

Unlike MAE, positive and negative errors can cancel each other out.

---

## 6. R² (Coefficient of Determination)

Measures how much of the variance in the target variable is explained by the model.

$$
R^2 =
1-
\frac
{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

where:

$$
\bar{y} =
\frac{1}{n}
\sum_{i=1}^{n}y_i
$$

Typical interpretation:

- **$R^2 = 1$** → Perfect predictions.
- **$R^2 = 0$** → No improvement over predicting the mean.
- **$R^2 < 0$** → Worse than simply predicting the mean.
- **Higher $R^2$** → Better fit.

## 7. MAPE (Mean Absolute Percentage Error)

Measures the average absolute percentage difference between the actual and predicted values.

$$
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}
{y_i}
\right|
$$

**Lower is better.**

For example, a MAPE of $10\%$ means that the predictions have an average absolute percentage error of approximately $10\%$.

> **Note:** MAPE is problematic when actual values $y_i$ are zero or very close to zero, because the percentage error becomes undefined or extremely large.