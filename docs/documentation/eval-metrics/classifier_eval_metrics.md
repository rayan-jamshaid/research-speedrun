# Classification Evaluation Metrics

## Confusion Matrix

For binary classification:

|                       | Actual Positive | Actual Negative |
|-----------------------|-----------------|-----------------|
| **Predicted Positive** | TP              | FP              |
| **Predicted Negative** | FN              | TN              |

Where:

- **TP (True Positive):** Positive class correctly predicted as positive.
- **TN (True Negative):** Negative class correctly predicted as negative.
- **FP (False Positive):** Negative class incorrectly predicted as positive.
- **FN (False Negative):** Positive class incorrectly predicted as negative.

---

## 1. Accuracy

The proportion of all predictions that are correctly classified.

$$
\text{Accuracy} =
\frac{TP + TN}
{TP + TN + FP + FN}
$$

---

## 2. Sensitivity / Recall / TPR

The proportion of actual positive cases that are correctly identified.

Also known as:

- **Sensitivity**
- **Recall**
- **True Positive Rate (TPR)**

$$
\text{Sensitivity} =
\text{Recall} =
\text{TPR} =
\frac{TP}
{TP + FN}
$$

---

## 3. Specificity / TNR

The proportion of actual negative cases that are correctly identified.

Also known as:

- **Specificity**
- **True Negative Rate (TNR)**

$$
\text{Specificity} =
\text{TNR} =
\frac{TN}
{TN + FP}
$$

---

## 4. Precision / PPV

The proportion of cases predicted as positive that are actually positive.

Also known as:

- **Precision**
- **Positive Predictive Value (PPV)**

$$
\text{Precision} =
\text{PPV} =
\frac{TP}
{TP + FP}
$$

---

## 5. NPV

The proportion of cases predicted as negative that are actually negative.

**NPV** = Negative Predictive Value.

$$
\text{NPV} =
\frac{TN}
{TN + FN}
$$

---

## 6. F1 Score

The harmonic mean of Precision and Recall.

It provides a balance between false positives and false negatives.

$$
F_1 =
2 \times
\frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
$$

Using the confusion matrix directly:

$$
F_1 =
\frac{2TP}
{2TP + FP + FN}
$$

---

## 7. F-beta Score

The F-beta score is a generalized version of the F1 score that allows different importance to be assigned to Precision and Recall.

$$
F_\beta =
(1+\beta^2)
\frac{\text{Precision} \times \text{Recall}}
{\beta^2\text{Precision} + \text{Recall}}
$$

Where:

- $\beta = 1$ → **F1 score**, equal importance to Precision and Recall.
- $\beta > 1$ → Gives **more importance to Recall**.
- $\beta < 1$ → Gives **more importance to Precision**.

For example:

$$
F_2
$$

places greater emphasis on Recall/Sensitivity than Precision.

---

## 8. ROC-AUC

**ROC-AUC** = Area Under the Receiver Operating Characteristic Curve.

The ROC curve plots:

$$
\text{TPR (Sensitivity)}
$$

against

$$
\text{FPR (False Positive Rate)}
$$

across different classification thresholds.

The False Positive Rate is:

$$
\text{FPR} =
\frac{FP}
{FP + TN}
$$

The **AUC** is the area under the ROC curve:

$$
\text{ROC-AUC}
=
\int_0^1 \text{TPR}(x)\,dx
$$

ROC-AUC measures the model's ability to distinguish between the positive and negative classes across different classification thresholds.

A value of:

- **1.0** → Perfect discrimination
- **0.5** → Random discrimination
- **< 0.5** → Worse than random

---

## 9. PR-AUC

**PR-AUC** = Area Under the Precision-Recall Curve.

The Precision-Recall curve plots:

$$
\text{Precision}
$$

against

$$
\text{Recall}
$$

across different classification thresholds.

The area under this curve is:

$$
\text{PR-AUC}
=
\int_0^1
\text{Precision}(r)\,dr
$$

PR-AUC is particularly useful for **imbalanced classification problems**, because it focuses on the performance of the positive class through Precision and Recall.

---

## 10. FPR

**FPR** = False Positive Rate.

It measures the proportion of actual negative cases that are incorrectly classified as positive.

$$
\text{FPR} =
\frac{FP}
{FP + TN}
$$

It is also related to Specificity:

$$
\text{FPR} = 1 - \text{Specificity}
$$

---

## 11. FNR

**FNR** = False Negative Rate.

It measures the proportion of actual positive cases that are incorrectly classified as negative.

$$
\text{FNR} =
\frac{FN}
{FN + TP}
$$

It is also related to Sensitivity:

$$
\text{FNR} = 1 - \text{Sensitivity}
$$