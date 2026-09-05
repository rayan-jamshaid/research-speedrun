# Models Class

A simple Python class for training and evaluating machine learning models on tabular classification datasets.

**Current Supported Model**

* CatBoost Classifier

The class automatically:

* Trains the model
* Evaluates on validation and test datasets
* Computes common classification metrics
* Saves Confusion Matrix images
* Saves ROC Curve images
* Returns predictions, probabilities, metrics, image paths, and the trained model

---

# Installation

Install the required packages.

```bash
pip install numpy pandas matplotlib scikit-learn catboost
```

---

# Import

```python
from models import Models
```

---

# Initialize

```python
models = Models()
```

By default, all plots are saved to

```
../results/
```

You can specify another location.

```python
models = Models(results_dir="./results")
```

---

# Method

```python
catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    **kwargs
)
```

---

# Parameters

| Parameter | Description                    |
| --------- | ------------------------------ |
| X_train   | Training features              |
| y_train   | Training labels                |
| X_val     | Validation features            |
| y_val     | Validation labels              |
| X_test    | Test features                  |
| y_test    | Test labels                    |
| **kwargs  | Additional CatBoost parameters |

---

# Default CatBoost Parameters

```python
{
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "random_seed": 42
}
```

You can override any parameter.

Example:

```python
results = models.catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    iterations=1000,
    depth=8,
    learning_rate=0.03
)
```

---

# Example

```python
from models import Models

models = Models(results_dir="./results")

results = models.catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
)

print(results["validation"]["metrics"])
print(results["test"]["metrics"])
```

---

# Returned Dictionary

```python
results = {

    "model": trained_model,

    "validation": {

        "metrics": {

            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "roc_auc": ...

        },

        "predictions": ...,

        "probabilities": ...,

        "images": {

            "confusion_matrix": "...",

            "roc_curve": "..."

        }

    },

    "test": {

        "metrics": {

            "accuracy": ...,
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "roc_auc": ...

        },

        "predictions": ...,

        "probabilities": ...,

        "images": {

            "confusion_matrix": "...",

            "roc_curve": "..."

        }

    }

}
```

---

# Accessing Metrics

Validation Accuracy

```python
results["validation"]["metrics"]["accuracy"]
```

Validation Precision

```python
results["validation"]["metrics"]["precision"]
```

Validation Recall

```python
results["validation"]["metrics"]["recall"]
```

Validation F1 Score

```python
results["validation"]["metrics"]["f1"]
```

Validation ROC-AUC

```python
results["validation"]["metrics"]["roc_auc"]
```

Test Accuracy

```python
results["test"]["metrics"]["accuracy"]
```

Test ROC-AUC

```python
results["test"]["metrics"]["roc_auc"]
```

---

# Accessing Predictions

Validation Predictions

```python
results["validation"]["predictions"]
```

Test Predictions

```python
results["test"]["predictions"]
```

---

# Accessing Prediction Probabilities

Validation

```python
results["validation"]["probabilities"]
```

Test

```python
results["test"]["probabilities"]
```

---

# Accessing Saved Images

Validation Confusion Matrix

```python
results["validation"]["images"]["confusion_matrix"]
```

Validation ROC Curve

```python
results["validation"]["images"]["roc_curve"]
```

Test Confusion Matrix

```python
results["test"]["images"]["confusion_matrix"]
```

Test ROC Curve

```python
results["test"]["images"]["roc_curve"]
```

---

# Output Directory

The class automatically creates the following directory structure.

```
results/

└── CatBoost/

    ├── validation_confusion_matrix.png

    ├── validation_roc_curve.png

    ├── test_confusion_matrix.png

    └── test_roc_curve.png
```

---

# Custom CatBoost Parameters

Any CatBoostClassifier parameter can be passed directly.

Example:

```python
results = models.catboost(

    X_train,
    y_train,

    X_val,
    y_val,

    X_test,
    y_test,

    iterations=1200,
    depth=10,
    learning_rate=0.02,
    l2_leaf_reg=5,
    bootstrap_type="Bayesian",
    random_strength=2

)
```

These parameters override the default values.

---

# Current Metrics

The current implementation computes:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# Planned Features

Future versions of the `Models` class will include:

* Random Forest
* XGBoost
* LightGBM
* Extra Trees
* Decision Tree
* AdaBoost
* Gradient Boosting
* HistGradientBoosting

Additional evaluation metrics:

* Balanced Accuracy
* MCC
* Cohen's Kappa
* Log Loss
* Sensitivity
* Specificity
* Precision-Recall Curve
* Feature Importance
* SHAP Explanations
* Model Saving
* Cross Validation
* Hyperparameter Tuning Support
