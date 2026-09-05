import os
from abc import ABC, abstractmethod
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import optuna

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    auc
)


###############################################################################
# GPU Helper
###############################################################################

def _has_gpu() -> bool:
    """
    Return True if a CUDA-capable GPU is available.
    Uses PyTorch for detection; falls back to False if unavailable.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Secondary fallback: CatBoost GPU counter
    try:
        from catboost.utils import get_gpu_device_count
        return get_gpu_device_count() > 0
    except Exception:
        return False


###############################################################################
# Abstract Base Model
###############################################################################

class BaseModel(ABC):
    """
    Abstract base class for all ML model wrappers.
    Follows SOLID principles by providing a common interface and shared
    evaluation logic for every concrete model implementation.
    """

    def __init__(self, results_dir: str = "../results"):
        """
        Parameters
        ----------
        results_dir : str
            Directory where evaluation plots will be saved.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_external=None,
        y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model and evaluate it on validation, test, and
        (optionally) external datasets.

        Parameters
        ----------
        X_train, y_train
            Training data.
        X_val, y_val
            Validation data.
        X_test, y_test
            Test data.
        X_external, y_external
            External dataset for evaluation (optional).
        kwargs
            Additional model-specific hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", and optionally "external".
            Each dataset entry contains "metrics", "predictions",
            "probabilities", "confusion_matrix", and "images".
        """

    # ------------------------------------------------------------------
    # Shared private helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model,
        X,
        y,
        dataset_name: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model and save diagnostic plots.

        Returns
        -------
        dict
            Contains metrics, predictions, probabilities, confusion matrix
            values, and saved image paths.
        """

        save_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)

        y_pred = model.predict(X)

        # CatBoost can return shape (n, 1)
        y_pred = np.asarray(y_pred).astype(int).flatten()

        y_prob = model.predict_proba(X)[:, 1]

        ###################################################################
        # Metrics
        ###################################################################

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "accuracy":  accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall":    recall_score(y, y_pred),
            "f1":        f1_score(y, y_pred),
            "f2_score":  fbeta_score(y, y_pred, beta=2),
            "roc_auc":   roc_auc_score(y, y_prob),
        }

        # Specificity (TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # NPV (Negative Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # FPR (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # FNR (False Negative Rate)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall_vals, precision_vals)

        metrics.update({
            "specificity": specificity,
            "npv":         npv,
            "fpr":         fpr,
            "fnr":         fnr,
            "pr_auc":      pr_auc,
        })

        ###################################################################
        # Confusion Matrix plot
        ###################################################################

        cm_path = os.path.join(save_dir, f"{dataset_name}_confusion_matrix.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap="Blues", ax=ax)
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        cm_values = {
            "true_negative":  int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive":  int(tp),
        }

        ###################################################################
        # ROC Curve plot
        ###################################################################

        roc_path = os.path.join(save_dir, f"{dataset_name}_roc_curve.png")

        fig, ax = plt.subplots(figsize=(6, 6))
        RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        ###################################################################
        # PR-AUC Curve plot
        ###################################################################

        pr_path = os.path.join(save_dir, f"{dataset_name}_pr_curve.png")

        fig, ax = plt.subplots(figsize=(6, 6))
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
        ax.plot(recall_vals, precision_vals, label=f'PR-AUC = {pr_auc:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()

        ###################################################################
        # Return
        ###################################################################

        return {
            "metrics":          metrics,
            "predictions":      y_pred,
            "probabilities":    y_prob,
            "confusion_matrix": cm_values,
            "images": {
                "confusion_matrix": cm_path,
                "roc_curve":        roc_path,
                "pr_curve":         pr_path,
            },
        }

    # ------------------------------------------------------------------
    # Shared result-building helper
    # ------------------------------------------------------------------

    def _build_result(
        self,
        model,
        X_val, y_val,
        X_test, y_test,
        X_external, y_external,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on all provided splits and return a unified result dict.
        """

        validation_results = self._evaluate(model, X_val,  y_val,  "validation", model_name)
        test_results       = self._evaluate(model, X_test, y_test, "test",       model_name)

        result = {
            "model":      model,
            "validation": validation_results,
            "test":       test_results,
        }

        if X_external is not None and y_external is not None:
            result["external"] = self._evaluate(
                model, X_external, y_external, "external", model_name
            )

        return result


###############################################################################
# Concrete Model Classes
###############################################################################

class CatBoostModel(BaseModel):
    """
    CatBoost classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a CatBoost classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional CatBoost hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "iterations":    500,
            "learning_rate": 0.05,
            "depth":         6,
            "loss_function": "Logloss",
            "eval_metric":   "AUC",
            "verbose":       False,
            "random_seed":   42,
        }

        if _has_gpu():
            default_params["task_type"] = "GPU"

        default_params.update(kwargs)

        model = CatBoostClassifier(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "CatBoost"
        )


class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate an XGBoost classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional XGBoost hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "n_estimators":  500,
            "learning_rate": 0.05,
            "max_depth":     6,
            "objective":     "binary:logistic",
            "eval_metric":   "auc",
            "random_state":  42,
        }

        if _has_gpu():
            default_params["device"] = "cuda"

        # Map CatBoost-style kwargs to XGBoost equivalents
        mapped_kwargs = {}
        for k, v in kwargs.items():
            if k == "iterations":
                mapped_kwargs["n_estimators"] = v
            elif k == "depth":
                mapped_kwargs["max_depth"] = v
            elif k in ("loss_function", "verbose"):
                pass  # not applicable to XGBoost
            elif k == "random_seed":
                mapped_kwargs["random_state"] = v
            else:
                mapped_kwargs[k] = v

        default_params.update(mapped_kwargs)

        model = XGBClassifier(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "XGBoost"
        )


class XGBoostRandomSearchModel(BaseModel):
    """
    XGBoost classifier wrapper with RandomizedSearchCV for hyperparameter tuning.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        param_distributions=None,
        n_iter=10,
        cv=5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate an XGBoost classifier using RandomizedSearchCV.
        """

        if param_distributions is None:
            param_distributions = {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }

        base_params = {
            "objective":     "binary:logistic",
            "eval_metric":   "auc",
            "random_state":  42,
        }

        if _has_gpu():
            base_params["device"] = "cuda"

        model = XGBClassifier(**base_params)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        return self._build_result(
            best_model, X_val, y_val, X_test, y_test, X_external, y_external, "XGBoost_RandomSearch"
        )


class XGBoostOptunaModel(BaseModel):
    """
    XGBoost classifier wrapper with Optuna for hyperparameter optimization.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        n_trials=20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate an XGBoost classifier using Optuna.
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
            }
            if _has_gpu():
                params["device"] = "cuda"

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_val_prob = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_val_prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "auc"
        best_params["random_state"] = 42
        if _has_gpu():
            best_params["device"] = "cuda"

        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)

        return self._build_result(
            best_model, X_val, y_val, X_test, y_test, X_external, y_external, "XGBoost_Optuna"
        )


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a Random Forest classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional RandomForest hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "n_estimators": 100,
            "max_depth":    6,
            "random_state": 42,
            "n_jobs":       -1,
        }

        default_params.update(kwargs)

        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "RandomForest"
        )


class DecisionTreeModel(BaseModel):
    """
    Decision Tree classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a Decision Tree classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional DecisionTree hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "max_depth":    6,
            "random_state": 42,
        }

        default_params.update(kwargs)

        model = DecisionTreeClassifier(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "DecisionTree"
        )


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a Logistic Regression classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional LogisticRegression hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "max_iter":     1000,
            "random_state": 42,
        }

        default_params.update(kwargs)

        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "LogisticRegression"
        )


class LightGBMModel(BaseModel):
    """
    LightGBM classifier wrapper.
    """

    def run(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train and evaluate a LightGBM classifier.

        Parameters
        ----------
        X_train, y_train : Training data.
        X_val,   y_val   : Validation data.
        X_test,  y_test  : Test data.
        X_external, y_external : External dataset (optional).
        kwargs : Additional LightGBM hyperparameters.

        Returns
        -------
        dict
            Keys: "model", "validation", "test", optionally "external".
        """

        default_params = {
            "n_estimators":  100,
            "learning_rate": 0.05,
            "max_depth":     6,
            "random_state":  42,
            "verbosity":     -1,
        }

        if _has_gpu():
            default_params["device"] = "gpu"

        default_params.update(kwargs)

        model = LGBMClassifier(**default_params)
        model.fit(X_train, y_train)

        return self._build_result(
            model, X_val, y_val, X_test, y_test, X_external, y_external, "LightGBM"
        )


###############################################################################
# Facade — backward-compatible Models class
###############################################################################

class Models:
    """
    Facade class that preserves the original interface.

    Delegates each method call to the corresponding concrete model class,
    following the same pattern as the Writer / BaseWriter design.

    Current Models
    --------------
    - CatBoost          → CatBoostModel
    - XGBoost           → XGBoostModel
    - XGBoost Random Search → XGBoostRandomSearchModel
    - XGBoost Optuna    → XGBoostOptunaModel
    - Random Forest     → RandomForestModel
    - Decision Tree     → DecisionTreeModel
    - Logistic Regression → LogisticRegressionModel
    - LightGBM          → LightGBMModel
    """

    def __init__(self, results_dir: str = "../results"):
        """
        Parameters
        ----------
        results_dir : str
            Directory where plots will be saved.
        """
        self.results_dir = results_dir
        self._catboost           = CatBoostModel(results_dir)
        self._xgboost            = XGBoostModel(results_dir)
        self._xgboost_random_search = XGBoostRandomSearchModel(results_dir)
        self._xgboost_optuna     = XGBoostOptunaModel(results_dir)
        self._random_forest      = RandomForestModel(results_dir)
        self._decision_tree      = DecisionTreeModel(results_dir)
        self._logistic_regression = LogisticRegressionModel(results_dir)
        self._lightgbm           = LightGBMModel(results_dir)

    # ------------------------------------------------------------------
    # Public methods — identical signatures to the original Models class
    # ------------------------------------------------------------------

    def catboost(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate CatBoost. Delegates to CatBoostModel."""
        return self._catboost.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def xgboost(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate XGBoost. Delegates to XGBoostModel."""
        return self._xgboost.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def xgboost_random_search(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate XGBoost with RandomizedSearchCV. Delegates to XGBoostRandomSearchModel."""
        return self._xgboost_random_search.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def xgboost_optuna(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate XGBoost with Optuna. Delegates to XGBoostOptunaModel."""
        return self._xgboost_optuna.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def random_forest(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate Random Forest. Delegates to RandomForestModel."""
        return self._random_forest.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def decision_tree(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate Decision Tree. Delegates to DecisionTreeModel."""
        return self._decision_tree.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def logistic_regression(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate Logistic Regression. Delegates to LogisticRegressionModel."""
        return self._logistic_regression.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )

    def lightgbm(
        self,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_external=None, y_external=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate LightGBM. Delegates to LightGBMModel."""
        return self._lightgbm.run(
            X_train, y_train, X_val, y_val, X_test, y_test,
            X_external, y_external, **kwargs
        )