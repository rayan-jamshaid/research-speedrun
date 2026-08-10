import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


class Models:
    """
    A collection of Machine Learning models.

    Current Models
    --------------
    - CatBoost
    """

    def __init__(self, results_dir: str = "../results"):
        """
        Parameters
        ----------
        results_dir : str
            Directory where plots will be saved.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    ###########################################################################
    # Private Methods
    ###########################################################################

    def _evaluate(
        self,
        model,
        X,
        y,
        dataset_name: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Returns
        -------
        Dictionary containing metrics, predictions,
        probabilities and saved image paths.
        """

        save_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)

        y_pred = model.predict(X)

        # CatBoost returns shape (n,1)
        y_pred = np.asarray(y_pred).astype(int).flatten()

        y_prob = model.predict_proba(X)[:, 1]

        #######################################################################
        # Metrics
        #######################################################################

        metrics = {

            "accuracy":
                accuracy_score(y, y_pred),

            "precision":
                precision_score(y, y_pred),

            "recall":
                recall_score(y, y_pred),

            "f1":
                f1_score(y, y_pred),

            "roc_auc":
                roc_auc_score(y, y_prob)
        }

        #######################################################################
        # Confusion Matrix
        #######################################################################

        cm_path = os.path.join(
            save_dir,
            f"{dataset_name}_confusion_matrix.png"
        )

        fig, ax = plt.subplots(figsize=(5, 5))

        ConfusionMatrixDisplay.from_predictions(
            y,
            y_pred,
            cmap="Blues",
            ax=ax
        )

        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        # Extract raw confusion matrix values
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cm_values = {
            "true_negative":  int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive":  int(tp),
        }

        #######################################################################
        # ROC Curve
        #######################################################################

        roc_path = os.path.join(
            save_dir,
            f"{dataset_name}_roc_curve.png"
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        RocCurveDisplay.from_predictions(
            y,
            y_prob,
            ax=ax
        )

        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        #######################################################################
        # Return
        #######################################################################

        return {

            "metrics": metrics,

            "predictions": y_pred,

            "probabilities": y_prob,

            "confusion_matrix": cm_values,

            "images": {

                "confusion_matrix": cm_path,

                "roc_curve": roc_path
            }

        }

    ###########################################################################
    # Public Methods
    ###########################################################################

    def catboost(
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
        Train and evaluate CatBoost.

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
            Additional CatBoost parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        # Check if GPU is present for CatBoost
        has_gpu = False
        try:
            from catboost.utils import get_gpu_device_count
            has_gpu = get_gpu_device_count() > 0
        except Exception:
            pass

        default_params = {

            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": 42

        }

        if has_gpu:
            default_params["task_type"] = "GPU"

        default_params.update(kwargs)

        model = CatBoostClassifier(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "CatBoost"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "CatBoost"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "CatBoost"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }
        if external_results is not None:
            result["external"] = external_results

        return result



    def xgboost(
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
        Train and evaluate XGBoost.

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
            Additional XGBoost parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        # Check if GPU is present for XGBoost
        has_gpu = False
        try:
            from catboost.utils import get_gpu_device_count
            has_gpu = get_gpu_device_count() > 0
        except Exception:
            pass

        default_params = {

            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42

        }

        if has_gpu:
            default_params["device"] = "cuda"

        # Map incompatible kwargs from list_processor.py to XGBoost parameters
        mapped_kwargs = {}
        for k, v in kwargs.items():
            if k == "iterations":
                mapped_kwargs["n_estimators"] = v
            elif k == "depth":
                mapped_kwargs["max_depth"] = v
            elif k == "loss_function":
                pass
            elif k == "random_seed":
                mapped_kwargs["random_state"] = v
            elif k == "verbose":
                pass
            else:
                mapped_kwargs[k] = v

        default_params.update(mapped_kwargs)

        model = XGBClassifier(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "XGBoost"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "XGBoost"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "XGBoost"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }
        if external_results is not None:
            result["external"] = external_results

        return result

    def random_forest(
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
        Train and evaluate Random Forest.

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
            Additional Random Forest parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1
        }

        default_params.update(kwargs)

        model = RandomForestClassifier(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "RandomForest"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "RandomForest"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "RandomForest"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }
        if external_results is not None:
            result["external"] = external_results

        return result

    def decision_tree(
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
        Train and evaluate Decision Tree.

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
            Additional Decision Tree parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        default_params = {
            "max_depth": 6,
            "random_state": 42
        }

        default_params.update(kwargs)

        model = DecisionTreeClassifier(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "DecisionTree"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "DecisionTree"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "DecisionTree"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }
        if external_results is not None:
            result["external"] = external_results

        return result

    def logistic_regression(
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
        Train and evaluate Logistic Regression.

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
            Additional Logistic Regression parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        default_params = {
            "max_iter": 1000,
            "random_state": 42
        }

        default_params.update(kwargs)

        model = LogisticRegression(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "LogisticRegression"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "LogisticRegression"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "LogisticRegression"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }

        if external_results is not None:
            result["external"] = external_results

        return result


    def lightgbm(
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
        Train and evaluate LightGBM.

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
            Additional LightGBM parameters.

        Returns
        -------
        dict
            Validation metrics,
            Test metrics,
            External metrics (if provided),
            Image paths,
            Trained model.
        """

        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42,
            "verbosity": -1
        }

        default_params.update(kwargs)

        model = LGBMClassifier(
            **default_params
        )

        model.fit(
            X_train,
            y_train
        )

        validation_results = self._evaluate(
            model,
            X_val,
            y_val,
            "validation",
            "LightGBM"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "LightGBM"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "LightGBM"
            )

        result = {
            "model": model,
            "validation": validation_results,
            "test": test_results,
        }

        if external_results is not None:
            result["external"] = external_results

        return result