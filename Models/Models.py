import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

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

        default_params = {

            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": 42

        }

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

        default_params = {

            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "auc",
            "verbose": False,
            "random_seed": 42

        }

        default_params.update(kwargs)

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



