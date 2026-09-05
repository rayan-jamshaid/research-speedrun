import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


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

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor


class ModelsRegression:
    """
    A collection of Machine Learning models for regression tasks.

    Current Models
    --------------
    - CatBoost
    - XGBoost
    - Random Forest
    - Decision Tree
    - Linear Regression
    - LightGBM
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
        Evaluate a trained regression model.

        Returns
        -------
        Dictionary containing metrics, predictions,
        and saved image paths.
        """

        save_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)

        y_pred = model.predict(X)
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y).flatten()

        #######################################################################
        # Metrics
        #######################################################################

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mean_absolute_error(y_true, y_pred),
            "medae": median_absolute_error(y_true, y_pred),
            "mbe": np.mean(y_true - y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred)
        }

        #######################################################################
        # Actual vs Predicted Scatter Plot
        #######################################################################

        avp_path = os.path.join(
            save_dir,
            f"{dataset_name}_actual_vs_predicted.png"
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="k", linewidths=0.5)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{model_name} - Actual vs Predicted ({dataset_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(avp_path)
        plt.close()

        #######################################################################
        # Residual Plot
        #######################################################################

        residuals = y_true - y_pred

        res_path = os.path.join(
            save_dir,
            f"{dataset_name}_residuals.png"
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(y_pred, residuals, alpha=0.6, s=30, edgecolors="k", linewidths=0.5)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (Actual - Predicted)")
        ax.set_title(f"{model_name} - Residual Plot ({dataset_name})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(res_path)
        plt.close()

        #######################################################################
        # Return
        #######################################################################

        return {

            "metrics": metrics,

            "predictions": y_pred,

            "images": {
                "actual_vs_predicted": avp_path,
                "residuals": res_path
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
        Train and evaluate CatBoost Regressor.

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
        has_gpu = _has_gpu()

        default_params = {

            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": False,
            "random_seed": 42

        }

        if has_gpu:
            default_params["task_type"] = "GPU"

        default_params.update(kwargs)

        model = CatBoostRegressor(
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
        Train and evaluate XGBoost Regressor.

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
        has_gpu = _has_gpu()

        default_params = {

            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
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

        model = XGBRegressor(
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
        Train and evaluate Random Forest Regressor.

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

        model = RandomForestRegressor(
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
        Train and evaluate Decision Tree Regressor.

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

        model = DecisionTreeRegressor(
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

    def linear_regression(
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
        Train and evaluate Linear Regression.

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
            Additional Linear Regression parameters.

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
            "n_jobs": -1
        }

        default_params.update(kwargs)

        model = LinearRegression(
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
            "LinearRegression"
        )

        test_results = self._evaluate(
            model,
            X_test,
            y_test,
            "test",
            "LinearRegression"
        )

        # Evaluate on external dataset if provided
        external_results = None
        if X_external is not None and y_external is not None:
            external_results = self._evaluate(
                model,
                X_external,
                y_external,
                "external",
                "LinearRegression"
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
        Train and evaluate LightGBM Regressor.

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
            "verbosity": -1,
            "objective": "regression"
        }

        # Enable GPU acceleration when a CUDA device is available
        if _has_gpu():
            default_params["device"] = "gpu"

        default_params.update(kwargs)

        model = LGBMRegressor(
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
