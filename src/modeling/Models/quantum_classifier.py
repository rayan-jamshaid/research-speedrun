"""PennyLane variational quantum classifier.

This module intentionally depends on PennyLane lazily so the existing
classical models remain importable when the optional quantum stack is absent.
The public ``run`` method follows the result contract used by ``Models``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PennyLaneQuantumClassifier:
    """Small variational quantum classifier for binary tabular targets.

    Features are standardized and projected onto ``n_qubits`` rotation
    angles.  The circuit uses angle embedding followed by trainable strongly
    entangling layers and measures the first qubit's Z expectation value.
    Training uses PennyLane's differentiable Torch interface.
    """

    def __init__(
        self,
        results_dir: str = "../results",
        n_qubits: int = 4,
        n_layers: int = 2,
        epochs: int = 30,
        learning_rate: float = 0.02,
        batch_size: int = 32,
        random_state: int = 42,
        device_name: str = "default.qubit",
    ) -> None:
        if n_qubits < 1 or n_layers < 1 or epochs < 1:
            raise ValueError("n_qubits, n_layers, and epochs must be positive")
        self.results_dir = results_dir
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.device_name = device_name
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._model = None
        os.makedirs(results_dir, exist_ok=True)

    def _load_pennylane(self):
        try:
            import pennylane as qml
            import torch
        except ImportError as exc:
            raise ImportError(
                "PennyLaneQuantumClassifier requires pennylane and torch. "
                "Install them with: pip install pennylane torch"
            ) from exc
        return qml, torch

    def _build_model(self, n_features: int):
        qml, torch = self._load_pennylane()
        torch.manual_seed(self.random_state)
        n_wires = self.n_qubits
        qdevice = qml.device(self.device_name, wires=n_wires)

        @qml.qnode(qdevice, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        weight_shapes = {"weights": (self.n_layers, n_wires, 3)}
        layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # A classical projection makes arbitrary-width tables compatible with
        # a fixed-size quantum circuit while keeping the quantum part explicit.
        projection = torch.nn.Linear(n_features, n_wires)

        class Hybrid(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = projection
                self.quantum = layer

            def forward(self, x):
                return self.quantum(torch.tanh(self.projection(x)))

        return Hybrid(), torch

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float32)
        if X_array.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric array")
        y_encoded = self.label_encoder.fit_transform(np.asarray(y))
        if len(self.label_encoder.classes_) != 2:
            raise ValueError("PennyLaneQuantumClassifier currently supports binary targets only")
        X_scaled = self.scaler.fit_transform(X_array)
        self._model, torch = self._build_model(X_scaled.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        features = torch.tensor(X_scaled, dtype=torch.float32)
        targets = torch.tensor(y_encoded, dtype=torch.float32)
        generator = torch.Generator().manual_seed(self.random_state)

        self._model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(features), generator=generator)
            for start in range(0, len(features), self.batch_size):
                batch = order[start : start + self.batch_size]
                # Expectation values are in [-1, 1]; convert to logits.
                logits = self._model(features[batch]) * 4.0
                loss = loss_fn(logits, targets[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError("Call fit before predict_proba")
        _, torch = self._load_pennylane()
        features = torch.tensor(self.scaler.transform(np.asarray(X, dtype=np.float32)))
        self._model.eval()
        with torch.no_grad():
            logits = self._model(features) * 4.0
            positive = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        positive = self.predict_proba(X)[:, 1]
        encoded = (positive >= 0.5).astype(int)
        return self.label_encoder.inverse_transform(encoded)

    def _evaluate(self, X, y, name: str) -> Dict[str, Any]:
        y_true = self.label_encoder.transform(np.asarray(y))
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision_values, recall_values, _ = precision_recall_curve(y_true, y_prob)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "npv": tn / (tn + fn) if tn + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0,
            "pr_auc": auc(recall_values, precision_values),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

        stem = os.path.join(self.results_dir, f"quantum_{name}")
        paths = {}
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=ax, cmap="Blues")
        fig.tight_layout(); paths["confusion_matrix"] = stem + "_confusion_matrix.png"; fig.savefig(paths["confusion_matrix"]); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6))
        if metrics["roc_auc"] is not None:
            RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
        fig.tight_layout(); paths["roc_curve"] = stem + "_roc_curve.png"; fig.savefig(paths["roc_curve"]); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(recall_values, precision_values)
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        fig.tight_layout(); paths["pr_curve"] = stem + "_pr_curve.png"; fig.savefig(paths["pr_curve"]); plt.close(fig)
        return {"metrics": metrics, "predictions": y_pred, "probabilities": y_prob,
                "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                                     "false_negative": int(fn), "true_positive": int(tp)},
                "images": paths}

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test,
            X_external: Optional[Any] = None, y_external: Optional[Any] = None,
            **kwargs) -> Dict[str, Any]:
        self.fit(X_train, y_train)
        result = {"model": self,
                  "validation": self._evaluate(X_val, y_val, "validation"),
                  "test": self._evaluate(X_test, y_test, "test")}
        if X_external is not None and y_external is not None:
            result["external"] = self._evaluate(X_external, y_external, "external")
        return result
