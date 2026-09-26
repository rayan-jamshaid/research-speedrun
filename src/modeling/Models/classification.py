"""Classification model entry points.

The legacy classical collection remains in :mod:`Models`; quantum models are
kept in dedicated modules so optional PennyLane dependencies stay isolated.
"""

from .Models import Models
from .quantum_classifier import PennyLaneQuantumClassifier

__all__ = ["Models", "PennyLaneQuantumClassifier"]
