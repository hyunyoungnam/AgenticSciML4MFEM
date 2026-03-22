"""
Base classes for surrogate models.

Defines the interface for surrogate models that predict FEM outputs
from input parameters without running full simulations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SurrogateType(Enum):
    """Types of surrogate models."""
    DEEPONET = auto()


@dataclass
class SurrogateConfig:
    """
    Configuration for surrogate model.

    Attributes:
        model_type: Type of surrogate model
        branch_layers: Layer sizes for branch network (encodes input functions)
        trunk_layers: Layer sizes for trunk network (encodes coordinates)
        activation: Activation function name
        learning_rate: Initial learning rate
        batch_size: Training batch size
        epochs: Maximum training epochs
        patience: Early stopping patience
        output_dim: Dimension of output field
        checkpoint_dir: Directory for saving checkpoints
    """
    model_type: SurrogateType = SurrogateType.DEEPONET
    branch_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    trunk_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    activation: str = "tanh"
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10000
    patience: int = 1000
    output_dim: int = 1
    checkpoint_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type.name,
            "branch_layers": self.branch_layers,
            "trunk_layers": self.trunk_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "output_dim": self.output_dim,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
        }


@dataclass
class PredictionResult:
    """
    Result of surrogate model prediction.

    Attributes:
        values: Predicted field values at query points
        uncertainty: Uncertainty estimate (if available)
        coordinates: Query point coordinates
        metadata: Additional prediction metadata
    """
    values: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> np.ndarray:
        """Get mean prediction."""
        return self.values

    @property
    def std(self) -> Optional[np.ndarray]:
        """Get standard deviation (uncertainty)."""
        return self.uncertainty

    def max_uncertainty_indices(self, top_k: int = 10) -> np.ndarray:
        """Get indices of points with highest uncertainty."""
        if self.uncertainty is None:
            return np.array([])
        return np.argsort(self.uncertainty.flatten())[-top_k:][::-1]


class SurrogateModel(ABC):
    """
    Abstract base class for surrogate models.

    A surrogate model learns to predict FEM simulation outputs
    (displacement, stress, temperature, etc.) from input parameters
    (geometry, material properties, boundary conditions).
    """

    def __init__(self, config: SurrogateConfig):
        """
        Initialize surrogate model.

        Args:
            config: Model configuration
        """
        self.config = config
        self._is_trained = False
        self._model = None

    @abstractmethod
    def build(
        self,
        input_dim: int,
        coord_dim: int,
        num_sensors: int
    ) -> None:
        """
        Build the model architecture.

        Args:
            input_dim: Dimension of input parameters (branch input)
            coord_dim: Dimension of coordinates (trunk input, typically 2 or 3)
            num_sensors: Number of sensor points (for branch network)
        """
        pass

    @abstractmethod
    def train(
        self,
        branch_inputs: np.ndarray,
        trunk_inputs: np.ndarray,
        outputs: np.ndarray,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the surrogate model.

        Args:
            branch_inputs: Input function values at sensor points (N, num_sensors)
            trunk_inputs: Coordinate points (N * num_points, coord_dim)
            outputs: Target field values (N * num_points, output_dim)
            validation_split: Fraction of data for validation

        Returns:
            Training history dictionary
        """
        pass

    @abstractmethod
    def predict(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with the trained model.

        Args:
            branch_input: Input function values (1, num_sensors) or (N, num_sensors)
            trunk_input: Query coordinates (num_points, coord_dim)

        Returns:
            PredictionResult with predictions and uncertainty
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    def compute_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute prediction errors.

        Args:
            predictions: Predicted values
            targets: Ground truth values

        Returns:
            Dictionary of error metrics
        """
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # Relative errors (avoid division by zero)
        target_norm = np.linalg.norm(targets)
        if target_norm > 1e-10:
            relative_l2 = np.linalg.norm(predictions - targets) / target_norm
        else:
            relative_l2 = float('inf')

        # Max error
        max_error = np.max(np.abs(predictions - targets))

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "relative_l2": float(relative_l2),
            "max_error": float(max_error),
        }
