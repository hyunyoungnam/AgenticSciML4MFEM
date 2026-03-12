"""
Surrogate model trainer.

Handles the training workflow for surrogate models including
data preparation, training, validation, and model checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import SurrogateConfig, SurrogateModel
from .deeponet import DeepONetSurrogate, DeepONetEnsemble


@dataclass
class TrainingConfig:
    """
    Configuration for surrogate training workflow.

    Attributes:
        surrogate_config: Configuration for the surrogate model
        use_ensemble: Whether to use ensemble for uncertainty
        n_ensemble: Number of models in ensemble
        normalize_inputs: Whether to normalize input features
        normalize_outputs: Whether to normalize output targets
        train_test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        save_dir: Directory to save trained model
        log_dir: Directory for training logs
    """
    surrogate_config: SurrogateConfig = field(default_factory=SurrogateConfig)
    use_ensemble: bool = True
    n_ensemble: int = 5
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    train_test_split: float = 0.1
    random_seed: int = 42
    save_dir: Optional[Path] = None
    log_dir: Optional[Path] = None


@dataclass
class TrainingResult:
    """
    Result of surrogate training.

    Attributes:
        success: Whether training completed successfully
        train_loss: Final training loss
        test_loss: Final test loss
        metrics: Dictionary of evaluation metrics
        history: Training history
        model_path: Path to saved model
        normalization_params: Parameters for input/output normalization
    """
    success: bool
    train_loss: float = 0.0
    test_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[Path] = None
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class SurrogateTrainer:
    """
    Trainer for surrogate models.

    Handles the complete training workflow:
    1. Data preprocessing and normalization
    2. Train/test splitting
    3. Model building and training
    4. Validation and evaluation
    5. Model checkpointing
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self._model: Optional[SurrogateModel] = None
        self._input_normalizer: Optional[Normalizer] = None
        self._output_normalizer: Optional[Normalizer] = None

    def prepare_data(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
        outputs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.

        Args:
            parameters: Input parameters shape (N_samples, n_params)
            coordinates: Query coordinates shape (N_points, coord_dim)
            outputs: Output field values shape (N_samples, N_points, output_dim)

        Returns:
            Tuple of (train_params, train_outputs, test_params, test_outputs,
                     coordinates, normalization_params)
        """
        np.random.seed(self.config.random_seed)

        n_samples = parameters.shape[0]
        n_test = int(n_samples * self.config.train_test_split)
        n_train = n_samples - n_test

        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Split data
        train_params = parameters[train_idx]
        test_params = parameters[test_idx]
        train_outputs = outputs[train_idx]
        test_outputs = outputs[test_idx]

        # Normalize if configured
        if self.config.normalize_inputs:
            self._input_normalizer = Normalizer()
            train_params = self._input_normalizer.fit_transform(train_params)
            test_params = self._input_normalizer.transform(test_params)

        if self.config.normalize_outputs:
            self._output_normalizer = Normalizer()
            # Flatten for normalization, then reshape back
            train_shape = train_outputs.shape
            test_shape = test_outputs.shape
            train_outputs = self._output_normalizer.fit_transform(
                train_outputs.reshape(-1, train_outputs.shape[-1])
            ).reshape(train_shape)
            test_outputs = self._output_normalizer.transform(
                test_outputs.reshape(-1, test_outputs.shape[-1])
            ).reshape(test_shape)

        return (
            train_params,
            train_outputs,
            test_params,
            test_outputs,
            coordinates,
            self._get_normalization_params()
        )

    def _get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters for saving/loading."""
        params = {}
        if self._input_normalizer:
            params["input_mean"] = self._input_normalizer.mean.tolist()
            params["input_std"] = self._input_normalizer.std.tolist()
        if self._output_normalizer:
            params["output_mean"] = self._output_normalizer.mean.tolist()
            params["output_std"] = self._output_normalizer.std.tolist()
        return params

    def train(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
        outputs: np.ndarray,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> TrainingResult:
        """
        Train surrogate model.

        Args:
            parameters: Input parameters shape (N_samples, n_params)
            coordinates: Query coordinates shape (N_points, coord_dim)
            outputs: Output field values shape (N_samples, N_points, output_dim)
            callback: Optional callback(epoch, loss) for progress reporting

        Returns:
            TrainingResult with training metrics and model path
        """
        try:
            # Prepare data
            (
                train_params,
                train_outputs,
                test_params,
                test_outputs,
                coords,
                norm_params
            ) = self.prepare_data(parameters, coordinates, outputs)

            # Determine dimensions
            n_params = train_params.shape[1]
            coord_dim = coords.shape[1]
            output_dim = train_outputs.shape[-1] if train_outputs.ndim > 2 else 1

            # Update config output dimension
            self.config.surrogate_config.output_dim = output_dim

            # Create model
            if self.config.use_ensemble:
                self._model = DeepONetEnsemble(
                    self.config.surrogate_config,
                    n_models=self.config.n_ensemble
                )
            else:
                self._model = DeepONetSurrogate(self.config.surrogate_config)

            # Build model
            # For DeepONet, we treat each parameter as a "sensor"
            # So num_sensors = 1 and input_dim = n_params
            self._model.build(
                input_dim=1,
                coord_dim=coord_dim,
                num_sensors=n_params
            )

            # Train
            history = self._model.train(
                branch_inputs=train_params,
                trunk_inputs=coords,
                outputs=train_outputs,
                validation_split=0.1
            )

            # Evaluate on test set
            if len(test_params) > 0:
                predictions = self._model.predict(test_params, coords)

                # Denormalize for error computation
                pred_values = predictions.values
                if self.config.normalize_outputs and self._output_normalizer:
                    pred_values = self._output_normalizer.inverse_transform(
                        pred_values.reshape(-1, output_dim)
                    ).reshape(pred_values.shape)
                    test_outputs_denorm = self._output_normalizer.inverse_transform(
                        test_outputs.reshape(-1, output_dim)
                    ).reshape(test_outputs.shape)
                else:
                    test_outputs_denorm = test_outputs

                errors = self._model.compute_error(
                    pred_values.flatten(),
                    test_outputs_denorm.flatten()
                )
            else:
                errors = {}

            # Save model if directory specified
            model_path = None
            if self.config.save_dir:
                model_path = self.config.save_dir / "surrogate_model"
                self._model.save(model_path)

                # Save normalization params
                import json
                norm_path = self.config.save_dir / "normalization.json"
                with open(norm_path, "w") as f:
                    json.dump(norm_params, f, indent=2)

            return TrainingResult(
                success=True,
                train_loss=history.get("best_loss", 0.0),
                test_loss=errors.get("mse", 0.0),
                metrics=errors,
                history=history,
                model_path=model_path,
                normalization_params=norm_params,
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                error_message=str(e),
            )

    @property
    def model(self) -> Optional[SurrogateModel]:
        """Get the trained model."""
        return self._model

    def predict(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with trained model.

        Handles normalization/denormalization automatically.

        Args:
            parameters: Input parameters
            coordinates: Query coordinates

        Returns:
            Predicted field values (denormalized)
        """
        if self._model is None or not self._model.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Normalize inputs
        if self._input_normalizer:
            parameters = self._input_normalizer.transform(parameters)

        # Predict
        result = self._model.predict(parameters, coordinates)
        predictions = result.values

        # Denormalize outputs
        if self._output_normalizer:
            output_dim = predictions.shape[-1] if predictions.ndim > 2 else 1
            predictions = self._output_normalizer.inverse_transform(
                predictions.reshape(-1, output_dim)
            ).reshape(predictions.shape)

        return predictions


class Normalizer:
    """Simple mean-std normalizer."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        """Fit normalizer to data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-10, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform to original scale."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted.")
        return data * self.std + self.mean
