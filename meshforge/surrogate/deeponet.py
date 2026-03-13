"""
DeepONet implementation using DeepXDE.

DeepONet (Deep Operator Network) learns mappings between function spaces,
making it ideal for predicting FEM outputs from input parameters.

Reference:
    Lu et al., "Learning nonlinear operators via DeepONet based on the
    universal approximation theorem of operators", Nature Machine Intelligence, 2021.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import PredictionResult, SurrogateConfig, SurrogateModel

# Lazy imports for optional dependencies
_deepxde = None
_tf = None


def _get_deepxde():
    """Lazy import of deepxde module."""
    global _deepxde
    if _deepxde is None:
        try:
            import deepxde as dde
            _deepxde = dde
        except ImportError:
            raise ImportError(
                "DeepXDE is required for DeepONet surrogate. "
                "Install with: pip install deepxde"
            )
    return _deepxde


def _get_backend():
    """Get the DeepXDE backend (TensorFlow, PyTorch, etc.)."""
    dde = _get_deepxde()
    backend = dde.backend.backend_name

    if "tensorflow" in backend:
        import tensorflow as tf
        return tf
    elif "pytorch" in backend:
        import torch
        return torch
    else:
        raise RuntimeError(f"Unsupported DeepXDE backend: {backend}")


class DeepONetSurrogate(SurrogateModel):
    """
    DeepONet surrogate model using DeepXDE.

    This model uses the operator learning paradigm:
    - Branch network: Encodes input function (geometry params, material props, etc.)
    - Trunk network: Encodes output coordinates (spatial locations)
    - Output: Field value at the query location for given input function

    For FEM surrogates:
    - Branch input: [delta_R, E, nu, load_magnitude, ...] or sensor values
    - Trunk input: [x, y] or [x, y, z] coordinates
    - Output: displacement, stress, or temperature at that location
    """

    def __init__(self, config: SurrogateConfig):
        """
        Initialize DeepONet surrogate.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._input_dim: Optional[int] = None
        self._coord_dim: Optional[int] = None
        self._num_sensors: Optional[int] = None
        self._model = None
        self._train_state = None

    def build(
        self,
        input_dim: int,
        coord_dim: int,
        num_sensors: int
    ) -> None:
        """
        Build the DeepONet architecture.

        Args:
            input_dim: Dimension of input parameters per sensor
            coord_dim: Dimension of coordinates (2 for 2D, 3 for 3D)
            num_sensors: Number of sensor points for branch network
        """
        dde = _get_deepxde()

        self._input_dim = input_dim
        self._coord_dim = coord_dim
        self._num_sensors = num_sensors

        # Branch network input size: num_sensors * input_dim
        branch_input_size = num_sensors * input_dim

        # Build layer configurations
        branch_layers = [branch_input_size] + self.config.branch_layers
        trunk_layers = [coord_dim] + self.config.trunk_layers

        # Create DeepONet
        # The last layer of branch and trunk must have the same size
        # DeepXDE handles this automatically with the 'size' parameter

        net = dde.nn.DeepONetCartesianProd(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation=self.config.activation,
            kernel_initializer="Glorot normal",
        )

        self._net = net
        self._is_trained = False

    def train(
        self,
        branch_inputs: np.ndarray,
        trunk_inputs: np.ndarray,
        outputs: np.ndarray,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the DeepONet model.

        Args:
            branch_inputs: Input function values shape (N_samples, num_sensors * input_dim)
            trunk_inputs: Coordinate points shape (N_points, coord_dim) - same for all samples
            outputs: Target values shape (N_samples, N_points, output_dim)
            validation_split: Fraction for validation

        Returns:
            Training history
        """
        dde = _get_deepxde()

        if self._net is None:
            raise RuntimeError("Model not built. Call build() first.")

        # Reshape outputs for DeepXDE: (N_samples, N_points * output_dim)
        n_samples = outputs.shape[0]
        n_points = outputs.shape[1]
        output_dim = outputs.shape[2] if len(outputs.shape) > 2 else 1
        outputs_flat = outputs.reshape(n_samples, -1)

        # Create DeepXDE Triple data
        # DeepONetCartesianProd expects:
        # - X_train = (branch_inputs, trunk_inputs)
        # - y_train = outputs
        data = dde.data.TripleCartesianProd(
            X_train=(branch_inputs, trunk_inputs),
            y_train=outputs_flat,
            X_test=(branch_inputs, trunk_inputs),
            y_test=outputs_flat,
        )

        # Create model
        self._model = dde.Model(data, self._net)

        # Compile with optimizer
        self._model.compile(
            optimizer="adam",
            lr=self.config.learning_rate,
            loss="mse",
            metrics=["mean l2 relative error"],
        )

        # Train with early stopping
        checker = dde.callbacks.EarlyStopping(
            min_delta=1e-6,
            patience=self.config.patience,
        )

        # Model checkpoint if directory specified
        callbacks = [checker]
        if self.config.checkpoint_dir:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model_save = dde.callbacks.ModelCheckpoint(
                str(self.config.checkpoint_dir / "model"),
                save_better_only=True,
                period=100,
            )
            callbacks.append(model_save)

        # Train
        loss_history, train_state = self._model.train(
            iterations=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            display_every=1000,
        )

        self._train_state = train_state
        self._is_trained = True

        # Return training history
        return {
            "loss_train": loss_history.loss_train,
            "loss_test": loss_history.loss_test,
            "metrics_train": loss_history.metrics_train,
            "metrics_test": loss_history.metrics_test,
            "steps": loss_history.steps,
            "best_step": train_state.best_step,
            "best_loss": train_state.best_loss,
        }

    def predict(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with trained model.

        Args:
            branch_input: Input parameters shape (N_samples, num_sensors * input_dim)
                         or (num_sensors * input_dim,) for single sample
            trunk_input: Query coordinates shape (N_points, coord_dim)

        Returns:
            PredictionResult with predictions
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Ensure correct shapes
        if branch_input.ndim == 1:
            branch_input = branch_input.reshape(1, -1)

        # Predict
        predictions = self._model.predict((branch_input, trunk_input))

        # Reshape to (N_samples, N_points, output_dim) if needed
        n_samples = branch_input.shape[0]
        n_points = trunk_input.shape[0]

        if predictions.ndim == 1:
            predictions = predictions.reshape(n_samples, n_points, -1)
        elif predictions.ndim == 2 and n_samples == 1:
            predictions = predictions.reshape(n_samples, n_points, -1)

        return PredictionResult(
            values=predictions,
            coordinates=trunk_input,
            metadata={
                "n_samples": n_samples,
                "n_points": n_points,
            }
        )

    def predict_with_uncertainty(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray,
        n_ensemble: int = 5,
        dropout_rate: float = 0.1
    ) -> PredictionResult:
        """
        Make predictions with uncertainty estimation using MC Dropout.

        Args:
            branch_input: Input parameters
            trunk_input: Query coordinates
            n_ensemble: Number of forward passes for uncertainty
            dropout_rate: Dropout rate for uncertainty estimation

        Returns:
            PredictionResult with mean predictions and uncertainty
        """
        # For now, return deterministic prediction
        # TODO: Implement MC Dropout or ensemble for uncertainty
        result = self.predict(branch_input, trunk_input)

        # Placeholder uncertainty (to be implemented)
        result.uncertainty = np.zeros_like(result.values) + 0.1
        result.metadata["uncertainty_method"] = "placeholder"

        return result

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model (directory)
        """
        if self._model is None:
            raise RuntimeError("No model to save. Train the model first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model using DeepXDE
        self._model.save(str(path / "model"))

        # Save configuration
        import json
        config_path = path / "config.json"
        config_dict = self.config.to_dict()
        config_dict["input_dim"] = self._input_dim
        config_dict["coord_dim"] = self._coord_dim
        config_dict["num_sensors"] = self._num_sensors

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from (directory)
        """
        dde = _get_deepxde()
        path = Path(path)

        # Load configuration
        import json
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        self._input_dim = config_dict.get("input_dim")
        self._coord_dim = config_dict.get("coord_dim")
        self._num_sensors = config_dict.get("num_sensors")

        # Rebuild model architecture
        if all([self._input_dim, self._coord_dim, self._num_sensors]):
            self.build(self._input_dim, self._coord_dim, self._num_sensors)

        # Load weights
        # Note: DeepXDE model loading requires the model to be compiled first
        # This is a simplified version; full implementation would need
        # to recreate the data object as well
        self._is_trained = True


class DeepONetEnsemble(SurrogateModel):
    """
    Ensemble of DeepONet models for uncertainty quantification.

    Uses multiple independently trained models to estimate
    prediction uncertainty through variance of ensemble predictions.
    """

    def __init__(self, config: SurrogateConfig, n_models: int = 5):
        """
        Initialize DeepONet ensemble.

        Args:
            config: Model configuration
            n_models: Number of models in ensemble
        """
        super().__init__(config)
        self.n_models = n_models
        self._models: List[DeepONetSurrogate] = []

    def build(
        self,
        input_dim: int,
        coord_dim: int,
        num_sensors: int
    ) -> None:
        """Build ensemble of models."""
        self._models = []
        for i in range(self.n_models):
            model = DeepONetSurrogate(self.config)
            model.build(input_dim, coord_dim, num_sensors)
            self._models.append(model)

    def train(
        self,
        branch_inputs: np.ndarray,
        trunk_inputs: np.ndarray,
        outputs: np.ndarray,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """Train all models in ensemble with bootstrap sampling."""
        histories = []
        n_samples = branch_inputs.shape[0]

        for i, model in enumerate(self._models):
            print(f"Training ensemble model {i+1}/{self.n_models}")

            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            branch_boot = branch_inputs[indices]
            outputs_boot = outputs[indices]

            history = model.train(
                branch_boot,
                trunk_inputs,
                outputs_boot,
                validation_split
            )
            histories.append(history)

        self._is_trained = True

        return {
            "ensemble_histories": histories,
            "n_models": self.n_models,
        }

    def predict(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with uncertainty from ensemble.

        Returns mean prediction and standard deviation across ensemble.
        """
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained.")

        predictions = []
        for model in self._models:
            result = model.predict(branch_input, trunk_input)
            predictions.append(result.values)

        predictions = np.stack(predictions, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return PredictionResult(
            values=mean_pred,
            uncertainty=std_pred,
            coordinates=trunk_input,
            metadata={
                "n_ensemble": self.n_models,
                "uncertainty_method": "ensemble_std",
            }
        )

    def predict_with_epistemic_uncertainty(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with detailed epistemic uncertainty analysis.

        Epistemic uncertainty represents model uncertainty due to
        limited training data - this is what active learning targets.

        Returns:
            PredictionResult with detailed uncertainty metrics
        """
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained.")

        # Get predictions from each ensemble member
        predictions = []
        for model in self._models:
            result = model.predict(branch_input, trunk_input)
            predictions.append(result.values)

        predictions = np.stack(predictions, axis=0)  # (n_models, n_samples, n_points, dim)

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Epistemic uncertainty (model disagreement)
        epistemic_std = np.std(predictions, axis=0)

        # Variance decomposition
        # Total variance = mean of variances + variance of means
        # Epistemic = variance of means (what we reduce with more data)
        variance_of_means = np.var(predictions, axis=0)

        # Coefficient of variation (relative uncertainty)
        cv = epistemic_std / (np.abs(mean_pred) + 1e-10)

        # Prediction intervals (approximate 95% CI)
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)

        return PredictionResult(
            values=mean_pred,
            uncertainty=epistemic_std,
            coordinates=trunk_input,
            metadata={
                "n_ensemble": self.n_models,
                "uncertainty_method": "epistemic",
                "epistemic_variance": float(np.mean(variance_of_means)),
                "mean_cv": float(np.mean(cv)),
                "lower_95": lower_bound,
                "upper_95": upper_bound,
            }
        )

    def get_ensemble_predictions(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray
    ) -> np.ndarray:
        """
        Get raw predictions from all ensemble members.

        Useful for detailed uncertainty analysis or custom
        acquisition functions.

        Args:
            branch_input: Input parameters
            trunk_input: Query coordinates

        Returns:
            Array of shape (n_models, n_samples, n_points, dim)
        """
        if not self._is_trained:
            raise RuntimeError("Ensemble not trained.")

        predictions = []
        for model in self._models:
            result = model.predict(branch_input, trunk_input)
            predictions.append(result.values)

        return np.stack(predictions, axis=0)

    def compute_disagreement(
        self,
        branch_input: np.ndarray,
        trunk_input: np.ndarray,
        metric: str = "std"
    ) -> np.ndarray:
        """
        Compute ensemble disagreement (for Query-by-Committee).

        Args:
            branch_input: Input parameters
            trunk_input: Query coordinates
            metric: Disagreement metric ("std", "range", "entropy")

        Returns:
            Disagreement values per sample/point
        """
        predictions = self.get_ensemble_predictions(branch_input, trunk_input)

        if metric == "std":
            disagreement = np.std(predictions, axis=0)
        elif metric == "range":
            disagreement = np.ptp(predictions, axis=0)
        elif metric == "cv":
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            disagreement = std_pred / (np.abs(mean_pred) + 1e-10)
        else:
            disagreement = np.std(predictions, axis=0)

        return disagreement

    def save(self, path: Union[str, Path]) -> None:
        """Save all ensemble models."""
        path = Path(path)
        for i, model in enumerate(self._models):
            model.save(path / f"model_{i}")

    def load(self, path: Union[str, Path]) -> None:
        """Load all ensemble models."""
        path = Path(path)
        self._models = []
        for i in range(self.n_models):
            model = DeepONetSurrogate(self.config)
            model.load(path / f"model_{i}")
            self._models.append(model)
        self._is_trained = True
