"""
Surrogate model trainer.

Handles the training workflow for surrogate models including
data preparation, training, validation, and model checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import TransolverConfig, EnsembleConfig, SurrogateModel
from .transolver import TransolverModel
from .ensemble import EnsembleModel


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
    surrogate_config: TransolverConfig = field(default_factory=TransolverConfig)
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
            num_points = coords.shape[0]

            # Update config output dimension
            self.config.surrogate_config.output_dim = output_dim

            # Create model
            if self.config.use_ensemble:
                ensemble_config = EnsembleConfig(
                    n_members=self.config.n_ensemble,
                    member_config=self.config.surrogate_config
                )
                model = EnsembleModel(ensemble_config)
            else:
                model = TransolverModel(self.config.surrogate_config)

            model.build(n_params, coord_dim, num_points)

            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            # Convert data to tensors
            train_params_t = torch.tensor(train_params, dtype=torch.float32)
            train_outputs_t = torch.tensor(train_outputs, dtype=torch.float32)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

            test_params_t = torch.tensor(test_params, dtype=torch.float32, device=device)
            test_outputs_t = torch.tensor(test_outputs, dtype=torch.float32, device=device)

            # Create data loader
            dataset = TensorDataset(train_params_t, train_outputs_t)
            loader = DataLoader(
                dataset,
                batch_size=self.config.surrogate_config.batch_size,
                shuffle=True
            )

            # Setup optimizer and loss
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.surrogate_config.learning_rate
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20, verbose=False
            )
            criterion = nn.MSELoss()

            # Training loop
            history = {'train_loss': [], 'test_loss': []}
            best_test_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(self.config.surrogate_config.epochs):
                # Train epoch
                model.train()
                epoch_loss = 0.0

                for batch_params, batch_outputs in loader:
                    batch_params = batch_params.to(device)
                    batch_outputs = batch_outputs.to(device)

                    # Expand coords for batch
                    batch_coords = coords_t.unsqueeze(0).expand(
                        batch_params.shape[0], -1, -1
                    )

                    optimizer.zero_grad()
                    predictions = model.forward(batch_params, batch_coords)
                    loss = criterion(predictions, batch_outputs)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_params.shape[0]

                train_loss = epoch_loss / len(train_params)
                history['train_loss'].append(train_loss)

                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    test_coords = coords_t.unsqueeze(0).expand(
                        test_params_t.shape[0], -1, -1
                    )
                    test_preds = model.forward(test_params_t, test_coords)
                    test_loss = criterion(test_preds, test_outputs_t).item()
                    history['test_loss'].append(test_loss)

                scheduler.step(test_loss)

                # Early stopping
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config.surrogate_config.patience:
                    break

                # Callback
                if callback:
                    callback(epoch, train_loss)

            # Restore best model
            if best_state:
                model.load_state_dict(best_state)
                model.to(device)

            model._is_trained = True
            self._model = model

            # Save model if configured
            model_path = None
            if self.config.save_dir:
                model_path = self.config.save_dir / "surrogate_model.pt"
                model.save(model_path)

            # Compute final metrics
            model.eval()
            with torch.no_grad():
                test_coords = coords_t.unsqueeze(0).expand(
                    test_params_t.shape[0], -1, -1
                )
                final_preds = model.forward(test_params_t, test_coords)

            metrics = model.compute_error(
                final_preds.cpu().numpy(),
                test_outputs
            )

            return TrainingResult(
                success=True,
                train_loss=history['train_loss'][-1],
                test_loss=best_test_loss,
                metrics=metrics,
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
