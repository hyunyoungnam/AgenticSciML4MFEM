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

from .base import TransolverConfig, EnsembleConfig, SurrogateModel
from .transolver import TransolverModel
from .ensemble import EnsembleModel
from .pino_loss import PINOElasticityLoss


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
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray], Dict]:
        """
        Prepare data for training.

        Args:
            parameters:  Input parameters (N_samples, n_params)
            coordinates: Per-sample coordinates, each (N_i, coord_dim)
            outputs:     Per-sample outputs,     each (N_i, output_dim)

        Returns:
            (train_params, train_coords, test_params, test_coords,
             train_outputs, test_outputs, norm_params)
        """
        np.random.seed(self.config.random_seed)

        n_samples = parameters.shape[0]
        n_test = max(1, int(n_samples * self.config.train_test_split))
        n_train = n_samples - n_test

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_params  = parameters[train_idx]
        test_params   = parameters[test_idx]
        train_coords  = [coordinates[i] for i in train_idx]
        test_coords   = [coordinates[i] for i in test_idx]
        train_outputs = [outputs[i] for i in train_idx]
        test_outputs  = [outputs[i] for i in test_idx]

        if self.config.normalize_inputs:
            self._input_normalizer = Normalizer()
            train_params = self._input_normalizer.fit_transform(train_params)
            test_params  = self._input_normalizer.transform(test_params)

        if self.config.normalize_outputs:
            self._output_normalizer = Normalizer()
            # Fit on all training output values concatenated
            all_train = np.concatenate([o.reshape(-1, o.shape[-1]) for o in train_outputs], axis=0)
            self._output_normalizer.fit(all_train)
            train_outputs = [
                self._output_normalizer.transform(o.reshape(-1, o.shape[-1])).reshape(o.shape)
                for o in train_outputs
            ]
            test_outputs = [
                self._output_normalizer.transform(o.reshape(-1, o.shape[-1])).reshape(o.shape)
                for o in test_outputs
            ]

        return (
            train_params,
            train_coords,
            test_params,
            test_coords,
            train_outputs,
            test_outputs,
            self._get_normalization_params(),
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
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingResult:
        """
        Train surrogate model with per-sample coordinates.

        Each sample may have a different number of mesh nodes (N_i). Gradients
        are accumulated over `batch_size` samples before each optimizer step,
        which is equivalent to mini-batch training without requiring a fixed N.

        Args:
            parameters:  Input parameters (N_samples, n_params)
            coordinates: Per-sample node coords, each (N_i, coord_dim)
            outputs:     Per-sample field values, each (N_i, output_dim)
            callback:    Optional callback(epoch, train_loss)

        Returns:
            TrainingResult with training metrics and model path
        """
        try:
            (
                train_params,
                train_coords,
                test_params,
                test_coords,
                train_outputs,
                test_outputs,
                norm_params,
            ) = self.prepare_data(parameters, coordinates, outputs)

            n_train = len(train_params)
            n_test  = len(test_params)

            # Dimensions from first sample (coord_dim is fixed; num_points is metadata only)
            n_params  = train_params.shape[1]
            coord_dim = train_coords[0].shape[1]
            output_dim = train_outputs[0].shape[-1]
            num_points = train_coords[0].shape[0]

            self.config.surrogate_config.output_dim = output_dim

            if self.config.use_ensemble:
                ensemble_config = EnsembleConfig(
                    n_members=self.config.n_ensemble,
                    member_config=self.config.surrogate_config,
                )
                model = EnsembleModel(ensemble_config)
            else:
                model = TransolverModel(self.config.surrogate_config)

            model.build(n_params, coord_dim, num_points)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            batch_size = self.config.surrogate_config.batch_size
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.surrogate_config.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20
            )
            criterion = nn.MSELoss()

            cfg = self.config.surrogate_config
            use_pino = (
                output_dim >= 2
                and coord_dim == 2
                and (cfg.pino_weight > 0 or cfg.pino_eq_weight > 0)
            )
            pino_fn = (
                PINOElasticityLoss(
                    E=cfg.pino_E,
                    nu=cfg.pino_nu,
                    eq_weight=cfg.pino_eq_weight,
                    energy_weight=cfg.pino_weight,
                ).to(device)
                if use_pino
                else None
            )

            history = {'train_loss': [], 'test_loss': [], 'pino_loss': []}
            best_test_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(self.config.surrogate_config.epochs):
                model.train()
                epoch_loss = 0.0
                epoch_pino_loss = 0.0
                indices = np.random.permutation(n_train)

                optimizer.zero_grad()
                accum = 0

                for idx in indices:
                    params_t = torch.tensor(
                        train_params[idx:idx+1], dtype=torch.float32, device=device
                    )
                    coords_t = torch.tensor(
                        train_coords[idx], dtype=torch.float32, device=device
                    ).unsqueeze(0)  # (1, N_i, coord_dim)
                    output_t = torch.tensor(
                        train_outputs[idx], dtype=torch.float32, device=device
                    ).unsqueeze(0)  # (1, N_i, output_dim)

                    pred = model.forward(params_t, coords_t)
                    data_loss = criterion(pred, output_t) / batch_size
                    if pino_fn is not None:
                        physics_loss = pino_fn(pred[0], output_t[0], coords_t[0]) / batch_size
                    else:
                        physics_loss = torch.tensor(0.0, device=device)
                    loss = data_loss + physics_loss
                    loss.backward()
                    epoch_loss += data_loss.item() * batch_size
                    epoch_pino_loss += physics_loss.item() * batch_size
                    accum += 1

                    if accum >= batch_size:
                        optimizer.step()
                        optimizer.zero_grad()
                        accum = 0

                if accum > 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss = epoch_loss / n_train
                history['train_loss'].append(train_loss)

                # Test evaluation
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for idx in range(n_test):
                        params_t = torch.tensor(
                            test_params[idx:idx+1], dtype=torch.float32, device=device
                        )
                        coords_t = torch.tensor(
                            test_coords[idx], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        output_t = torch.tensor(
                            test_outputs[idx], dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        pred = model.forward(params_t, coords_t)
                        test_loss += criterion(pred, output_t).item() / n_test
                history['test_loss'].append(test_loss)
                history['pino_loss'].append(epoch_pino_loss / n_train)

                scheduler.step(test_loss)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config.surrogate_config.patience:
                    break

                if callback:
                    callback(epoch, train_loss)

            if best_state:
                model.load_state_dict(best_state)
                model.to(device)

            model._is_trained = True
            self._model = model

            model_path = None
            if self.config.save_dir:
                model_path = self.config.save_dir / "surrogate_model.pt"
                model.save(model_path)

            # Final metrics on test set (denormalized)
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for idx in range(n_test):
                    params_t = torch.tensor(
                        test_params[idx:idx+1], dtype=torch.float32, device=device
                    )
                    coords_t = torch.tensor(
                        test_coords[idx], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    pred = model.forward(params_t, coords_t)
                    all_preds.append(pred.cpu().numpy().flatten())
                    all_targets.append(test_outputs[idx].flatten())

            metrics = model.compute_error(
                np.concatenate(all_preds),
                np.concatenate(all_targets),
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

    def predict_with_uncertainty(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return (mean, uncertainty) both in original (denormalized) scale.

        Args:
            parameters:  Input parameters (N_samples, n_params) or (n_params,)
            coordinates: Query coordinates (num_points, coord_dim)

        Returns:
            Tuple of (mean, uncertainty), each (num_points, output_dim)
            uncertainty is None if model does not support it.
        """
        if self._model is None or not self._model.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if self._input_normalizer:
            parameters = self._input_normalizer.transform(parameters)

        result = self._model.predict(parameters, coordinates)
        mean = result.values
        unc = result.uncertainty

        if self._output_normalizer:
            output_dim = mean.shape[-1] if mean.ndim > 1 else 1
            mean = self._output_normalizer.inverse_transform(
                mean.reshape(-1, output_dim)
            ).reshape(mean.shape)
            if unc is not None:
                unc = (
                    unc.reshape(-1, output_dim) * self._output_normalizer.std
                ).reshape(unc.shape)

        return mean, unc

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
