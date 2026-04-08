"""
Ensemble model for uncertainty quantification.

Uses an ensemble of Transolver models to estimate prediction uncertainty
via disagreement among ensemble members.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .base import SurrogateModel, EnsembleConfig, PredictionResult, TransolverConfig
from .transolver import TransolverModel


class EnsembleModel(SurrogateModel, nn.Module):
    """
    Ensemble of Transolver models for uncertainty quantification.

    Trains multiple independent models and uses their disagreement
    (standard deviation of predictions) as an uncertainty estimate.
    This is required by active learning for acquisition functions.
    """

    def __init__(self, config: EnsembleConfig):
        """
        Initialize ensemble model.

        Args:
            config: Ensemble configuration
        """
        nn.Module.__init__(self)
        # Use member_config for base class
        SurrogateModel.__init__(self, config.member_config)
        self.ensemble_config = config

        self._models: List[TransolverModel] = []
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self, input_dim: int, coord_dim: int, num_points: int) -> None:
        """
        Build all ensemble members.

        Args:
            input_dim: Dimension of input parameters
            coord_dim: Dimension of coordinates
            num_points: Number of mesh points
        """
        self._models = []

        for i in range(self.ensemble_config.n_members):
            # Create member with slightly different initialization
            member_config = TransolverConfig(
                slice_num=self.ensemble_config.member_config.slice_num,
                n_heads=self.ensemble_config.member_config.n_heads,
                d_model=self.ensemble_config.member_config.d_model,
                n_layers=self.ensemble_config.member_config.n_layers,
                mlp_ratio=self.ensemble_config.member_config.mlp_ratio,
                dropout=self.ensemble_config.member_config.dropout,
                learning_rate=self.ensemble_config.member_config.learning_rate,
                batch_size=self.ensemble_config.member_config.batch_size,
                epochs=self.ensemble_config.member_config.epochs,
                patience=self.ensemble_config.member_config.patience,
                output_dim=self.ensemble_config.member_config.output_dim,
                checkpoint_dir=self.ensemble_config.member_config.checkpoint_dir,
            )

            model = TransolverModel(member_config)
            model.build(input_dim, coord_dim, num_points)
            self._models.append(model)

        # Register as modules for proper parameter tracking
        self.members = nn.ModuleList(self._models)
        self.to(self._device)

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass returning mean of ensemble predictions.

        Args:
            params: Input parameters (B, n_params)
            coords: Mesh coordinates (B, N, coord_dim)

        Returns:
            Mean predictions (B, N, output_dim)
        """
        predictions = torch.stack([
            model.forward(params, coords) for model in self._models
        ], dim=0)  # (M, B, N, output_dim)

        return predictions.mean(dim=0)

    def forward_with_uncertainty(
        self,
        params: torch.Tensor,
        coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and std of ensemble predictions.

        Args:
            params: Input parameters (B, n_params)
            coords: Mesh coordinates (B, N, coord_dim)

        Returns:
            Tuple of (mean, std) predictions, each (B, N, output_dim)
        """
        predictions = torch.stack([
            model.forward(params, coords) for model in self._models
        ], dim=0)  # (M, B, N, output_dim)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std

    def predict(
        self,
        params: np.ndarray,
        coords: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with uncertainty estimates.

        Args:
            params: Input parameters (N_samples, n_params) or (n_params,)
            coords: Query coordinates (num_points, coord_dim)

        Returns:
            PredictionResult with mean predictions and uncertainty
        """
        self.eval()

        # Handle single sample
        if params.ndim == 1:
            params = params[np.newaxis, :]

        # Convert to tensors
        params_t = torch.tensor(params, dtype=torch.float32, device=self._device)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self._device)

        # Expand coords for batch: (N, coord_dim) -> (B, N, coord_dim)
        if coords_t.ndim == 2:
            coords_t = coords_t.unsqueeze(0).expand(params_t.shape[0], -1, -1)

        with torch.no_grad():
            mean, std = self.forward_with_uncertainty(params_t, coords_t)

        values = mean.cpu().numpy()
        uncertainty = std.cpu().numpy()

        # Squeeze if single sample
        if values.shape[0] == 1:
            values = values[0]
            uncertainty = uncertainty[0]

        return PredictionResult(
            values=values,
            uncertainty=uncertainty,
            coordinates=coords,
        )

    def get_member(self, idx: int) -> TransolverModel:
        """Get a specific ensemble member."""
        return self._models[idx]

    @property
    def n_members(self) -> int:
        """Number of ensemble members."""
        return len(self._models)

    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'ensemble_config': {
                'n_members': self.ensemble_config.n_members,
                'member_config': self.ensemble_config.member_config.to_dict(),
            },
            'state_dict': self.state_dict(),
            'is_trained': self._is_trained,
        }

        # Save member-specific info from first model
        if self._models:
            checkpoint['input_dim'] = self._models[0]._input_dim
            checkpoint['coord_dim'] = self._models[0]._coord_dim
            checkpoint['num_points'] = self._models[0]._num_points

        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load ensemble from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device)

        # Reconstruct config
        member_dict = checkpoint['ensemble_config']['member_config']
        member_config = TransolverConfig(
            slice_num=member_dict['slice_num'],
            n_heads=member_dict['n_heads'],
            d_model=member_dict['d_model'],
            n_layers=member_dict['n_layers'],
            mlp_ratio=member_dict['mlp_ratio'],
            dropout=member_dict['dropout'],
            learning_rate=member_dict['learning_rate'],
            batch_size=member_dict['batch_size'],
            epochs=member_dict['epochs'],
            patience=member_dict['patience'],
            output_dim=member_dict['output_dim'],
        )
        self.ensemble_config = EnsembleConfig(
            n_members=checkpoint['ensemble_config']['n_members'],
            member_config=member_config,
        )

        # Rebuild model
        self.build(
            checkpoint['input_dim'],
            checkpoint['coord_dim'],
            checkpoint['num_points']
        )

        self.load_state_dict(checkpoint['state_dict'])
        self._is_trained = checkpoint['is_trained']
