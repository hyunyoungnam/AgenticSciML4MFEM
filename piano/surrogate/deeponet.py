"""
DeepONet surrogate model.

Learns the operator mapping: parameters -> field values via
    output(x) = sum_k  branch_k(params) * trunk_k(x)  +  bias

Branch encodes *what* (parameter dependence).
Trunk encodes *where* (spatial basis functions).

Best suited for fixed-geometry, small-dataset problems where Transolver
over-parameterises the output space.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .base import SurrogateModel, PredictionResult


@dataclass
class DeepONetConfig:
    """Configuration for DeepONet surrogate."""
    arch_type: str = "deeponet"
    hidden_dim: int = 64      # width of branch and trunk MLPs
    n_basis: int = 32         # number of shared basis functions (p)
    n_layers: int = 3         # depth of each MLP (branch and trunk)
    dropout: float = 0.0      # branch dropout (parameter encoding — keep low)
    trunk_dropout: float = 0.1  # trunk dropout (spatial basis — regularises oscillatory modes)
    learning_rate: float = 1e-3
    batch_size: int = 4
    epochs: int = 200
    patience: int = 50
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    activation: str = "gelu"
    output_dim: int = 1       # 1 for scalar (von Mises), 2 for displacement
    # Kept at 0 — PINO losses not applicable to DeepONet
    pino_weight: float = 0.0
    pino_eq_weight: float = 0.0
    tip_weight: float = 0.0
    checkpoint_dir: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _DeepONetNet(nn.Module):
    """
    Branch-trunk DeepONet network.

    For output_dim=D:
      branch: n_params -> n_basis * D   (D independent branch heads)
      trunk:  coord_dim -> n_basis      (shared spatial basis)
      output: einsum(branch, trunk) -> (B, N, D)
    """

    def __init__(self, n_params: int, coord_dim: int, config: DeepONetConfig):
        super().__init__()
        self.n_basis = config.n_basis
        self.output_dim = config.output_dim

        self.branch = _build_mlp(
            n_params, config.hidden_dim, config.n_basis * config.output_dim,
            config.n_layers, config.activation, config.dropout,
        )
        self.trunk = _build_mlp(
            coord_dim, config.hidden_dim, config.n_basis,
            config.n_layers, config.activation, config.trunk_dropout,
        )
        self.bias = nn.Parameter(torch.zeros(config.output_dim))

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        params: (B, n_params)
        coords: (B, N, coord_dim)  or  (N, coord_dim)
        returns: (B, N, output_dim)
        """
        B = params.shape[0]
        xy = coords[0] if coords.ndim == 3 else coords  # (N, coord_dim)

        b = self.branch(params)              # (B, n_basis * output_dim)
        t = self.trunk(xy)                   # (N, n_basis)

        # reshape branch to (B, output_dim, n_basis)
        b = b.view(B, self.output_dim, self.n_basis)
        # (B, D, P) x (N, P) -> (B, D, N) -> (B, N, D)
        out = torch.einsum("bdp,np->bdn", b, t).permute(0, 2, 1)
        return out + self.bias               # (B, N, output_dim)


class DeepONetModel(SurrogateModel, nn.Module):
    """SurrogateModel wrapper around _DeepONetNet."""

    def __init__(self, config: DeepONetConfig):
        nn.Module.__init__(self)
        SurrogateModel.__init__(self, config)
        self.config = config
        self._net: Optional[_DeepONetNet] = None
        self._input_dim: Optional[int] = None
        self._coord_dim: Optional[int] = None
        self._num_points: Optional[int] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, input_dim: int, coord_dim: int, num_points: int) -> None:
        self._input_dim = input_dim
        self._coord_dim = coord_dim
        self._num_points = num_points
        self._net = _DeepONetNet(input_dim, coord_dim, self.config)
        self.net = self._net  # register as submodule

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self._net(params, coords)

    def predict(self, params: np.ndarray, coords: np.ndarray) -> PredictionResult:
        if params.ndim == 1:
            params = params[np.newaxis, :]
        self.eval()
        params_t = torch.tensor(params, dtype=torch.float32, device=self._device)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self._device)
        if coords_t.ndim == 2:
            coords_t = coords_t.unsqueeze(0).expand(params_t.shape[0], -1, -1)
        with torch.no_grad():
            preds = self.forward(params_t, coords_t)
        values = preds.cpu().numpy()
        if values.shape[0] == 1:
            values = values[0]
        return PredictionResult(values=values, uncertainty=None, coordinates=coords)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "input_dim": self._input_dim,
            "coord_dim": self._coord_dim,
            "num_points": self._num_points,
            "is_trained": self._is_trained,
        }, path)

    def load(self, path: Union[str, Path]) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self._device)
        self.build(ckpt["input_dim"], ckpt["coord_dim"], ckpt["num_points"])
        self.load_state_dict(ckpt["state_dict"])
        self._is_trained = ckpt["is_trained"]
