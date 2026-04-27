"""
Fracture mechanics PINO loss for crack / V-notch problems.

Four complementary physics-informed loss terms:

1. K_I Consistency
   Extracts the stress intensity factor from the predicted displacement field
   via least-squares Williams correlation and enforces it equals the input K_I.

2. Crack Face Traction-Free BC
   Enforces σ_yy = σ_xy = 0 on elements adjacent to the crack face
   (y ≈ tip_y, x < tip_x).

3. Williams Asymptotic Residual
   Enforces that the near-tip displacement field (r < r_williams) matches the
   Mode I Williams expansion exactly.

4. J-Integral Conservation
   Uses the domain form of the J-integral and enforces J = K_I²/E (plane stress).

All terms operate in **physical (SI) units**. The caller (trainer) is responsible
for denormalizing predicted displacements before passing them here.

Reference:
  Williams (1957), "On the stress distribution at the base of a stationary crack"
  Li et al. (2021), "Physics-Informed Neural Operator for Learning PDEs", ICLR 2024
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay

from .pino_loss import _compute_B_matrices


class CrackFractureLoss(nn.Module):
    """
    PINO loss for Mode I fracture mechanics problems.

    Assumes:
    - Dominant Mode I loading (K_II ≈ 0)
    - Plane stress conditions
    - Crack/notch bisector along y = tip_y, extending from x = 0 to x = tip_x
    - Crack propagation direction: +x

    Args:
        tip_x:          Crack tip x-coordinate
        tip_y:          Crack tip y-coordinate
        r_ki_min:       Inner radius of K_I extraction annulus
        r_ki_max:       Outer radius of K_I extraction annulus
        r_williams:     Radius of Williams matching zone (r < r_williams)
        r_j:            Outer radius of J-integral domain
        crack_face_tol: Half-width tolerance for detecting crack-face elements
        ki_weight:      Weight for K_I consistency loss
        bc_weight:      Weight for crack face BC loss
        williams_weight: Weight for Williams asymptotic loss
        j_weight:       Weight for J-integral conservation loss
    """

    def __init__(
        self,
        tip_x: float,
        tip_y: float,
        r_ki_min: float = 0.02,
        r_ki_max: float = 0.10,
        r_williams: float = 0.05,
        r_j: float = 0.15,
        crack_face_tol: float = 0.02,
        ki_weight: float = 1.0,
        bc_weight: float = 1.0,
        williams_weight: float = 1.0,
        j_weight: float = 1.0,
    ):
        super().__init__()
        self.tip_x = tip_x
        self.tip_y = tip_y
        self.r_ki_min = r_ki_min
        self.r_ki_max = r_ki_max
        self.r_williams = r_williams
        self.r_j = r_j
        self.crack_face_tol = crack_face_tol
        self.ki_weight = ki_weight
        self.bc_weight = bc_weight
        self.williams_weight = williams_weight
        self.j_weight = j_weight

        # Mesh topology cache — computed once per unique coordinate set
        self._coords_hash: int = -1
        self._elems: torch.Tensor = None
        self._B: torch.Tensor = None
        self._areas: torch.Tensor = None
        self._centroids: torch.Tensor = None

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------

    def _ensure_mesh_cache(self, coords: torch.Tensor, device: torch.device) -> None:
        """Triangulate coords and cache B-matrices (runs once per unique mesh)."""
        coords_np = coords.detach().cpu().numpy()
        h = hash(coords_np.tobytes())

        if h == self._coords_hash:
            # Move cached tensors to current device if needed
            self._elems = self._elems.to(device)
            self._B = self._B.to(device)
            self._areas = self._areas.to(device)
            self._centroids = self._centroids.to(device)
            return

        tri = Delaunay(coords_np)
        elems_np = tri.simplices  # (M, 3)

        xy = torch.tensor(coords_np[elems_np], dtype=torch.float32, device=device)
        B, areas = _compute_B_matrices(xy)

        self._elems = torch.tensor(elems_np, dtype=torch.long, device=device)
        self._B = B
        self._areas = areas
        self._centroids = xy.mean(dim=1)  # (M, 2)
        self._coords_hash = h

    def _build_C(self, E: float, nu: float, device: torch.device) -> torch.Tensor:
        """Plane-stress constitutive matrix C (3×3, Voigt notation)."""
        factor = E / (1.0 - nu ** 2)
        C = torch.tensor(
            [
                [1.0,  nu,             0.0],
                [nu,   1.0,            0.0],
                [0.0,  0.0, (1.0 - nu) / 2.0],
            ],
            dtype=torch.float32,
            device=device,
        ) * factor
        return C

    @staticmethod
    def _kappa(nu: float) -> float:
        """Kolosov constant for plane stress."""
        return (3.0 - nu) / (1.0 + nu)

    # ------------------------------------------------------------------
    # Williams expansion
    # ------------------------------------------------------------------

    def _williams_displacement(
        self,
        coords: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Mode I Williams displacement field at every mesh node.

        u_x = K_I/(2μ) * sqrt(r/2π) * cos(θ/2) * (κ - 1 + 2sin²(θ/2))
        u_y = K_I/(2μ) * sqrt(r/2π) * sin(θ/2) * (κ + 1 - 2cos²(θ/2))

        Returns: (N, 2)
        """
        mu = E / (2.0 * (1.0 + nu))
        kappa = self._kappa(nu)

        dx = coords[:, 0] - self.tip_x
        dy = coords[:, 1] - self.tip_y
        r = (dx ** 2 + dy ** 2).sqrt().clamp(min=1e-12)
        theta = torch.atan2(dy, dx)

        sqrt_r_2pi = (r / (2.0 * torch.pi)).sqrt()
        cos_h = torch.cos(theta / 2.0)
        sin_h = torch.sin(theta / 2.0)

        coeff = K_I / (2.0 * mu) * sqrt_r_2pi
        u_x = coeff * cos_h * (kappa - 1.0 + 2.0 * sin_h ** 2)
        u_y = coeff * sin_h * (kappa + 1.0 - 2.0 * cos_h ** 2)

        return torch.stack([u_x, u_y], dim=-1)  # (N, 2)

    # ------------------------------------------------------------------
    # Term 1: K_I Consistency
    # ------------------------------------------------------------------

    def _ki_consistency(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        coords: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Least-squares K_I extraction from u_y in annulus [r_ki_min, r_ki_max].

        K_I_fit = 2μ * Σ(u_y_i * f_y_i) / Σ(f_y_i²)
        where f_y(r,θ) = sqrt(r/2π) * sin(θ/2) * (κ+1 - 2cos²(θ/2))

        Loss: ((K_I_fit - K_I_input) / |K_I_input|)²
        """
        mu = E / (2.0 * (1.0 + nu))
        kappa = self._kappa(nu)

        dx = coords[:, 0] - self.tip_x
        dy = coords[:, 1] - self.tip_y
        r = (dx ** 2 + dy ** 2).sqrt()

        mask = (r >= self.r_ki_min) & (r <= self.r_ki_max)
        if mask.sum() < 3:
            return torch.tensor(0.0, device=u_pred.device)

        r_m = r[mask].clamp(min=1e-12)
        theta = torch.atan2(dy[mask], dx[mask])

        sqrt_r_2pi = (r_m / (2.0 * torch.pi)).sqrt()
        sin_h = torch.sin(theta / 2.0)
        cos_h = torch.cos(theta / 2.0)

        # Williams basis for u_y (Mode I)
        f_y = sqrt_r_2pi * sin_h * (kappa + 1.0 - 2.0 * cos_h ** 2)

        u_y = u_pred[mask, 1]
        denom = (f_y ** 2).sum().clamp(min=1e-30)
        K_I_fit = 2.0 * mu * (u_y * f_y).sum() / denom

        return ((K_I_fit - K_I) / (abs(K_I) + 1e-10)) ** 2

    # ------------------------------------------------------------------
    # Term 2: Crack Face Traction-Free BC
    # ------------------------------------------------------------------

    def _crack_face_bc(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        E: float,
        nu: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        σ_yy = 0 and σ_xy = 0 on crack-face elements.

        Crack face: centroid satisfies |y_c - tip_y| < tol, x_c < tip_x.
        Loss: Σ A_e(σ_yy² + σ_xy²) / (Σ A_e * E²)   [dimensionless]
        """
        cx = self._centroids[:, 0]
        cy = self._centroids[:, 1]
        mask = (torch.abs(cy - self.tip_y) < self.crack_face_tol) & (
            cx < self.tip_x - self.crack_face_tol
        )

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        C = self._build_C(E, nu, device)

        u_elem = u_pred[self._elems[mask]].reshape(-1, 6)
        eps = torch.einsum("mij,mj->mi", self._B[mask], u_elem)  # (M_cf, 3)
        sig = torch.einsum("ij,mj->mi", C, eps)                  # (M_cf, 3)

        a_cf = self._areas[mask]
        total_a = a_cf.sum().clamp(min=1e-30)
        L_bc = (a_cf * (sig[:, 1] ** 2 + sig[:, 2] ** 2)).sum() / (
            total_a * E ** 2
        )
        return L_bc

    # ------------------------------------------------------------------
    # Term 3: Williams Asymptotic Residual
    # ------------------------------------------------------------------

    def _williams_residual(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        coords: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Near-tip (r < r_williams) displacement must match the Williams expansion.

        Loss: mean||u_pred - u_williams||² / mean||u_williams||²
        """
        dx = coords[:, 0] - self.tip_x
        dy = coords[:, 1] - self.tip_y
        r = (dx ** 2 + dy ** 2).sqrt()
        mask = r < self.r_williams

        if mask.sum() == 0:
            return torch.tensor(0.0, device=u_pred.device)

        # Williams GT — detach so we don't backprop through the formula itself
        u_wil = self._williams_displacement(coords, K_I, E, nu).detach()

        u_pred_nt = u_pred[mask]
        u_wil_nt = u_wil[mask]

        residual_sq = ((u_pred_nt - u_wil_nt) ** 2).mean()
        normalization = (u_wil_nt ** 2).mean().clamp(min=1e-30)

        return residual_sq / normalization

    # ------------------------------------------------------------------
    # Term 4: J-Integral Conservation
    # ------------------------------------------------------------------

    def _j_integral(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Domain form of J-integral: J = ∫_Ω [σ_ij ∂u_i/∂x₁ - W δ₁j] ∂q/∂x_j dΩ

        Hat function: q(r) = 1 - r/r_j  (q=1 at tip, q=0 at r=r_j)
        Reference:    J_ref = K_I²/E  (plane stress)
        Loss:         ((J - J_ref) / |J_ref|)²
        """
        cx = self._centroids[:, 0]
        cy = self._centroids[:, 1]
        dx_c = cx - self.tip_x
        dy_c = cy - self.tip_y
        r_c = (dx_c ** 2 + dy_c ** 2).sqrt().clamp(min=1e-12)

        in_domain = r_c < self.r_j
        if in_domain.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Gradient of hat function q(r) = 1 - r/r_j at element centroids
        # ∂q/∂x = -1/r_j * (x_c - tip_x) / r_c   (zero outside domain)
        dq_dx = torch.where(in_domain, -dx_c / (self.r_j * r_c), torch.zeros_like(dx_c))
        dq_dy = torch.where(in_domain, -dy_c / (self.r_j * r_c), torch.zeros_like(dy_c))

        C = self._build_C(E, nu, device)

        # All-element strain, stress, and strain energy
        u_elem = u_pred[self._elems].reshape(-1, 6)          # (M, 6)
        eps = torch.einsum("mij,mj->mi", self._B, u_elem)    # (M, 3): [ε_xx, ε_yy, γ_xy]
        sig = torch.einsum("ij,mj->mi", C, eps)              # (M, 3): [σ_xx, σ_yy, σ_xy]
        W = 0.5 * (eps * sig).sum(dim=-1)                    # (M,)  strain energy density

        # Displacement gradients ∂u_x/∂x and ∂u_y/∂x
        # B[:,0,0::2] = [∂N₁/∂x, ∂N₂/∂x, ∂N₃/∂x]  (from row 0 of B, even cols)
        dN_dx = self._B[:, 0, 0::2]                  # (M, 3)
        u_x_nodes = u_elem[:, 0::2]                  # (M, 3)  x-displacements at nodes
        u_y_nodes = u_elem[:, 1::2]                  # (M, 3)  y-displacements at nodes

        du_x_dx = (dN_dx * u_x_nodes).sum(-1)        # (M,)  ∂u_x/∂x
        du_y_dx = (dN_dx * u_y_nodes).sum(-1)        # (M,)  ∂u_y/∂x

        sig_xx, sig_yy, sig_xy = sig[:, 0], sig[:, 1], sig[:, 2]

        # Integrand per element (domain form, x₁ = x):
        # f_x = (σ_xx ∂u_x/∂x + σ_xy ∂u_y/∂x - W) * ∂q/∂x
        # f_y = (σ_xy ∂u_x/∂x + σ_yy ∂u_y/∂x)     * ∂q/∂y
        integrand = (
            (sig_xx * du_x_dx + sig_xy * du_y_dx - W) * dq_dx
            + (sig_xy * du_x_dx + sig_yy * du_y_dx) * dq_dy
        )

        J = (self._areas * integrand).sum()

        J_ref = K_I ** 2 / E
        return ((J - J_ref) / (abs(J_ref) + 1e-10)) ** 2

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        u_pred: torch.Tensor,  # (N, 2)  physical displacement
        coords: torch.Tensor,  # (N, 2)  mesh coordinates
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Compute combined fracture mechanics physics loss.

        Args:
            u_pred:  Predicted displacement field in physical units (N, 2)
            coords:  Mesh node coordinates (N, 2)
            K_I:     Mode I stress intensity factor [Pa√m]
            E:       Young's modulus [Pa]
            nu:      Poisson's ratio

        Returns:
            Scalar loss (dimensionless, weighted sum of all active terms)
        """
        device = u_pred.device
        self._ensure_mesh_cache(coords, device)

        total = torch.tensor(0.0, device=device)

        if self.ki_weight > 0.0:
            total = total + self.ki_weight * self._ki_consistency(
                u_pred, coords, K_I, E, nu
            )

        if self.bc_weight > 0.0:
            total = total + self.bc_weight * self._crack_face_bc(
                u_pred, E, nu, device
            )

        if self.williams_weight > 0.0:
            total = total + self.williams_weight * self._williams_residual(
                u_pred, coords, K_I, E, nu
            )

        if self.j_weight > 0.0:
            total = total + self.j_weight * self._j_integral(
                u_pred, K_I, E, nu, device
            )

        return total
