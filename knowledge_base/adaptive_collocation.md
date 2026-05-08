# Adaptive Collocation Near Singularities

## Category
Physics loss / training strategy

## Problem
PINO collocation points (where PDE residuals are evaluated) are typically placed uniformly. Near a crack tip the residual magnitude is O(r^{-3/2}), which means uniform points miss the high-residual zone and the physics loss is dominated by far-field, low-residual regions where the network already performs well.

## Technique
Use non-uniform collocation: concentrate 60–80% of collocation points within a disk of radius r_inner around the crack tip. The remaining 20–40% cover the far field uniformly.

Optionally: use residual-adaptive refinement — after each training epoch, compute the PDE residual at candidate points and keep only the top-k highest-residual points for the next epoch.

## When to Apply
- PINO equilibrium loss is decreasing slowly despite active `pino_eq_weight`
- Physics loss is < 0.5% of data loss (too diluted by far-field low-residual points)
- Analyst reports near-tip zone still has high relative error after 5+ physics rounds

## Implementation Notes
- Combine with tip-weighted MSE for consistent near-tip emphasis
- `r_inner` ≈ 0.1 * crack_half_length is a practical starting point
- Do NOT use r_inner < mesh element size — subgrid resolution has no physics meaning
- Re-compute `(r, θ)` and Williams features at the adapted points

## Expected Impact
1.5–3× faster convergence of the elasticity equilibrium loss term.

## References
Karniadakis et al. (2021) Nature Reviews Physics; Gao et al. (2023) Adaptive PINN.
